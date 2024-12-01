import numpy as np
import torch
import pandas as pd
from typing import Callable, Dict, Optional
import json
import os
from datetime import datetime
from pathlib import Path
# from src.models.video import AuVi2LSTMModel
from src.train import optimize, to_device, accuracy
from src.helpers import train_test_split, load_model,device
from src.data import create_dataloaders



def create_experiment_dir(base_suffix: str) -> Path:
    """
    Create a directory for the experiment with timestamp.
    
    Args:
        base_suffix: Base suffix for the experiment name
        
    Returns:
        Path object pointing to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/{timestamp}_{base_suffix}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    (exp_dir / "fold_predictions").mkdir(exist_ok=True)
    (exp_dir / "model_checkpoints").mkdir(exist_ok=True)
    
    return exp_dir

def save_experiment_config(exp_dir: Path, **kwargs) -> None:
    """
    Save experiment configuration to a JSON file.
    
    Args:
        exp_dir: Directory to save the configuration
        kwargs: Configuration parameters to save
    """
    config = {
        k: str(v) if isinstance(v, (Path, type)) else v
        for k, v in kwargs.items()
        if  not isinstance(v, (pd.DataFrame, torch.optim.Optimizer))
    }
    
    with open(exp_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

def evaluate_model_with_predictions(model, dataloader, class_counts):
    """
    Evaluate model and return accuracies along with predictions and ground truths.
    
    Returns:
        tuple: (unweighted_acc, weighted_acc, predictions, ground_truths)
    """
    model.eval()
    total_unweighted = 0
    total_weighted = 0
    n_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = to_device(inputs, device)
            targets = to_device(targets, device)
            outputs = model(inputs)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            correct = (preds == targets).float()
            
            # Unweighted accuracy
            unweighted_acc = correct.mean().item()
            
            # Weighted accuracy
            class_accuracies = []
            for cls in range(len(class_counts)):
                mask = targets == cls
                if mask.sum() > 0:
                    class_accuracies.append(correct[mask].mean().item())
            weighted_acc = np.mean(class_accuracies) if class_accuracies else 0.0
            
            total_unweighted += unweighted_acc
            total_weighted += weighted_acc
            n_batches += 1
    
    return (
        total_unweighted / n_batches, 
        total_weighted / n_batches,
        np.array(all_preds),
        np.array(all_targets)
    )

def save_fold_predictions(exp_dir, fold, modality, predictions, ground_truths, valid_df):
    """
    Save predictions and ground truths for a specific fold and modality.
    
    Args:
        exp_dir: Path to experiment directory
        fold: Fold number
        modality: String indicating the modality (visual/audio/combined)
        predictions: Array of model predictions
        ground_truths: Array of ground truth labels
        valid_df: Validation DataFrame with video_ids
    """
    predictions_df = pd.DataFrame({
        'video_id': valid_df['video_id'].values,
        'ground_truth': ground_truths,
        'prediction': predictions
    })
    
    output_path = exp_dir / "fold_predictions" / f"{modality}_fold{fold}_predictions.csv"
    predictions_df.to_csv(output_path, index=False)

def sequential_cross_validate(
    df: pd.DataFrame,
    model_class: Callable,
    n_splits: int = 5,
    valid_size: float = 0.2,
    batch_size: int = 16,
    num_epochs_visual: int = 30,
    num_epochs_audio: int = 30,
    num_epochs_combined: int = 10,
    num_workers: int = 8,
    learning_rate: float = 0.001,
    weight_decay: float = 0.004,
    accumulation_steps: int = 4,
    class_weight_power: float = 0.3,
    label_smoothing: float = 0.03,
    seed: int = 42,
    base_suffix: str = "sequential",
):
    exp_dir = create_experiment_dir(base_suffix)
    checkpoints_dir = str(exp_dir / "model_checkpoints")
    s_epoch=1
    
    # Initialize accuracy tracking
    results = {
        'visual': {'unweighted': [], 'weighted': []},
        'audio': {'unweighted': [], 'weighted': []},
        'combined': {'unweighted': [], 'weighted': []}
    }
    
    # Save experiment configuration
    save_experiment_config(
        exp_dir,
        n_splits=n_splits,
        valid_size=valid_size,
        batch_size=batch_size,
        num_epochs_visual=num_epochs_visual,
        num_epochs_audio=num_epochs_audio,
        num_epochs_combined=num_epochs_combined,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    modality_dim = {"audio":384, "visual":384 }
    for fold in range(n_splits):
        current_seed = seed + fold
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        
        # Split data
        train_fold, valid_fold = train_test_split(
            df, test_size=valid_size,
            stratify=df['stratify_on'],
            random_state=current_seed
        )
        
        # Calculate class weights
        stratify_on = train_fold['emotion'].to_numpy()
        unique_classes, class_counts = np.unique(stratify_on, return_counts=True)
        n_samples = len(stratify_on)
        class_weights = {}
        for cls, count in zip(unique_classes, class_counts):
            class_weights[cls] = n_samples / (len(unique_classes) * count) ** 0.5
        sample_weights = [class_weights[t]/sum(class_weights.values()) for t in stratify_on]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(train_fold), replacement=True)

        class_weights = torch.tensor(
            [len(df)/(class_counts[i]*len(class_counts))**class_weight_power 
             for i in range(len(class_counts))],
            dtype=torch.float32,
            device='cuda'
        )
        class_weights /= class_weights.sum()
        
        def loss_fn(output, target):
            return torch.nn.functional.cross_entropy(
                output,
                target,
                label_smoothing=label_smoothing,
                weight=class_weights
            )
        
        # --- Stage 1: Visual-only training ---
        print("\nTraining Visual-only model")
        model = model_class(num_classes=len(class_weights), hidden_sizes=modality_dim,mode="visual").cuda()
        
        data_loaders = create_dataloaders(
            df=train_fold,
            batch_size=batch_size,
            valid_df=valid_fold,
            num_workers=num_workers,
            mode='visual',
            sampler=sampler
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=1, threshold=1e-3, verbose=True)
        step = lambda loss, epoch=None: scheduler.step(loss)
        
        model_name = f"visual_fold{fold}"
        optimize(
            data_loaders=data_loaders,
            model=model,
            optimizer=optimizer,
            loss=loss_fn,
            step=step,
            s_epoch=s_epoch,
            n_epochs=num_epochs_visual,
            model_name=model_name,
            accumulation_steps=accumulation_steps,
            checkpoints_dir=checkpoints_dir
        )
        
        load_model(model_name=model_name, model=model, checkpoints_dir=checkpoints_dir, loading_order=['best', 'last'])
        visual_backbone_state = {
            # 'rnn': model.v_rnn.state_dict(),
            'classifier': model.classifier.weight.data,
            'classifier_bias': model.classifier.bias.data
        }
        
        # Evaluate and save predictions
        unweighted_acc, weighted_acc, predictions, ground_truths = evaluate_model_with_predictions(
            model, data_loaders['valid'], class_counts
        )
        save_fold_predictions(exp_dir, fold, 'visual', predictions, ground_truths, valid_fold)
        results['visual']['unweighted'].append(unweighted_acc)
        results['visual']['weighted'].append(weighted_acc)
        
        del model, optimizer
        torch.cuda.empty_cache()
        
        # --- Stage 2: Audio-only training ---
        print("\nTraining Audio-only model")
        model = model_class(num_classes=len(class_weights), hidden_sizes=modality_dim, mode="audio").cuda()
        
        data_loaders = create_dataloaders(
            df=train_fold,
            batch_size=batch_size,
            valid_df=valid_fold,
            num_workers=num_workers,
            mode='audio',
            sampler=sampler
        )
        
        optimizer = torch.optim.AdamW([
            {'params': model.a_rnn.parameters(), 'lr': 0.0001},       # Audio RNN
            {'params': model.classifier.parameters(), 'lr': 0.001}  # Classifier
        ], weight_decay=1e-4) 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=1, threshold=1e-3, verbose=True)
        step = lambda loss, epoch=None: scheduler.step(loss)
        
        model_name = f"audio_fold{fold}"
        optimize(
            data_loaders=data_loaders,
            model=model,
            optimizer=optimizer,
            loss=loss_fn,
            step=step,
            s_epoch=s_epoch,
            n_epochs=num_epochs_audio,
            model_name=model_name,
            accumulation_steps=accumulation_steps,
            checkpoints_dir=checkpoints_dir
        )
        
        load_model(model_name=model_name, model=model, checkpoints_dir=checkpoints_dir, loading_order=['best', 'last'])
        audio_backbone_state = {
            'rnn': model.a_rnn.state_dict(),
            'classifier': model.classifier.weight.data,
            'classifier_bias': model.classifier.bias.data
        }
        
        # Evaluate and save predictions
        unweighted_acc, weighted_acc, predictions, ground_truths = evaluate_model_with_predictions(
            model, data_loaders['valid'], class_counts
        )
        save_fold_predictions(exp_dir, fold, 'audio', predictions, ground_truths, valid_fold)
        results['audio']['unweighted'].append(unweighted_acc)
        results['audio']['weighted'].append(weighted_acc)
        
        del model, optimizer
        torch.cuda.empty_cache()
        
        # --- Stage 3: Combined model with frozen features ---
        print("\nTraining Combined model")
        model = model_class(num_classes=len(class_weights), hidden_sizes=modality_dim, mode="both").cuda()
        
        # model.v_rnn.load_state_dict(visual_backbone_state['rnn'])
        model.a_rnn.load_state_dict(audio_backbone_state['rnn'])
        
        visual_weight = visual_backbone_state['classifier']
        audio_weight = audio_backbone_state['classifier']
        
        with torch.no_grad():
            model.classifier.weight.data[:, :modality_dim['visual']] = visual_weight
            model.classifier.weight.data[:,modality_dim['visual']:] = audio_weight
            model.classifier.bias.data = (visual_backbone_state['classifier_bias'] + 
                                        audio_backbone_state['classifier_bias']) / 2
        
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        
        data_loaders = create_dataloaders(
            df=train_fold,
            batch_size=batch_size,
            valid_df=valid_fold,
            num_workers=num_workers,
            mode='both',
            sampler=sampler
        )
        
        optimizer = torch.optim.AdamW(
            model.classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=1, threshold=1e-3, verbose=True)
        step = lambda loss, epoch=None: scheduler.step(loss)
        
        model_name = f"combined_fold{fold}"
        optimize(
            data_loaders=data_loaders,
            model=model,
            optimizer=optimizer,
            loss=loss_fn,
            step=step,
            s_epoch=s_epoch,
            n_epochs=num_epochs_combined,
            model_name=model_name,
            accumulation_steps=accumulation_steps,
            checkpoints_dir=checkpoints_dir
        )
        
        load_model(model_name=model_name, model=model, checkpoints_dir=checkpoints_dir, loading_order=['best', 'last'])
        
        # Evaluate and save predictions
        unweighted_acc, weighted_acc, predictions, ground_truths = evaluate_model_with_predictions(
            model, data_loaders['valid'], class_counts
        )
        save_fold_predictions(exp_dir, fold, 'combined', predictions, ground_truths, valid_fold)
        results['combined']['unweighted'].append(unweighted_acc)
        results['combined']['weighted'].append(weighted_acc)
        
        del model, optimizer, visual_backbone_state, audio_backbone_state
        torch.cuda.empty_cache()
    
    # Calculate and save summary statistics
    summary = {}
    for modality in results:
        summary[modality] = {
            'unweighted': {
                'mean': np.mean(results[modality]['unweighted']),
                'std': np.std(results[modality]['unweighted']),
                'values': results[modality]['unweighted']
            },
            'weighted': {
                'mean': np.mean(results[modality]['weighted']),
                'std': np.std(results[modality]['weighted']),
                'values': results[modality]['weighted']
            }
        }
    
    with open(exp_dir / "accuracy_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAccuracy Summary:")
    for modality in summary:
        print(f"\n{modality.capitalize()} Results:")
        print(f"Unweighted: {summary[modality]['unweighted']['mean']:.2f}% (±{summary[modality]['unweighted']['std']:.2f})")
        print(f"Weighted: {summary[modality]['weighted']['mean']:.2f}% (±{summary[modality]['weighted']['std']:.2f})")
    
    return summary


from typing import Dict, Literal

class AuViLSTMModel(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        mode: Literal["audio", "visual", "both"] = "visual",
        hidden_sizes: Dict = {"audio":384, "visual":384 },
        rnn_num_layers: int = 2,
        backbone_feat_size: int = 768
    ):
        """
        Simplified AuViLSTMModel for emotion recognition.
        
        Args:
            num_classes: Number of emotion classes
            mode: Training mode ("audio", "visual", or "both")
            rnn_hidden_size: Hidden size for LSTM layers
            rnn_num_layers: Number of LSTM layers
            backbone_feat_size: Feature size from backbone models
        """
        super().__init__()
        self.mode = mode
        
        # Visual components
        if mode in ["visual", "both"]:
            
            from transformers import AutoModelForImageClassification 

            self.v_backbone = AutoModelForImageClassification.from_pretrained(
                "dima806/facial_emotions_image_detection"
            )
            self.v_backbone.classifier = torch.nn.Identity()  # Remove classifier
            self.v_rnn = torch.nn.GRU(
                input_size=backbone_feat_size,
                hidden_size=hidden_sizes['visual'],
                num_layers=rnn_num_layers,
                batch_first=True
            )
            
            # Freeze backbone
            for param in self.v_backbone.parameters():
                param.requires_grad = False
        
        # Audio components
        if mode in ["audio", "both"]:
            from funasr import AutoModel
            audio_model = AutoModel(model="iic/emotion2vec_plus_base")
            self.a_backbone = audio_model.model
            self.a_rnn = torch.load("GRU.pt")
            
            # Freeze backbone
            for param in self.a_backbone.parameters():
                param.requires_grad = False
        
        # Final classifier
        input_size = hidden_sizes[mode] if mode != "both" else hidden_sizes["visual"]+hidden_sizes["audio"]
        self.classifier = torch.nn.Linear(input_size, num_classes)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            batch: Dictionary containing 'frames' and/or 'audio' depending on mode
            
        Returns:
            torch.Tensor: Classification logits
        """
        features = []
        
        # Process visual input
        if self.mode in ["visual", "both"]:
            frames = batch['frames']
            batch_size = frames.shape[0]
            seq_len = frames.shape[1]
            
            # Reshape for backbone
            frames = frames.view(-1, *frames.shape[-3:])
            
            # Extract features
            with torch.inference_mode():
                visual_feats = self.v_backbone(frames).logits
            
            # Reshape back to sequence
            visual_feats = visual_feats.view(batch_size, seq_len, -1)
            
            # Process through GRU
            _, h_n = self.v_rnn(visual_feats)
            features.append(h_n[-1])  # Take last layer's hidden state
            # features.append(visual_feats.mean(dim=1))
        
        # Process audio input
        if self.mode in ["audio", "both"]:
            audio = batch['audio'].squeeze(1)  # Remove channel dimension
            
            # Normalize audio
            audio = torch.nn.functional.layer_norm(audio, [audio.shape[-1]])
            
            # Extract features
            with torch.inference_mode():
                audio_feats = self.a_backbone.extract_features(audio)['x']
            
            # Process through GRU
            _, h_n = self.a_rnn(audio_feats)
            features.append(h_n[-1])  # Take last layer's hidden state
        
        # Combine features and classify
        combined_features = torch.cat(features, dim=-1) if len(features) > 1 else features[0]
        return self.classifier(combined_features)
    
if __name__ == "__main__":
    df = pd.read_csv("data/metadata.csv")
    df = df[~(df['video_id'].apply(lambda x:x[:2]=='02')& (df['dataset']=='RAVDESS'))].reset_index(drop=True)
    df['stratify_on']=df['dataset']+'_'+df['emotion']
    summary = sequential_cross_validate(
        df=df,
        model_class=AuViLSTMModel,
        # batch_size=32,
        n_splits=1,
        num_epochs_visual=15,
        num_epochs_audio=8,
        num_epochs_combined=4,
        learning_rate=0.001,
        num_workers=12,
        base_suffix="sequential_training"
    )
