import numpy as np
import torch
from torch import nn
import pandas as pd
from typing import Callable, Dict, Optional
import json
from datetime import datetime
from pathlib import Path
# from src.models.video import AuVi2LSTMModel
from src.train import optimize, to_device, accuracy
from src.helpers import train_test_split, load_model,device
from src.data import  audio_data_loader
from sklearn.model_selection import StratifiedKFold



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
        "path" :valid_df['Path'].values,
        "dataset":valid_df['Dataset'].values,
        'ground_truth': ground_truths,
        'prediction': predictions
    })
    
    output_path = exp_dir / "fold_predictions" / f"{modality}_fold{fold}_predictions.csv"
    predictions_df.to_csv(output_path, index=False)


class FullEm2vecNPLSTM384h(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        from funasr import AutoModel
        
        model = AutoModel(model="iic/emotion2vec_plus_base")
        self.backbone = model.model
        
        # Initially freeze all parameters
        self.freeze_backbone()
        
        # Unfreeze only last 2 blocks initially
        # self.unfreeze_last_n_blocks(2)
        
        self.backbone.proj = None
        self.rnn = nn.GRU(768, 768//2, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(768//2, num_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_blocks(self, n):
        for param in self.backbone.blocks[-n:].parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            x = nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
        
        x = self.backbone.extract_features(x, padding_mask=None)['x']
        o, hn = self.rnn(x)
        x = self.classifier(hn[-1])
        return x

def sequential_cross_validate(
    df: pd.DataFrame,
    model_class: Callable,
    n_splits: int = 5,
    valid_size: float = 0.2,
    batch_size: int = 16,
    initial_epochs: int = 15,  # First phase training epochs
    final_epochs: int = 15,    # Second phase training epochs
    num_workers: int = 8,
    initial_lr: float = 0.001,
    final_lr: float = 0.0001,  # Lower learning rate for fine-tuning
    weight_decay: float = 0.004,
    accumulation_steps: int = 4,
    class_weight_power: float = 0.3,
    label_smoothing: float = 0.03,
    seed: int = 42,
    base_suffix: str = "sequential",
):
    exp_dir = create_experiment_dir(base_suffix)
    checkpoints_dir = str(exp_dir / "model_checkpoints")
    
    results = {
        'phase1': {'unweighted': [], 'weighted': []},
        'phase2': {'unweighted': [], 'weighted': []}
    }
    
    # Save experiment configuration
    save_experiment_config(
        exp_dir,
        n_splits=n_splits,
        valid_size=valid_size,
        batch_size=batch_size,
        initial_epochs=initial_epochs,
        final_epochs=final_epochs,
        initial_lr=initial_lr,
        final_lr=final_lr,
        weight_decay=weight_decay
    )
    df['stratify_on'] = df['Dataset']+'_'+df['Emotion']
    df['Emotion'] = df['Emotion'].astype("category")
    df['target']=df['Emotion'].cat.codes.astype(np.int64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(skf.split(df, df['stratify_on'])):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")

        # Split data
        train_fold = df.iloc[train_index].reset_index(drop=True)
        valid_fold = df.iloc[test_index].reset_index(drop=True)
        # Calculate class weights
        stratify_on = train_fold['Emotion'].to_numpy()
        unique_classes, class_counts = np.unique(stratify_on, return_counts=True)
        n_samples = len(stratify_on)
        class_sample_weights = {}
        for cls, count in zip(unique_classes, class_counts):
            class_sample_weights[cls] = n_samples / (len(unique_classes) * count) ** 0.6
        sample_weights = [class_sample_weights[t]/sum(class_sample_weights.values()) for t in stratify_on]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(train_fold), replacement=True)

        unique_classes, class_counts = np.unique(df['Emotion'].to_numpy(), return_counts=True)
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
        
        # Phase 1: Initial training with most layers frozen
        print(f"\nPhase 1: Initial training")
        model = model_class(len(class_counts))
        
        data_loaders = {
            'train': audio_data_loader(train_fold, sampler=sampler, batch_size=batch_size, num_workers=num_workers),
            'valid': audio_data_loader(valid_fold, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        }
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=initial_lr,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, threshold=1e-3, verbose=True
        )
        step = lambda loss, epoch=None: scheduler.step(loss)
        
        # Train Phase 1
        model_name = f"phase1_fold{fold}"
        optimize(
            data_loaders=data_loaders,
            model=model,
            optimizer=optimizer,
            loss=loss_fn,
            step=step,
            s_epoch=1,
            n_epochs=initial_epochs,
            model_name=model_name,
            accumulation_steps=accumulation_steps,
            checkpoints_dir=checkpoints_dir
        )
        
        # Evaluate Phase 1
        unweighted_acc, weighted_acc, predictions, ground_truths = evaluate_model_with_predictions(
            model, data_loaders['valid'], class_counts
        )
        results['phase1']['unweighted'].append(unweighted_acc)
        results['phase1']['weighted'].append(weighted_acc)
        
        # Phase 2: Unfreeze more layers and continue training
        print(f"\nPhase 2: Fine-tuning")
        # model.unfreeze_last_n_blocks(6)
        stratify_on = train_fold['stratify_on'].to_numpy()
        unique_classes, class_counts = np.unique(stratify_on, return_counts=True)
        n_samples = len(stratify_on)
        class_sample_weights = {}
        for cls, count in zip(unique_classes, class_counts):
            class_sample_weights[cls] = n_samples / (len(unique_classes) * count) ** 0.9
        sample_weights = [class_sample_weights[t]/sum(class_sample_weights.values()) for t in stratify_on]
        sampler.weights = torch.tensor(sample_weights)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=final_lr,
            weight_decay=weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, threshold=1e-3, verbose=True
        )
        step = lambda loss, epoch=None: scheduler.step(loss)
        
        # Train Phase 2
        model_name = f"phase2_fold{fold}"
        optimize(
            data_loaders=data_loaders,
            model=model,
            optimizer=optimizer,
            loss=loss_fn,
            step=step,
            s_epoch=1,
            n_epochs=final_epochs,
            model_name=model_name,
            accumulation_steps=accumulation_steps,
            checkpoints_dir=checkpoints_dir
        )
        
        # Final evaluation
        unweighted_acc, weighted_acc, predictions, ground_truths = evaluate_model_with_predictions(
            model, data_loaders['valid'], class_counts
        )
        save_fold_predictions(exp_dir, fold, 'combined', predictions, ground_truths, valid_fold)
        results['phase2']['unweighted'].append(unweighted_acc)
        results['phase2']['weighted'].append(weighted_acc)
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Calculate and save summary statistics
    summary = {}
    for phase in results:
        summary[phase] = {
            'unweighted': {
                'mean': np.mean(results[phase]['unweighted']),
                'std': np.std(results[phase]['unweighted']),
                'values': results[phase]['unweighted']
            },
            'weighted': {
                'mean': np.mean(results[phase]['weighted']),
                'std': np.std(results[phase]['weighted']),
                'values': results[phase]['weighted']
            }
        }
    
    with open(exp_dir / "accuracy_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nAccuracy Summary:")
    for phase in summary:
        print(f"\n{phase.capitalize()} Results:")
        print(f"Unweighted: {summary[phase]['unweighted']['mean']:.2f}% (±{summary[phase]['unweighted']['std']:.2f})")
        print(f"Weighted: {summary[phase]['weighted']['mean']:.2f}% (±{summary[phase]['weighted']['std']:.2f})")
    
    return summary

if __name__ == "__main__":
    df = pd.read_csv("data/metadata.csv")
    df = df =df[~((df['Emotion']=='surprise') | (df['Emotion']=='disgust')| (df['Emotion']=='fear') |(df['Dataset']=='EYASE')|  (df['Dataset']=='RAVDESS')|(df['Dataset']=='SaudiEMO') )].reset_index(drop=True)
    summary = sequential_cross_validate(
        df=df,
        model_class=FullEm2vecNPLSTM384h,
        batch_size=16,
        accumulation_steps=8,
        n_splits=5,
        initial_epochs=20,
        final_epochs=10,
        num_workers=12,
        initial_lr=0.001,
        final_lr=0.0001,
        base_suffix="two_phase_training"
    )
