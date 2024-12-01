import multiprocessing
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset, default_collate
import numpy as np
from pathlib import Path
from src.helpers import load_audio, train_test_split, seed
import torchaudio.transforms as AT
import pandas as pd
import torchvision.transforms.v2 as VT

class RAVDESSDataset(Dataset):
    """Dataset class for processed RAVDESS data with faces and audio with mode selection"""
    
    def __init__(
        self,
        df,
        mode='both',  # 'visual', 'audio', or 'both'
        audio_transforms=None,
        image_transforms=None
    ):
        self.df = df
        self.mode = mode
        self.audio_transforms = audio_transforms
        
        # Default image transforms if none provided
        self.image_transforms = image_transforms or VT.Compose([
            VT.ToImage(),
            VT.Grayscale(num_output_channels=3),
            VT.ToDtype(torch.float32, scale=True),
            VT.Normalize(mean=[0.5, 0.5, 0.5], 
                        std=[0.5, 0.5, 0.5])
        ])
        
        if mode not in ['visual', 'audio', 'both']:
            raise ValueError("Mode must be one of: 'visual', 'audio', 'both'")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        result = {}
        
        # Load visual data if needed
        if self.mode in ['visual', 'both']:
            frame_paths = sorted(Path(row['frame_dir']).glob("*.png"))
            frames = []
            for frame_path in frame_paths:
                try:
                    frame = read_image(str(frame_path))
                except Exception as e:
                    print(frame_path)
                if self.image_transforms:
                    frame = self.image_transforms(frame)
                frames.append(frame)
            result['frames'] = torch.stack(frames)
            
        # Load audio data if needed
        if self.mode in ['audio', 'both']:
            try:
                audio = torch.from_numpy(load_audio(row['audio_path'])[None, :])
            except Exception as e:
                print(row['audio_path'])
            if self.audio_transforms:
                audio = self.audio_transforms(audio)
            result['audio'] = audio
            
        return result, row['target']
    


class AudioAugmentor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    @torch.no_grad()
    def forward(self, waveform):
        # Very mild amplitude scaling (15% chance)
        # We keep this mild because amplitude patterns are crucial for emotion
        if np.random.random() < 0.5:
            # Smaller range to preserve emotional intensity
            waveform = waveform * np.random.uniform(0.8, 1.2)
        
        # Extremely selective time masking (10% chance)
        # Only mask very short segments to avoid losing emotional context
        if np.random.random() < 0.3:
            # Much shorter masks (20-50ms assuming 16kHz sample rate)
            mask_size = np.random.randint(320, 800)  
            mask_start = np.random.randint(0, waveform.shape[-1] - mask_size)
            waveform[..., mask_start:mask_start + mask_size] = 0
        
        # Background noise (10% chance)
        # Use very clean SNR to preserve emotional content
        if np.random.random() < 0.3:
            signal_power = torch.mean(waveform ** 2)
            # Higher SNR range (20-30dB) to keep emotion clear
            snr = np.random.uniform(20, 30)
            noise = torch.randn_like(waveform) * torch.sqrt(signal_power / (10 ** (snr / 10)))
            waveform += noise

        return waveform
    
    

def create_dataloaders(
    df = (lambda : pd.read_csv("data/preprocessed_faces/metadata.csv",dtype={'class':'category'}))() ,
    batch_size: int = 32,
    valid_size: float = 0.2,
    num_workers: int = 4,
    mode : str ='both',
    valid_df =None,
    sampler =None,
    seed: int = seed,
) -> dict:
    """Create train and validation dataloaders"""
    
    if valid_df is None:
        # Stratified split
        train_df, valid_df = train_test_split(
            df,
            test_size=valid_size,
            stratify=df['stratify_on'],
            random_state=seed
        )

    else:
        train_df =df
    # Audio transforms (same as original)
    audio_transforms = {
        "train": AudioAugmentor(),
    }
    vi_transforms =VT.Compose([
            VT.ToImage(),
            VT.RandomHorizontalFlip(),
            VT.RandomAutocontrast(),
            # VT.RandomAdjustSharpness(1.3),
            # VT.RandomEqualize(),
            VT.TrivialAugmentWide(),
            VT.RandomErasing(p=0.1),
            VT.Grayscale(num_output_channels=3),
            VT.ToDtype(torch.float32,scale=True),
            VT.Normalize(mean=[0.5, 0.5, 0.5], 
                        std=[0.5, 0.5, 0.5])
        ])
    # Create datasets
    train_dataset = RAVDESSDataset(
        train_df,
        mode=mode,
        audio_transforms=audio_transforms['train'],
        image_transforms=vi_transforms
    )
    
    valid_dataset = RAVDESSDataset(
        valid_df,
        mode=mode
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False if sampler else True ,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'valid': valid_loader
    }







def collate_fn(batch):
    """
    Collate function that pads audio and frame tensors with safety checks.
    Skips padding if keys don't exist in the sample dictionary.
    """
    # Initialize with None in case keys don't exist
    max_audio_length = max_num_frames = 0
    
    # Safely get maximum lengths
    for sample, _ in batch:
        if 'audio' in sample:
            max_audio_length = max(max_audio_length, sample['audio'].shape[-1])
        if 'frames' in sample:
            max_num_frames = max(max_num_frames, sample['frames'].shape[0])
    
    # Pad samples if needed
    for sample, _ in batch:
        # Pad audio if it exists
        if 'audio' in sample and max_audio_length > 0:
            padding_size = max_audio_length - sample['audio'].shape[-1]
            if padding_size > 0:
                sample['audio'] = torch.nn.functional.pad(
                    sample['audio'],
                    (padding_size, 0),
                    value=0.0
                )
        
        # Pad frames if they exist
        if 'frames' in sample and max_num_frames > 0:
            frames_diff = max_num_frames - sample['frames'].shape[0]
            if frames_diff > 0:
                frames_padd = torch.zeros((frames_diff, *sample['frames'].shape[1:]))
                sample['frames'] = torch.concat((sample['frames'], frames_padd))
    
    return default_collate(batch)


