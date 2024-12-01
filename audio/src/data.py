import torch
from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd
from .helpers import train_test_split, load_audio, seed 
import torchaudio.transforms as AT
import random
from torch.utils.data import default_collate


class RAVDESSDataset(torch.utils.data.Dataset):
    """Dataset class for processed RAVDESS data with faces and audio"""
    
    def __init__(
        self,
        df,
        audio_transforms=None,
        sample_rate: int = 16000
    ):
        self.df = df
        self.audio_transforms = audio_transforms
        self.sample_rate = sample_rate
        
        # Initialize Silero VAD
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                    model='silero_vad',
                                    force_reload=False)
        self.vad_model = model
        self.get_speech_timestamps = utils[0]  # The function is the first element in utils
        self.vad_model.eval()
    
    def remove_silence(self, audio: torch.Tensor) -> torch.Tensor:
        """Remove silence from audio using Silero VAD"""
        # Handle input shape (1, num_samples)
        audio = audio.squeeze(0)  # (num_samples,)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio,
            self.vad_model,
            sampling_rate=self.sample_rate,
            return_seconds=False
        )
        
        if not speech_timestamps:
            return audio.unsqueeze(0)  # Return original if no speech detected
        
        # Concatenate speech segments
        speech_segments = []
        for segment in speech_timestamps:
            start_frame = segment['start']
            end_frame = segment['end']
            speech_segments.append(audio[start_frame:end_frame])
        
        # Combine all segments
        processed_audio = torch.cat(speech_segments, dim=0)
        
        # Restore shape (1, num_samples)
        return processed_audio.unsqueeze(0)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load and process audio
        audio = torch.from_numpy(load_audio(row['Path'])[None, :])
        
        # Remove silence using Silero VAD for above 3s audio
        if audio.shape[-1]>self.sample_rate*3:
            audio = self.remove_silence(audio)
        
        if self.audio_transforms:
            audio = self.audio_transforms(audio)
        
        return audio, row['target']
    

class AudioAugmentor(torch.nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        
    @torch.no_grad()
    def forward(self, waveform):
        # Amplitude scaling (30% chance)
        # Keeping this as it's computationally light
        if np.random.random() < 0.5:
            waveform = waveform * np.random.uniform(0.7, 1.3)
        
        # Time masking (20% chance)
        # Efficient operation, just zeroing out values
        if np.random.random() < 0.5:
            mask_size = np.random.randint(320, 800)  # Reduced max mask size
            mask_start = np.random.randint(0, waveform.shape[-1] - mask_size)
            waveform[..., mask_start:mask_start + mask_size] = 0
        
        # Speed perturbation (20% chance)
        # Using simple interpolation for efficiency
        if np.random.random() < 0.2:
            speed_factor = np.random.uniform(0.95, 1.05)  # Smaller range
            orig_len = waveform.size(-1)
            new_len = int(orig_len / speed_factor)
            waveform =torch.nn.functional.interpolate(
                waveform.unsqueeze(0), 
                size=new_len, 
                mode='linear', 
                align_corners=False
            ).squeeze(0)
            # Resize to original length
            if waveform.size(-1) > orig_len:
                waveform = waveform[..., :orig_len]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, orig_len - waveform.size(-1)))

        # Background noise (20% chance)
        # Simple Gaussian noise, computationally efficient
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.001, 0.01)  # Reduced noise level
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise

        return waveform
    

def audio_data_loader(
    df, shuffle =True, sampler=None,batch_size: int = 32, num_workers: int = -1,
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()


    train_data = RAVDESSDataset(df,  AudioAugmentor() if (sampler or shuffle) else None ) 


    return torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False if sampler else shuffle ,
        sampler=sampler,
     collate_fn=collate_fn,
    #    pin_memory=True,
    )

def collate_fn(batch):
    # [(data, target), (data, target), .... (data, target)]

    max_length=max([tensor.shape[-1] for tensor,target in batch])
    tensors = [(torch.nn.functional.pad(tensor,(max_length-tensor.shape[-1],0),value=0.0), target) for tensor,target in batch]
                                            #  (before,after)
    return default_collate(tensors) 
