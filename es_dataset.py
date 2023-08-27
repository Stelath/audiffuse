import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning.pytorch as pl

from laion_clap.training.data import get_audio_features, int16_to_float32, float32_to_int16

DEFAULT_AUDIO_CFG = {'audio_length': 1024,
                    'clip_samples': 480000,
                    'mel_bins': 64,
                    'sample_rate': 48000,
                    'window_size': 1024,
                    'hop_size': 480,
                    'fmin': 50,
                    'fmax': 14000,
                    'class_num': 527,
                    'model_type': 'HTSAT',
                    'model_name': 'tiny'}

class EpidemicSoundDataset(Dataset):
    def __init__(self, dataset: str, audio_cfg: map=DEFAULT_AUDIO_CFG):
        self.dataset = dataset
        self.metadata = pd.read_parquet(os.path.join(dataset, 'metadata.parquet'))
        self.audio_cfg = audio_cfg

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        song = self.metadata.iloc[idx]
        song_id = song['id']
        
        album_art = Image.open(os.path.join(self.dataset, 'album_art', f'{song_id}.jpg'))
        album_art = self.to_tensor(album_art) * 2 - 1 # Convert image to tensor and set range of image from -1 to 1

        audio = np.load(os.path.join(self.dataset, 'songs_npy', f'{song_id}.npy'))
        audio = self.process_audio(audio)

        return {'album_art': album_art, 'audio': audio}

    def process_audio(self, audio):
        """Generate Mel-spectrogram from Audio WaveForm

        Parameters
        ----------
        audio: np.ndarray
            Audio WaveForm
        """
        audio_waveform = int16_to_float32(float32_to_int16(audio))
        audio_waveform = torch.from_numpy(audio_waveform).float()

        audio_dict = {}
        audio_dict = get_audio_features(
            audio_dict, audio_waveform, 480000, 
            data_truncating='fusion', 
            data_filling='repeatpad',
            audio_cfg=self.audio_cfg,
            require_grad=False
        )

        return audio_dict


class EpidemicSoundDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.dataset = EpidemicSoundDataset(self.data_dir)
        self.es_train, self.es_val = random_split(self.dataset, [len(self.dataset) - 16, 16])

    def train_dataloader(self):
        return DataLoader(self.es_train, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.es_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.es_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.es_predict, batch_size=self.batch_size)
