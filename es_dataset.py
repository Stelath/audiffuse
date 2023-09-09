import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
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

        self.transform = transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        song = self.metadata.iloc[idx]
        song_id = song['id']
        
        album_art = Image.open(os.path.join(self.dataset, 'album_art', f'{song_id}.jpg'))
        album_art = self.transform(album_art) * 2 - 1 # Convert image to tensor and set range of image from -1 to 1

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
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers_train: int = 0, num_workers_val: int = 0, num_workers_val_gen: int = 0, val_gen_images: int = 16, percent_val: float = 0.025, clip_samples: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers_train = num_workers_train
        self.num_workers_val = num_workers_val
        self.num_workers_val_gen = num_workers_val_gen

        self.val_gen_images = val_gen_images
        self.percent_val = percent_val

        self.clip_samples = clip_samples

    def setup(self, stage: str):
        self.dataset_lg = EpidemicSoundDataset(self.data_dir)

        if self.clip_samples > 0:
            indices = range(self.clip_samples)
            self.dataset = Subset(self.dataset_lg, indices)
        else:
            self.dataset = self.dataset_lg

        val_images = round(len(self.dataset) * self.percent_val)
        val_gen_images = self.val_gen_images
        train_images = len(self.dataset) - val_images - val_gen_images

        static_generator = torch.Generator().manual_seed(42)
        self.es_train, self.es_val, self.es_val_gen_imgs = random_split(self.dataset, [train_images, val_images, val_gen_images], generator=static_generator)

    def train_dataloader(self):
        return DataLoader(self.es_train, batch_size=self.batch_size, num_workers=self.num_workers_train)
    
    def val_dataloader(self):
        return [DataLoader(self.es_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers_val), DataLoader(self.es_val_gen_imgs, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers_val_gen)]

    def test_dataloader(self):
        return DataLoader(self.es_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.es_predict, batch_size=self.batch_size)


# def collate_fn(batch):
#     data = {}
#     for key in batch[0].keys():
#         if key == 'audio':
#             data[key] = [item[key] for item in batch]
#         else:
#             data[key] = torch.stack([item[key] for item in batch])
#     return data
