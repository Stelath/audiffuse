import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms

import torchaudio
import torchaudio.transforms as T

import lightning.pytorch as pl
from lightning.pytorch.utilities import CombinedLoader

from transformers import Wav2Vec2FeatureExtractor

class EpidemicSoundDataset(Dataset):
    def __init__(self, dataset: str, max_audio_length: int = 120):#180
        self.dataset = dataset
        self.metadata = pd.read_parquet(os.path.join(dataset, 'metadata.parquet'))
        self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        self.max_audio_length = max_audio_length * self.audio_processor.sampling_rate

        self.transform = transforms.Compose([
            transforms.Resize((512, 512), antialias=True),
            transforms.ToTensor(),
        ])
        
        with open(os.path.join(dataset, "songs_npy", "sample_rate.txt"), "r") as file:
            self.sample_rate = int(file.read())

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        song = self.metadata.iloc[idx]
        song_id = song['id']
        
        album_art = Image.open(os.path.join(self.dataset, 'album_art', f'{song_id}.jpg'))
        album_art = self.transform(album_art) * 2 - 1 # Convert image to tensor and set range of image from -1 to 1

        audio = np.load(os.path.join(self.dataset, 'songs_npy', f'{song_id}.npy'))
        audio = self.process_audio(audio, self.sample_rate)

        return {'album_art': album_art, 'audio': audio}

    def process_audio(self, audio, sampling_rate):
        """Generate Mel-spectrogram from Audio WaveForm

        Parameters
        ----------
        audio: np.ndarray
            Audio WaveForm
        """
        
        resample_rate = self.audio_processor.sampling_rate
        
        if resample_rate != sampling_rate:
            resampler = T.Resample(sampling_rate, resample_rate)
            audio = resampler(audio)
        
        processed_audio = self.audio_processor(audio, sampling_rate=resample_rate, return_tensors="pt", padding='max_length', max_length=self.max_audio_length, truncation=True)
        
        processed_audio['input_values'] = processed_audio['input_values'][0]
        
        return processed_audio


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
        iterables = {'main_val': DataLoader(self.es_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers_val), 'img_val': DataLoader(self.es_val_gen_imgs, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers_val_gen)}
        combined_loader = CombinedLoader(iterables, mode='sequential')
        _ = iter(combined_loader)
        return combined_loader

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
