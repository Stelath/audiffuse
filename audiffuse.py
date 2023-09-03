import os
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
import laion_clap
from laion_clap.clap_module.htsat import create_htsat_model
from laion_clap.clap_module.model import CLAPAudioCfp

@dataclass
class DiffuserConfig:
    sample_size=64
    in_channels=4
    out_channels=4
    layers_per_block=2
    block_out_channels=(320, 640, 1280, 1280)
    down_block_types = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D')
    up_block_types = ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D')
    cross_attention_dim=768

class Audiffuse(pl.LightningModule):
    def __init__(self, diffuser_config: DiffuserConfig = DiffuserConfig(), lr: float = 1e-4, loss_func: str = 'mse', scheduler_timesteps: int = 1000):
        super().__init__()
        self.save_hyperparameters()
        self.instantiate_first_stage()
        self.instantiate_cond_stage()

        self.diffuser_config = diffuser_config
        self.instantiate_diffuser()

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=scheduler_timesteps)

        self.lr = lr
        if loss_func == 'mse':
            self.loss_func = F.mse_loss

        self.test_epoch = 0
        self.test_images = []

    def instantiate_first_stage(self):
        model = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
        model.eval()

        self.first_stage_model = model
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self):
        audio_cfg = CLAPAudioCfp(model_type='HTSAT', model_name='tiny', sample_rate=48000, audio_length=1024, window_size=1024, hop_size=480, fmin=50, fmax=14000, class_num=527, mel_bins=64, clip_samples=480000)
        model = create_htsat_model(audio_cfg, enable_fusion=True, fusion_type='aff_2d')
        model.eval()

        # model = laion_clap.CLAP_Module(enable_fusion=True)
        # model.load_ckpt(model_id=3) #, verbose=True)
        # model.text_branch = None # Remove text branch from CLAP model
        # model = model.model.audio_branch # Set to the audio branch and remove the rest
        # model.eval()

        self.cond_stage_model = model
        for param in self.cond_stage_model.patch_embed.parameters():
            param = param.contiguous()
            param.requires_grad = False

        for param in self.cond_stage_model.parameters():
            param = param.contiguous()
            param.requires_grad = False
    
    def instantiate_diffuser(self):
        model = UNet2DConditionModel(
            sample_size=self.diffuser_config.sample_size,
            in_channels=self.diffuser_config.in_channels,
            out_channels=self.diffuser_config.out_channels,
            layers_per_block=self.diffuser_config.layers_per_block,
            block_out_channels=self.diffuser_config.block_out_channels,
            down_block_types=self.diffuser_config.down_block_types,
            up_block_types=self.diffuser_config.up_block_types,
            cross_attention_dim=self.diffuser_config.cross_attention_dim
        )

        self.diffuser_model = model

    def encode_images(self, images):
        return self.first_stage_model.encode(images).latent_dist
    
    def encode_audio(self, audio):
        # print('bagels')
        # print(audio['waveform'].dtype)
        # print(audio['mel_fusion'].dtype)
        audio_embeds = self.cond_stage_model(audio, device=audio['waveform'].device)["fine_grained_embedding"]
        
        audio_embeds_avg_pool = F.avg_pool1d(audio_embeds.permute(0, 2, 1), kernel_size=4, padding=1).permute(0, 2, 1)
        audio_embeds_max_pool = F.max_pool1d(audio_embeds.permute(0, 2, 1), kernel_size=4, padding=1).permute(0, 2, 1)

        audio_embeds = audio_embeds_avg_pool + audio_embeds_max_pool

        return audio_embeds
    
    def forward(self, *args, **kwargs):
        return self.diffuser_model(*args, **kwargs)
    
    def step(self, batch, batch_idx):
        images, audio = batch['album_art'], batch['audio']

        pdt = torch.float16 if self.trainer.precision == '16-mixed' else torch.float32
        audio['waveform'] = audio['waveform'].type(pdt)
        audio['mel_fusion'] = audio['mel_fusion'].type(pdt)

        # # Change the type of all the dictionaries in audio to pdt
        # for i, audio_dict in enumerate(audio):
        #     for key in audio_dict:
        #         if isinstance(audio_dict[key], torch.Tensor):
        #             audio[i][key] = audio_dict[key].type(torch.float32)

        # Encode images
        latents = self.encode_images(images)
        latents = latents.sample().type(pdt)

        # Encode audio
        encoded_audio = self.encode_audio(audio)

        # Add noise
        noise = torch.randn(latents.shape, dtype=pdt).to(latents.device)
        bs = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bs,), device=latents.device, dtype=pdt).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        pred_noise = self(noisy_latents, timesteps, encoded_audio, return_dict=True)[0]
        
        loss = self.loss_func(pred_noise, noise)
        
        return loss

    def training_step(self, batch, batch_idx):
        self.diffuser_model.train()

        loss = self.step(batch, batch_idx)

        self.log("train/loss", loss.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.diffuser_model.eval()

        loss = self.step(batch, batch_idx)

        self.log("val/loss", loss.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.diffuser_model.eval()
        images, audio = batch['album_art'], batch['audio']

        pdt = torch.float16 if self.trainer.precision == '16-mixed' else torch.float32
        audio['waveform'] = audio['waveform'].type(pdt)
        audio['mel_fusion'] = audio['mel_fusion'].type(pdt)

        static_generator = torch.Generator()
        static_generator.manual_seed(42)
        latent_size = images.shape / 8
        latent_size[1] = 4
        latents = torch.randn(latent_size, generator=static_generator, dtype=pdt).to(images.device)

        encoded_audio = self.encode_audio(audio.to(self.device))
        
        for t in self.noise_scheduler.timesteps:
            pred_noise = self(latents, t, encoded_audio).sample
            latents = self.noise_scheduler.step(pred_noise, t, latents).prev_sample
        
        images = self.ae.decode(latents)
        images = images.detach().cpu().numpy()
        
        self.test_images = np.concatenate((self.test_images, images), axis=0)
        
        return images 
    
    def on_test_epoch_end(self):
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break
        
        for img_idx, image in self.test_images:
            tb_logger.add_image(f"Image/{img_idx}", image, self.test_epoch)
            
            image = Image.fromarray(image)

            save_dir = os.path.join(self.trainer.log_dir, "images/{img_idx:02}")
            os.makedirs(save_dir, exist_ok=True)
            image.save(f"images/{img_idx:02}/{self.test_epoch:02}.png")

        self.test_epoch += 1
        self.test_images = []

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.diffuser_model.parameters(), lr=self.lr)
        optimizer = FusedAdam(self.diffuser_model.parameters(), lr=self.lr)
        return optimizer
    
    @torch.no_grad()
    def gen_images(self, audio, height=512, width=512, num_images=4, num_inference_steps=50):
        self.noise_scheduler.set_timesteps(num_inference_steps)

        encoded_audio = self.encode_audio(audio.to(self.device))
        
        latents = torch.randn((num_images, 4, height / 8, width / 8)).to(self.device)
        
        for t in tqdm(self.noise_scheduler.timesteps):
            pred_noise = self(latents, t, encoded_audio).sample
            latents = self.noise_scheduler.step(pred_noise, t, latents).prev_sample
        
        images = self.ae.decode(latents)
        
        return images 