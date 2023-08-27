import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from tqdm import tqdm
from dataclasses import dataclass

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler

import laion_clap

@dataclass
class DiffuserConfig:
    sample_size=64
    in_channels=4
    out_channels=4
    layers_per_block=2
    block_out_channels=(320, 640, 1280, 1280)
    down_block_types = ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D')
    up_block_types = ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D')
    cross_attention_dim=512

class Audiffuse(pl.LightningModule):
    def __init__(self, diffuser_config: DiffuserConfig = DiffuserConfig(), loss_func: str = 'mse', scheduler_timesteps: int = 1000):
        super().__init__()
        self.save_hyperparameters()
        self.instantiate_first_stage()
        self.instantiate_cond_stage()

        self.diffuser_config = diffuser_config
        self.instantiate_diffuser()

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=scheduler_timesteps)

        if loss_func == 'mse':
            self.loss_func = nn.MSELoss()

    def instantiate_first_stage(self):
        model = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
        model.eval()

        self.first_stage_model = model
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self):
        model = laion_clap.CLAP_Module(enable_fusion=True)
        model.load_ckpt(model_id=3)#, verbose=True)
        model = model.model
        model.text_branch = None # Remove text branch from CLAP model
        model.eval()

        self.cond_stage_model = model
        for param in self.cond_stage_model.parameters():
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
        return self.first_stage_model.encode(images)
    
    def encode_audio(self, audio):
        return self.cond_stage_model.get_audio_embedding_from_data(x = audio, use_tensor=False)

    def training_step(self, batch, batch_idx):
        pdt = torch.float16 if self.trainer.precision == '16-mixed' else torch.float32
        images, audio = batch

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
        
        self.log("train/loss", loss.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    # def validation_step(self, batch, batch_idx):


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    
    @torch.no_grad()
    def gen_images(self, audio, height=512, width=512, num_images=4, num_inference_steps=50):
        self.noise_scheduler.set_timesteps(num_inference_steps)

        audio = self.encode_audio(audio.to(self.device))
        
        latents = torch.randn((num_images, 4, height / 8, width / 8)).to(self.device)
        
        for t in tqdm(self.noise_scheduler.timesteps):
            pred_noise = self(latents, t, encoded_images).sample
            latents = self.noise_scheduler.step(pred_noise, t, latents).prev_sample
        
        triplanes = self.ae.decode(latents)
        
        return triplanes 