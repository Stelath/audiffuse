import os
import numpy as np
from PIL import Image
from dataclasses import dataclass
from tqdm.auto import tqdm
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ConstantLR
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

from torch.distributed.fsdp.wrap import wrap
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
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
    def __init__(self, first_stage_ckpt: str, cond_stage_ckpt: str, diffuser_config: DiffuserConfig = DiffuserConfig(), lr: float = 1e-4, use_lr_scheduler: bool = True, loss_func: str = 'mse', noise_scheduler_timesteps: int = 1000, scale_by_std: bool = True, scale_factor: float = 1.0, val_gen_freq: int = 5, freeze_cond_stage: bool = True):
        super().__init__()
        self.save_hyperparameters()

        self.scale_by_std = scale_by_std
        self.scale_factor = scale_factor
        self.instantiate_first_stage(first_stage_ckpt)

        self.freeze_cond_stage = freeze_cond_stage
        self.instantiate_cond_stage(cond_stage_ckpt)

        self.diffuser_config = diffuser_config
        self.instantiate_diffuser()

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=noise_scheduler_timesteps)

        self.lr = lr
        self.use_lr_scheduler = use_lr_scheduler
        if loss_func == 'mse':
            self.loss_func = F.mse_loss

        self.val_runs = 0
        self.val_gen_runs = 0
        self.val_gen_freq = val_gen_freq
        self.val_gen_ran = False
        self.val_images = None

        self.restarted_from_ckpt = False

    def instantiate_first_stage(self, first_stage_ckpt):
        model = AutoencoderKL.from_pretrained(first_stage_ckpt)
        model.eval()

        self.first_stage_model = model
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, cond_stage_ckpt):
        audio_cfg = CLAPAudioCfp(model_type='HTSAT', model_name='tiny', sample_rate=48000, audio_length=1024, window_size=1024, hop_size=480, fmin=50, fmax=14000, class_num=527, mel_bins=64, clip_samples=480000)
        model = create_htsat_model(audio_cfg, enable_fusion=True, fusion_type='aff_2d')
        model.load_state_dict(torch.load(cond_stage_ckpt, map_location=torch.device('cpu')))
        model.eval()

        self.cond_stage_model = model

        if self.freeze_cond_stage:
            for param in model.parameters():
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

    def encode_images(self, images, sample = True, scale = True):
        encoding = self.first_stage_model.encode(images).latent_dist

        if sample:
            encoding = encoding.sample()
        
        if scale and sample:
            encoding *= self.scale_factor
        elif scale and not sample:
            raise ValueError("Cannot scale encoding if not sampling")

        return encoding
    
    def decode_latents(self, latents):
        return self.first_stage_model.decode((1 / self.scale_factor) * latents).sample
    
    def encode_audio(self, audio):
        audio_embeds = self.cond_stage_model(audio, device=audio['waveform'].device)["fine_grained_embedding"]
        
        audio_embeds_avg_pool = F.avg_pool1d(audio_embeds.permute(0, 2, 1), kernel_size=4, padding=1).permute(0, 2, 1)
        audio_embeds_max_pool = F.max_pool1d(audio_embeds.permute(0, 2, 1), kernel_size=4, padding=1).permute(0, 2, 1)

        audio_embeds = audio_embeds_avg_pool + audio_embeds_max_pool

        return audio_embeds
    
    def forward(self, *args, **kwargs):
        return self.diffuser_model(*args, **kwargs)
    
    def step(self, batch, batch_idx):
        pdt = torch.float16 if self.trainer.precision == '16-mixed' else torch.float32
        images, audio = batch['album_art'], batch['audio']

        audio['waveform'] = audio['waveform'].to(pdt)
        audio['mel_fusion'] = audio['mel_fusion'].to(pdt)

        # Encode images
        latents = self.encode_images(images, sample=True, scale=True).to(pdt)
        # print('SCALE FACTOR:', self.scale_factor)

        # Encode audio
        encoded_audio = self.encode_audio(audio)

        # Add noise
        noise = torch.randn(latents.shape, dtype=pdt).to(latents.device)
        bs = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=latents.device).long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        pred_noise = self(noisy_latents, timesteps, encoded_audio, return_dict=True)[0]
        
        loss = self.loss_func(pred_noise, noise)

        # Ensure all GPU operations are completed before the next iteration
        # torch.cuda.synchronize()
        
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            images = batch['album_art']
            images = images.to(self.device)
            latents = self.encode_images(images, scale=False)
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / latents.flatten().std())

    def training_step(self, batch, batch_idx):
        self.diffuser_model.train()

        loss = self.step(batch, batch_idx)

        self.log("train/loss", loss.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        self.diffuser_model.eval()

        if dataloader_idx == 0:
            loss = self.step(batch, batch_idx)

            self.log("val/loss", loss.item(), prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            self.val_runs += 1
        
        if dataloader_idx == 1 and self.val_runs % self.val_gen_freq == 0:
            images = self.gen_val_images(batch)
            if self.val_images is None:
                self.val_images = images
            else:
                self.val_images = np.concatenate((self.val_images, images), axis=0)
            
            self.val_gen_runs += 1
            self.val_gen_ran = True
        

    @torch.no_grad()
    def gen_val_images(self, batch):
        self.diffuser_model.eval()
        images, audio = batch['album_art'], batch['audio']

        pdt = torch.float16 if self.trainer.precision == '16-mixed' else torch.float32
        audio['waveform'] = audio['waveform'].to(pdt)
        audio['mel_fusion'] = audio['mel_fusion'].to(pdt)

        static_generator = torch.Generator()
        static_generator.manual_seed(42)
        latent_size = (images.shape[0], 4, images.shape[-2] // 8, images.shape[-1] // 8)
        latents = torch.randn(latent_size, generator=static_generator, dtype=pdt).to(images.device)

        encoded_audio = self.encode_audio(audio)
        
        for t in self.noise_scheduler.timesteps:
            pred_noise = self(latents, t, encoded_audio).sample
            latents = self.noise_scheduler.step(pred_noise, t, latents).prev_sample

            # Ensure all GPU operations are completed before the next iteration
            torch.cuda.synchronize()
        
        images = self.decode_latents(latents)
        images = images.detach().cpu().numpy()
        
        return images 
    
    def on_validation_epoch_end(self):
        if not self.val_runs % self.val_gen_freq == 0 and not self.val_gen_ran:
            return

        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break
        
        for img_idx, image in enumerate(self.val_images):
            image = (((np.transpose(image, (1, 2, 0)) * 0.5) + 0.5) * 255).astype(np.uint8)

            tb_logger.add_image(f"Image/gr{self.global_rank:02}_idx{img_idx:02}", image, self.val_gen_runs, dataformats='HWC')
            image = Image.fromarray(image)

            save_dir = os.path.join(self.trainer.log_dir, f"images/gr{self.global_rank:02}_idx{img_idx:02}")
            os.makedirs(save_dir, exist_ok=True)
            image.save(os.path.join(save_dir, f"{self.val_gen_runs:02}.png"))

        self.val_gen_ran = False
        self.val_images = None
    
    # def configure_sharded_model(self):
    #     # print("DEVICE:", self.device)
    #     self.first_stage_model = self.first_stage_model.to('cuda')
    #     self.cond_stage_model = self.cond_stage_model.to('cuda')
    #     self.diffuser_model = wrap(self.diffuser_model)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.diffuser_model.parameters(), lr=self.lr)
        if self.freeze_cond_stage:
            optimizer = FusedAdam(chain(self.diffuser_model.parameters(), self.cond_stage_model.parameters()), lr=self.lr)
        else:
            optimizer = FusedAdam(self.diffuser_model.parameters(), lr=self.lr)

        # if self.use_lr_scheduler:
        #     print("Setting up ConstantLR scheduler...")
        #     scheduler = [
        #         {
        #             'scheduler': ConstantLR(optimizer, ),
        #             'interval': 'step',
        #             'frequency': 1
        #         }]
        #     return [optimizer], scheduler
        
        return optimizer
    
    def on_load_checkpoint(self, checkpoint):
        self.restarted_from_ckpt = True

    @torch.no_grad()
    def gen_images(self, audio, height=512, width=512, num_images=4, num_inference_steps=50):
        self.diffuser_model.eval()
        self.noise_scheduler.set_timesteps(num_inference_steps)

        # Change the type of all the dictionaries in audio to pdt
        for key in audio:
            if isinstance(audio[key], torch.Tensor):
                audio[key] = audio[key].to(self.device)

        encoded_audio = self.encode_audio(audio)
        
        latents = torch.randn((num_images, 4, height // 8, width // 8)).to(self.device)
        
        for t in tqdm(self.noise_scheduler.timesteps):
            pred_noise = self(latents, t, encoded_audio).sample
            latents = self.noise_scheduler.step(pred_noise, t, latents).prev_sample
        
        images = self.decode_latents(latents)
        
        return images
    
    def make_album_art(self, song_path, save_path = os.getcwd()):
        pass