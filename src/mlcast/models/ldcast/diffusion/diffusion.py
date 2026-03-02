import torch
import torch.nn as nn
import pytorch_lightning as L
from typing import Any
import contextlib
from ...base import NowcastingLightningModule
import numpy as np
from .utils import extract_into_tensor
from .ema import EMA 

class LatentDiffusion(nn.Module):
    def __init__(self, conditioner, denoiser, parametrization = "eps"):
        super().__init__()
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.parametrization = parametrization

    def forward(self, x, n_timesteps = 4):
        # during training, noisy should be x_t
        # during inference, noisy should be noise
        t, noisy, latent_inputs = x
        condition = self.conditioner(latent_inputs)

        # if parametrization is eps, out is the predicted noise
        # if parametrization is x0, out is the guessed x0
        # if parametrization is v, out is the guessed v
        out = self.denoiser(noisy, t, context = condition)
        return out


class LatentDiffusionLightning(NowcastingLightningModule):
    def __init__(self, ldm, loss, scheduler, ema_config = {'use': True}):
        super().__init__(ldm, loss)
        self.scheduler = scheduler

        # register the schedules (i.e. the values of alpha, beta etc).
        self.register_schedule()

        if ema_config['use']:
            self.ema = EMA(self.net.denoiser, **ema_config['kwargs'])
            
    def register_schedule(self):
        
        schedule = self.scheduler.schedule(torch.float32, next(self.net.parameters()).device)
        
        # check if the ldm has already some saved buffers
        saved_buffers = dict(self.net.named_buffers())
        already_saved_and_different = [name for name in schedule.keys() 
                                       if (name in saved_buffers.keys() and (schedule[name] != saved_buffers[name]).any())
                                      ]
        if len(already_saved_and_different) > 0:
            raise AttributeError(f'The denoiser has already some different values for {already_saved_and_different}')
        
        for k in schedule.keys():
            self.net.denoiser.register_buffer(k, schedule[k])
    
    def training_logic(self, batch, batch_idx):
        latent_inputs, latent_true = batch
        x0 = latent_true
        
        t, noise, x_t = self.q_sample(x0)
        
        if self.net.parametrization == 'eps':
            target = noise
        if self.net.parametrization == 'x0':
            target = x0
        
        model_output = self.net((t, x_t, latent_inputs))

        return self.loss(model_output, target)

    def q_sample(self, x0, noise = None, t = None):
        '''generate noise target for training'''
        if noise is None:
            noise = torch.randn_like(x0)
        if t is None:
            t = torch.randint(0, self.scheduler.timesteps, (x0.shape[0],), device=x0.device).long()
        
        x_noisy = extract_into_tensor(self.net.denoiser.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
            extract_into_tensor(self.net.denoiser.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        
        return t, noise, x_noisy

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, 'ema'):
            self.ema.update()

    def on_validation_start(self):
        if hasattr(self, 'ema'):
            self.ema.apply_shadow()

    def on_validation_end(self):
        if hasattr(self, 'ema'):
            self.ema.restore()

    def on_test_start(self):
        if hasattr(self, 'ema'):
            self.ema.apply_shadow()

    def on_test_end(self):
        if hasattr(self, 'ema'):
            self.ema.restore()
    
    def on_predict_start(self):
        if hasattr(self, 'ema'):
            self.ema.apply_shadow()

    def on_predict_end(self):
        if hasattr(self, 'ema'):
            self.ema.restore()
