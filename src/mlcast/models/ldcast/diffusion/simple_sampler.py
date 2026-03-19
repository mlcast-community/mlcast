# from https://github.com/mfroelund/ldcast-dmi-public/blob/master/ldcast/models/diffusion/diffusion.py, but reworked

"""
From https://github.com/CompVis/latent-diffusion/main/ldm/models/diffusion/ddpm.py
Pared down to simplify code.

The original file acknowledges:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
"""

import torch
import torch.nn as nn
import pytorch_lightning as L
from functools import partial
import numpy as np

from .utils import make_beta_schedule, extract_into_tensor


class SimpleSampler(L.LightningModule):
    '''Sampler used for training (the PLMSSampler is used for inference). The sample method is not consistent with the sample method of PLMSSampler'''
    def __init__(self,
        timesteps=1000,
        beta_schedule="linear",
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",  # all assuming fixed variance schedules
    ):
        super().__init__()

        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        self.beta_schedule = beta_schedule
        self.timesteps = timesteps
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.cosine_s = cosine_s
        
    def register_schedule(self, denoiser):

        # check if the denoiser has already some saved buffers
        buffer_names = ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod'] 
        already_saved = [n for n in buffer_names if n in dict(denoiser.named_buffers()).keys()]
        if len(already_saved) > 0:
            raise AttributeError(f'The denoiser has already some saved values for {already_saved}')
        
        betas = make_beta_schedule(
            self.beta_schedule, self.timesteps,
            linear_start=self.linear_start, linear_end=self.linear_end,
            cosine_s=self.cosine_s
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        
        assert alphas_cumprod.shape[0] == self.timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32, device = next(denoiser.parameters()).device)
        
        denoiser.register_buffer('betas', to_torch(betas))
        denoiser.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        denoiser.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        denoiser.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        denoiser.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        
        
    def q_sample(self, denoiser, x_start, noise=None):
        '''generate noise target for training'''
        if noise is None:
            noise = torch.randn_like(x_start)
        t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device).long()
        x_noisy = extract_into_tensor(denoiser.sqrt_alphas_cumprod, t, x_start.shape) * x_start + \
            extract_into_tensor(denoiser.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        return t, noise, x_noisy

    def sample(self, denoiser, conditioning, num_diffusion_iters = 50):
        '''sampling for inference, should maybe be implemented to be consistent with the PLMSSampler class'''
        pass
    