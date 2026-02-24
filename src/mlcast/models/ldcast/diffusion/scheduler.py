from functools import partial
from src.mlcast.models.ldcast.diffusion.utils import make_beta_schedule
import numpy as np
import torch

class Scheduler():
    def __init__(self,
                 timesteps = 1000,
                 beta_schedule = "linear",
                 linear_start = 1e-4,
                 linear_end = 2e-2,
                 cosine_s = 8e-3,
                ):
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.cosine_s = cosine_s
    
    def schedule(self, dtype, device):
        
        betas = make_beta_schedule(
            self.beta_schedule, self.timesteps,
            linear_start=self.linear_start, linear_end=self.linear_end,
            cosine_s=self.cosine_s
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        
        assert alphas_cumprod.shape[0] == self.timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype = dtype, device = device)

        return {'betas': to_torch(betas),
                'alphas_cumprod': to_torch(alphas_cumprod),
                'alphas_cumprod_prev': to_torch(alphas_cumprod_prev),
                'sqrt_alphas_cumprod': to_torch(np.sqrt(alphas_cumprod)),
                'sqrt_one_minus_alphas_cumprod': to_torch(np.sqrt(1. - alphas_cumprod))}