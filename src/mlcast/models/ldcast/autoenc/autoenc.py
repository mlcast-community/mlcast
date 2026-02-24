# from https://github.com/MeteoSwiss/ldcast/blob/master/ldcast/models/autoenc/autoenc.py

import pytorch_lightning as pl
import torch
from torch import nn
from .encoder import SimpleConvEncoder, SimpleConvDecoder

from ..distributions import (
    ensemble_nll_normal,
    kl_from_standard_normal,
    sample_from_standard_normal,
)

class autoenc_loss(nn.Module):
    def __init__(self, kl_weight = 0.01):
        super().__init__()
        self.kl_weight = kl_weight
        
    def forward(self, predictions, y):
        (y_pred, mean, log_var) = predictions
    
        rec_loss = (y - y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)
    
        total_loss = rec_loss + self.kl_weight * kl_loss
        
        return {'total_loss': total_loss, 'rec_loss': rec_loss, 'kl_loss': kl_loss}


class AutoencoderKLNet(pl.LightningModule):
    def __init__(
        self,
        encoder = SimpleConvEncoder(),
        decoder = SimpleConvDecoder(),
        kl_weight=0.01,
        encoded_channels=64,
        hidden_width=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_width = hidden_width
        self.to_moments = nn.Conv3d(encoded_channels, 2 * hidden_width, kernel_size=1)
        self.to_decoder = nn.Conv3d(hidden_width, encoded_channels, kernel_size=1)
        self.log_var = nn.Parameter(torch.zeros(size=()))
        self.kl_weight = kl_weight

    def encode(self, x, return_log_var = False):
        if len(x.shape) < 5:
            x = x[None]
        h = self.encoder(x)
        (mean, log_var) = torch.chunk(self.to_moments(h), 2, dim=1)
        if return_log_var:
            return (mean, log_var)
        else:
            return mean

    def decode(self, z):
        z = self.to_decoder(z)
        dec = self.decoder(z)
        return dec

    def forward(self, x, n_timesteps, sample_posterior=True):
        (mean, log_var) = self.encode(x, return_log_var = True)
        if sample_posterior:
            z = sample_from_standard_normal(mean, log_var)
        else:
            z = mean
        dec = self.decode(z)
        return (dec, mean, log_var)