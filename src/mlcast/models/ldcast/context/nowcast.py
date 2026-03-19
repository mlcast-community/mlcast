# from https://github.com/MeteoSwiss/ldcast/blob/master/ldcast/models/nowcast/nowcast.py, but removed the Nowcaster, AFNONowcastNetBasic and AFNONowcastNet classes because they were not used. Reworked also the two remaining classes (FusionBlock3D and AFNONowcastNetBase) to simplify the code by removing the unused parts.

import collections

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from ..blocks.afno import AFNOBlock3d
from ..blocks.attention import positional_encoding, TemporalTransformer

class FusionBlock3d(nn.Module):
    def __init__(self, dim, size_ratio, dim_out=None, afno_fusion=False):
        super().__init__()

        if dim_out is None:
            dim_out = dim
        
        if size_ratio == 1:
            scale = nn.Identity()
        else:
            scale = []
            while size_ratio > 1:
                scale.append(nn.ConvTranspose3d(
                    dim[i], dim_out if size_ratio==2 else dim[i],
                    kernel_size=(1,3,3), stride=(1,2,2),
                    padding=(0,1,1), output_padding=(0,1,1)
                ))
                size_ratio //= 2
            scale = nn.Sequential(*scale)
        self.scale = scale
        
        self.afno_fusion = afno_fusion
        
        if self.afno_fusion:
            self.fusion = nn.Identity()
        
    def resize_proj(self, x):
        x = x.permute(0,4,1,2,3)
        x = self.scale(x)
        x = x.permute(0,2,3,4,1)
        return x

    def forward(self, x):
        x = self.resize_proj(x)
        return x


class AFNONowcastNetBase(nn.Module):
    def __init__(
        self,
        autoencoder_dim,
        embed_dim=128,
        embed_dim_out=None,
        analysis_depth=4,
        forecast_depth=4,
        input_patches=1,
        input_size_ratios=1,
        output_patches=2,
        afno_fusion=False
    ):
        super().__init__()
        
        if embed_dim_out is None:
            embed_dim_out = embed_dim
        self.embed_dim = embed_dim
        self.embed_dim_out = embed_dim_out
        self.output_patches = output_patches

        self.proj = nn.Conv3d(autoencoder_dim, embed_dim, kernel_size=1)
        
        self.analysis = nn.Sequential(
            *(AFNOBlock3d(embed_dim) for _ in range(analysis_depth))
        )

        # temporal transformer
        self.use_temporal_transformer = input_patches != output_patches
        if self.use_temporal_transformer:
            self.temporal_transformer = TemporalTransformer(embed_dim)

        # data fusion
        self.fusion = FusionBlock3d(embed_dim, input_size_ratios,
            afno_fusion=afno_fusion, dim_out=embed_dim_out)

        # forecast
        self.forecast = nn.Sequential(
            *(AFNOBlock3d(embed_dim_out) for _ in range(forecast_depth))
        )
    
    def add_pos_enc(self, x, t):
        '''not sure the this does what it was supposed to do in the original LDCast code'''
        if t.shape[1] != x.shape[1]:
            # this can happen if x has been compressed 
            # by the autoencoder in the time dimension
            ds_factor = t.shape[1] // x.shape[1]
            t = F.avg_pool1d(t.unsqueeze(1), ds_factor)[:,0,:]

        pos_enc = positional_encoding(t, x.shape[-1], add_dims=(2,3))
        return x + pos_enc

    def forward(self, z, timesteps):
        '''z is the latent representation of the conditioning and timesteps is contains the timesteps of the input frames (it is [-3, -2, -1, 0])'''
        z = self.proj(z)
        z = z.permute(0,2,3,4,1)
        z = self.analysis(z)
        
        if self.use_temporal_transformer:
            # add positional encoding
            z = self.add_pos_enc(z, timesteps)
            
            # transform to output shape and coordinates
            expand_shape = z.shape[:1] + (-1,) + z.shape[2:]
            pos_enc_output = positional_encoding(
                torch.arange(1,self.output_patches+1, device=z.device), 
                self.embed_dim, add_dims=(0,2,3)
            )
            pe_out = pos_enc_output.expand(*expand_shape)
            z = self.temporal_transformer(pe_out, z)

        
        # merge inputs
        z = self.fusion(z)
        # produce prediction
        z = self.forecast(z)
        return z.permute(0,4,1,2,3) # to channels-first order