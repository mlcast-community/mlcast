# new file with respect to original code

from ..base import NowcastingModelBase
import pytorch_lightning as L
from .data import LatentDataset, AutoencoderDataset, DataModule
from torch.utils.data import DataLoader
import torch
import contextlib
from torch.utils.data import TensorDataset


class LDCast(NowcastingModelBase):
    def __init__(self, ldm_lightning, autoencoder, sampler):
        super().__init__()
        self.ldm_lightning = ldm_lightning
        self.autoencoder = autoencoder
        self.sampler = sampler
    
    def fit(self, sampled_radar_dataset, dataloader_kwargs = {}, trainer_kwargs = {}):
        '''dataset should contains pairs of (inputs, true), with
        inputs.shape = (batch_size, 1, 4, 256, 256)
        true.shape = (batch_size, 1, 20, 256, 256)
        '''
        print('Training autoencoder')
        self.fit_autoencoder(sampled_radar_dataset, dataloader_kwargs = dataloader_kwargs, trainer_kwargs = trainer_kwargs)

        print('Training ldm')
        self.fit_ldm(sampled_radar_dataset, dataloader_kwargs = dataloader_kwargs, trainer_kwargs = trainer_kwargs)

    def fit_ldm(self, sampled_radar_dataset, dataloader_kwargs = {}, trainer_kwargs = {}):
        self.autoencoder.net.eval()
        self.ldm_lightning.net.train()

        dataset = LatentDataset(sampled_radar_dataset, self.autoencoder.net)
        datamodule = DataModule(dataset, **dataloader_kwargs)
        trainer = L.Trainer(**trainer_kwargs)
        trainer.fit(self.ldm_lightning, datamodule)
    
    def fit_autoencoder(self, sampled_radar_dataset, dataloader_kwargs = {}, trainer_kwargs = {}):
        self.autoencoder.net.train()

        dataset = AutoencoderDataset(sampled_radar_dataset)
        datamodule = DataModule(dataset, **dataloader_kwargs)
        trainer = L.Trainer(**trainer_kwargs)
        trainer.fit(self.autoencoder, datamodule)

    def predict(self, inputs, num_diffusion_iters = 50, verbose = True):
        '''inputs.shape = (batch_size, 1, 4, 256, 256)'''

        assert False, 'prediction should be implemented with a trainer, to take into account the switches of ema weights for example'''
        
        latent_inputs = self.autoencoder.net.encode(inputs)
        condition = self.ldm_lightning.net.conditioner(latent_inputs)

        gen_shape = (32, 5, 256//4, 256//4)
        batch_size = len(latent_inputs)

        with contextlib.redirect_stdout(None):
            (s, intermediates) = self.sampler.sample(
                num_diffusion_iters, 
                batch_size,
                gen_shape,
                condition,
                progbar = verbose)

        return s
        
        latent_pred = self.ldm_lightning(latent_inputs)
        return self.autoencoder.net.decode(latent_pred)
        
    def save(self, folder):
        torch.save(self.autoencoder.net.state_dict(), f'{folder}/autoencoder.pt')
        torch.save(self.ldm_lightning.net.conditioner.state_dict(), f'{folder}/conditioner.pt')
        torch.save(self.ldm_lightning.net.denoiser.state_dict(), f'{folder}/denoiser.pt')

        if hasattr(self.ldm_lightning, 'ema'):
            self.ldm_lightning.ema.save(f'{folder}/ema.pt')

    def load(self, folder):
        self.autoencoder.net.load_state_dict(torch.load(f'{folder}/autoencoder.pt'))
        self.ldm_lightning.net.conditioner.load_state_dict(torch.load(f'{folder}/conditioner.pt'))
        self.ldm_lightning.net.denoiser.load_state_dict(torch.load(f'{folder}/denoiser.pt'))

        if hasattr(self.ldm_lightning, 'ema'):
            self.ldm_lightning.ema.load(f'{folder}/ema.pt')

    @classmethod
    def from_config(cls, config):
        
        if isinstance(config, str):
            import yaml
            with open(config, 'r') as file:
                config = yaml.safe_load(file)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        from .autoenc.autoenc import AutoencoderKLNet, autoenc_loss
        from ..base import NowcastingLightningModule
        from .diffusion.unet import UNetModel
        from .context.context import AFNONowcastNetCascade
        from .diffusion.diffusion import LatentDiffusion, LatentDiffusionLightning
        from torch.nn import L1Loss
        from .diffusion.scheduler import Scheduler
        from .diffusion.plms import PLMSSampler
        
        autoencoder = NowcastingLightningModule(AutoencoderKLNet(), autoenc_loss()).to(device)
        conditioner = AFNONowcastNetCascade(**config['conditioner']).to(device)
        denoiser = UNetModel(**config['denoiser']).to(device)
        ldm = LatentDiffusion(conditioner, denoiser)
        ldm_lightning = LatentDiffusionLightning(ldm, L1Loss(), Scheduler(), ema_config = config['ema'])
        sampler = PLMSSampler(denoiser)
        
        return cls(ldm_lightning, autoencoder, sampler)

        
        
    