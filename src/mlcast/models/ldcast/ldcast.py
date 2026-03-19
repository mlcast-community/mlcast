# new file with respect to original code

from ..base import NowcastingModelBase
import pytorch_lightning as L
from .data import LatentDataset, AutoencoderDataset, DataModule, load_in_memory
import torch
import contextlib

#torch.multiprocessing.set_start_method('spawn')

class LDCast(NowcastingModelBase):
    def __init__(self, ldm, autoencoder, sampler):
        super().__init__()
        self.ldm = ldm
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

        #assert False, 'need to add a trainer instance in the LatentDataset class to automatically move the autoencoder to cuda etc.'
        self.autoencoder.net.eval()

        dataset = LatentDataset(sampled_radar_dataset, self.autoencoder)
        datamodule = DataModule(dataset, **dataloader_kwargs)
        trainer = L.Trainer(**trainer_kwargs)
        trainer.fit(self.ldm, datamodule)
    
    def fit_autoencoder(self, sampled_radar_dataset, dataloader_kwargs = {}, trainer_kwargs = {}):
        
        dataset = AutoencoderDataset(sampled_radar_dataset)
        datamodule = DataModule(dataset, **dataloader_kwargs)
        trainer = L.Trainer(**trainer_kwargs)
        trainer.fit(self.autoencoder, datamodule)

    def predict(self, inputs, num_diffusion_iters = 50, verbose = True):
        '''inputs.shape = (batch_size, 1, 4, 256, 256)'''

        assert False, 'prediction should be implemented with a trainer, to take into account the switches of ema weights for example'''
        
        latent_inputs = self.autoencoder.encode(inputs)
        condition = self.ldm.net.conditioner(latent_inputs)

        gen_shape = (32, 5, 256//4, 256//4)
        batch_size = len(latent_inputs)

        # this could also be put in the LatentDiffusion class, by overriding the predict_step method (https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#inference)
        with contextlib.redirect_stdout(None):
            (s, intermediates) = self.sampler.sample(
                num_diffusion_iters, 
                batch_size,
                gen_shape,
                condition,
                progbar = verbose)

        return s
        
        latent_pred = self.ldm(latent_inputs)
        return self.autoencoder.net.decode(latent_pred)
        
    def save(self, folder):
        torch.save(self.autoencoder.net.state_dict(), f'{folder}/autoencoder.pt')
        torch.save(self.ldm.net.conditioner.state_dict(), f'{folder}/conditioner.pt')
        torch.save(self.ldm.net.denoiser.state_dict(), f'{folder}/denoiser.pt')

        if hasattr(self.ldm, 'ema'):
            self.ldm.ema.save(f'{folder}/ema.pt')

    def load(self, folder):
        self.autoencoder.net.load_state_dict(torch.load(f'{folder}/autoencoder.pt'))
        self.ldm.net.conditioner.load_state_dict(torch.load(f'{folder}/conditioner.pt'))
        self.ldm.net.denoiser.load_state_dict(torch.load(f'{folder}/denoiser.pt'))

        if hasattr(self.ldm, 'ema'):
            self.ldm.ema.load(f'{folder}/ema.pt')

    @classmethod
    def from_config(cls, config):
        
        from .autoenc.autoenc import Autoencoder
        from .diffusion.diffusion import LatentDiffusion
        from .diffusion.plms import PLMSSampler
        
        autoencoder = Autoencoder.from_config(config['autoencoder'])
        ldm = LatentDiffusion.from_config(config['ldm'], autoencoder)
        sampler = PLMSSampler(ldm.net.denoiser)
        
        return cls(ldm, autoencoder, sampler)

        
        
    