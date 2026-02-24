import torch
import torch.nn as nn
import pytorch_lightning as L
from typing import Any
import contextlib
from src.mlcast.models.base import NowcastingLightningModule
import numpy as np
from src.mlcast.models.ldcast.diffusion.utils import extract_into_tensor

print('take care of ema scope, which was used as context manager each exactly when denoiser.forward was called, so it should be a taken care of in the code code about the denoiser or about the diffuser (nothing to do with samplers)')  

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
    def __init__(self, ldm, loss, scheduler):
        super().__init__(ldm, loss)
        self.scheduler = scheduler

        # register the schedules (i.e. the values of alpha, beta etc).
        self.register_schedule()

    def register_schedule(self):
        
        schedule = self.scheduler.schedule(torch.float32, next(self.net.parameters()).device)
        
        # check if the ldm has already some saved buffers
        saved_buffers = dict(self.net.named_buffers()).keys()
        already_saved = [name for name in schedule.keys() if name in saved_buffers]
        if len(already_saved) > 0:
            raise AttributeError(f'The denoiser has already some saved values for {already_saved}')
        
        for k, v in schedule.items():
            self.net.register_buffer(k, v)
    
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
        
        x_noisy = extract_into_tensor(self.net.sqrt_alphas_cumprod, t, x0.shape) * x0 + \
            extract_into_tensor(self.net.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        
        return t, noise, x_noisy

    def load_original_weights(self, ldm_weights_fn):
        '''
        load the weights of the denoiser and the conditioner, from the way they were saved originally
        at the moment, the ema scope is not taken into account
        returns the weights which were not loaded in one of the two nets (should be only those related to the ema scope)
        '''
        ldm_state_dict = torch.load(ldm_weights_fn)

        # track the keys
        ldm_keys = list(ldm_state_dict.keys())
        
        # remove the weights of the autoencoder
        for k in ldm_keys.copy():
            if k.startswith('autoencoder.') or k.startswith('context_encoder.autoencoder.'):
                ldm_keys.remove(k)

        # extract the keys of the denoiser (it was called 'model' in the original code)
        denoiser_state_dict = {}
        for k in ldm_keys.copy():
            if k.startswith('model.'):
                new_key = k.replace('model.', '')
                denoiser_state_dict[new_key] = ldm_state_dict[k]
                ldm_keys.remove(k)
        
        # extract the keys of the conditioner (it was called 'context_encoder' in the original code)
        conditioner_state_dict = {}
        for k in ldm_keys.copy():
            if k.startswith('context_encoder.'):
                    new_key = k.replace('context_encoder.', '')
                    conditioner_state_dict[new_key] = ldm_state_dict[k]
                    ldm_keys.remove(k)

        # proj, temporal_transformer and analysis were lists one only element, I simplified this
        # the keys have to be adapted
        new_conditioner_state_dict = {}
        for k, v in conditioner_state_dict.items():
            new_key = k
            if k.startswith('proj.0.'):
                new_key = k.replace('proj.0.', 'proj.')
            if k.startswith('temporal_transformer.0.'):
                new_key = k.replace('temporal_transformer.0.', 'temporal_transformer.')
            if k.startswith('analysis.0.'):
                new_key = k.replace('analysis.0.', 'analysis.')
            new_conditioner_state_dict[new_key] = v
        conditioner_state_dict = new_conditioner_state_dict
        
        self.net.conditioner.load_state_dict(conditioner_state_dict)
        self.net.denoiser.load_state_dict(denoiser_state_dict)

        # check that the buffers saved in self.net are the same than the original ones
        for buffer in self.net.named_buffers():
            name, value = buffer
            assert (value == ldm_state_dict[name].to(value.device)).all()
            ldm_keys.remove(name)
        
        return ldm_keys

        

class LatentNowcaster(L.LightningModule):
    """Base class for PyTorch Lightning modules used in nowcasting models.

    This class provides a standard interface for training and validation
    steps, as well as optimizer configuration.
    """

    def __init__(
        self,
        conditioner: nn.Module,
        denoiser: nn.Module,
        loss: nn.Module,
        training_sampler: nn.Module,
        inference_sampler: nn.Module,
        optimizer_class: Any | None = None,
        optimizer_kwargs: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["denoiser", "conditioner", "training_sampler", "inference_sampler", "loss"])
        self.conditioner = conditioner
        self.denoiser = denoiser
        self.loss = loss
        self.training_sampler = training_sampler
        self.inference_sampler = inference_sampler
        self.optimizer_class = torch.optim.Adam if optimizer_class is None else optimizer_class

        training_sampler.register_schedule(denoiser)

    def infer(self, latent_inputs, num_diffusion_iters = 50, verbose = True):

        condition = self.conditioner(latent_inputs)
        
        gen_shape = (32, 5, 256//4, 256//4)
        batch_size = len(list(condition.values())[0])
        with contextlib.redirect_stdout(None):
            (s, intermediates) = self.inference_sampler.sample(
                num_diffusion_iters, 
                batch_size,
                gen_shape,
                condition,
                progbar=verbose
            )
        return s

    def model_step(self, latent_batch: Any, batch_idx: int, step_name: str = "train") -> torch.Tensor:
        """Generic model step for training or validation.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch

        Returns:
            Loss value for the current batch
        """
        latent_inputs, latent_targets = latent_batch
        
        condition = self.conditioner(latent_inputs)
        t, noise, latent_target_noisy = self.training_sampler.q_sample(self.denoiser, latent_targets)
        guessed_noise = self.denoiser(latent_target_noisy, t, context = condition)
        loss = self.loss(guessed_noise, noise)
        
        if isinstance(loss, dict):
            # append step name to loss keys for logging
            loss = {f"{step_name}/{k}": v for k, v in loss.items()}
            self.log_dict(loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            loss = loss.get(f"{step_name}/total_loss", None)
            if loss is None:
                raise ValueError(f"Loss is None for step {step_name}. Ensure loss function returns a valid tensor.")
        else:
            self.log(f"{step_name}/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Training step for a single batch.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch

        Returns:
            Loss value for the current batch
        """
        return self.model_step(batch, batch_idx, step_name="train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Validation step for a single batch.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch

        Returns:
            Loss value for the current batch
        """
        return self.model_step(batch, batch_idx, step_name="val")
        
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for training.

        Returns:
            Optimizer instance to use for training
        """
        return self.optimizer_class(self.parameters(), **(self.hparams.optimizer_kwargs or {}))
    

    def on_train_start(self):
        self._current_sampler = self.training_sampler
        super().on_train_start()
    
    def on_validation_start(self):
        self._current_sampler_mode = self.training_sampler
        super().on_validation_start()
    
    def on_predict_start(self):
        self._current_sampler_mode = self.inference_sampler
        super().on_predict_start()
    
    def on_test_start(self):
        # training or inference sampler ???
        self._current_sampler_mode = self.training_sampler
        super().on_test_start()
