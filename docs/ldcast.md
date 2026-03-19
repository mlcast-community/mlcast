# Main LDCast class documentation

1. [LDCast class](#ldcast-class)
2. [Inference](#inference)
3. [Loading/saving weights](#loading/saving-weights)
4. [Training](#training)

## LDCast class

The `LDCast` class is a subclass of `NowcastingModelBase` and takes three arguments
 - the `ldm` (typically, an instance of `LatentDiffusion`)
 - the `autoencoder` (typically, an instance of `Autoencoder`)
 - the `sampler`

An instance can be created from a `dict` containing the configuration, based on the architecture of LDCast:
```python
from mlcast.models.ldcast.ldcast import LDCast
ldcast = LDCast.from_config(config)
```
A config very close to what was used in the original code is in 'config.yaml'. It should be loaded as
```python
from omegaconf import OmageConf
OmegaConf.register_new_resolver("as_class", lambda class_name: eval(class_name))
config = OmegaConf.load('config.yaml')
```

## Inference

Predictions can be produced with
```python
import torch
inputs = torch.randn(1, 1, 4, 256, 256, device = 'cuda') # fake data
ldcast.predict(inputs)
```
**Do not use for the moment, since the EMA weights (if used) are not automatically used for inference**

## Loading/saving weights
To load from a folder containing in different files the weights of the autoencoder, of the denoiser and of the conditioner (and possibly ema weights):
```python
ldcast.load('/path/to/folder')
```
To save in a folder:
```python
ldcast.save('/path/to/folder')
```

## Training

If `sampled_radar_dataset` is a `SampledRadarDataset` built with Gabriele's code (https://github.com/DSIP-FBK/ConvGRU-Ensemble/blob/main/convgru_ensemble/datamodule.py), the autoencoder can be trained with
```python
ldcast.fit_autoencoder(sampled_radar_dataset)
```
and the ldm can be trained
```python
ldcast.fit_ldm(sampled_radar_dataset)
```
Keyword arguments can be passed to the trainer and the dataloader through the `trainer_kwargs` and `dataloader_kwargs` keywords.