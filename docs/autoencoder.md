# Autoencoder documentation

1. [Autoencoder class](#autoencoder-class)
2. [Tensor shapes](#tensor-shapes)
3. [Encoding and decoding](#encoding-and-decoding)
4. [Loading original weights](#loading-original-weights)
5. [Antialiasing](#antialiasing)
6. [Autoencoder training dataset](#autoencoder-training-dataset)
7. [Background on variational autoencoders](#background-on-variational-autoencoders)

## Autoencoder class

The `Autoencoder` class is a subclass of `NowcastingLightningModule`, and takes three arguments:
 - the `net` (an instance of `AutoencoderKLNet` for LDCast), which is the neural network of the autoencoder, containing the decoder and the autoencoder
 - the `loss` (an instance of `AutoencoderLoss` for LDCast)
Options for the optimizer and the learning rate scheduler can be passed as well.

An instance can be created from a `dict` containing the configuration, based on the architecture of LDCast's autoencoder:
```python
from mlcast.models.ldcast.autoencoder.autoencoder import Autoencoder
autoencoder = Autoencoder.from_config(config)
```

## Tensor shapes

The autoencoder encodes sequences of radar images (not image by image). The number of radar images encoded at once is given by `autoenc_time_ratio` and was set to 4 in the original code (and kept here). `Conv3d` layers are used for the encoding, so input tensors have shape
```
(batch_size, n_channels, autoenc_time_ratio,) + spatial shape
```
`n_channels` is always 1 for radar images.

In latent space, the tensors have shape `(batch_size, 32, n, 64, 64)`, where 32 is the `hidden_width` of the `autoencoder` and `n` is the number of consecutive encoded radar images divided by `autoenc_time_ratio`. **I should still clarify which of these parameters can be changed freely, and how it affects other shapes. Can `autoencoder.net` encode a e.g. 8 images at once (in which case `n` is 2) ?**


## Encoding and decoding

Doing the following
```python
import torch
inputs = torch.randn(1, 1, 4, 256, 256, device = 'cuda') # fake sample
autoencoder(inputs)
```
is equivalent to `autoencoder.net(inputs)` and computes the whole forward pass through the `net` (encoding + decoding). To encode only, one needs to do
```python
autoencoder.net.encode(inputs).
```
If `encoded` is an encoded sample, it can be decoded as
```python
autoencoder.net.decode(encoded)
```

## Laoding original weights

The original weights can be loaded directly as
```python
autoenc_weights_fn = '/path/to/original/autoencoder/weights'
autoencoder.net.load_state_dict(torch.load(autoenc_weights_fn))
```

## Antialiasing

As in the original code, antialiasing is applied by default (by an Antialiaser object) to the inputs before being fed to the `net`.

## Autoencoder training dataset

Gabriele's code produces a dataset whose samples are sequences of `steps` images (`steps` is usually set to 24, to have 4 input images and 20 ground truth images).

But the autoencoder needs samples which are sequences of only 4 images, so each sample in `SampledRadarDataset` needs to be divided in 6 samples. This is done by the `AutoencoderDataset`. Its samples are tuple `(x, y)` where `y = x` since we want the autoencoder to reconstruct the sequences.

**The current implementation of this class is not the most efficient since, when going through the `AutoencoderDataset`, each sample of the `SampledRadarDataset` is loaded 6 times.**

## Background on variational autoencoders

The autoencoder used in LDCast is a variational autoencoder. Here is some background on that kind of autoencoder.

Source https://medium.com/@jpark7/finally-a-clear-derivation-of-the-vae-kl-loss-4cb38d2e47b3.

Variational autoencoders encode the data through a normal distribution in latent space: each sample is represented by the mean and the standard deviation of the normal distribution. When decoding the sample, a new sample is created resembling the original sample, but is not quite the same. The degree to which we force the decoded samples to resemble the original ones is tuned by the `kl_weight` parameter of the KL loss function.

When using the encoded sample (for example to produce a condition with the conditioner), only the mean is used. In the original code, `autoencoder.net.decode` was returning a tuple `(mean, log_var)`, so that one had to select the mean with `autoencoder.net.decode(x)[0]`, which is not very clear. I replaced this by adding a keyword `return_log_var` in `autoencoder.net.decode`.