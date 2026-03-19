# Latent diffusion documentation

1. [LatentDiffusion class](#latentdiffusion-class)
2. [LatentDiffusionNet](#latentdiffusionnet)
3. [Training vs inference modes](#training-vs-inference-modes)
4. [Loading original weights](#loading-original-weights)
5. [Exponential Moving Average](#exponential-moving-average)
6. [LatentDataset](#latentdataset)
7. [Background on diffusion models](#background-on-diffusion-models)

## LatentDiffusion class

The `LatentDiffusion` class is a subclass of `NowcastingLightningModule` and takes three arguments
 - the `net` (typically, an instance of `LatentDiffusionNet`)
 - the `loss` (a `torch.nn.MSELoss` for LDCast)
 - the `scheduler`, scheduling the diffusion process
Options for the optimizer and the learning rate scheduler can be passed as well.

An instance can be created from a `dict` containing the configuration, based on the corresponding part of the architecture of LDCast:
```python
from mlcast.models.ldcast.diffusion.diffusion import LatentDiffusion
ldm = LatentDiffusion.from_config(config)
```

## LatentDiffusionNet

The `LatentDiffusionNet` class combines two elements: the `conditioner` and the `denoiser`.

The `denoiser` takes some noise and performs the backward diffusion process to produce samples (in latent space). Since we want a nowcast based on input images, the denoiser needs with some condition based on the input images.

The role of the `conditioner` is to provide this condition to the denoiser. It takes input images (encoded in latent space) and returns a condition (also called context ?) to help the denoiser to produce relevant predictions. The `conditioner` could also be called a forecaster.

In the original LDCast code, the `conditioner` was called `analysis_net` and the `denoiser` was called `denoiser` or `model`.

As in the original code, the `conditioner` is an instance of `AFNONowcastNet` and the `denoiser` is an instance of `UNetModel`.

**check the output shape of the conditioner and the input shape of the denoiser**

## Training vs inference modes

The `net` combines the `conditioner` and the `denoiser` in the way they should be trained (i.e. the `denoiser` is always called after the `conditioner`). The inference process is however different: once the input has been converted in latent space, it is passed to the conditioner to produce a condition; then, the denoiser is repeatedly called to iteratively denoise a completely noisy image according to a scheme defined by a sampler (see [Background on diffusion models](background-on-diffusion-models)). During the inference, `net.forward` is thus never called.

## Loading original weights

The structure of this part of LDCast has changed a little with respect to the original code, so the weights need to be reorganized before being loaded. The `convert_original_weights` function does this:
```python
from mlcast.models.ldcast.original_weights import convert_original_weights
ldm_weights_fn = '/path/to/original/ldm/genforecast/weights'
state_dict = convert_original_weights(ldm_weights_fn)
```
`state_dict`is a `dict` whose keys are `conditioner`, `denoiser`, `ema` and `unmatched`. `state_dict['unmatched']` contains the elements which were not matched in `convert_original_weights` (should be empty). The weights of the conditioner, of the denoiser and the EMA can then be loaded as
```
ldm.net.conditioner.load_state_dict(state_dict['conditioner'])
ldm.net.denoiser.load_state_dict(state_dict['denoiser'])
ldm.ema.load(state_dict['ema'])
```

## Exponential Moving Average

The original code included an Exponential Moving Average (EMA) of the weights of the denoiser. This seems quite common for diffusion models.

The idea is two versions of the weights of the models:
 1. the usual weights, which are updated through the optimization of the training loss
 2. the EMA weights, which are computed as an average of the last values of the usual weights

The average is exponentially weighted, so that the latest weights are more taken into account.

This is useful because the usual weights are quite unstable, while the EMA weights are more stable because of the average.

The EMA is switched on by default, and can be switched off when creating the `LatentDiffusion` class (setting the keyword `ema_config` to `{'use': False}`). When switched on, the original weights have to be loaded into the model for the computation of the training loss, but the loss with the EMA weights should be computed during validation (and also after each training step ?). The EMA weights should also be used during inference. Everything is handled automatically if a `pl.Trainer` is used (through lightning hooks on the `LatentDiffusion` class). This means that one also needs to use a `pl.Trainer` at inference.

## LatentDataset

Gabriele's `SampledRadarDataset` returns sequence of `steps` radar images (with `steps` usually set to 24). However, the training of the `ldm` requires samples in latent space. This is handled by the `LatentDataset` class: it returns samples `(x, y)` with `x` being the latent encoding of the input radar images, and `y` being the latent encoding of the radar images to predict. The `LatentDataset` thus needs the trained `autoencoder.net` !

## Background on diffusion models

See https://huggingface.co/blog/annotated-diffusion for some notations and formulas.

During training, we start from a sample $x_0$ and create a series of samples $x_0, x_1, ..., x_T$ according to the formula (forward diffusion)

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha_t}}\epsilon_t, \quad t = 1, ..., T
$$

where $\epsilon_t \sim \mathcal{N}(0, 1)$. The constants $\bar{\alpha}_t$ are chosen, but they need the property that $\bar{\alpha}_t \to 0$ as $t \to T$, so that $x_T \sim \mathcal{N}(0, 1)$. These constants are computed with an algorithm called a scheduler.

From a given $x_t$, the model can either be trained to predict $x_0$, $\epsilon_t$ or the velocity $v_t$. The latter is defined as

$$
v_t = \sqrt{\bar{\alpha}_t} \epsilon_t - \sqrt{1-\bar{\alpha}_t} x_0,
$$

which is equivalent to

$$
x_0 = \sqrt{\bar{\alpha}_t} x_t - \sqrt{1-\bar{\alpha}_t} v_t.
$$

The model is also given the timestep $t$. The loss is computed by comparing the target quantity ($\epsilon_t$, $x_0$ or $v_t$) with the predicted quantity by the model. Choosing to predict $x_0$, $\epsilon_t$ or $v_t$ is conceptually equivalent, the difference is in the numerical properties of the scheme (like in ODE integration schemes).

The validation and test steps are done in the same way.

So the model is trained to predict $x_0$ (or something from which we can compute $x_0$) from $x_T\sim \mathcal{N}(0, 1)$. But for large values of $t$, this prediction is actually quite bad. During actual prediction, the prediction is usually iteratively refined with sampler schemes. The idea is that, from the noise predicted based on $x_T$, the sampler scheme allows to compute $x_{T - \Delta t}$. The model is then used to predict the noise based on this estimation of $x_{T - \Delta t}$, and the sampler scheme allows to deduce $x_{T - 2\Delta t}$, etc. $\Delta t$ is usually taken of the order of 50 (while $T$ is usually 1000).

In Hugging Face Diffusers library, the scheduler and the sampler parts are often combined in one object called a scheduler, but the sampler part is only used during inference.