# MLCast implementation of LDCast

see main branch https://github.com/mlcast-community/mlcast for details.

```python
future_timesteps = 20
autoenc_time_ratio = 4 # number of timesteps encoded in the autoencoder
```
Here, 4 consecutive radar images are encoded at once.

# Autoencoder

```python
from src.mlcast.models.ldcast.autoenc.autoenc import AutoencoderKLNet, autoenc_loss
from src.mlcast.models.base import NowcastingLightningModule
autoencoder = NowcastingLightningModule(AutoencoderKLNet(), autoenc_loss()).to('cuda')
```
The autoencoder is an instance of the NowcastingLightningModule. Training the autoencoder:
```python
# create fake data
inputs = torch.randn(2, 1, 4, 256, 256, device = 'cuda')

with torch.no_grad():
    # the forward pass of the autoencoder returns also the encoding
    # so [0] is needed to select the decoded part only
    y = autoencoder(x, 4)[0]
batch = (x, y)

import pytorch_lightning as L
trainer = L.Trainer()
trainer.fit(autoencoder, batch)
```
The inputs tensors have shape `(batch_size, n_channels, number of input radar images,) + spatial shape`. In latent space, the tensors have shape `(batch_size, 32, n, 64, 64)`, where 32 is the `hidden_width` of the `autoencoder` and `n` is the number of consecutive encoded radar images divided by `autoenc_time_ratio` (set to 4).

The original weights can be loaded directly as
```python
autoenc_weights_fn = '/path/to/original/autoencoder/weights'
autoencoder.net.load_state_dict(torch.load(autoenc_weights_fn))
```

# Latent diffusion (= conditioner + denoiser)
The `LatentDiffusion` class is a `nn.Module` combining the conditioner and the denoiser.
```python
# setup forecaster
conditioner = AFNONowcastNetCascade(
    32,
    train_autoenc=False,
    output_patches=future_timesteps//autoenc_time_ratio,
    cascade_depth=3,
    embed_dim=128,
    analysis_depth=4
).to('cuda')

# setup denoiser
from src.mlcast.models.ldcast.diffusion.unet import UNetModel
denoiser = UNetModel(in_channels=autoencoder.net.hidden_width,
    model_channels=256, out_channels=autoencoder.net.hidden_width,
    num_res_blocks=2, attention_resolutions=(1,2), 
    dims=3, channel_mult=(1, 2, 4), num_heads=8,
    num_timesteps=future_timesteps//autoenc_time_ratio,
    context_ch=[128, 256, 512] # context channels (= analysis_net.cascade_dims)
                    ).to('cuda')

from src.mlcast.models.ldcast.diffusion.diffusion import LatentDiffusion
ldm = LatentDiffusion(conditioner, denoiser)
```
The `LatentDiffusion` class has a forward pass: it takes the noise, the timesteps of the diffusion and the encoded inputs
```python
latent_inputs = autoencoder.net.encode(inputs)
noise = torch.randn(2, 32, 5, 64, 64, device = latent_inputs.device)
t = torch.tensor([2, 3], device = latent_inputs.device)
ldm((t, noise, latent_inputs))
```
The noise has to have the shape true radar images encoded in latent space.

Create fake data to train the ldm:
```python
from torch.utils.data import TensorDataset
true = torch.randn(2, 1, future_timesteps, 256, 256, device = 'cuda')
dataset = TensorDataset(inputs, true)
```
Create a ```Dataset``` which convert the samples in latent space with the autoencoder
```
self.autoencoder.net.eval()
latent_dataset = LatentDataset(dataset, autoencoder.net)
dataloader = DataLoader(latent_dataset, batch_size=2)
```
Put `ldm` in a `LightningModule` and train:
```python
from torch.nn import L1Loss
from src.mlcast.models.ldcast.diffusion.scheduler import Scheduler
from src.mlcast.models.ldcast.diffusion.diffusion import LatentDiffusionLightning

ldm_lightning = LatentDiffusionLightning(ldm, L1Loss(), Scheduler())
trainer = L.Trainer()
trainer.fit(ldm_lightning, dataloader)
```

The original weights can not be directly loaded because the models are structured a little differently, but the original weights can be loaded with
```python
ldm_weights_fn = '/path/to/original/ldm/genforecast/weights'
unexpected_keys = ldm_lightning.load_original_weights(ldm_weights_fn)
```
`unexpected_keys` contains the keys that were not loaded (only the ema weights because I did not take care of the ema scope for the moment)

# Main LDCast class

```python
from src.mlcast.models.ldcast.ldcast import LDCast
ldcast = LDCast(ldm_lightning, autoencoder)
```

# TO DO

During training, an EMA scope was used for the weights of the denoiser, I removed this for the moment, but it should reincluded in some way.

The 'timesteps' variable sometimes refers to the timesteps of the diffusion process (= 1000 during training) and sometimes refers to the nowcasting timesteps (where each time step = 5 minutes). Better to have different names.

I have understood that samplers are only used in inference ! The training (and validation) step is always done by predicting the noise (or a quantity which is related to it by a simple formula). What I called previously the SimpleSampler is actually simply a scheduler (which determines the values of alphas and betas, and add the noise on the latent samples during training)

We might integrate this code within the Hugging Face Diffusers Library.

It remains mainly to write code in the main LDCast class (in `ldcast.py`)

# Basics on diffusion models

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

The original code was using antialiasing before feeding the samples to the model (at least during inference), I should add this

# The variational autoencoder

Source https://medium.com/@jpark7/finally-a-clear-derivation-of-the-vae-kl-loss-4cb38d2e47b3.

Variational autoencoders encode the data through a normal distribution in latent space: each sample is represented by the mean and the standard deviation of the normal distribution. When decoding the sample, a new sample is created resembling the original sample, but is not quite the same. The degree to which we force the decoded samples to resemble the original ones is tuned by the `kl_weight` parameter of the KL loss function.

When using the encoded sample (for example to produce a condition with the conditioner), only the mean is used. In the original code, `autoencoder.decode` was returning a tuple `(mean, log_var)`, so that one had to select the mean with `autoencoder.decode(x)[0]`, which is not very clear. I replaced this by adding a keyword `return_log_var` in `autoencoder.decode`.
