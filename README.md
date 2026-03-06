# MLCast implementation of LDCast

see main branch https://github.com/mlcast-community/mlcast for context.

## Code structure

There is one main `LDCast` class, subclassing the `NowcastingModelBase` class. There are three main nets in LDCast:
 - the autoencoder
 - the conditioner
 - the denoiser

The `NowcastingLightningModule` is subclassed by the smaller composites of nets that should be trained at once. This gives two subclasses in this case:
 - the autoencoder (encoder + decoder) has to be trained on its own, so there is one subclass of `NowcastingLightningModule` called `Autoencoder`
 - the conditioner and the denoiser have to be trained together, so they are combined into one neural network (the `LatentDiffusionNet` class), whose training is handled by the `LatentDiffusion` subclass of the `NowcastingLightningModule`

## Documentation

See `docs` folder for some documenation on the main `LDCast` class, on the autoencoder and on the latent diffusion part.

## TO DO

reorganize the `LatentDiffusion` class ? for the moment, `LatentDiffusionNet.forward` is never called during inference because the inference process is quite different than in training (see `docs/ldm.md). It might be maybe a bit clearer to reorganize that by implementing explicitly different training and inference step methods in the `LatentDiffusion` class (that being said, `AutoencoderKLNet.forward` is never called either during inference)

The 'timesteps' variable sometimes refers to the timesteps of the diffusion process (= 1000 during training) and sometimes refers to the nowcasting timesteps (where each time step = 5 minutes). Better to have different names.

We might integrate this code within the Hugging Face Diffusers Library.

It remains mainly to write code in the main LDCast class (in `ldcast.py`)

It would be nice to rewrite the PLMS sampler, it is a little messy

implement different parametrization than 'eps'

use ZarrDataModule and ZarrDataset !

add the computation of the EMA loss during the ldm training, change the LDCast.predict method so that EMA weights are automatically used during inference

add in the code (and in the doc) the input and output shapes of the nets

understand which parameters can be changed, which have to be adapted when others change

make the implementation of the `AutoencoderDataset` more efficient ? (see docs/autoencoder)