import fiddle as fdl

from mlcast.config import (
    convgru_training_experiment,
    latent_diffusion_experiment,
    set_variables,
    toggle_masking,
    use_random_sampler,
    use_ratio_splits,
)
from mlcast.data.sequence import SourceDataPrecomputedSequenceDataset, SourceDataRandomSequenceDataset


def test_fiddler_set_variables() -> None:
    """Verify set_variables syncs dataset variables and network input_channels."""
    cfg = convgru_training_experiment.as_buildable()

    set_variables(cfg, ["rainfall_rate", "rainfall_flux"])

    assert cfg.data.sequence_dataset_factory.standard_names == ["rainfall_rate", "rainfall_flux"]
    assert cfg.pl_module.network.input_channels == 2


def test_fiddler_toggle_masking() -> None:
    """Verify toggle_masking syncs dataset mask return and module masked_loss."""
    cfg = convgru_training_experiment.as_buildable()

    toggle_masking(cfg, False)
    assert cfg.data.return_mask is False
    assert cfg.pl_module.masked_loss is False

    toggle_masking(cfg, True)
    assert cfg.data.return_mask is True
    assert cfg.pl_module.masked_loss is True


def test_fiddler_set_variables_on_latent_diffusion() -> None:
    """Verify set_variables applies to both stages of a LatentDiffusionTrainingExperiment."""
    cfg = latent_diffusion_experiment.as_buildable()

    set_variables(cfg, ["rainfall_rate", "rainfall_flux", "rainfall_intensity"])

    # Both stages share the same sequence_dataset_factory object
    expected_names = ["rainfall_rate", "rainfall_flux", "rainfall_intensity"]
    assert cfg.stage1.data.sequence_dataset_factory.standard_names == expected_names
    assert cfg.stage2.data.sequence_dataset_factory.standard_names == expected_names

    # Encoder (inside AutoencoderNet) has input_channels and should be updated
    assert cfg.stage1.pl_module.network.encoder.input_channels == 3
    assert cfg.stage2.pl_module.autoencoder.encoder.input_channels == 3


def test_fiddler_use_random_sampler_on_latent_diffusion() -> None:
    """Verify use_random_sampler applies to both stages of LatentDiffusionTrainingExperiment."""
    cfg = latent_diffusion_experiment.as_buildable()

    assert fdl.get_callable(cfg.stage1.data.sequence_dataset_factory) is SourceDataPrecomputedSequenceDataset
    assert fdl.get_callable(cfg.stage2.data.sequence_dataset_factory) is SourceDataPrecomputedSequenceDataset

    use_random_sampler(cfg)

    assert fdl.get_callable(cfg.stage1.data.sequence_dataset_factory) is SourceDataRandomSequenceDataset
    assert fdl.get_callable(cfg.stage2.data.sequence_dataset_factory) is SourceDataRandomSequenceDataset


def test_fiddler_use_ratio_splits_on_latent_diffusion() -> None:
    """Verify use_ratio_splits applies to both stages of LatentDiffusionTrainingExperiment."""
    cfg = latent_diffusion_experiment.as_buildable()

    use_ratio_splits(cfg, train=0.6, val=0.2)

    assert cfg.stage1.data.splits == {"time": {"train": 0.6, "val": 0.2, "test": 0.2}}
    assert cfg.stage2.data.splits == {"time": {"train": 0.6, "val": 0.2, "test": 0.2}}
