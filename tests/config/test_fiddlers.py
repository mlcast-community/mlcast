from mlcast.config import set_variables, toggle_masking, training_experiment


def test_fiddler_set_variables():
    """Verify set_variables syncs dataset variables and network input_channels."""
    cfg = training_experiment.as_buildable()

    # Apply fiddler
    set_variables(cfg, ["rainfall_rate", "rainfall_flux"])

    # Check sync
    assert cfg.data.dataset_factory.standard_names == ["rainfall_rate", "rainfall_flux"]
    assert cfg.pl_module.network.input_channels == 2


def test_fiddler_toggle_masking():
    """Verify toggle_masking syncs dataset mask return and module masked_loss."""
    cfg = training_experiment.as_buildable()

    # Disable masking
    toggle_masking(cfg, False)
    assert cfg.data.dataset_factory.return_mask is False
    assert cfg.pl_module.masked_loss is False

    # Enable masking
    toggle_masking(cfg, True)
    assert cfg.data.dataset_factory.return_mask is True
    assert cfg.pl_module.masked_loss is True
