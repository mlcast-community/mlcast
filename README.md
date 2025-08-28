# mlcast

The MLCast Community is a collaborative effort bringing together meteorological services, research institutions, and academia across Europe to develop a unified Python package for AI-based nowcasting. This is an initiative of the E-AI WG6 (Nowcasting) of EUMETNET.

This repo contains the `mlcast` package for machine learning-based weather nowcasting.

## Project Status

⚠️ **Under Development** - This package is currently in early development stages and not usable by end users. The API and functionality are subject to change.

## Installation
```bash
# Install from pypi
pip install mlcast
```

or
```bash
# Install from source
git clone https://github.com/mlcast-community/mlcast
cd mlcast
uv pip install -e .

# For development
uv pip install -e ".[dev]"
```

## Project Structure

```
mlcast/
├── src/mlcast/          # Main package source code
│   ├── __init__.py      # Package initialization and version
│   ├── data/            # Data loading and preprocessing
│   │   ├── zarr_datamodule.py   # PyTorch Lightning data module for Zarr
│   │   └── zarr_dataset.py      # PyTorch dataset for Zarr arrays
│   ├── models/          # Lightning model implementations
│   │   └── base.py      # Abstract base classes for nowcasting models
│   └── modules/         # Pure PyTorch neural network modules
│       └── convgru_modules.py   # ConvGRU encoder-decoder modules
├── examples/            # Example scripts and notebooks
│   └── scripts/
│       └── simple_train.py      # Basic training example
├── pyproject.toml       # Project metadata and dependencies
├── LICENSE              # Apache 2.0 license
└── README.md            # This file
```

## Development

This project uses `uv` for dependency management. To set up the development environment:

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run pre-commit hooks
uv run pre-commit install
```

## Contributing

Please feel free to raise issues or PRs if you have any suggestions or questions.

## Links to presentations for discussion about the API

- [2024/02/04 first design discussions](https://docs.google.com/presentation/d/1oWmnyxOfUMWgeQi0XyX4fX9YDMX1vl6h/edit?usp=drive_link&rtpof=true&sd=true)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
