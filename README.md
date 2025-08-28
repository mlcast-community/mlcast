# mlcast


The MLCast Community is a collaborative effort bringing together meteorological services, research institutions, and academia across Europe to develop a unified Python package for AI-based nowcasting. This is an initiative of the E-AI WG6 (Nowcasting) of EUMETNET.

This repo contains the `mlcast` package for machine learning-based precipitation nowcasting.

## Project Status

⚠️ **Under Development** - This package is currently not usable and is being actively developed. The API and functionality are subject to change.

## Project Structure

- `src/mlcast/` - Main package source code
  - `models/` - Core model definitions
    - `base.py` - (Abstract) base classes for nowcasting models
  - `modules/` - Neural network torch modules and components
    - `convgru_modules.py` - ConvGRU-based encoder-decoder modules
- `api_reference/` - Reference implementations and API examples
  - `pysteps_ref.py` - PySteps reference implementation
  - `pl_ref.py` - PyTorch Lightning reference implementation
  - `mlcast_api.py` - Proposed API design example

Please feel free to raise issues or PRs if you have any suggestions or questions.

## Links to presentations for discussion about the API

- [2024/02/04 first design discussions](https://docs.google.com/presentation/d/1oWmnyxOfUMWgeQi0XyX4fX9YDMX1vl6h/edit?usp=drive_link&rtpof=true&sd=true)
