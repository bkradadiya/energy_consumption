# Energy Forecasting – Runner

## Quickstart
1. `python -m venv .`
2. Activate the venv
3. `pip install -r requirements.txt`
4. `python -m run -h`   # see CLI help

### Commands
- `python -m run data`                  # preview dataset
- `python -m run train --timesteps 240 --model-dir app/models_seq`
- `python -m run predict --model-dir app/models_seq`
