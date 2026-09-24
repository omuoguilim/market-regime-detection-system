# Regime Lab

A market-regime research app by Oluchi Muoguilim, with a Python pipeline and an interactive HTML dashboard.

## Repository status

The working source code is now included. The historical dataset, saved research results and fitted models have not been published in this repository.

## Run with your own data

Use Python 3.12. Clone this repository, then run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
```

Place a CSV at `data/sp500_clean.csv` with Date, Open, High, Low, Close and Volume columns and enough history for training. Then run:

```sh
python launch.py --retrain
```

The launcher generates a timestamped result folder and opens its dashboard. Use `--no-browser` to build without opening a browser.

## Source map

- `pipeline.py`: features, model fitting and backtest execution.
- `reliability.py`: state alignment and model diagnostics.
- `analyze.py`: comparisons and report generation.
- `dashboard.py`, `dashboard.html`: dashboard builder and interface.
- `launch.py`: launcher and retraining command.

This is an offline research project, not a live trading service or investment recommendation. Historical results are not claims of future performance.
