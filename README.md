# Regime Lab

A Python research app for exploring how market conditions change over time.

Regime Lab lets you look through market history, inspect model uncertainty and compare strategy behavior in an interactive dashboard. The interesting part is being able to question a model's outputs, rather than only seeing its final label.

## What you can explore

- Historical dates, market labels and model probabilities.
- Strategy equity curves, drawdowns and trading costs.
- Uncertainty and disagreement between model fits.
- Training windows and state history.
- Downloadable outputs and a self-contained HTML dashboard.

These views are available after generating a run with your own data.

## A few things to try after running it

1. **Compare two dates.** Look at the labels and how confident the model appears on each.
2. **Look beyond the ending balance.** Compare the strategy curves with their drawdowns and costs.
3. **Find an uncertain period.** Explore the uncertainty view and inspect where the fits disagree.
4. **Follow a state over time.** Use the training and state-history views to see how its description changes.
5. **Export something you noticed.** Download an output and compare it with the dashboard.

## What is available here?

The working source is included. The historical dataset, saved results and fitted models are not currently published here, so this checkout does not include a ready-to-open historical demo.

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
