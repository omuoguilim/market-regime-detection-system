# Market Regime Detection

A research project exploring how a Gaussian Hidden Markov Model can describe changing conditions in the S&P 500 and inform an adaptive-exposure backtest.

## Repository status

**This repository currently contains a scaffold, not the runnable research project.** The Python modules, data-collection notebook, dashboard and requirements file are empty placeholders. The working notebooks, dataset, trained models and walk-forward pipeline have not been uploaded here.

There are no reproducible performance results in this checkout. Earlier experimental results should not be treated as validated out-of-sample performance.

## Research direction

The project investigates daily returns, rolling volatility, momentum and volume changes as inputs to a three-state model. The intended comparison is between buy-and-hold and a strategy whose exposure changes with the inferred market state.

The fuller research workflow needs chronological training and evaluation, transaction costs, careful timing of signals and a reproducible data source. State names such as bull, bear and transition are interpretations of the learned states.

## Planned layout

| Path | Intended role | Current contents |
| --- | --- | --- |
| `notebooks/01_data_collection.ipynb` | Data collection | Empty placeholder |
| `src/data_loader.py` | Load and clean market data | Empty placeholder |
| `src/features.py` | Construct model inputs | Empty placeholder |
| `src/model.py` | Fit and evaluate the regime model | Empty placeholder |
| `src/strategy.py` | Simulate exposure and costs | Empty placeholder |
| `dashboard/app.py` | Explore research outputs | Empty placeholder |
| `requirements.txt` | Reproducible dependencies | Empty placeholder |

## Before this can be run

Upload the working implementation, document the data source and date range, add dependency versions, and include chronological evaluation outputs and commands. Until those files are present, installing the empty requirements file or launching the dashboard will not run the project.

This repository documents a research project, not a trading recommendation.
