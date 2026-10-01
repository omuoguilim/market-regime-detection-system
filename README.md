# Market Regime Detection / Regime Lab

I built Regime Lab to study how market conditions change over time and whether a strategy reacts usefully to those changes. I started with a hidden Markov model, then added a walk-forward research pipeline and a dashboard so I could inspect the results rather than rely on one performance number.

## What I can investigate

- Historical regime labels and state probabilities.
- Hard regime allocation versus probability-weighted allocation and buy-and-hold.
- Equity curves, drawdowns, turnover and trading costs.
- Uncertainty, disagreements and changes between model fits.
- Training windows and state history.
- Saved research outputs in an HTML dashboard.

I treat this as retrospective research, not a live forecast or a promise of better returns. Lower drawdown alone does not establish a better strategy, especially when exposure and trading costs differ.

## Run a research pass

I use Python 3.12 and the pinned research dependencies.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python launch.py --retrain
```

I place the input at `data/sp500_clean.csv` with `Date`, `Open`, `High`, `Low`, `Close` and `Volume` columns. Historical data and completed run outputs are not bundled in this repository.

`python launch.py --retrain --no-browser` generates the dashboard without opening it. After a completed run, `python launch.py` opens the most recent available analysis.

## Code map

| File | What I use it for |
| --- | --- |
| `pipeline.py` | Features, model fits and strategy evaluation |
| `reliability.py` | Reliability diagnostics |
| `analyze.py` | Run comparisons |
| `dashboard.py`, `dashboard.html` | Dashboard generation and interface |
| `launch.py` | Training and opening a run |

## What I look at first

I compare costs and exposure before comparing returns. Then I check the probability history around unstable periods and inspect how a refit changed the state interpretation.

My existing walk-forward work is retrospective; it is not an untouched holdout. I still need independent validation before making claims about generalization.
