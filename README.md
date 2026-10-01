# Regime Lab

I built Regime Lab to study changing market conditions and check how useful a model’s assessments really are. I started with a hidden Markov model for the S&P 500, then added a walk-forward pipeline, strategy comparisons and a dashboard.

The question I’m exploring now is: **can the model recognize when it should hold back?**

[Open the research demo](https://oluchi-muoguilim.superct3663.chatgpt.site/demos/regime-lab/)

## Start with the first screen

I start with a day in the timeline. The opening panel explains the leading pattern, the reasons for withholding an assessment and the estimated size of daily price moves. The probability bars compare historical patterns; they are not the odds of a future gain.

I then check what changed in the inputs and compare earlier, similar days. Moving backward hides later prices in the Explorer. Strategy results and the full-period experiment remain retrospective views.

Bull means the stronger-momentum training group, Bear the weaker-momentum group and Transition the middle group. These labels are relative to the training window.

## What’s included

- Daily HMM probabilities with forward-only filtering.
- A plain-language opening view, feature changes and historical examples.
- An uncertain status for spread-out probabilities, fit disagreement, unusual inputs, repeated label changes or imputed volume.
- A withholding experiment with coverage, pending outcomes and matched-date baseline comparisons.
- Strategy curves, drawdowns, transaction costs and state history across refits.
- A completed-daily-bar adapter for Alpaca, with local credentials and symbol/feed checks.
- A separate, append-only assessment journal with issuance times, model fingerprints and hash-chain checks.
- Exports of the historical study and run data.

## What I learned from the current experiment

In the existing 2,733-session retrospective run, the rules kept about 97.7% of assessments. Withheld days had larger risk-forecast errors, but the trailing-30-session baseline still beat the HMM risk forecast both overall and on kept days. I keep that result visible rather than presenting the filter as a proven improvement.

The target is annualized RMS daily returns over the next five sessions, a measure of movement size. It is not market direction, investment return or objectively correct bull/bear classification. The final five sessions stay unscored until their outcomes exist.

The rules were chosen after historical exploration. This is not an untouched holdout or a live track record. My next validation step is collecting forward assessments without changing the policy in response to their outcomes. [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) documents the definitions.

## Run the historical research

I use Python 3.12 and the pinned dependencies.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m unittest -v test_assessment test_journal test_current_data
python launch.py --retrain
```

I place the source at `data/sp500_clean.csv` with `Date`, `Open`, `High`, `Low`, `Close` and `Volume` columns. Historical data, fitted models and completed runs are not bundled in this repository.

`python launch.py --run results/MY_RUN --no-browser` rebuilds an existing run. New training runs record state return variances for the risk experiment. Older runs without those variances leave risk forecasts unavailable unless I explicitly supply locally generated, trusted model files with `assessment.py --trusted-models`. I never use that option for downloaded pickle files.

## Connect current daily data

The public demo is a saved historical run. I have not connected a live account to it, and the adapter is not a tick-by-tick stream.

I keep `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` in my local environment. No API key belongs in a portfolio page or committed source. Feed access depends on the provider account.

```sh
python current_data.py --symbol SPY --feed iex --out data/spy_daily.csv
python launch.py --retrain --data data/spy_daily.csv --symbol SPY --feed iex
```

I use a new filename for each download so older research inputs stay intact. Today’s bar is included only after a conservative 20:15 Eastern cutoff; otherwise the latest completed prior session is used. Vendor corrections can still occur. I train and assess the same instrument and feed: SPY cannot replace the S&P 500 index silently.

After training, I issue an assessment with a locally generated model from that run:

```sh
python observe.py --data data/spy_daily.csv --model results/MY_RUN/model_XXX.joblib --journal journal/assessments.jsonl
```

For a later source file from the same instrument/feed, I score matured outcomes separately:

```sh
python journal.py --journal journal/assessments.jsonl --data data/spy_daily_later.csv --out journal/evaluation.json
```

Repeated identical assessments do not add duplicate entries. Conflicting revisions are rejected. Changed historical inputs, mismatched instruments and broken journal hashes are rejected before scoring. Backfilled assessments are not counted as forward evidence. Hash checks are local integrity checks, not an external guarantee of issuance time.

## Code map

| File | What I use it for |
| --- | --- |
| `pipeline.py` | Daily features, model fits and strategy evaluation |
| `assessment.py` | Warning rules, risk forecasts, comparisons and historical examples |
| `current_data.py` | Completed daily bars from Alpaca |
| `observe.py` | Issuing a current daily assessment |
| `journal.py` | Append-only records and later outcome evaluation |
| `reliability.py` | Fit disagreement and state matching |
| `analyze.py` | Strategy analysis |
| `dashboard.py`, `dashboard.html` | Research dashboard |
| `launch.py` | Training and opening a run |

I still need authenticated provider testing and a forward evaluation period. Lower error on selected days alone does not establish better generalization or a better trading strategy.
