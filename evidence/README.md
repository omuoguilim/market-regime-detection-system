# Historical result evidence

I saved the original run manifest and the hard-state HMM and buy-and-hold execution ledgers here so my reported results can be checked without training a model or trusting a screenshot.

From the repository root:

```sh
python3 verify_evidence.py
python3 -m unittest -v test_evidence
```

The verifier uses only Python's standard library. It recomputes compound growth, annualized return, maximum drawdown and zero-risk-free-rate Sharpe from each net-return series, checks every saved NAV, checks causal and matching execution dates, and compares the results to the saved manifest.

| Measure | Hard-state HMM | Buy-and-hold |
| --- | ---: | ---: |
| Annualized return | 4.6848% | 11.5202% |
| Maximum drawdown magnitude | 15.7465% | 32.2365% |
| Sharpe, risk-free rate assumed zero | 0.6147 | 0.7724 |
| Final value of $1 | $1.6436 | $3.2653 |

There are 4,023 input observations and 3,993 usable feature rows. The first 1,260 feature rows form the initial training window. That leaves 2,733 assessment sessions. The final two signals do not have complete execution periods, leaving 2,731 strategy periods. The `assessment-summary.json` artifact records the saved aggregate withholding/risk-forecast experiment (per-day records are omitted from this evidence bundle); a strategy drawdown is not a risk-forecast accuracy score.

The source checksum, package versions, 63-session refit cadence, 5 bps fee and 2 bps slippage assumptions are in `walk-forward/run.json`. The ledgers are saved outputs, not independent proof that the original source prices were correct. Raw vendor prices and fitted models remain separate. Re-running training requires a matching source CSV and the pinned ML environment described in the main README.

This was explored historical data. Lower drawdown came with lower annualized return and lower Sharpe. It is not evidence of profitable live trading, an untouched holdout, or a guaranteed future outcome.
