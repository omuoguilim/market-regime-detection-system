# Assessment experiment

I separate three things: the HMM’s state label, warning rules that can withhold an assessment, and a measurable risk forecast.

## Rules frozen for this implementation

I withhold when normalized probability entropy exceeds 0.75; accepted-seed vote disagreement exceeds 0.34; fewer than two accepted fits are available; an input’s maximum absolute standardized distance exceeds the training window’s 99th percentile; more than two label flips occur in the last five assessments; or volume was imputed. Missing seed evidence is not agreement.

Each standardization and unusual-input limit uses the current fold’s training data only. These thresholds are post-hoc research choices, not validated confidence levels.

## Forecast and outcome

I propagate the filtered state probabilities through the fitted transition matrix over five sessions. Each state contributes its unscaled daily-return variance plus squared mean return. The forecast is the square root of their mean, multiplied by the square root of 252.

The outcome is the annualized root mean square of the next five observed close-to-close returns. The baseline is the annualized RMS of the trailing 30 returns available at assessment time. RMS includes the return mean; it is not demeaned standard deviation.

## Comparison

I report root mean squared forecast error for all, kept and withheld groups. Both forecasts are scored on the same matured dates within each group. Coverage is the kept fraction of all assessments. Incomplete outcomes remain pending.

Five-session outcomes overlap, so dates are not independent observations. I do not report a significance claim or a confidence interval from an independent-observation assumption. Selection can remove harder, higher-risk conditions; lower selected error does not establish better modeling. This experiment does not estimate trading profitability.

## Forward evidence

I record issuance time, symbol, feed, source fingerprint, model fingerprint, probabilities, reasons and forecasts. A separate evaluation checks that the observed history is unchanged and that issuance preceded the first outcome session. Old-session backfills stay unscored as forward evidence. Local files can be rewritten by their owner, so a hash chain is not a third-party timestamp service.

## Historical examples

I compare standardized feature distance with earlier sessions whose entire five-session outcome was already available by the selected date. Examples are separated by at least 30 source sessions and exclude the most recent 30 sessions. I show five examples as context, not a probability estimate or a forecast.

## Data boundary

The existing public run is retrospective S&P 500 index research. A current-data run must use one instrument and feed throughout training and observation. Credentials remain outside source code. The daily adapter does not supply a separately trained intraday model or enable public real-time streaming.
