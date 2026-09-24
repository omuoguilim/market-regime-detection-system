"""Phase 1: causal features, filtered HMM inference, walk-forward simulation."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
import joblib

FEATURES = ['Daily_Returns', 'Volatility', 'Momentum', 'Volume_Change']
LABELS = ['Bear', 'Transition', 'Bull']


def prepare(path):
    data = pd.read_csv(path, index_col='Date', parse_dates=True)
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError('Dates must be unique and increasing; input is not silently reordered.')
    if data.index.hasnans:
        raise ValueError('Invalid dates.')
    for col in ['Open', 'Close', 'High', 'Low', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='raise')
        invalid = (data[col] < 0) if col == 'Volume' else (data[col] <= 0)
        if not np.isfinite(data[col]).all() or invalid.any():
            raise ValueError(f'{col} must contain finite positive values.')
    if ((data.High < data[['Open', 'Close', 'Low']].max(axis=1)) |
            (data.Low > data[['Open', 'Close', 'High']].min(axis=1))).any():
        raise ValueError('Inconsistent OHLC prices.')
    data['Daily_Returns'] = data.Close.pct_change(fill_method=None)
    data['Volatility'] = data.Daily_Returns.rolling(30).std()
    data['Momentum'] = data.Close.pct_change(30, fill_method=None)
    data['Volume_imputed'] = data.Volume.eq(0)
    volume_for_feature = data.Volume.mask(data.Volume.eq(0)).ffill()
    if volume_for_feature.isna().any():
        raise ValueError('Cannot impute leading missing volume without future data.')
    data['Volume_Change'] = volume_for_feature.pct_change(fill_method=None)
    # Keep every raw date for execution. Warm-up rows are excluded only from training.
    return data


def emission_log(model, x):
    return np.column_stack([multivariate_normal.logpdf(
        x, mean=mean, cov=cov) for mean, cov in zip(model.means_, model.covars_)])


def filter_states(model, x, previous=None):
    """P(state_t | x_1,...,x_t). No backward pass and no future observations."""
    emissions = emission_log(model, x)
    result = []
    for likelihood in emissions:
        prior = model.startprob_ if previous is None else previous @ model.transmat_
        with np.errstate(divide='ignore'):
            weights = np.log(prior) + likelihood
        previous = np.exp(weights - logsumexp(weights))
        if not np.isfinite(previous).all():
            raise ValueError('Invalid filter probabilities.')
        result.append(previous)
    return np.asarray(result)


def train_model(train, seeds=(7, 42, 123), return_pool=False):
    from hmmlearn.hmm import GaussianHMM
    scaler = StandardScaler().fit(train[FEATURES])
    x = scaler.transform(train[FEATURES])
    best = None
    diagnostics = []
    accepted_models = []
    for seed in seeds:
        try:
            model = GaussianHMM(n_components=3, covariance_type='full',
                                n_iter=300, tol=1e-3, random_state=seed)
            model.fit(x)
            history = list(model.monitor_.history)
            improvement = history[-1] - history[-2] if len(history) > 1 else None
            converged = improvement is not None and -1e-6 <= improvement < model.tol
            score = float(model.score(x))
            diagnostics.append(dict(seed=seed, score=score, iterations=model.monitor_.iter,
                                    improvement=improvement, accepted=bool(converged)))
            if converged and np.isfinite(score):
                accepted_models.append((seed, model))
            if converged and np.isfinite(score) and (best is None or score > best[0]):
                best = (score, model, seed)
        except (ValueError, np.linalg.LinAlgError) as exc:
            diagnostics.append(dict(seed=seed, accepted=False, error=str(exc)))
    if best is None:
        raise RuntimeError(f'No converged seed. Do not silently use a failed fit: {diagnostics}')
    _, model, seed = best
    raw_means = scaler.inverse_transform(model.means_)
    # Transparent semantic naming rule, learned only from the training window.
    # These are relative momentum groups, not verified economic ground truth.
    order = np.argsort(raw_means[:, FEATURES.index('Momentum')], kind='stable')
    result = (scaler, model, order, seed, diagnostics)
    return (*result, accepted_models) if return_pool else result


def simulate(data, signals, target, fee_bps=5, slippage_bps=2):
    """Rebalance at next open; hold through following open. Start and finish in cash.

    Turnover uses actual drifted portfolio weight, not yesterday's target weight.
    Fees and slippage are proportional approximations charged against pre-trade NAV.
    """
    if not np.isfinite([fee_bps, slippage_bps]).all() or min(fee_bps, slippage_bps) < 0:
        raise ValueError('Costs must be finite and nonnegative.')
    if signals.index.has_duplicates or not signals.index.is_monotonic_increasing:
        raise ValueError('Signal dates must be unique and increasing.')
    positions = data.index.get_indexer(signals.index)
    if (positions < 0).any() or (len(positions)>1 and not (np.diff(positions)==1).all()):
        raise ValueError('Signal dates must cover consecutive source sessions; do not skip held returns.')
    rows = []
    weight = 0.0
    nav = 1.0
    cost_rate = (fee_bps + slippage_bps) / 10000
    for date, row in signals.iterrows():
        i = data.index.get_loc(date)
        if i + 2 >= len(data):
            continue
        desired = float(target(row))
        if not 0 <= desired <= 1:
            raise ValueError('Long-only exposure must be in [0,1].')
        turnover = abs(desired - weight)
        cost = turnover * cost_rate
        asset_return = float(data.Open.iloc[i+2] / data.Open.iloc[i+1] - 1)
        net_return = (1-cost) * (1+desired*asset_return) - 1
        nav *= 1+net_return
        weight = desired*(1+asset_return)/(1+desired*asset_return)
        rows.append(dict(signal_date=str(date.date()), execution_date=str(data.index[i+1].date()),
                         end_date=str(data.index[i+2].date()), exposure=desired,
                         turnover=turnover, cost_fraction=cost, net_return=net_return, nav=nav))
    if not rows:
        raise ValueError('No complete executable periods.')
    # Terminal liquidation is included for every strategy, including buy-and-hold.
    last_cost = weight*cost_rate
    rows[-1]['net_return'] = (1+rows[-1]['net_return'])*(1-last_cost)-1
    rows[-1]['nav'] *= 1-last_cost
    rows[-1]['turnover'] += weight
    rows[-1]['cost_fraction'] += last_cost
    return pd.DataFrame(rows)


def metrics(ledger):
    r = ledger.net_return.to_numpy()
    wealth = np.r_[1.0, np.cumprod(1+r)]
    dd = wealth / np.maximum.accumulate(wealth) - 1
    vol = r.std(ddof=1) * np.sqrt(252)
    elapsed = (pd.Timestamp(ledger.end_date.iloc[-1])-pd.Timestamp(ledger.execution_date.iloc[0])).days/365.25
    return dict(final_value=float(wealth[-1]), total_return=float(wealth[-1]-1),
                cagr=float(wealth[-1]**(1/elapsed)-1), annual_volatility=float(vol),
                sharpe_zero_rf=float(r.mean()*252/vol) if vol > 0 else None,
                max_drawdown=float(dd.min()), turnover=float(ledger.turnover.sum()),
                sum_cost_fractions=float(ledger.cost_fraction.sum()))


def run(path, out, initial_train=1260, block=63, fee_bps=5, slippage_bps=2):
    if initial_train < 100 or block < 1 or min(fee_bps, slippage_bps) < 0:
        raise ValueError('Invalid training, block, or cost settings.')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    data = prepare(path)
    features = data.dropna(subset=FEATURES)
    if len(features) <= initial_train + 2:
        raise ValueError('Not enough observations for training and execution.')
    from reliability import match_profiles, ensemble_diagnostics
    predictions, folds, seed_records = [], [], []
    for start in range(initial_train, len(features), block):
        train = features.iloc[:start]
        test = features.iloc[start:start+block]
        scaler, model, order, seed, diagnostics, pool = train_model(train, return_pool=True)
        previous = filter_states(model, scaler.transform(train[FEATURES]))[-1]
        probs = filter_states(model, scaler.transform(test[FEATURES]), previous)
        fold_id = len(folds)
        artifact = dict(model=model, scaler=scaler, state_order=order, feature_names=FEATURES,
                        train_end=str(train.index[-1].date()))
        joblib.dump(artifact, out/f'model_{fold_id:03d}.joblib')
        aligned = []
        for candidate_seed, candidate in pool:
            # Align candidate states to the selected model's semantic order.
            mapping, distance, margin = match_profiles(model.means_[order], candidate.means_, np.ones(4))
            last = filter_states(candidate, scaler.transform(train[FEATURES]))[-1]
            candidate_probs = filter_states(candidate, scaler.transform(test[FEATURES]), last)[:, mapping]
            aligned.append(candidate_probs)
            joblib.dump(dict(model=candidate, scaler=scaler, aligned_order=mapping,
                             train_end=str(train.index[-1].date())),
                        out/f'model_{fold_id:03d}_seed_{candidate_seed}.joblib')
            for date, cp in zip(test.index, candidate_probs):
                seed_records.append(dict(Date=date, fold=fold_id, seed=candidate_seed,
                    p_bear=cp[0], p_transition=cp[1], p_bull=cp[2],
                    alignment_mean_distance=float(distance.mean()), alignment_margin=float(margin)))
        ensemble = ensemble_diagnostics(np.stack(aligned))
        for j, (date, p) in enumerate(zip(test.index, probs)):
            canonical = p[order]
            entropy = -np.sum(canonical*np.log(np.clip(canonical, 1e-300, 1)))/np.log(3)
            predictions.append(dict(Date=date, fold=fold_id,
                                    train_end=str(train.index[-1].date()),
                                    p_bear=canonical[0], p_transition=canonical[1], p_bull=canonical[2],
                                    label=LABELS[int(canonical.argmax())],
                                    volume_imputed=bool(data.loc[date, 'Volume_imputed']),
                                    uncertainty_entropy=entropy,
                                    switch_probability=float(1-p@np.diag(model.transmat_)),
                                    seed_count=len(pool),
                                    seed_vote_disagreement=ensemble['vote_disagreement'][j],
                                    seed_js_disagreement=ensemble['js_disagreement'][j],
                                    seed_exposure_std=ensemble['exposure_std'][j]))
        fold = dict(fold=fold_id, train_start=str(train.index[0].date()),
                    train_end=str(train.index[-1].date()), test_start=str(test.index[0].date()),
                    test_end=str(test.index[-1].date()), selected_seed=seed,
                    state_order=order.tolist(), raw_means=scaler.inverse_transform(model.means_).tolist(),
                    transitions=model.transmat_.tolist(), diagnostics=diagnostics)
        folds.append(fold)
        print(f'Fold {fold_id}: trained through {fold["train_end"]}, predicted through {fold["test_end"]}', flush=True)
    signals = pd.DataFrame(predictions).set_index('Date')
    signals.to_csv(out/'predictions.csv')
    pd.DataFrame(seed_records).to_csv(out/'seed_predictions.csv', index=False)
    (out/'folds.json').write_text(json.dumps(folds, indent=2))
    rules = dict(buy_hold=lambda row: 1.0,
                 stocks60_cash40=lambda row: .6,
                 hard_hmm=lambda row: {'Bear':0, 'Transition':.5, 'Bull':1}[row.label],
                 probability_hmm=lambda row: row.p_bull+.5*row.p_transition)
    summary = {}
    for name, rule in rules.items():
        ledger = simulate(data, signals, rule, fee_bps, slippage_bps)
        ledger.to_csv(out/f'{name}_ledger.csv', index=False)
        summary[name] = metrics(ledger)
    versions = {name:importlib.metadata.version(name) for name in
                ['numpy', 'pandas', 'scipy', 'scikit-learn', 'hmmlearn', 'joblib']}
    manifest = dict(source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    python=platform.python_version(), packages=versions,
                    input_rows=len(data), feature_rows=len(features),
                    volume_imputed_dates=[str(d.date()) for d in data.index[data.Volume_imputed]],
                    initial_train=initial_train, refit_every=block,
                    fee_bps=fee_bps, slippage_bps=slippage_bps,
                    results=summary, status='retrospective walk-forward research, not untouched holdout')
    (out/'run.json').write_text(json.dumps(manifest, indent=2))
    from analyze import analyze
    analyze(path, out, out/'analysis')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/sp500_clean.csv')
    parser.add_argument('--out', default='results/run_002')
    parser.add_argument('--initial-train', type=int, default=1260)
    parser.add_argument('--block', type=int, default=63)
    parser.add_argument('--fee-bps', type=float, default=5)
    parser.add_argument('--slippage-bps', type=float, default=2)
    args = parser.parse_args()
    run(args.data, args.out, args.initial_train, args.block, args.fee_bps, args.slippage_bps)
