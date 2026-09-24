"""Analyze phase-1 or phase-2 results without retraining or loading pickle files."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import prepare, simulate, metrics, FEATURES
from reliability import lineage


def validate_predictions(data, signals, folds):
    if signals.empty or signals.index.has_duplicates or not signals.index.is_monotonic_increasing:
        raise ValueError('Prediction dates must be nonempty, unique and increasing.')
    p = signals[['p_bear','p_transition','p_bull']].to_numpy()
    if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-9):
        raise ValueError('Invalid probabilities.')
    if not (pd.to_datetime(signals.train_end)<signals.index).all():
        raise ValueError('Training cutoff leaks into prediction period.')
    expected_labels = np.array(['Bear','Transition','Bull'])[p.argmax(axis=1)]
    if not (signals.label.to_numpy()==expected_labels).all():
        raise ValueError('Labels disagree with probability maxima.')
    if len(set(f['fold'] for f in folds))!=len(folds):
        raise ValueError('Duplicate fold IDs.')
    if set(signals.fold.unique())!={f['fold'] for f in folds}:
        raise ValueError('Fold IDs do not match predictions.')
    last_end = None
    for f in folds:
        start, cutoff, end = map(pd.Timestamp,[f['test_start'],f['train_end'],f['test_end']])
        if not cutoff < start <= end or (last_end is not None and start <= last_end):
            raise ValueError('Fold timing is invalid.')
        part = signals[signals.fold==f['fold']]
        if part.index.min()!=start or part.index.max()!=end or not (pd.to_datetime(part.train_end)==cutoff).all():
            raise ValueError('Fold coverage does not match predictions.')
        if sorted(f['state_order'])!=[0,1,2]:
            raise ValueError('Invalid semantic state order.')
        last_end = end
    positions = data.index.get_indexer(signals.index)
    if (positions<0).any() or not (np.diff(positions)==1).all():
        raise ValueError('Prediction dates must be consecutive source sessions.')


def diagnostic_buckets(signals, ledgers):
    """Descriptive future-outcome association. Not a trading rule or causal test."""
    hard = ledgers['hard_hmm'].set_index('signal_date')
    baseline = ledgers['stocks60_cash40'].set_index('signal_date')
    frame = signals.copy()
    frame.index = frame.index.strftime('%Y-%m-%d')
    frame = frame.loc[hard.index].copy()
    frame['net_excess_next_period'] = hard.net_return-baseline.net_return
    rows = []
    for col in ['uncertainty_entropy','seed_vote_disagreement','seed_js_disagreement']:
        if col not in frame or frame[col].notna().sum()==0:
            continue
        bins = [-1e-12,.25,.5,.75,1.00000001]
        groups = pd.cut(frame[col],bins,labels=['0–0.25','0.25–0.50','0.50–0.75','0.75–1.00'])
        for bucket, subset in frame.groupby(groups,observed=True):
            rows.append(dict(diagnostic=col,bucket=str(bucket),observations=len(subset),
                mean_next_period_excess=float(subset.net_excess_next_period.mean()),
                underperformed60_fraction=float((subset.net_excess_next_period<0).mean())))
    return pd.DataFrame(rows)


def analyze(data_path, run_dir, output):
    run_dir, output = Path(run_dir),Path(output)
    manifest = json.loads((run_dir/'run.json').read_text())
    expected_hash = manifest['source_sha256']
    if hashlib.sha256(Path(data_path).read_bytes()).hexdigest()!=expected_hash:
        raise ValueError('Source checksum differs from run.json.')
    fee, slippage = manifest['fee_bps'],manifest['slippage_bps']
    data = prepare(data_path)
    signals = pd.read_csv(run_dir/'predictions.csv',index_col='Date',parse_dates=['Date'])
    folds = json.loads((run_dir/'folds.json').read_text())
    validate_predictions(data, signals, folds)
    training = data.loc[:folds[0]['train_end']].dropna(subset=FEATURES)
    scale = training[FEATURES].std(ddof=0).to_numpy()
    line = lineage(folds,scale)
    rules = dict(buy_hold=lambda r:1.,stocks60_cash40=lambda r:.6,
                 hard_hmm=lambda r:{'Bear':0.,'Transition':.5,'Bull':1.}[r.label],
                 probability_hmm=lambda r:r.p_bull+.5*r.p_transition)
    ledgers = {name:simulate(data,signals,rule,fee,slippage) for name,rule in rules.items()}
    summary = {name:metrics(ledger) for name,ledger in ledgers.items()}
    # Verify results against the imported manifest if original metrics are present.
    for name, original in manifest.get('results',{}).items():
        if name in summary and not np.isclose(original['final_value'],summary[name]['final_value'],rtol=1e-9):
            raise ValueError(f'{name}: results do not reproduce the supplied run.')
    buckets = diagnostic_buckets(signals,ledgers)
    seed_available = 'seed_count' in signals and bool(signals.seed_count.ge(2).any())
    output.mkdir(parents=True,exist_ok=False)
    line.to_csv(output/'lineage.csv',index=False)
    buckets.to_csv(output/'diagnostic_buckets.csv',index=False)
    for name,ledger in ledgers.items():
        ledger.to_csv(output/f'{name}_ledger.csv',index=False)
    record = dict(source_sha256=expected_hash,fee_bps=fee,slippage_bps=slippage,
                  seed_diagnostics_available=seed_available,results=summary,
                  status='Descriptive retrospective analysis. No untouched holdout.')
    (output/'comparison.json').write_text(json.dumps(record,indent=2))
    table = pd.DataFrame(summary).T[['final_value','cagr','annual_volatility','sharpe_zero_rf','max_drawdown','turnover']]
    table.columns=['Value of $1','CAGR','Annual volatility','Sharpe (RF=0)','Max drawdown','Turnover']
    table_html = table.to_html(formatters={c:(lambda x:f'{x:.2%}') for c in ['CAGR','Annual volatility','Max drawdown']},float_format=lambda x:f'{x:.3f}')
    biggest = line[line.fold>0].nlargest(12,'profile_distance')
    display = biggest[['available_from','lineage','current_label','profile_distance','assignment_margin']]
    seed_text = ('Seed diagnostics are present. Agreement is only across initializations of the same model, not independent model families.'
                 if seed_available else 'Seed disagreement is unavailable in this imported run. Run the updated pipeline to collect all accepted seeds; missing disagreement is not zero disagreement.')
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regime Reliability Lab: research report</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 22px;color:#202135;background:#f7f9fc}}h1{{font-size:36px;line-height:1.15}}h2{{margin-top:40px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto}}.note{{border-left:4px solid #6053bc;padding:14px 18px;background:#eeecfc}}summary{{cursor:pointer;font-weight:bold}}a{{color:#5142a7}}</style>
<h1>Regime Reliability Lab</h1><p>Historical signals: {signals.index[0].date()} through {signals.index[-1].date()}. {len(folds)} fits, {len(signals):,} signals.</p>
<p class="note">Research report, not a live market forecast. Costs: {fee:g} bps fees + {slippage:g} bps assumed slippage per unit traded. The last two signals have no complete execution period and are excluded from performance.</p>
<h2>Does the HMM add value beyond holding less stock?</h2><div class="scroll">{table_html}</div>
<p>60/40 means 60% stocks and 40% zero-yield cash, rebalanced daily, NOT a stock/bond portfolio. It is an approximate exposure comparator suggested after seeing phase-1 results, not an independently preregistered strategy. All strategies use the same executable dates, proportional costs and terminal liquidation.</p>
<h2>Regime lineage: largest profile changes</h2><p>Stable IDs L0/L1/L2 start as Bear/Transition/Bull. New states are matched one-to-one by standardized mean-profile distance. Scale is fixed from the first training window. State numbers may change without profiles changing.</p>
<div class="scroll">{display.to_html(index=False,float_format=lambda x:f'{x:.4f}')}</div>
<p>Distances and assignment margins are not probabilities or validated drift alerts. Small assignment margins mean alternative matches have similar costs. Mean matching does not measure covariance changes, state splitting, or prove economic identity.</p>
<h2>Seed disagreement</h2><p>{seed_text}</p>
<h2>Reliability and subsequent outcomes</h2><p>Fixed diagnostic bins; compare the HMM's next open-to-open net return with the 60/40 baseline. These are descriptive conditional averages, not compounded strategy returns, causal effects, accuracy scores, or evidence of statistical significance. Market conditions and exposure can confound them.</p>
<div class="scroll">{buckets.to_html(index=False,float_format=lambda x:f'{x:.6f}')}</div>
<details><summary>All lineage records</summary><div class="scroll">{line.to_html(index=False,float_format=lambda x:f'{x:.5f}')}</div></details>
<h2>Limitations</h2><p>The S&amp;P 500 index is not directly tradable; prices omit dividends and the simulator omits taxes and cash yield. The source's zero-volume entry is causally imputed and must be checked against the provider. This history was previously explored. No fragility score is validated here.</p>
<p>Method references: <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html">SciPy state matching</a>, <a href="https://hmmlearn.readthedocs.io/en/stable/api.html">hmmlearn</a>.</p></html>'''
    (output/'report.html').write_text(report)
    print('Report:',output/'report.html')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/sp500_clean.csv')
    parser.add_argument('--run',default='example_phase1')
    parser.add_argument('--out',default='results/review_001')
    args=parser.parse_args()
    analyze(args.data,args.run,args.out)
