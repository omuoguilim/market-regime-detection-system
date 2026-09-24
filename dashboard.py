"""Build an offline dashboard from actual CSV/JSON results, never model pickles."""
import csv
import hashlib
import json
import math
from pathlib import Path


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def build(root, run):
    root, run = Path(root), Path(run)
    manifest = json.loads((run/'run.json').read_text())
    source = root/'data/sp500_clean.csv'
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['source_sha256']:
        raise ValueError('The source CSV does not match the selected run. Dashboard not rebuilt.')
    analysis = run/'analysis'
    if not analysis.exists() and run == root/'example_phase1':
        analysis = root/'results/phase1_review'
    comparison = json.loads((analysis/'comparison.json').read_text())
    predictions = rows(run/'predictions.csv')
    prices = {r['Date']:number(r['Close']) for r in rows(source)}
    fields = ['p_bear','p_transition','p_bull','uncertainty_entropy','switch_probability','seed_count','seed_vote_disagreement','seed_js_disagreement','seed_exposure_std']
    signals = [dict(date=r['Date'], label=r['label'], fold=int(r['fold']), train_end=r['train_end'], close=prices.get(r['Date']), imputed=r.get('volume_imputed','').lower()=='true', **{k:number(r.get(k)) for k in fields}) for r in predictions]
    ledgers = {}
    for name in comparison['results']:
        ledger = rows(analysis/(name+'_ledger.csv'))
        wealth, peak = 1., 1.
        values = [dict(date=ledger[0]['execution_date'], value=1., drawdown=0.)]
        for r in ledger:
            wealth *= 1+float(r['net_return'])
            peak = max(peak, wealth)
            values.append(dict(date=r['end_date'], value=wealth, drawdown=wealth/peak-1))
        ledgers[name] = values
    payload = dict(run=run.name, manifest=manifest, comparison=comparison, signals=signals, ledgers=ledgers,
                   folds=json.loads((run/'folds.json').read_text()), lineage=rows(analysis/'lineage.csv'),
                   buckets=rows(analysis/'diagnostic_buckets.csv'))
    template = (root/'dashboard.html').read_text()
    # Escape script terminators even if an imported file contains unusual text.
    encoded = json.dumps(payload,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    output = run/'dashboard.html'
    output.write_text(template.replace('/*__DATA__*/', 'const DATA = '+encoded+';'))
    return output.resolve()
