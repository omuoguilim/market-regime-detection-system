"""Recompute published strategy metrics from CSV ledgers, without ML packages."""
import argparse
import csv
from datetime import date
import json
import math
from pathlib import Path
import statistics


def ledger_metrics(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError('At least two execution periods are required')
    wealth = peak = 1.0
    drawdown = 0.0
    returns = []
    previous_end = None
    for row in rows:
        start, end = date.fromisoformat(row['execution_date']), date.fromisoformat(row['end_date'])
        signal = date.fromisoformat(row['signal_date'])
        if not signal < start < end or (previous_end is not None and start != previous_end):
            raise ValueError('Dates must be causal and execution periods consecutive')
        previous_end = end
        r = float(row['net_return'])
        cost = float(row['cost_fraction'])
        if not math.isfinite(r) or r <= -1 or not math.isfinite(cost) or cost < 0:
            raise ValueError('Invalid returns or costs')
        wealth *= 1 + r
        if not math.isclose(wealth, float(row['nav']), rel_tol=1e-9):
            raise ValueError('NAV does not match compounded returns')
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1)
        returns.append(r)
    elapsed = (date.fromisoformat(rows[-1]['end_date']) - date.fromisoformat(rows[0]['execution_date'])).days / 365.25
    volatility = statistics.stdev(returns) * math.sqrt(252)
    return {'final_value': wealth, 'cagr': wealth ** (1 / elapsed) - 1,
            'max_drawdown': drawdown, 'sharpe_zero_rf': statistics.mean(returns) * 252 / volatility,
            'execution_periods': len(rows)}


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'run.json').read_text())
    results = {}
    dates = None
    for name in ['hard_hmm', 'buy_hold']:
        path = directory / f'{name}_ledger.csv'
        calculated = ledger_metrics(path)
        with path.open(newline='') as stream:
            current_dates = [(r['execution_date'], r['end_date']) for r in csv.DictReader(stream)]
        if dates is not None and dates != current_dates:
            raise ValueError('Strategies must use identical execution dates')
        dates = current_dates
        for key in ['final_value', 'cagr', 'max_drawdown', 'sharpe_zero_rf']:
            if not math.isclose(calculated[key], manifest['results'][name][key], rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError(f'{name}: published {key} does not reproduce')
        results[name] = calculated
    signals = manifest['feature_rows'] - manifest['initial_train']
    return {'input_rows': manifest['input_rows'], 'feature_rows': manifest['feature_rows'],
            'assessment_sessions': signals, 'results': results,
            'status': 'Historical explored sample; not an untouched holdout or live track record.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default=str(Path(__file__).parent / 'evidence/walk-forward'))
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
