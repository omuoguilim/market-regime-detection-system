"""Open the bundled research dashboard; no third-party packages required."""
import argparse
import datetime as dt
import subprocess
import sys
import webbrowser
from pathlib import Path
from dashboard import build

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrain', action='store_true', help='Fit a fresh, timestamped run using installed research dependencies.')
    parser.add_argument('--run', type=Path, help='Open a different completed run folder.')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    run = args.run
    if args.retrain:
        run = ROOT/'results'/('run_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        subprocess.run([sys.executable, str(ROOT/'pipeline.py'), '--data', str(ROOT/'data/sp500_clean.csv'), '--out', str(run)], cwd=ROOT, check=True)
    if run is None:
        candidates = [p.parent for p in (ROOT/'results').glob('*/run.json') if (p.parent/'analysis/comparison.json').exists()]
        run = max(candidates, key=lambda p:(p/'run.json').stat().st_mtime) if candidates else ROOT/'example_phase1'
    if not run.is_absolute():
        run = ROOT/run
    output = build(ROOT, run)
    print('Dashboard:', output)
    if not args.no_browser:
        webbrowser.open(output.as_uri())


if __name__ == '__main__':
    main()
