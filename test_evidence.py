import csv
from pathlib import Path
import shutil
import tempfile
import unittest
from verify_evidence import verify

EVIDENCE = Path(__file__).parent / 'evidence/walk-forward'


class EvidenceTests(unittest.TestCase):
    def test_committed_results_reproduce(self):
        result = verify(EVIDENCE)
        self.assertEqual(result['input_rows'], 4023)
        self.assertEqual(result['assessment_sessions'], 2733)
        self.assertAlmostEqual(result['results']['hard_hmm']['max_drawdown'], -.15746452027306623)
        self.assertAlmostEqual(result['results']['buy_hold']['cagr'], .11520168875464232)

    def test_modified_return_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'evidence'
            shutil.copytree(EVIDENCE, target)
            path = target / 'hard_hmm_ledger.csv'
            with path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                fields, rows = reader.fieldnames, list(reader)
            rows[1]['net_return'] = '0.9'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, 'NAV'):
                verify(target)


if __name__ == '__main__':
    unittest.main()
