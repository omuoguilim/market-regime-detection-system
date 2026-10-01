import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from journal import append,read_journal,evaluate,digest

class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'journal.jsonl'
        self.a=dict(mode='forward',symbol='SPY',feed='iex',asof='2026-01-02',train_end='2026-01-01',model_version='a',status='uncertain')
        self.now=dt.datetime(2026,1,3,2,tzinfo=dt.timezone.utc)
    def test_identical_retry_does_not_rewrite_or_duplicate(self):
        one=append(self.path,self.a,self.now);original=self.path.read_bytes()
        self.assertEqual(append(self.path,self.a,self.now),one);self.assertEqual(self.path.read_bytes(),original)
    def test_conflicting_revision_and_historical_backfill_rejected(self):
        append(self.path,self.a,self.now)
        with self.assertRaises(ValueError):append(self.path,{**self.a,'status':'assessment available'},self.now)
        with self.assertRaises(ValueError):append(self.path,{**self.a,'mode':'retrospective'},self.now)
    def test_modified_entry_is_detected_and_blocks_append(self):
        append(self.path,self.a,self.now);self.path.write_text(self.path.read_text().replace('uncertain','changed'))
        with self.assertRaisesRegex(ValueError,'hash chain'):read_journal(self.path)
        with self.assertRaises(ValueError):append(self.path,{**self.a,'model_version':'b'},self.now)
    def test_forward_scoring_rejects_backfill_and_revised_history(self):
        import hashlib
        import numpy as np
        import pandas as pd
        data=pd.DataFrame(dict(Open=100.,High=102.,Low=99.,Close=np.arange(100.,109.),Volume=100.,Daily_Returns=.01),index=pd.bdate_range('2026-01-01',periods=9))
        fingerprint=hashlib.sha256(data.loc[:self.a['asof'],['Open','High','Low','Close','Volume']].to_csv().encode()).hexdigest()
        a={**self.a,'observed_source_hash':fingerprint,'forecast_risk':.1,'baseline_risk':.2}
        append(self.path,a,self.now)
        result=evaluate(self.path,data,'SPY','iex');self.assertEqual(result['comparison']['all']['scored'],1)
        with self.assertRaises(ValueError):evaluate(self.path,data,'QQQ','iex')
        changed=data.copy();changed.iloc[0,0]=90
        with self.assertRaises(ValueError):evaluate(self.path,changed,'SPY','iex')
        other=Path(self.tmp.name)/'backfill.jsonl';append(other,a,dt.datetime(2026,1,12,tzinfo=dt.timezone.utc))
        back=evaluate(other,data,'SPY','iex');self.assertEqual(back['excluded_backfills'],1);self.assertEqual(back['comparison']['all']['assessments'],0)

    def test_invalid_training_cutoff_and_future_session_rejected(self):
        with self.assertRaises(ValueError):append(self.path,{**self.a,'train_end':self.a['asof']},self.now)
        with self.assertRaises(ValueError):append(self.path,{**self.a,'asof':'2027-01-01'},self.now)

if __name__=='__main__':unittest.main()
