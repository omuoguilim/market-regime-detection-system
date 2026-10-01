import datetime as dt
import os
import unittest
from unittest.mock import patch
from current_data import completed_bars,fetch

class CurrentDataTests(unittest.TestCase):
    def setUp(self):
        self.bars=[dict(t='2026-10-01T04:00:00Z',o=1,h=2,l=1,c=2,v=5),dict(t='2026-10-02T04:00:00Z',o=2,h=3,l=2,c=3,v=5)]
    def test_partial_daily_bar_excluded_in_eastern_timezone(self):
        frame=completed_bars(self.bars,dt.datetime(2026,10,2,19,tzinfo=dt.timezone.utc))
        self.assertEqual(list(frame.index),['2026-10-01'])
    def test_today_accepted_only_after_conservative_extended_session_cutoff(self):
        frame=completed_bars(self.bars,dt.datetime(2026,10,3,1,tzinfo=dt.timezone.utc))
        self.assertEqual(len(frame),2)
    def test_duplicate_sessions_rejected(self):
        with self.assertRaises(ValueError):completed_bars([self.bars[0]]*2,dt.datetime(2026,10,3,tzinfo=dt.timezone.utc))
    def test_missing_credentials_fail_before_request(self):
        with patch.dict(os.environ,{},clear=True),patch('urllib.request.urlopen') as request:
            with self.assertRaisesRegex(ValueError,'Set APCA'):fetch('SPY','2016-01-01')
            request.assert_not_called()

if __name__=='__main__':unittest.main()
