import unittest
import numpy as np
import pandas as pd
from assessment import assess, compare, future_risk, historical_analogs, risk_forecast, training_reference
from pipeline import FEATURES

class AssessmentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(5)
        self.data=pd.DataFrame(rng.normal(0,.01,(90,4)),columns=FEATURES,index=pd.bdate_range('2026-01-01',periods=90))
        self.data['Close']=100*np.cumprod(1+self.data.Daily_Returns)
        self.ref=training_reference(self.data.iloc[:40])
        self.row=dict(p_bear=.05,p_transition=.05,p_bull=.9,seed_count=3,seed_vote_disagreement=0)
    def test_missing_seed_evidence_withholds_instead_of_claiming_agreement(self):
        row={**self.row,'seed_count':1,'seed_vote_disagreement':None}
        a=assess(row,self.data[FEATURES].iloc[40],self.ref,[])
        self.assertEqual(a['status'],'uncertain');self.assertIn('Too few model fits to check agreement',a['reasons'])
    def test_spread_probabilities_unusual_inputs_and_flips(self):
        row={**self.row,'p_bear':1/3,'p_transition':1/3,'p_bull':1/3}
        a=assess(row,np.full(4,100),self.ref,['Bull','Bear','Bull','Bear'])
        self.assertGreaterEqual(len(a['reasons']),3)
    def test_forecast_constant_variance_and_invalid_matrix(self):
        self.assertAlmostEqual(risk_forecast([.2,.3,.5],np.eye(3),np.zeros((3,4)),np.full(3,.0001)),np.sqrt(.0001*252))
        with self.assertRaises(ValueError):risk_forecast([.2,.3,.5],np.zeros((3,3)),np.zeros((3,4)),np.ones(3))
    def test_future_edits_do_not_change_past_assessment_or_analogs(self):
        date=self.data.index[60];prefix=self.data.loc[:date]
        before=historical_analogs(self.data,date,self.ref)
        modified=self.data.copy();modified.loc[modified.index>date,FEATURES]=1000
        self.assertEqual(before,historical_analogs(modified,date,self.ref))
        self.assertEqual(before,historical_analogs(prefix,date,self.ref))
        for r in before:self.assertLessEqual(self.data.index.get_loc(pd.Timestamp(r['date']))+5,60)
    def test_pending_outcomes_are_not_zero_and_cohorts_share_dates(self):
        self.assertIsNone(future_risk(self.data,self.data.index[-5]))
        r=[dict(status='uncertain',forecast_risk=.1,baseline_risk=.2,realized_risk=.15),dict(status='assessment available',forecast_risk=.1,baseline_risk=.1,realized_risk=None)]
        c=compare(r);self.assertEqual(c['kept']['scored'],0);self.assertEqual(c['pending'],1);self.assertEqual(c['coverage'],.5)
    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):assess({**self.row,'p_bull':5},self.data[FEATURES].iloc[40],self.ref,[])

if __name__=='__main__':unittest.main()
