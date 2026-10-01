"""Issue one daily assessment from a locally trained, trusted model artifact."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib
import numpy as np
from pipeline import prepare, FEATURES, filter_states
from assessment import assess, training_reference, risk_forecast
from reliability import match_profiles, ensemble_diagnostics
from journal import append


def prefix_hash(data, cutoff):
    return hashlib.sha256(data.loc[:cutoff,['Open','High','Low','Close','Volume']].to_csv().encode()).hexdigest()


def observe(data_path, model_path, journal_path):
    data_path,model_path=Path(data_path),Path(model_path)
    metadata=json.loads(Path(str(data_path)+'.meta.json').read_text())
    # Joblib must only load model files produced locally by this project.
    a=joblib.load(model_path)
    if metadata['symbol']!=a.get('symbol') or metadata['feed']!=a.get('feed'):
        raise ValueError('Model and incoming data must use the same symbol and feed; SPY is not the S&P 500 index')
    data=prepare(data_path);features=data.dropna(subset=FEATURES)
    if prefix_hash(data,a['train_end'])!=a.get('training_source_hash'):
        raise ValueError('Training history changed; refit rather than mixing revised inputs with this model')
    date=features.index[-1]
    now=dt.datetime.now(ZoneInfo('America/New_York'))
    if date.date()>now.date() or (date.date()==now.date() and now.time()<dt.time(20,15)):raise ValueError('Today’s daily bar is not complete yet')
    if date.date()<dt.datetime.now(ZoneInfo('America/New_York')).date()-dt.timedelta(days=4):raise ValueError('Latest data is stale; refresh the daily source first')
    if not a['train_end']<str(date.date()):raise ValueError('Training must end before the assessment')
    model,scaler,order=a['model'],a['scaler'],a['state_order']
    p=filter_states(model,scaler.transform(features[FEATURES]))
    aligned=[]
    for file in sorted(model_path.parent.glob(model_path.stem+'_seed_*.joblib')):
        seed=joblib.load(file)
        if seed['train_end']!=a['train_end']:raise ValueError('Seed training cutoffs differ')
        mapping,_,_=match_profiles(model.means_[order],seed['model'].means_,np.ones(4))
        aligned.append(filter_states(seed['model'],scaler.transform(features[FEATURES]))[-1,mapping])
    diag=ensemble_diagnostics(np.array(aligned)[:,None,:]) if aligned else {'vote_disagreement':np.array([np.nan])}
    row=dict(zip(['p_bear','p_transition','p_bull'],p[-1,order]))
    row.update(seed_count=len(aligned),seed_vote_disagreement=diag['vote_disagreement'][0],
               volume_imputed=bool(data.Volume_imputed.iloc[-1]))
    ref=training_reference(features.loc[:a['train_end']])
    labels=np.array(['Bear','Transition','Bull'])[p[:-1,order].argmax(axis=1)].tolist()
    assessment=assess(row,features[FEATURES].iloc[-1],ref,labels,features[FEATURES].iloc[-2])
    variance=model.covars_[:,0,0]*scaler.scale_[0]**2
    risk=risk_forecast(p[-1],model.transmat_,scaler.inverse_transform(model.means_),variance)
    result=dict(**assessment,mode='forward',asof=str(date.date()),train_end=a['train_end'],
                symbol=a['symbol'],feed=a['feed'],model_version=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                data_hash=hashlib.sha256(data_path.read_bytes()).hexdigest(),observed_source_hash=prefix_hash(data,date),probabilities=row,
                forecast_risk=risk,baseline_risk=float(np.sqrt(np.mean(data.Daily_Returns.iloc[-30:]**2)*252)))
    # Missing disagreement is represented as null in a strict JSON record.
    result['probabilities']['seed_vote_disagreement']=float(row['seed_vote_disagreement']) if np.isfinite(row['seed_vote_disagreement']) else None
    return append(journal_path,result)

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--data',required=True);cli.add_argument('--model',required=True);cli.add_argument('--journal',default='journal/assessments.jsonl')
    a=cli.parse_args();print(json.dumps(observe(a.data,a.model,a.journal),indent=2,allow_nan=False))
