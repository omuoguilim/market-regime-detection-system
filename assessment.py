"""Causal abstention diagnostics and a fixed-horizon risk forecast experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import FEATURES, LABELS, prepare

POLICY = dict(entropy_max=.75, vote_disagreement_max=.34,
              unusual_training_quantile=.99, flips_max=2, flip_window=5, horizon=5)


def training_reference(train):
    x = train[FEATURES].to_numpy(dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-12)
    distance = np.abs((x-mean)/scale).max(axis=1)
    return dict(mean=mean.tolist(), scale=scale.tolist(),
                unusual_limit=float(np.quantile(distance, POLICY['unusual_training_quantile'])))


def assess(row, feature, reference, previous_labels, previous_feature=None):
    p = np.array([row['p_bear'], row['p_transition'], row['p_bull']], dtype=float)
    if not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):
        raise ValueError('Invalid state probabilities')
    x = np.asarray(feature, dtype=float)
    if x.shape != (4,) or not np.isfinite(x).all():
        raise ValueError('Four finite daily features required')
    z = np.abs((x-np.array(reference['mean']))/np.array(reference['scale']))
    entropy = float(-np.sum(p*np.log(np.clip(p,1e-300,1)))/np.log(3))
    label = LABELS[int(p.argmax())]
    history = (list(previous_labels)+[label])[-POLICY['flip_window']:]
    flips = sum(a!=b for a,b in zip(history,history[1:]))
    reasons = []
    if entropy>POLICY['entropy_max']: reasons.append('State probabilities are spread out')
    disagreement = row.get('seed_vote_disagreement')
    if int(row.get('seed_count',0))<2 or disagreement is None or not np.isfinite(disagreement):
        reasons.append('Too few model fits to check agreement')
    elif float(disagreement)>POLICY['vote_disagreement_max']:
        reasons.append('Accepted model fits disagree')
    if float(z.max())>reference['unusual_limit']: reasons.append('Inputs are unusual compared with training')
    if flips>POLICY['flips_max']: reasons.append('The leading state keeps changing')
    if bool(row.get('volume_imputed',False)): reasons.append('Volume needed an imputed value')
    changed = []
    if previous_feature is not None:
        delta = (x-np.asarray(previous_feature))/np.array(reference['scale'])
        changed = sorted([dict(feature=k, value=float(v), change_training_sd=float(d))
                          for k,v,d in zip(FEATURES,x,delta)],
                         key=lambda item:abs(item['change_training_sd']),reverse=True)
    return dict(status='uncertain' if reasons else 'assessment available',label=label,
                reasons=reasons,entropy=entropy,flips=flips,input_distance=float(z.max()),
                unusual_limit=reference['unusual_limit'],changes=changed)


def risk_forecast(probabilities, transitions, raw_means, return_variances, horizon=5):
    """Forecast RMS daily returns, annualized, NOT direction or state correctness."""
    p=np.asarray(probabilities,dtype=float)
    t=np.asarray(transitions,dtype=float)
    m=np.asarray(raw_means,dtype=float)[:,0]
    v=np.asarray(return_variances,dtype=float)
    if horizon<1 or t.shape!=(3,3) or v.shape!=(3,) or not np.isfinite(v).all() or (v<0).any():
        raise ValueError('Invalid risk forecast parameters')
    if p.shape!=(3,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):
        raise ValueError('Invalid forecast probabilities')
    if not np.isfinite(t).all() or (t<0).any() or not np.allclose(t.sum(axis=1),1) or not np.isfinite(m).all():
        raise ValueError('Invalid transition matrix or means')
    moments=[]
    for _ in range(horizon):
        p=p@t
        moments.append(float(p@(v+m*m)))
    return float(np.sqrt(np.mean(moments)*252))


def future_risk(data, date, horizon=5):
    i=data.index.get_loc(pd.Timestamp(date))
    if i+horizon>=len(data): return None
    r=data.Daily_Returns.iloc[i+1:i+horizon+1].to_numpy()
    if not np.isfinite(r).all(): return None
    return float(np.sqrt(np.mean(r*r)*252))


def compare(records):
    """Every cohort compares both forecasts on exactly the same matured dates."""
    out={}
    for name, group in [('all',records),('kept',[r for r in records if r['status']!='uncertain']),
                        ('withheld',[r for r in records if r['status']=='uncertain'])]:
        complete=[r for r in group if r.get('realized_risk') is not None and r.get('forecast_risk') is not None]
        out[name]=dict(assessments=len(group),scored=len(complete))
        if complete:
            truth=np.array([r['realized_risk'] for r in complete])
            out[name].update(model_rmse=float(np.sqrt(np.mean((np.array([r['forecast_risk'] for r in complete])-truth)**2))),
                             baseline_rmse=float(np.sqrt(np.mean((np.array([r['baseline_risk'] for r in complete])-truth)**2))),
                             mean_realized_risk=float(truth.mean()))
    out['coverage']=out['kept']['assessments']/len(records) if records else None
    out['pending']=sum(r.get('realized_risk') is None for r in records)
    return out


def historical_analogs(data, asof, reference, count=5, horizon=5):
    """Only candidates whose whole outcome is known by the selected date."""
    end=data.index.get_loc(pd.Timestamp(asof))
    current=data[FEATURES].iloc[end].to_numpy(dtype=float)
    eligible=data.iloc[:max(0,end-max(30,horizon)+1)].dropna(subset=FEATURES)
    distances=np.linalg.norm((eligible[FEATURES].to_numpy(dtype=float)-current)/np.array(reference['scale']),axis=1)
    ranked=[(float(distances[j]),eligible.index[j]) for j in np.argsort(distances)]
    chosen=[]
    for distance,date in sorted(ranked):
        if any(abs(data.index.get_loc(date)-data.index.get_loc(c['date']))<30 for c in chosen):continue
        i=data.index.get_loc(date)
        if i+horizon>end:continue
        chosen.append(dict(date=str(date.date()),distance=distance,
                           following_return=float(data.Close.iloc[i+horizon]/data.Close.iloc[i]-1),
                           following_risk=future_risk(data,date,horizon)))
        if len(chosen)==count:break
    return chosen


def study(data_path, run_dir, trusted_models=False):
    run=Path(run_dir);manifest=json.loads((run/'run.json').read_text())
    if hashlib.sha256(Path(data_path).read_bytes()).hexdigest()!=manifest['source_sha256']:
        raise ValueError('Study input differs from the run source')
    data=prepare(data_path);signals=pd.read_csv(run/'predictions.csv',index_col='Date',parse_dates=True)
    folds=json.loads((run/'folds.json').read_text());refs={};models={}
    for f in folds:
        train=data.loc[:f['train_end']].dropna(subset=FEATURES)
        refs[f['fold']]=training_reference(train)
        variance=f.get('return_variances')
        # Older runs can be inspected, but no risk forecast is fabricated when
        # state variances were not recorded. Local trusted model files may supply them.
        artifact=run/f"model_{f['fold']:03d}.joblib"
        if variance is None and trusted_models and artifact.exists():
            import joblib
            a=joblib.load(artifact)  # opt-in for locally generated, trusted artifacts only
            if a['train_end']!=f['train_end'] or not np.allclose(a['scaler'].inverse_transform(a['model'].means_),f['raw_means']):
                raise ValueError('Model artifact does not match its fold')
            variance=(a['model'].covars_[:,0,0]*a['scaler'].scale_[0]**2).tolist()
        models[f['fold']]=(f,variance)
    history=[];records=[]
    for date,row in signals.iterrows():
        if not pd.Timestamp(row.train_end)<date:raise ValueError('Training cutoff is not before assessment')
        i=data.index.get_loc(date);ref=refs[int(row.fold)];f,variance=models[int(row.fold)]
        assessment=assess(row,data.loc[date,FEATURES],ref,history,data[FEATURES].iloc[i-1])
        history.append(assessment['label'])
        rawp=np.zeros(3)
        for state,k in zip(f['state_order'],['p_bear','p_transition','p_bull']):rawp[state]=row[k]
        forecast=risk_forecast(rawp,f['transitions'],f['raw_means'],variance) if variance is not None else None
        baseline=float(np.sqrt(np.mean(data.Daily_Returns.iloc[max(0,i-29):i+1]**2)*252))
        records.append(dict(date=str(date.date()),**assessment,forecast_risk=forecast,
                            baseline_risk=baseline,realized_risk=future_risk(data,date),
                            analogs=historical_analogs(data,date,ref)))
    result=dict(source_sha256=manifest['source_sha256'],mode='retrospective',target='Next 5 sessions: annualized RMS daily returns',
                policy=POLICY,policy_status='Chosen after historical exploration, not a validated threshold',
                comparison=compare(records),records=records)
    (run/'assessment.json').write_text(json.dumps(result,allow_nan=False))
    return result

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--data',required=True);cli.add_argument('--run',required=True)
    cli.add_argument('--trusted-models',action='store_true',help='Only for model files generated locally, never downloaded pickles')
    a=cli.parse_args();result=study(a.data,a.run,a.trusted_models);print(json.dumps(result['comparison'],indent=2))
