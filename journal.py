"""Append-only local assessment log with hash-chain tamper detection."""
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read_journal(path):
    path=Path(path)
    if not path.exists():return []
    records=[];previous='0'*64
    for line in path.read_text().splitlines():
        row=json.loads(line);stored=row.pop('hash')
        if row.get('previous_hash')!=previous or digest(row)!=stored:
            raise ValueError('Journal hash chain failed verification')
        row['hash']=stored;previous=stored;records.append(row)
    return records


def append(path, assessment, now=None):
    """Idempotent per model/source/session. Conflicting revisions are rejected."""
    if assessment.get('mode')!='forward':raise ValueError('Historical assessments belong in study outputs, not the forward journal')
    now=now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:raise ValueError('An aware issuance time is required')
    asof=dt.date.fromisoformat(assessment['asof'])
    if asof>now.date() or dt.date.fromisoformat(assessment['train_end'])>=asof:
        raise ValueError('Invalid assessment timing')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    identity=digest({k:assessment[k] for k in ['symbol','feed','asof','model_version']})
    # A separate lock protects readers/writers without truncating the log.
    with Path(str(path)+'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        existing=read_journal(path)
        for record in existing:
            if record['id']==identity:
                if record['assessment']!=assessment:raise ValueError('This assessment already exists with different inputs')
                return record
        record=dict(id=identity,issued_at=now.isoformat(),previous_hash=existing[-1]['hash'] if existing else '0'*64,
                    assessment=assessment)
        record['hash']=digest(record)
        with path.open('a') as stream:
            stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
            import os
            os.fsync(stream.fileno())
        return record


def evaluate(path, data, symbol, feed):
    """Score only outcomes that occur after issuance and have fully matured."""
    from assessment import future_risk, compare
    from zoneinfo import ZoneInfo
    records=[]
    for row in read_journal(path):
        a=row['assessment'];date=a['asof']
        if a['symbol']!=symbol or a['feed']!=feed:raise ValueError('Journal/data instrument or feed mismatch')
        fingerprint=hashlib.sha256(data.loc[:date,['Open','High','Low','Close','Volume']].to_csv().encode()).hexdigest()
        if fingerprint!=a['observed_source_hash']:raise ValueError('Observed history was revised; outcomes not scored')
        issued=dt.datetime.fromisoformat(row['issued_at']).astimezone(ZoneInfo('America/New_York'))
        # Backfilled old sessions are retained but never counted as forward evidence.
        eligible=dt.date.fromisoformat(date)>=issued.date()-dt.timedelta(days=4)
        future=data.index[data.index>date]
        if len(future) and issued>=dt.datetime.combine(future[0].date(),dt.time(),tzinfo=ZoneInfo('America/New_York')):eligible=False
        outcome=future_risk(data,date) if eligible else None
        records.append(dict(date=date,status=a['status'],forecast_risk=a['forecast_risk'],
                            baseline_risk=a['baseline_risk'],realized_risk=outcome,
                            forward_eligible=eligible,issued_at=row['issued_at']))
    eligible=[r for r in records if r['forward_eligible']]
    return dict(mode='forward journal',comparison=compare(eligible),excluded_backfills=len(records)-len(eligible),records=records)


if __name__=='__main__':
    import argparse
    from pipeline import prepare
    cli=argparse.ArgumentParser();cli.add_argument('--journal',required=True);cli.add_argument('--data',required=True);cli.add_argument('--out',required=True)
    a=cli.parse_args();meta=json.loads(Path(str(a.data)+'.meta.json').read_text())
    result=evaluate(a.journal,prepare(a.data),meta['symbol'],meta['feed'])
    Path(a.out).write_text(json.dumps(result,allow_nan=False,indent=2));print(json.dumps(result['comparison'],indent=2))
