"""Fetch completed daily bars. Credentials stay in environment, never the dashboard."""
import argparse
import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

HOST='https://data.alpaca.markets'


def completed_bars(bars, now=None):
    now=now or dt.datetime.now(dt.timezone.utc)
    local=now.astimezone(ZoneInfo('America/New_York'))
    today=local.date()
    out=[]
    for b in bars:
        stamp=dt.datetime.fromisoformat(b['t'].replace('Z','+00:00'))
        date=stamp.astimezone(ZoneInfo('America/New_York')).date()
        # Conservative: never infer a daily regime from today's partial bar.
        if date>today or (date==today and local.time()<dt.time(20,15)):continue
        out.append(dict(Date=str(date),Open=b['o'],High=b['h'],Low=b['l'],Close=b['c'],Volume=b['v']))
    if not out:raise ValueError('No completed daily sessions in the response')
    frame=pd.DataFrame(out).set_index('Date')
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('Provider returned duplicate or unordered daily sessions')
    return frame


def fetch(symbol,start,feed='iex'):
    if feed not in ['iex','sip']:raise ValueError('Unsupported feed')
    if not symbol.isascii() or not symbol.replace('.','').isalnum():raise ValueError('Invalid equity/ETF symbol')
    key=os.environ.get('APCA_API_KEY_ID');secret=os.environ.get('APCA_API_SECRET_KEY')
    if not key or not secret:raise ValueError('Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the local environment')
    page=None;bars=[];seen=set()
    for _ in range(100):
        args=dict(timeframe='1Day',start=start,limit=10000,adjustment='raw',feed=feed,sort='asc')
        if page:args['page_token']=page
        request=urllib.request.Request(HOST+'/v2/stocks/'+symbol+'/bars?'+urllib.parse.urlencode(args),
            headers={'APCA-API-KEY-ID':key,'APCA-API-SECRET-KEY':secret})
        try:
            with urllib.request.urlopen(request,timeout=30) as response:result=json.load(response)
        except urllib.error.HTTPError as exc:raise RuntimeError(f'Market data request failed (HTTP {exc.code})') from None
        bars.extend(result.get('bars') or [])
        page=result.get('next_page_token')
        if not page:return completed_bars(bars)
        if page in seen:raise ValueError('Repeated provider pagination token')
        seen.add(page)
    raise ValueError('Provider pagination limit reached')

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--symbol',default='SPY');cli.add_argument('--start',default='2016-01-01')
    cli.add_argument('--feed',choices=['iex','sip'],default='iex');cli.add_argument('--out',required=True)
    a=cli.parse_args();out=Path(a.out)
    if out.exists():raise SystemExit('Choose a new output filename; existing research inputs are not overwritten.')
    frame=fetch(a.symbol,a.start,a.feed);out.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(out)
    metadata=dict(symbol=a.symbol,feed=a.feed,provider='Alpaca',frequency='daily',adjustment='raw',
                  retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),last_completed_session=frame.index[-1],
                  status='Completed daily bars; today included only after 20:15 Eastern; no streaming connection')
    Path(str(out)+'.meta.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
