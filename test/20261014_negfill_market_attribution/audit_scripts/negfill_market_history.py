from pathlib import Path
import json
import numpy as np
import polars as pl

out=Path('/tmp/negfill_history_20261014');out.mkdir(exist_ok=True)
root=Path('/media/kevin/SSD2/Data/marketData')
rows=[];issues=[]
for f in sorted(root.glob('*_marketData.parquet')):
    d=f.name[:8]
    if not '20240101'<=d<='20261013':continue
    q=pl.scan_parquet(f);schema=q.collect_schema()
    need=['quote_code','ins_type','opening_ref_price','open_price','close_price','trading_volume','trading_turnover']
    if any(c not in schema for c in need):issues.append({'date':d,'missing':[c for c in need if c not in schema]});continue
    q=q.select(need).filter(pl.col('quote_code').str.contains(r'^[1-9]\d{3}$')&(pl.col('ins_type')=='stock')&(pl.col('open_price')>0)&(pl.col('close_price')>0)&(pl.col('opening_ref_price')>0)&(pl.col('trading_volume')>=500000))
    z=q.with_columns(((pl.col('close_price')/pl.col('open_price')-1)*10000).alias('oc_bp'),((pl.col('close_price')/pl.col('opening_ref_price')-1)*10000).alias('cc_bp'),(pl.col('open_price')/pl.col('opening_ref_price')-1).alias('gap')).collect()
    assert z.height>100,(d,z.height)
    row={'Date':d,'n':z.height,'stock_ew_long_oc_bp':z['oc_bp'].mean(),'stock_ew_long_cc_bp':z['cc_bp'].mean(),'positive_oc_share':(z['oc_bp']>0).mean(),'open_high_long_oc_bp':z.filter(pl.col('gap')>.005)['oc_bp'].mean()}
    for code in ['2330']:
        a=z.filter(pl.col('quote_code')==code);row[code+'_oc_bp']=a['oc_bp'].item() if len(a) else None
    rows.append(row)
pl.DataFrame(rows).write_parquet(out/'broad_market_daily.parquet')
(out/'broad_market_coverage.json').write_text(json.dumps({'days':len(rows),'start':rows[0]['Date'],'end':rows[-1]['Date'],'issues':issues,'universe':'ordinary four-digit stocks with daily volume >=500,000 shares and valid open/close/reference; equal-weight; descriptive realized activity screen'},indent=2))
print('Market coverage',len(rows),'issues',issues)
