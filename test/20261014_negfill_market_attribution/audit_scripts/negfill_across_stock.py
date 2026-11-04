"""Descriptive cross-stock matching; not a capacity-matched strategy backtest."""
from pathlib import Path
import json
import polars as pl
import numpy as np

base=Path('/tmp/negfill_history_20261014')
keys=['Date','QuoteCode','ChannelSeq']
mk=['Date','time_cell','ref_cell']
def cols(x):
    return x.with_columns(
        pl.col('RemainSeconds').cut([3300,6900,10500,14100]).cast(pl.String).alias('time_cell'),
        pl.col('ToRef').cut([0,.01,.02,.03,.05]).cast(pl.String).alias('ref_cell'),
        (10000*(pl.col('BidPrice1')-pl.col('Close'))/pl.col('BidPrice1')+pl.col('txf_beta_60d')*pl.col('txf_to_1330_bp')-19.3).alias('resid_net_bp'),
        (pl.col('BidPrice1')*1000).alias('notional'))
tr=pl.read_parquet(base/'notebook_trades.parquet')
m=pl.read_parquet(base/'mother_events.parquet').with_columns(pl.col('Date').dt.strftime('%Y%m%d'))
pool=cols(m.join(tr.select(keys),on=keys,how='anti')).group_by(mk+['QuoteCode']).agg(pl.col('resid_net_bp').mean().alias('stock_bp')).group_by(mk).agg(pl.col('stock_bp').mean().alias('pool_bp'),pl.len().alias('stocks'))
z=cols(tr).join(pool,on=mk,how='left')
coverage=float((z['stocks'].fill_null(0)>=5).mean())
z=z.filter(pl.col('stocks')>=5)
d=z.group_by('Date').agg(
    ((pl.col('resid_net_bp')*pl.col('notional')).sum()/pl.col('notional').sum()).alias('selected_resid_net_bp'),
    ((pl.col('pool_bp')*pl.col('notional')).sum()/pl.col('notional').sum()).alias('same_time_ref_pool_resid_net_bp')
).with_columns((pl.col('selected_resid_net_bp')-pl.col('same_time_ref_pool_resid_net_bp')).alias('selected_minus_pool_bp')).sort('Date')
path=base/'matched_across_stock_daily.csv'
if path.exists():
    prior=pl.read_csv(path,schema_overrides={'Date':pl.String})
    assert d['Date'].to_list()==prior['Date'].to_list()
    np.testing.assert_allclose(d.select(pl.exclude('Date')).to_numpy(),prior.select(pl.exclude('Date')).to_numpy(),atol=1e-8,rtol=1e-10)
d.write_csv(path)
periods=[('20240101','20241231'),('20250101','20251231'),('20260101','20261013'),('20260910','20261007'),('20260915','20261013'),('20261001','20261013')]
rows=[]
for a,b in periods:
    g=d.filter(pl.col('Date').is_between(pl.lit(a),pl.lit(b)))
    rows.append({'start':a,'end':b,'days':len(g),**{c:g[c].mean() for c in d.columns if c!='Date'}})
(base/'matched_across_stock_summary.json').write_text(json.dumps({'trade_coverage':coverage,'periods':rows},indent=2))
print('Cross-stock results reproduced',coverage)
