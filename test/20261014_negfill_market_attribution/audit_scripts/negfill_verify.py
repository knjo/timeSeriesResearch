"""Reconcile analysis cash with the project's original BackTest."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import polars as pl

root=Path('/home/kevin/Project/HFT');sys.path.insert(0,str(root))
from src.research.backTest import BackTest
base=Path('/tmp/negfill_history_20261014')
t=pl.read_parquet(base/'notebook_trades.parquet')
bt=BackTest(t.to_pandas())
stats,d=bt.run_analysis(return_daily=True,plot=False)
a=pl.read_csv(base/'attribution_daily.csv',schema_overrides={'Date':pl.String})
p=pl.from_pandas(d.reset_index()).with_columns(pl.col('Date').dt.strftime('%Y%m%d'))
z=a.join(p.select('Date','PnL'),on='Date',how='left').with_columns(pl.col('PnL').fill_null(0))
np.testing.assert_allclose(z['net'],z['PnL'],atol=1e-6,rtol=1e-12)
f=pl.read_parquet('/tmp/negfill_recent_20261014/native_exit_factor.parquet')
nb=root/'src/research/timeSeries/time_series_research_model.ipynb'
sha=hashlib.sha256(nb.read_bytes()).hexdigest()
assert sha==json.loads((base/'replay_manifest.json').read_text())['notebook_sha256']
parity=json.loads((base/'recent_replay_parity.json').read_text())
assert parity['unmatched_full']==0 and parity['unmatched_260']==0
assert abs(parity['max_label_difference'])<1e-10
assert f['market_native_bp'].null_count()==0
out={'BackTest_daily_cash_max_abs_error_TWD':float((z['net']-z['PnL']).abs().max()),'accepted_trades':len(t),'source_calendar_days':len(a),'no_trade_days':a.filter(pl.col('trades')==0)['Date'].to_list(),'BackTest_active_days':len(p),'notebook_hash_unchanged':True,'notebook_sha256':sha,'recent_native_factor_rows':len(f),'recent_native_factor_missing':f['market_native_bp'].null_count(),'recent_exit_quote_max_age_ms':f['exit_factor_age_ms'].max(),'recent_parity':parity,'description':'Original BackTest, same accepted events/exit labels/19.3 bp fees. Main report Sharpe uses all 672 source sessions; BackTest itself reports active-day Sharpe.'}
(base/'verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
