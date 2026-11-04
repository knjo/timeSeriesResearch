from pathlib import Path
import sys,datetime
import polars as pl
root=Path('/home/kevin/Project/HFT');sys.path.insert(0,str(root))
from src.dataloader.txfDataLoader import TxfDataLoader
base=Path('/tmp/negfill_recent_20261014')
df=pl.read_parquet(base/'notebook_trades.parquet').filter(pl.col('Date')>='20260901')
loader=TxfDataLoader(Path('/media/kevin/SSD2/Data/txfTickData'))
parts=[]
for (date,),g in df.partition_by('Date',as_dict=True).items():
    day=loader.load_day(date)
    if day is None:raise ValueError(date)
    q=day.quotes.select(pl.col('_txf_quote_time').alias('factor_exit_time'),pl.col('_txf_mid').alias('factor_exit_mid')).sort('factor_exit_time')
    joined=g.sort('exit_time_diagnostic').join_asof(q,left_on='exit_time_diagnostic',right_on='factor_exit_time',strategy='backward',tolerance='10s')
    joined=joined.with_columns((pl.col('exit_time_diagnostic')-pl.col('factor_exit_time')).dt.total_milliseconds().alias('exit_factor_age_ms'),
        (-pl.col('txf_beta_60d')*(pl.col('factor_exit_mid')/pl.col('txf_entry_mid')-1)*10000).alias('market_native_bp'))
    parts.append(joined)
    print(date,'rows',len(g),'unmatched',joined['market_native_bp'].null_count(),flush=True)
pl.concat(parts,how='vertical_relaxed').write_parquet(base/'native_exit_factor.parquet')
