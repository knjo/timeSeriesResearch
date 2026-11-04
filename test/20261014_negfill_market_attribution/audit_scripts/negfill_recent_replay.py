"""Replay existing notebook model/risk cells; save only recent diagnostic inputs."""
import os,sys,json,ast,datetime,time,gc
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
root=Path('/home/kevin/Project/HFT');sys.path.insert(0,str(root))
base=root/'src/research/timeSeries';os.chdir(base)
out=Path('/tmp/negfill_recent_20261014');out.mkdir(exist_ok=True)
book=json.loads((base/'time_series_research_model.ipynb').read_text())
sources={i:''.join(c.get('source',[])) for i,c in enumerate(book['cells'])}
from src.research.rolling_model import train_and_predict_ridge_rolling
from src.research.backTest import BackTest
from src.research import tool
# Keep the same cache mechanism in an isolated location.
original_stop=tool.add_stop_loss_prod
def isolated_stop(df,**kwargs):
    return original_stop(df,cache_path=out/'stop_times.parquet',**kwargs)
tool.add_stop_loss_prod=isolated_stop
file_names=np.array(sorted(p.name for p in (base/'data').glob('*.parquet') if p.stem<='20261013'))
folder_path=str(base/'data')
selected=[2,4,5,6,8,10,11,12,13,14,23,24,27,30,34]
words=set()
for i in selected:
    tree=ast.parse(sources[i])
    words.update(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str))
    words.update(n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute))
schema=pl.scan_parquet(base/'data'/file_names[-1]).collect_schema()
extra={'Date','QuoteCode','ChannelSeq','TransTime','txf_beta_60d','txf_to_1330_bp','txf_entry_mid','txf_entry_quote_time','Close','RefPrice','FutureHigh','TickSize','TrialMatch','OTC','BidPrice1','avg_askLots1','avg_bidLots1','close_return_20d'}
read_columns=sorted((words|extra)&set(schema.names()))
print('Notebook projection',len(read_columns),'columns; source sessions',len(file_names[-260:]),file_names[-260],file_names[-1],flush=True)
src=sources[2].replace("pd.read_parquet( f'{folder_path}/{f}' )","pl.read_parquet(f'{folder_path}/{f}', columns=read_columns).to_pandas()")
exec(compile(src,'notebook_cell_2','exec'))
signal_df=pd.concat(data,ignore_index=True,copy=False);del data;gc.collect()
for i in [4,5,6,8,10,11,12,13,14,23,24]:
    print('Executing notebook cell',i,flush=True)
    src=sources[i]
    if i==14:
        src=src.replace("plot_model_dashboard(result, model_type='normal')",'')
    exec(compile(src,f'notebook_cell_{i}','exec'))
condition=condition & (signal_df.TakerSell_CloseBP_net_pred>signal_df.thr_net)&(signal_df.TakerSell_CloseBP_netM_pred>signal_df.thr_netM)
recent=signal_df.Date>=pd.Timestamp('2026-08-01')
cols=[c for c in ['Date','QuoteCode','ChannelSeq','TransTime','BidPrice1','Close','RefPrice','FutureHigh','TickSize','txf_beta_60d','txf_to_1330_bp','txf_entry_mid','txf_entry_quote_time','TakerSell_CloseBP','TakerSell_CloseBP_net','TakerSell_CloseBP_netM','TakerSell_CloseBP_net_pred','TakerSell_CloseBP_netM_pred','ToRef','RemainSeconds','OTC'] if c in signal_df]
pl.from_pandas(signal_df.loc[recent,cols]).write_parquet(out/'mother_events.parquet')
exec(compile(sources[27],'notebook_cell_27','exec'))
signal_df_model=signal_df_model.loc[signal_df_model.Date>=pd.Timestamp('2026-08-01')].copy().reset_index(drop=True)
del signal_df;gc.collect()
src=sources[34]
# Preserve row-matched exit timestamps before the notebook sorts/filter rows.
src=src.replace("signalwithRisk_df['sl_first'] = sl_first","signalwithRisk_df['sl_first'] = sl_first\n        signalwithRisk_df['exit_time_diagnostic'] = pd.to_datetime(signalwithRisk_df['Date']) + pd.Timedelta(hours=13,minutes=30)\n        signalwithRisk_df.loc[tp_first,'exit_time_diagnostic'] = t.loc[tp_first,'low_time']\n        signalwithRisk_df.loc[sl_first,'exit_time_diagnostic'] = t.loc[sl_first,'high_time']")
src=src.replace('results = bt.run_analysis()','results = bt.run_analysis(plot=False)')
exec(compile(src,'notebook_cell_34','exec'))
cols=list(dict.fromkeys(cols+['tp_first','sl_first','tpPrice','slPrice','exit_time_diagnostic','stopLossProd']))
pl.from_pandas(signalwithRisk_df[cols]).write_parquet(out/'notebook_trades.parquet')
pl.from_pandas(daily_ticker_df).write_parquet(out/'notebook_daily_ticker.parquet')
import hashlib
(out/'replay_manifest.json').write_text(json.dumps({'notebook_sha256':hashlib.sha256((base/'time_series_research_model.ipynb').read_bytes()).hexdigest(),'source_start':file_names[-260],'source_end':file_names[-1],'source_days':260,'cells':[2,4,5,6,8,10,11,12,13,14,23,24,27,34],'read_columns':read_columns,'diagnostic_start':'20260801','note':'Unchanged current notebook training and risk; omit plots; isolated stop cache; same row-matched exit timestamp retained. Notebook stop-day exclusion is commented out.'},indent=2))
print('SAVED',out,len(signalwithRisk_df),flush=True)
