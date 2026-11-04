"""Fixed 20-active-session zero crossing; next-session shadow gate diagnostic."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/home/kevin/Project/HFT');sys.path.insert(0,str(root))
from src.research.strategy_benchmark_analysis import build_daily_strategy
src=root/'src/research/timeSeries/test/20261014_negfill_market_attribution/artifacts/v1'
out=Path('/tmp/negfill_r20_gate_20261014');out.mkdir(exist_ok=True)

def stats(values):
    x=np.asarray(values,dtype=float);c=np.cumsum(x);dd=c-np.maximum.accumulate(np.r_[0.,c])[1:]
    return {'net':float(x.sum()),'mdd':float(dd.min()),'sharpe':float(x.mean()/x.std(ddof=1)*np.sqrt(252)) if x.std(ddof=1)>0 else None}

def gate_path(r20,confirmation):
    # Decisions made AFTER close affect the NEXT active session. Shadow returns
    # remain the unchanged full strategy even while its execution is disabled.
    run=0;disabled=False;execute=[];streak=[];decision=[]
    for value in r20:
        execute.append(not disabled)
        if np.isfinite(value):
            if value<0:
                run+=1
                if run>=confirmation:disabled=True
            else:
                run=0;disabled=False
        streak.append(run);decision.append(disabled)
    return np.asarray(execute),np.asarray(streak),np.asarray(decision)

tr=pl.read_parquet(src/'notebook_trades.parquet')
p,audit=build_daily_strategy(tr.to_pandas())
p['r20']=p.strategy_net_bp.rolling(20,min_periods=20).mean()
a=pl.from_pandas(p).with_columns(pl.col('Date').dt.strftime('%Y%m%d')).sort('Date')
# Match notebook black line independently using Polars aggregation.
q=tr.group_by('Date').agg((pl.col('BidPrice1')*(pl.col('TakerSell_CloseBP')-19.3)/10).sum().alias('net'),(pl.col('BidPrice1')*1000).sum().alias('notional')).sort('Date').with_columns((pl.col('net')/pl.col('notional')*10000).alias('bp')).with_columns(pl.col('bp').rolling_mean(20,min_samples=20).alias('r20'))
assert a['Date'].to_list()==q['Date'].to_list()
np.testing.assert_allclose(a['r20'].to_numpy(),q['r20'].to_numpy(),atol=1e-10,equal_nan=True)
r=a['r20'].to_numpy();dates=a['Date'].to_list();cash=a['strategy_net_pnl'].to_numpy()
for k in [1,5,10]:
    execute,streak,decision=gate_path(r,k)
    a=a.with_columns(pl.Series(f'execute_k{k}',execute),pl.Series(f'gate_net_k{k}',cash*execute),pl.Series(f'disabled_after_close_k{k}',decision))
a=a.with_columns(pl.Series('negative_streak',streak))
calendar=pl.read_csv(src/'attribution_daily.csv',schema_overrides={'Date':pl.String}).select('Date','net')
d=calendar.join(a,on='Date',how='left').with_columns(pl.col('strategy_net_pnl').fill_null(0),*[pl.col(f'gate_net_k{k}').fill_null(0) for k in [1,5,10]])
np.testing.assert_allclose(d['net'],d['strategy_net_pnl'],atol=1e-6)
a.write_csv(out/'active_daily.csv');d.write_csv(out/'calendar_daily.csv')

# Spell out every episode, including the positive-return day that makes R20
# recover: that day's signal is still skipped by a close-based re-entry rule.
eps=[];i=0
while i<len(r):
    if not r[i]<0:i+=1;continue
    start=i
    while i<len(r) and r[i]<0:i+=1
    end=i-1;recover=i if i<len(r) else None
    stop_end=i if recover is not None else end
    loss_window=cash[start+1:stop_end+1]
    row={'start':dates[start],'last_negative':dates[end],'negative_active_days':end-start+1,
         'recovery_observed':dates[recover] if recover is not None else None,
         'first_disabled_session':dates[start+1] if start+1<len(dates) else None,
         'last_disabled_session':dates[stop_end] if start+1<=stop_end else None,
         'first_reenabled_session':dates[recover+1] if recover is not None and recover+1<len(dates) else None,
         'minimum_r20':float(np.min(r[start:end+1])),
         'k1_skipped_net':float(loss_window.sum()),'k1_gate_gain':float(-loss_window.sum()),
         'skipped_positive_pnl':float(loss_window[loss_window>0].sum()),
         'skipped_negative_pnl':float(loss_window[loss_window<0].sum()),
         'start_day_pnl':float(cash[start]),'recovery_day_pnl':float(cash[recover]) if recover is not None else None}
    for h in [5,10,20]:
        f=cash[start+1:start+1+h];row[f'forward_{h}_available']=len(f);row[f'forward_{h}_net']=float(f.sum())
    for k in [5,10]:
        trigger=start+k-1
        if trigger<=end:
            off=cash[trigger+1:stop_end+1]
            row[f'k{k}_trigger_close']=dates[trigger]
            row[f'k{k}_first_disabled']=dates[trigger+1] if trigger+1<len(dates) else None
            row[f'k{k}_gate_gain']=float(-off.sum())
        else:
            row[f'k{k}_trigger_close']=None;row[f'k{k}_first_disabled']=None;row[f'k{k}_gate_gain']=None
    eps.append(row)
pl.DataFrame(eps,infer_schema_length=None).write_csv(out/'negative_episodes.csv')

annual=[]
for year in [2024,2025,2026]:
    z=a.filter(pl.col('Date').str.starts_with(str(year)));e=[x for x in eps if x['start'].startswith(str(year))]
    annual.append({'year':year,'active_days':len(z),'known_r20_days':z['r20'].len()-z['r20'].null_count(),'negative_days':z.filter(pl.col('r20')<0).height,'episodes':len(e),'longest_negative_run':max(x['negative_active_days'] for x in e)})
pl.DataFrame(annual).write_csv(out/'annual_negative_counts.csv')
periods=[('2024','20240101','20241231'),('2025','20250101','20251231'),('2026','20260101','20261013'),('all','20240101','20261013'),('current_after_trigger','20260916','20261013')]
summary=[]
for name,lo,hi in periods:
    z=d.filter(pl.col('Date').is_between(pl.lit(lo),pl.lit(hi)))
    base=stats(z['net'].to_numpy())
    summary.append({'period':name,'variant':'always_on','calendar_days':len(z),'skipped_active_days':0,'gate_gain':0.,**base})
    for k in [1,5,10]:
        x=z[f'gate_net_k{k}'].to_numpy();s=stats(x)
        summary.append({'period':name,'variant':f'confirm_{k}','calendar_days':len(z),'skipped_active_days':int((z[f'execute_k{k}']==False).sum()),'gate_gain':s['net']-base['net'],**s})
pl.DataFrame(summary).write_csv(out/'gate_summary.csv')

# Causal trigger checks: future value changes cannot alter current execution.
assert gate_path([1,-1,5],1)[0].tolist()==[True,True,False]
assert gate_path([1,-1,-2,1,2],1)[0].tolist()==[True,True,False,False,True]
assert gate_path([1,-1,-2,1,2],5)[0].tolist()==[True]*5
for k in [1,5,10]:
    execute=gate_path(r,k)[0]
    for cut in [25,200,500,650]:
        altered=r.copy();altered[cut:]=999.
        assert gate_path(altered,k)[0][:cut+1].tolist()==execute[:cut+1].tolist()
    gains=sum(x['k1_gate_gain'] for x in eps) if k==1 else sum(x[f'k{k}_gate_gain'] or 0 for x in eps)
    np.testing.assert_allclose(gains,(a[f'gate_net_k{k}']-a['strategy_net_pnl']).sum(),atol=1e-6)

import datetime
fig,axes=plt.subplots(3,1,figsize=(14,10))
for ax,year in zip(axes,[2024,2025,2026]):
    z=a.filter(pl.col('Date').str.starts_with(str(year)))
    t=[datetime.datetime.strptime(x,'%Y%m%d') for x in z['Date']];rr=z['r20'].to_numpy()
    ax.plot(t,rr,color='black',label='20-active-session mean signal short net BP')
    ax.fill_between(t,rr,0,where=rr<0,color='red',alpha=.2)
    ax.axhline(0,color='grey',linewidth=.8);ax.set_title(str(year)+(' through Oct 13' if year==2026 else ''))
    ax.set_ylabel('Mean daily BP');ax.grid(alpha=.2)
axes[0].legend();fig.suptitle('Same notebook black line | 19.3 bp costs | continuous windows across years')
fig.tight_layout();fig.savefig(out/'r20_by_year.png',dpi=140);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
t=[datetime.datetime.strptime(x,'%Y%m%d') for x in d['Date']]
for col,label in [('net','Always on')]+[(f'gate_net_k{k}',f'R20 < 0, confirm {k}, next session') for k in [1,5,10]]:
    values=d[col].to_numpy();c=np.cumsum(values);dd=c-np.maximum.accumulate(np.r_[0.,c])[1:]
    axes[0].plot(t,c/10000,label=label);axes[1].plot(t,dd/10000,label=label)
for ax in axes:ax.set_ylabel('TWD 10,000');ax.grid(alpha=.2)
axes[0].legend();axes[0].set_title('Fixed-rule diagnostic; shadow always runs; no capital reallocation')
axes[1].set_title('Drawdown with initial zero anchor')
fig.tight_layout();fig.savefig(out/'gate_equity.png',dpi=140);plt.close(fig)
manifest={'status':'scoped / RESEARCH_DIAGNOSTIC_ONLY','source':str(src/'notebook_trades.parquet'),'source_sha256':hashlib.sha256((src/'notebook_trades.parquet').read_bytes()).hexdigest(),'rows':len(tr),'calendar_days':len(d),'active_days':len(a),'warmup_unknown_active_days':19,'signal_definition':'20 active-session simple mean of daily notional-weighted net BP, costs 19.3 bp','rules':'fixed confirmations 1,5,10; stop on next active session after confirmed R20<0; restart session after shadow R20>=0; no search or strategy changes','checks':['exact helper/Polars rolling parity','cash reconciles original report','future perturbation leaves prior/current execution unchanged','episode gains reconcile total gate gains'],'limitations':['current-rule replay, not historical online OOS','original legacy global median preprocessing caveat persists','rule considered after seeing 2026 drawdown; historical comparison not untouched validation','Jan 2024 first19 active sessions unknown: keep always-on; no annual rolling reset','date endpoints 20261013, no live kernel access']}
(out/'verification.json').write_text(json.dumps(manifest,indent=2))
print('ANNUAL',json.dumps(annual,indent=2));print('GATE',json.dumps(summary,indent=2));print('LONG_EPISODES',json.dumps([e for e in eps if e['negative_active_days']>=5],indent=2))
