"""Descriptive attribution of the unchanged notebook replay; no strategy selection."""
from pathlib import Path
import json,sys
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

base=Path('/tmp/negfill_history_20261014')
root=Path('/home/kevin/Project/HFT')
fee=19.3
keys=['Date','QuoteCode','ChannelSeq']

def day_frame(trades):
    x=trades.with_columns((pl.col('BidPrice1')*1000).alias('notional'),
        (pl.col('BidPrice1')*pl.col('TakerSell_CloseBP')/10).alias('native_gross'),
        ((pl.col('BidPrice1')-pl.col('Close'))*1000).alias('close_gross'),
        (-pl.col('BidPrice1')*pl.col('txf_beta_60d')*pl.col('txf_to_1330_bp')/10).alias('market_close'),
        (pl.col('BidPrice1')*fee/10).alias('cost'))
    x=x.with_columns((pl.col('native_gross')-pl.col('cost')).alias('net'),
        (pl.col('native_gross')-pl.col('close_gross')).alias('exit_effect'),
        (pl.col('close_gross')-pl.col('market_close')).alias('stock_residual_gross'))
    assert x.select(pl.struct(keys).is_duplicated().sum()).item()==0
    vals=['notional','native_gross','close_gross','market_close','cost','net','exit_effect','stock_residual_gross']
    d=x.group_by('Date').agg(*[pl.col(c).sum() for c in vals],pl.len().alias('trades'),pl.col('QuoteCode').n_unique().alias('stocks')).sort('Date')
    expected=d['market_close']+d['stock_residual_gross']+d['exit_effect']-d['cost']
    np.testing.assert_allclose(d['net'],expected,rtol=1e-12,atol=1e-6)
    return x,d

def summary(g,name):
    p=g['net'].to_numpy();cum=np.cumsum(p);dd=cum-np.maximum.accumulate(np.r_[0.,cum])[1:]
    notion=float(g['notional'].sum())
    res={'period':name,'start':g['Date'][0],'end':g['Date'][-1],'days':len(g),'active_days':int((g['trades']>0).sum()),'trades':int(g['trades'].sum()),'net':float(np.sum(p)),'MDD':float(dd.min()),'positive_days':int((p>0).sum()),'negative_days':int((p<0).sum()),'notional':notion,'net_bp':float(np.sum(p)/notion*10000),'sharpe':float(np.mean(p)/np.std(p,ddof=1)*np.sqrt(252)) if len(p)>1 else None}
    for c in ['market_close','stock_residual_gross','exit_effect','cost','close_gross']:
        res[c]=float(g[c].sum());res[c+'_bp']=res[c]/notion*10000
    res['residual_close_net']=res['stock_residual_gross']-res['cost']
    res['residual_plus_exit_net']=res['net']-res['market_close']
    return res

def ci(x,B=2000,block=5):
    x=np.asarray(x);x=x[np.isfinite(x)];rng=np.random.default_rng(20261014)
    if len(x)<2:return [None,None]
    block=min(block,len(x));nb=int(np.ceil(len(x)/block));z=[]
    for _ in range(B):
        starts=rng.integers(0,len(x)-block+1,nb)
        z.append(x[np.concatenate([np.arange(t,t+block) for t in starts])[:len(x)]].mean())
    return np.quantile(z,[.05,.95]).tolist()

tr=pl.read_parquet(base/'notebook_trades.parquet')
assert tr['Date'].min()>='20240101' and tr['Date'].max()=='20261013'
events,d=day_frame(tr)
market=pl.read_parquet(base/'broad_market_daily.parquet')
calendar=market.select('Date')
d=calendar.join(d,on='Date',how='left').with_columns(pl.exclude('Date').fill_null(0)).sort('Date')
d=d.with_columns((pl.col('stock_residual_gross')-pl.col('cost')).alias('residual_close_net'),(pl.col('net')-pl.col('market_close')).alias('residual_plus_exit_net'))
d.write_csv(base/'attribution_daily.csv')
periods={str(y):(f'{y}0101',f'{y}1231') for y in [2024,2025,2026]}
periods.update({'2026-08':('20260801','20260831'),'2026-09':('20260901','20260930'),'2026-10截至13':('20261001','20261013'),'這波9/15起':('20260915','20261013'),'新增9/24起':('20260924','20261013')})
summ=[summary(d.filter(pl.col('Date').is_between(pl.lit(a),pl.lit(b))),n) for n,(a,b) in periods.items()]
pl.DataFrame(summ).write_csv(base/'period_summary.csv')
monthly=[summary(g,k[0]) for k,g in d.with_columns(pl.col('Date').str.slice(0,6).alias('month')).partition_by('month',as_dict=True).items()]
pl.DataFrame(monthly).write_csv(base/'monthly_summary.csv')

# Drawdown episodes with an initial zero-equity anchor; current unrecovered episode retained.
p=d['net'].to_numpy();cum=np.cumsum(p);dates=d['Date'].to_list();peak=0.;peakdate='initial';active=None;episodes=[]
for date,value in zip(dates,cum):
    if value>=peak:
        if active is not None:active['recovered']=date;episodes.append(active);active=None
        peak=value;peakdate=date
    else:
        if active is None:active={'peak_date':peakdate,'start':date,'trough':date,'dd':float(value-peak),'recovered':None}
        if value-peak<active['dd']:active['dd']=float(value-peak);active['trough']=date
if active:episodes.append(active)
episodes=sorted(episodes,key=lambda x:x['dd'])
for e in episodes[:10]:
    a=e['start'];b=e['trough'];g=d.filter(pl.col('Date').is_between(pl.lit(a),pl.lit(b)));e.update({k:float(g[k].sum()) for k in ['market_close','stock_residual_gross','exit_effect','cost']})
pl.DataFrame(episodes).write_csv(base/'drawdown_episodes.csv')

# Model ranking diagnostics on the original eligible mother events.
mother=pl.read_parquet(base/'mother_events.parquet').with_columns(pl.col('Date').dt.strftime('%Y%m%d'))
ic=mother.group_by('Date').agg(
    pl.corr(pl.col('TakerSell_CloseBP_net_pred').rank(),pl.col('TakerSell_CloseBP_net').rank()).alias('timing_IC'),
    pl.corr(pl.col('TakerSell_CloseBP_netM_pred').rank(),pl.col('TakerSell_CloseBP_netM').rank()).alias('selection_IC'),
    pl.corr(pl.col('TakerSell_CloseBP_netM_pred').rank(),pl.col('TakerSell_CloseBP').rank()).alias('selection_raw_IC')).sort('Date')
ic.write_csv(base/'model_ic_daily.csv')
ics=[]
for n,(a,b) in periods.items():
    z=ic.filter(pl.col('Date').is_between(pl.lit(a),pl.lit(b)));rec={'period':n,'days':len(z)}
    for c in ['timing_IC','selection_IC','selection_raw_IC']:rec[c]=z[c].mean();rec[c+'_ci90']=ci(z[c].to_numpy())
    ics.append(rec)
(base/'model_ic_periods.json').write_text(json.dumps(ics,indent=2))

# Match each admitted trade to the unselected original negFill event pool in the
# same date/stock/time x ToRef cell. This is diagnostic selection+filter effect.
def matching_columns(z):
    return z.with_columns(pl.col('RemainSeconds').cut([3300,6900,10500,14100]).cast(pl.String).alias('time_cell'),
        pl.col('ToRef').cut([0,.01,.02,.03,.05]).cast(pl.String).alias('ref_cell'),
        (10000*(pl.col('BidPrice1')-pl.col('Close'))/pl.col('BidPrice1')+pl.col('txf_beta_60d')*pl.col('txf_to_1330_bp')).alias('residual_close_bp'))
mk=['Date','QuoteCode','time_cell','ref_cell']
remaining=matching_columns(mother.join(tr.select(keys),on=keys,how='anti'))
pool=remaining.group_by(mk).agg(pl.len().alias('pool_n'),pl.col('residual_close_bp').mean().alias('pool_residual_bp'),pl.col('TakerSell_CloseBP').mean().alias('pool_stop108_bp'))
matched=matching_columns(events).join(pool,on=mk,how='left')
coverage=float((matched['pool_n'].fill_null(0)>=10).mean())
matched=matched.filter(pl.col('pool_n')>=10).with_columns((pl.col('residual_close_bp')-pl.col('pool_residual_bp')).alias('selection_excess_bp'))
md=matched.group_by('Date').agg(((pl.col('selection_excess_bp')*pl.col('notional')).sum()/pl.col('notional').sum()).alias('matched_excess_bp'),pl.len().alias('trades')).sort('Date')
md.write_csv(base/'matched_selection_daily.csv')
matching=[]
for n,(a,b) in periods.items():
    z=md.filter(pl.col('Date').is_between(pl.lit(a),pl.lit(b)));matching.append({'period':n,'days':len(z),'mean_daily_matched_excess_bp':z['matched_excess_bp'].mean(),'ci90':ci(z['matched_excess_bp'].to_numpy())})
(base/'matched_selection_summary.json').write_text(json.dumps({'trade_coverage_pool_ge10':coverage,'periods':matching},indent=2))

# Same dates, differing notebook history lengths: test late-period reproducibility.
recent=pl.read_parquet('/tmp/negfill_recent_20261014/notebook_trades.parquet')
overlap=tr.filter(pl.col('Date')>='20260801')
join=overlap.join(recent,on=keys,suffix='_260')
parity={'full_history_recent_rows':len(overlap),'original260_recent_rows':len(recent),'common_keys':len(join),'unmatched_full':overlap.join(recent.select(keys),on=keys,how='anti').height,'unmatched_260':recent.join(overlap.select(keys),on=keys,how='anti').height,'max_label_difference':float((join['TakerSell_CloseBP']-join['TakerSell_CloseBP_260']).abs().max())}
(base/'recent_replay_parity.json').write_text(json.dumps(parity,indent=2))

marketrows=[]
for n,(a,b) in periods.items():
    z=market.filter(pl.col('Date').is_between(pl.lit(a),pl.lit(b)))
    marketrows.append({'period':n,'days':len(z),**{c:float(z[c].mean()) for c in ['stock_ew_long_oc_bp','stock_ew_long_cc_bp','positive_oc_share','open_high_long_oc_bp']}})
pl.DataFrame(marketrows).write_csv(base/'market_periods.csv')
top=events.filter(pl.col('Date')>='20260915').group_by('QuoteCode').agg(pl.col('net').sum(),pl.col('market_close').sum(),pl.col('stock_residual_gross').sum(),pl.col('exit_effect').sum(),pl.len().alias('trades')).sort('net')
top.write_csv(base/'recent_stock_contributions.csv')
fig,ax=plt.subplots(3,1,figsize=(14,11))
dt=[datetime.strptime(x,'%Y%m%d') for x in dates]
ax[0].plot(dt,np.cumsum(d['net'].to_numpy())/10000,label='negFill notebook net')
ax[0].plot(dt,np.cumsum(d['market_close'].to_numpy())/10000,label='prior beta x entry-to-close TXF short')
ax[0].plot(dt,np.cumsum(d['residual_plus_exit_net'].to_numpy())/10000,label='stock residual + exit effect - costs')
ax[1].fill_between(dt,(cum-np.maximum.accumulate(np.r_[0.,cum])[1:])/10000,0,alpha=.4)
ax[1].set_title('Drawdown, initial equity anchored at zero')
r=d.filter(pl.col('Date')>='20260901');rd=[datetime.strptime(x,'%Y%m%d') for x in r['Date']]
for c in ['net','market_close','residual_plus_exit_net']:ax[2].plot(rd,np.cumsum(r[c].to_numpy())/10000,label=c)
ax[2].set_title('September onward: close-horizon decomposition; exit effect shown in residual remainder')
for a in ax:a.grid(alpha=.25);a.set_ylabel('TWD 10,000');a.legend()
fig.suptitle('Current notebook replay | 2024-01-02 to 2026-10-13 | cost 19.3 bp')
fig.tight_layout();fig.savefig(base/'history_attribution.png',dpi=160);plt.close(fig)
print('PERIODS',json.dumps(summ,ensure_ascii=False),flush=True)
print('EPISODES',json.dumps(episodes[:8],ensure_ascii=False),flush=True)
print('MATCHING',json.dumps(matching,ensure_ascii=False),flush=True)
print('IC',json.dumps(ics,ensure_ascii=False),flush=True)
print('PARITY',parity,flush=True)
