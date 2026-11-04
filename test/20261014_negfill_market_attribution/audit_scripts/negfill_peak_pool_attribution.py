"""Exact black-line accounting against the unchanged top-turnover pool."""
from pathlib import Path
import sys,json,hashlib,datetime
import numpy as np
import polars as pl
import pandas as pd
root=Path('/home/kevin/Project/HFT');sys.path.insert(0,str(root))
from src.research.strategy_benchmark_analysis import load_rank_basket_open_close_bp
from src.pipeline_storage import load_pipeline_storage
src=root/'src/research/timeSeries/test/20261014_negfill_market_attribution/artifacts'
out=Path('/tmp/negfill_peak_pool_20261015');out.mkdir(exist_ok=True)


def load_pool(dates):
    target=out/'pool_daily.csv'
    if target.exists():prior=pl.read_csv(target,schema_overrides={'Date':pl.String})
    else:
        cached=src/'benchmark_plot_v2/daily_compare.csv'
        prior=pl.read_csv(cached).select(pl.col('Date').str.replace_all('-','').str.slice(0,8),pl.col('rank_lt_100_close_return_bp').alias('pool_long_bp'),pl.col('rank_coverage').alias('coverage'))
    known={r['Date']:r for r in prior.to_dicts()};storage=load_pipeline_storage(project_root=root)
    missing=[d for d in dates if d not in known]
    print('Pool: cached',len(known),'load',len(missing),flush=True)
    for i,d in enumerate(missing):
        x=load_rank_basket_open_close_bp(pd.Timestamp(d),premarket_dir=storage.pre_market_dir,tick_data_dir=storage.tick_dir,stop_ratio=None)
        known[d]={'Date':d,'pool_long_bp':x['rank_lt_100_close_return_bp'],'coverage':x['rank_coverage']}
        if (i+1)%50==0:
            pl.DataFrame(list(known.values())).sort('Date').write_csv(target)
            print('Pool loaded',i+1,'/',len(missing),flush=True)
    result=pl.DataFrame(list(known.values())).sort('Date');result.write_csv(target)
    return result


def main():
    d=pl.read_csv(src/'r20_zero_crossing/active_daily.csv',schema_overrides={'Date':pl.String}).select('Date','strategy_net_bp','strategy_net_pnl','entry_notional','r20')
    pool=load_pool(d['Date'].to_list())
    d=d.join(pool,on='Date',how='left').with_columns((-pl.col('pool_long_bp')-19.3).alias('pool_short_net_bp'))
    assert d.filter(~pl.col('pool_short_net_bp').is_finite()|pl.col('pool_short_net_bp').is_null()).height==0,'Missing pool data: do not bridge rolling windows'
    d=d.with_columns((pl.col('strategy_net_bp')-pl.col('pool_short_net_bp')).alias('relative_excess_bp'))
    components=['strategy_net_bp','pool_short_net_bp','relative_excess_bp']
    d=d.with_columns(*[pl.col(c).rolling_mean(20,min_samples=20).alias(c+'_r20') for c in components])
    np.testing.assert_allclose(d['r20'].to_numpy(),d['strategy_net_bp_r20'].to_numpy(),atol=1e-10,equal_nan=True)
    np.testing.assert_allclose(d['r20'].to_numpy(),(d['pool_short_net_bp_r20']+d['relative_excess_bp_r20']).to_numpy(),atol=1e-10,equal_nan=True)
    for c in components:
        d=d.with_columns((pl.col(c)/20).alias(c+'_enter_contribution'),(-pl.col(c).shift(20)/20).alias(c+'_leave_contribution'),pl.col(c+'_r20').diff().alias(c+'_r20_change'))
        np.testing.assert_allclose(d[c+'_r20_change'].to_numpy()[20:],(d[c+'_enter_contribution']+d[c+'_leave_contribution']).to_numpy()[20:],atol=1e-10)
    # Same-sample close-horizon attribution as a robustness diagnostic, keeping
    # stock remainder and native-exit effects separate from the TXF factor.
    a=pl.read_csv(src/'v1/attribution_daily.csv',schema_overrides={'Date':pl.String}).select('Date',*[(pl.col(c)/pl.col('notional')*10000).alias(c+'_bp') for c in ['market_close','stock_residual_gross','exit_effect','cost']])
    d=d.join(a,on='Date',how='left').with_columns((pl.col('stock_residual_gross_bp')-pl.col('cost_bp')).alias('stock_residual_net_bp'))
    closecomponents=['market_close_bp','stock_residual_net_bp','exit_effect_bp']
    d=d.with_columns(*[pl.col(c).rolling_mean(20,min_samples=20).alias(c+'_r20') for c in closecomponents])
    np.testing.assert_allclose(d['r20'].to_numpy(),sum((d[c+'_r20'] for c in closecomponents)).to_numpy(),atol=1e-10,equal_nan=True)
    d.write_csv(out/'daily_decomposition.csv')
    dates=d['Date'].to_list();s=d['r20'].to_numpy();records=[]
    def event(i,j,kind):
        row={'kind':kind,'peak_date':dates[i],'end_date':dates[j],'active_sessions':j-i,'signal_peak_bp':float(s[i]),'signal_end_bp':float(s[j]),'signal_change_bp':float(s[j]-s[i])}
        for c in ['pool_short_net_bp','relative_excess_bp']+closecomponents:
            row[c+'_peak']=d[c+'_r20'][i];row[c+'_end']=d[c+'_r20'][j];row[c+'_change']=d[c+'_r20'][j]-d[c+'_r20'][i]
        for c in components:
            # Contributions from newly added vs expired observations over this
            # endpoint change. Overlap cancels; classify neither as a cause alone.
            row[c+'_new_contribution']=float(d[c+'_enter_contribution'][i+1:j+1].sum())
            row[c+'_expired_contribution']=float(d[c+'_leave_contribution'][i+1:j+1].sum())
            np.testing.assert_allclose(row[c+'_new_contribution']+row[c+'_expired_contribution'],d[c+'_r20'][j]-d[c+'_r20'][i],atol=1e-9)
        return row
    # Fixed descriptive definition: each year's R20 maximum, followed by its
    # lowest R20 before year end/cutoff. Future-known endpoints, NOT a signal.
    for year in [2024,2025,2026]:
        ix=np.array([i for i,x in enumerate(dates) if x.startswith(str(year)) and np.isfinite(s[i])]);i=int(ix[np.argmax(s[ix])]);later=ix[ix>=i];j=int(later[np.argmin(s[later])]);records.append(event(i,j,'annual_max_to_subsequent_min'))
    # Every completed upswing/downturn episode of R20 below its running peak,
    # tracked separately by year: no handpicked trough threshold.
    excursions=[]
    for year in [2024,2025,2026]:
        ix=[i for i,x in enumerate(dates) if x.startswith(str(year)) and np.isfinite(s[i])]
        peak=ix[0];trough=None
        for i in ix[1:]:
            if s[i]>=s[peak]:
                if trough is not None:
                    row=event(peak,trough,'year_running_high_drawdown');row['recovered_date']=dates[i];excursions.append(row)
                peak=i;trough=None
            elif trough is None or s[i]<s[trough]:trough=i
        if trough is not None:
            row=event(peak,trough,'year_running_high_drawdown');row['recovered_date']=None;excursions.append(row)
    pl.DataFrame(excursions).write_csv(out/'all_running_high_drawdowns.csv')
    idx={x:i for i,x in enumerate(dates)}
    for start,end in [('20260909','20260915'),('20260915','20261013')]:records.append(event(idx[start],idx[end],'recent_fixed_dates'))
    pl.DataFrame(records).write_csv(out/'peak_decomposition.csv')
    # Every daily downward step, not only selected large episodes.
    summary=[]
    for year in [2024,2025,2026]:
        z=d.filter(pl.col('Date').str.starts_with(str(year)) & (pl.col('strategy_net_bp_r20_change')<0))
        summary.append({'year':year,'down_steps':len(z),'total_signal_down_bp':z['strategy_net_bp_r20_change'].sum(),'pool_change_on_down_steps_bp':z['pool_short_net_bp_r20_change'].sum(),'excess_change_on_down_steps_bp':z['relative_excess_bp_r20_change'].sum(),'both_decline_days':z.filter((pl.col('pool_short_net_bp_r20_change')<0)&(pl.col('relative_excess_bp_r20_change')<0)).height,'pool_improves_but_signal_falls_days':z.filter(pl.col('pool_short_net_bp_r20_change')>0).height,'signal_enter_sum_bp':z['strategy_net_bp_enter_contribution'].sum(),'signal_expire_sum_bp':z['strategy_net_bp_leave_contribution'].sum()})
    pl.DataFrame(summary).write_csv(out/'all_down_steps_summary.csv')
    # Largest one-session black-line falls, including the original old-day date.
    d.with_columns(pl.col('Date').shift(20).alias('expired_date')).filter(pl.col('strategy_net_bp_r20_change').is_not_null()).sort('strategy_net_bp_r20_change').head(20).write_csv(out/'largest_down_steps.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(14,11))
    for ax,year in zip(axes,[2024,2025,2026]):
        z=d.filter(pl.col('Date').str.starts_with(str(year)));t=[datetime.datetime.strptime(x,'%Y%m%d') for x in z['Date']]
        for c,label,color in [('strategy_net_bp','Signal short net','black'),('pool_short_net_bp','Pure pool short net','tab:orange'),('relative_excess_bp','Signal minus pool (relative excess)','tab:blue')]:ax.plot(t,z[c+'_r20'],label=label,color=color)
        ax.axhline(0,color='grey',linestyle='--',linewidth=.7);ax.set_title(str(year));ax.set_ylabel('20-active-session mean BP');ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Exact identity: signal = same-direction pool + relative excess | through 2026-10-13')
    fig.tight_layout();fig.savefig(out/'pool_vs_excess_by_year.png',dpi=140);plt.close(fig)
    validation={'source_end':'20261013','execution_date':'20261015','active_days':len(d),'pool_missing':0,'minimum_pool_coverage':d['coverage'].min(),'rolling_window':20,'fee_bp':19.3,'identities':['signal equals pool short plus relative excess','rolling delta equals incoming/20 minus expired/20','signal equals close-horizon TXF factor plus stock residual net plus exit effect'],'peak_definition':'annual R20 maximum to subsequent same-year minimum; retrospective endpoints, not forecast','limits':['same current-rule replay and legacy median caveats','relative excess includes stock weights, timing and exits, not pure causal alpha','full-history beta factor uses entry-to-13:30; native exit effect separately retained'],'input_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src/'r20_zero_crossing/active_daily.csv',src/'v1/attribution_daily.csv',root/'src/research/strategy_benchmark_analysis.py']}}
    (out/'verification.json').write_text(json.dumps(validation,indent=2))
    print('PEAKS',json.dumps(records,indent=2));print('DOWNSTEPS',json.dumps(summary,indent=2));print('VERIFICATION',json.dumps(validation,indent=2))

if __name__=='__main__':main()
