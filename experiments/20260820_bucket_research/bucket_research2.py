"""Decompose composition tilt: selection layer (condition) vs cap layer (position limits).
Add A+timebucket hybrid. Everything strictly prior-data as before."""
import numpy as np, pandas as pd

S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
FEE = 19.3
df = pd.read_parquet(f'{S}/stage3_preds.parquet')
df = df.rename(columns={'TakerSell_CloseBP_net_pred': 'net_s3', 'TakerSell_CloseBP_netM_pred': 'netM_s3'})
s6 = pd.read_parquet(f'{S}/stage6_fhat_preds.parquet').rename(
    columns={'TakerSell_CloseBP_net_pred': 'net_ex', 'TakerSell_CloseBP_netM_pred': 'netM_ex'})
keys = ['Date','QuoteCode','TransTime','ChannelSeq']
df = df.merge(s6[keys + ['net_ex','netM_ex']], on=keys, how='left', validate='1:1')
df['D'] = df['Date'].astype(str).str.replace('-', '')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(keys).reset_index(drop=True)
DATES = sorted(df['D'].unique())

static = (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
          & (df.ToRef > 0) & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))
HOLD = (df.Date >= '2026-04-01')
TUNE = (df.Date >= '2025-12-05') & (df.Date <= '2026-03-31')
df['tb4'] = pd.cut(df.RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False)
df['rb4'] = pd.cut(df.ToRef, [0, .01, .02, .03, .05], labels=False)
base_sel = static & df.rb4.notna() & HOLD
base_2d = df[base_sel].groupby(['tb4','rb4']).size() / base_sel.sum()

def comp(m):
    sel = m.groupby(['tb4','rb4']).size() / len(m)
    tv = float((sel.reindex(base_2d.index, fill_value=0) - base_2d).abs().sum() / 2)
    return {'open30': (m.tb4 == 3).mean()*100, 'toref35': (m.rb4 == 3).mean()*100, 'TV': tv}

def thr_by_bucket(pred_col, bucket_ser, qs, n_days=3):
    d = pd.DataFrame({'D': df['D'], 'b': bucket_ser, 'v': df[pred_col]}).dropna()
    store = {k: g['v'].to_numpy() for k, g in d.groupby(['D','b'], observed=True)}
    buckets = sorted(d.b.unique()); maps = {q: {} for q in qs}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i-n_days):i]
        if not win: continue
        for b in buckets:
            pool = [store.get((w, b)) for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if not pool: continue
            for q, v in zip(qs, np.quantile(np.concatenate(pool), qs)): maps[q][(dt, b)] = v
    kb = list(zip(df['D'], bucket_ser))
    return {q: pd.Series([maps[q].get(k, np.nan) for k in kb], index=df.index) for q in qs}

def caps(m):
    m = m.sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    return m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
             & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
             & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]

def econ(m):
    pnl = m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5
    daily = pnl.groupby(m.Date).sum()
    return {'t/d': len(m)/m.Date.nunique(), 'capw': pnl.sum()/(m.BidPrice1/10).sum()*1e4,
            'pnl': pnl.sum(), 'shp': daily.mean()/daily.std()*np.sqrt(240) if daily.std() > 0 else np.nan}

configs = {}
tb = df['tb4']
configs['ref: s3+timebkt'] = (df.net_s3 > thr_by_bucket('net_s3', tb, [0.5])[0.5]) & (df.netM_s3 > thr_by_bucket('netM_s3', tb, [0.2])[0.2])
configs['A:  ex+global'] = (df.net_ex > thr_by_bucket('net_ex', pd.Series(0, index=df.index), [0.5])[0.5]) & (df.netM_ex > thr_by_bucket('netM_ex', pd.Series(0, index=df.index), [0.2])[0.2])
configs['A+t: ex+timebkt'] = (df.net_ex > thr_by_bucket('net_ex', tb, [0.5])[0.5]) & (df.netM_ex > thr_by_bucket('netM_ex', tb, [0.2])[0.2])
cell16 = df['tb4']*10 + df['rb4']
configs['A2: ex+2Dcell'] = (df.net_ex > thr_by_bucket('net_ex', cell16, [0.5])[0.5]) & (df.netM_ex > thr_by_bucket('netM_ex', cell16, [0.2])[0.2])
configs['R2: s3+2Dcell'] = (df.net_s3 > thr_by_bucket('net_s3', cell16, [0.5])[0.5]) & (df.netM_s3 > thr_by_bucket('netM_s3', cell16, [0.2])[0.2])

print(f"base universe (holdout, pre-cap): open30 {df[base_sel].tb4.eq(3).mean()*100:.1f} | toref35 {df[base_sel].rb4.eq(3).mean()*100:.1f}")
rows = []
for tag, c in configs.items():
    pre = df[c & static & HOLD]
    post = caps(pre)
    e = econ(post)
    et = econ(caps(df[c & static & TUNE]))
    rows.append({'config': tag,
                 **{f'pre_{k}': v for k, v in comp(pre).items()},
                 **{f'post_{k}': v for k, v in comp(post).items()},
                 'H_t/d': e['t/d'], 'H_capw': e['capw'], 'H_pnl': e['pnl'], 'H_shp': e['shp'],
                 'T_pnl': et['pnl'], 'T_shp': et['shp']})
t = pd.DataFrame(rows)
print('\n===== HOLDOUT: composition BEFORE caps vs AFTER caps + economics =====')
print(t.round(2).to_string(index=False))
t.to_csv(f'{S}/bucket_research2.csv', index=False)
