"""Option 3: two-tier per-stock position limits (deep 200 / shallow 100 or 50) on the current sample.
Also exports the adopted-baseline trade points."""
import numpy as np, pandas as pd, polars as pl
from pathlib import Path

S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
MERGED = Path('/home/kevin/Project/HFT/src/research/timeSeries/data')
FEE = 19.3

df = pd.read_parquet(f'{S}/stage3_preds.parquet')
df = df.rename(columns={'TakerSell_CloseBP_net_pred': 'net', 'TakerSell_CloseBP_netM_pred': 'netM'})
df['D'] = df['Date'].astype(str).str.replace('-', '')
df['Date'] = pd.to_datetime(df['Date'])

# ---- join deep flag from merged day files ----
parts = []
for f in sorted(MERGED.glob('*.parquet')):
    if not ('20250903' <= f.stem <= '20260817'):
        continue
    parts.append(pl.scan_parquet(f).select(
        'QuoteCode', 'ChannelSeq',
        ((pl.col('FillLots_atLow') / (pl.col('avg_bidLots1') + pl.col('avg_askLots1'))) < -2).alias('deep')
    ).with_columns(pl.lit(f.stem).alias('D')))
flag = pl.concat(parts).collect()
df = df.merge(flag.to_pandas(), on=['D','QuoteCode','ChannelSeq'], how='left', validate='1:1')
df['deep'] = df['deep'].fillna(False)
print('rows', len(df), '| deep share', round(df.deep.mean(), 3))
df['tb4'] = pd.cut(df.RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False)
df['rb4'] = pd.cut(df.ToRef, [0, .01, .02, .03, .05], labels=False)
df['cell'] = df.tb4 * 10 + df.rb4
df['kb'] = df['D'] + '_' + df['cell'].astype(str)
df = df.sort_values(['Date','QuoteCode','TransTime','ChannelSeq']).reset_index(drop=True)

def thr(col, q, n_days=3):
    d = df[['D','kb',col]].dropna()
    store = {k: g[col].to_numpy() for k, g in d.groupby('kb', observed=True)}
    DATES = sorted(df['D'].unique()); buckets = sorted(df.cell.dropna().unique())
    m = {}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i-n_days):i]
        if not win: continue
        for b in buckets:
            pool = [store.get(f'{w}_{b}') for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if pool: m[f'{dt}_{b}'] = np.quantile(np.concatenate(pool), q)
    return df['kb'].map(m)

static = (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
          & (df.ToRef > 0) & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))
cond = (df.net > thr('net', 0.3)) & (df.netM > thr('netM', 0.1)) & static

def pipeline(limit_deep, limit_shallow, mask, want=False):
    m = df[cond & mask].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m['limit'] = np.where(m.deep, limit_deep, limit_shallow)
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < m.limit) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]
    pnl = m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5
    daily = pnl.groupby(m.Date).sum()
    cap_d = (m.BidPrice1/10).groupby(m.Date).sum()
    eq = daily.cumsum(); mdd = (eq - eq.cummax()).min()
    st = {'pnl萬': pnl.sum(), 'Sharpe': daily.mean()/daily.std()*np.sqrt(240), 'MDD萬': mdd,
          '部位萬': cap_d.mean(), 'capw_bp': pnl.sum()/(m.BidPrice1/10).sum()*1e4,
          '筆/日': len(m)/m.Date.nunique(), '勝率%': (daily > 0).mean()*100}
    return (st, m) if want else (st, None)

print('\n===== 兩層額度對比 =====')
variants = [('baseline 200/200', 200, 200), ('兩層 200/100', 200, 100), ('兩層 200/50', 200, 50)]
for per, lo in [('全樣本', None), ('2026', '2026-01-01'), ('holdout', '2026-04-01')]:
    mask = df.Date >= (lo or '2000-01-01')
    out = {}
    for tag, ld, ls in variants:
        st, _ = pipeline(ld, ls, mask)
        out[tag] = st
    print(f'--- {per} ---')
    print(pd.DataFrame(out).T.round(2).to_string())

# holdout 時段分佈 (post-cap) for 200/100
_, m = pipeline(200, 100, df.Date >= '2026-04-01', want=True)
TB = {3: '09:00-09:30', 2: '09:30-10:30', 1: '10:30-11:30', 0: '11:30-12:30'}
print('\n兩層 200/100 holdout 時段(倉位後):',
      {TB[k]: round(v*100, 1) for k, v in m.tb4.value_counts(normalize=True).sort_index(ascending=False).items()})

# ---- export baseline points (adopted config, full range, post-cap) ----
_, mb = pipeline(200, 200, df.Date.notna(), want=True)
keep = ['Date','QuoteCode','TransTime','ChannelSeq','BidPrice1','TakerSell_CloseBP','deep',
        'net','netM','thr_dummy'] if False else \
       ['Date','QuoteCode','TransTime','ChannelSeq','BidPrice1','TakerSell_CloseBP','deep','net','netM','tb4','rb4']
out = mb[keep].rename(columns={'net': 'net_pred', 'netM': 'netM_pred'})
Path('/home/kevin/Project/HFT/src/research/timeSeries/baseline').mkdir(exist_ok=True)
out.to_parquet('/home/kevin/Project/HFT/src/research/timeSeries/baseline/timeseries_points.parquet', index=False)
print(f'\nbaseline points exported: {len(out):,} trades, {out.Date.nunique()} days')
