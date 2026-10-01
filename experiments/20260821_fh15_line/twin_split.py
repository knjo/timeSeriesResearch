"""FH15 book PnL split: twin trades (same-trade match with negFill) vs unique trades."""
import numpy as np, pandas as pd
S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
FEE = 19.3

def load(path):
    df = pd.read_parquet(path)
    df = df.rename(columns={'TakerSell_CloseBP_net_pred': 'net', 'TakerSell_CloseBP_netM_pred': 'netM'})
    df['D'] = df['Date'].astype(str).str.replace('-', '')
    df['Date'] = pd.to_datetime(df['Date'])
    df['tb4'] = pd.cut(df.RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False)
    df['rb4'] = pd.cut(df.ToRef, [0, .01, .02, .03, .05], labels=False)
    df['kb'] = df['D'] + '_' + (df.tb4*10 + df.rb4).astype(str)
    return df.sort_values(['Date','QuoteCode','TransTime','ChannelSeq']).reset_index(drop=True)

def thr(df, col, q, n_days=3):
    d = df[['D','kb',col]].dropna()
    store = {k: g[col].to_numpy() for k, g in d.groupby('kb', observed=True)}
    DATES = sorted(df['D'].unique()); buckets = sorted(set(k.split('_')[1] for k in store)); m = {}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i-n_days):i]
        if not win: continue
        for b in buckets:
            pool = [store.get(f'{w}_{b}') for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if pool: m[f'{dt}_{b}'] = np.quantile(np.concatenate(pool), q)
    return df['kb'].map(m)

def book(df, qn, qm):
    static = (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
              & (df.ToRef > 0) & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))
    c = (df.net > thr(df, 'net', qn)) & (df.netM > thr(df, 'netM', qm)) & static
    m = df[c].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]
    return m[['Date','QuoteCode','TransTime','ToRef','BidPrice1','TakerSell_CloseBP']].reset_index(drop=True)

A = book(load(f'{S}/stage3_preds.parquet'), 0.3, 0.1)
B = book(load(f'{S}/stage_fh15_preds.parquet'), 0.3, 0.1)

x = B.sort_values('TransTime').reset_index(drop=True)
mt = pd.merge_asof(x, A.sort_values('TransTime')[['Date','QuoteCode','TransTime']].assign(t2=lambda d: d.TransTime),
                   on='TransTime', by=['Date','QuoteCode'], direction='nearest')
ok_t = set(map(tuple, mt.loc[((mt.TransTime - mt.t2).abs().dt.total_seconds() <= 60).fillna(False),
                             ['Date','QuoteCode','TransTime']].values))
x2 = B.sort_values('ToRef').reset_index(drop=True)
mr = pd.merge_asof(x2, A.sort_values('ToRef')[['Date','QuoteCode','ToRef']].rename(columns={'ToRef':'r2'}).assign(ToRef=lambda d: d.r2),
                   on='ToRef', by=['Date','QuoteCode'], direction='nearest')
ok_r = set(map(tuple, mr.loc[((mr.ToRef - mr.r2).abs() <= 0.01).fillna(False),
                             ['Date','QuoteCode','TransTime']].values))
B['twin'] = B.apply(lambda r: (r.Date, r.QuoteCode, r.TransTime) in ok_t or (r.Date, r.QuoteCode, r.TransTime) in ok_r, axis=1)
B['pnl'] = B.BidPrice1 * (B.TakerSell_CloseBP - FEE) / 1e5

print('FH15 書拆解：攣生單(與 negFill 視同一筆) vs 獨有單')
for per, mask in [('全樣本', B.Date.notna()), ('2025', B.Date < '2026-01-01'),
                  ('2026', B.Date >= '2026-01-01'), ('holdout', B.Date >= '2026-04-01')]:
    s = B[mask]
    g = s.groupby('twin').apply(lambda d: pd.Series({
        '資本占比%': d.BidPrice1.sum()/s.BidPrice1.sum()*100,
        'capw_bp': d.pnl.sum()/(d.BidPrice1/10).sum()*1e4,
        'pnl萬': d.pnl.sum()}), include_groups=False).rename(index={True:'攣生', False:'獨有'})
    print(f'--- {per} ---'); print(g.round(1).to_string())
# 獨有單的日級 Sharpe（若當獨立小書）
u = B[~B.twin]
du = u.pnl.groupby(u.Date).sum()
print(f"\n獨有單當小書: 全樣本 pnl {du.sum():.0f}萬, Sharpe {du.mean()/du.std()*np.sqrt(240):.2f} | "
      f"holdout pnl {du[du.index>='2026-04-01'].sum():.0f}萬, "
      f"Sharpe {du[du.index>='2026-04-01'].mean()/du[du.index>='2026-04-01'].std()*np.sqrt(240):.2f}")
da = A.assign(pnl=A.BidPrice1*(A.TakerSell_CloseBP-FEE)/1e5).groupby('Date').pnl.sum()
j = pd.DataFrame({'u': du}).join(da.rename('a')).fillna(0)
print(f"獨有小書 vs negFill 日損益 corr: {j.u.corr(j.a):.3f}")
