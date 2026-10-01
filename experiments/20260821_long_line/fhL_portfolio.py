"""FHL long book (maker labels) + negFill baseline: ballast evaluation."""
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
    pnl = (m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5).groupby(m.Date).sum()
    cap = (m.BidPrice1 / 10).groupby(m.Date).sum()
    return pnl, cap

fh = load(f'{S}/stage_fhL_preds.parquet')
print('FHL rows:', len(fh))
best, bp = None, -1e18
for qn in [0.2, 0.3, 0.5]:
    for qm in [0.05, 0.1, 0.2]:
        p, c = book(fh, qn, qm)
        pt = p[(p.index >= '2025-12-05') & (p.index <= '2026-03-31')]
        if len(pt) > 30 and pt.sum() > bp:
            best, bp = (qn, qm), pt.sum()
print('FHL TUNE 最佳 q:', best, f'({bp:.0f}萬)')
pL, cL = book(fh, *best)

b = pd.read_parquet('/home/kevin/Project/HFT/src/research/intradaySignal/baseline/timeseries_points.parquet')
b['Date'] = pd.to_datetime(b['Date'])
pB = (b.BidPrice1*(b.TakerSell_CloseBP-FEE)/1e5).groupby(b.Date).sum()
cB = (b.BidPrice1/10).groupby(b.Date).sum()

idx = pB.index
pLa = pL.reindex(idx, fill_value=0); cLa = cL.reindex(idx, fill_value=0)
def sh(s): return s.mean()/s.std()*np.sqrt(240)
def stats(p, c, lo=None):
    if lo: p, c = p[p.index >= lo], c[c.index >= lo]
    eq = p.cumsum(); mdd = (eq-eq.cummax()).min()
    return {'pnl萬': round(p.sum()), 'Sharpe': round(sh(p), 2), 'MDD萬': round(mdd),
            '部位萬': round(c.mean()), 'capw': round(p.sum()/c.sum()*1e4, 1)}

print('\nFHL 書自身:')
for per, lo in [('全樣本', None), ('2026', '2026-01-01'), ('holdout', '2026-04-01')]:
    print(f'  {per}:', stats(pLa, cLa, lo))
print('負相關檢查: ρ(FHL, negFill) =', round(pLa.corr(pB), 3))
print('negFill 最差10%日: nf', round(pB[pB <= pB.quantile(.1)].mean()), '萬 | FHL', round(pLa[pB <= pB.quantile(.1)].mean()), '萬')

print('\n組合(權重掃描, w = FHL 資本占比):')
for w in [0.1, 0.2, 0.3, 0.4]:
    scale = w/(1-w) * cB.mean()/max(cLa.mean(), 1)
    pC = pB + pLa*scale; cC = cB + cLa*scale
    print(f'  w={w:.0%}: 全樣本 Sharpe {sh(pC):.2f} (nf {sh(pB):.2f}) | MDD {round((pC.cumsum()-pC.cumsum().cummax()).min())}萬 (nf {round((pB.cumsum()-pB.cumsum().cummax()).min())}) | '
          f'worst5日均 {round(pC.nsmallest(5).mean())} (nf {round(pB.nsmallest(5).mean())})')
