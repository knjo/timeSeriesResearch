"""FH15 book (own models+thresholds) + negFill baseline book → stacked portfolio."""
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
    DATES = sorted(df['D'].unique())
    buckets = sorted(set(k.split('_')[1] for k in store))
    m = {}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i-n_days):i]
        if not win: continue
        for b in buckets:
            pool = [store.get(f'{w}_{b}') for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if pool: m[f'{dt}_{b}'] = np.quantile(np.concatenate(pool), q)
    return df['kb'].map(m)

def book(df, qn, qm, want_m=False):
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
    return (pnl, cap, m) if want_m else (pnl, cap)

def stats(p, c, lo=None):
    if lo: p, c = p[p.index >= lo], c[c.index >= lo]
    eq = p.cumsum(); mdd = (eq - eq.cummax()).min()
    return {'pnl萬': round(p.sum(), 0), 'Sharpe': round(p.mean()/p.std()*np.sqrt(240), 2),
            'MDD萬': round(mdd, 0), '部位萬': round(c.mean(), 0),
            'capw_bp': round(p.sum()/c.sum()*1e4, 1), 'MDD/部位%': round(-mdd/c.mean()*100, 1),
            '勝率%': round((p > 0).mean()*100, 1)}

fh = load(f'{S}/stage_fh15_preds.parquet')
print('FH15 rows:', len(fh))
TUNE_LO, TUNE_HI = '2025-12-05', '2026-03-31'
best, bp = None, -1e18
for qn in [0.2, 0.3, 0.5]:
    for qm in [0.05, 0.1, 0.2]:
        p, c = book(fh, qn, qm)
        pt = p[(p.index >= TUNE_LO) & (p.index <= TUNE_HI)]
        if len(pt) > 30 and pt.sum() > bp:
            best, bp = (qn, qm), pt.sum()
print('FH15 TUNE 最佳 q:', best, f'({bp:.0f}萬)')
pF, cF, mF = book(fh, *best, want_m=True)

bl = load(f'{S}/stage3_preds.parquet')
pB, cB = book(bl, 0.3, 0.1)

idx = pB.index.union(pF.index)
pC = pB.reindex(idx, fill_value=0) + pF.reindex(idx, fill_value=0)
cC = cB.reindex(idx, fill_value=0) + cF.reindex(idx, fill_value=0)

for per, lo in [('全樣本', None), ('2026 Jan-Aug', '2026-01-01'), ('holdout Apr-Aug', '2026-04-01')]:
    out = {'negFill(baseline)': stats(pB, cB, lo), 'FH15(新書)': stats(pF, cF, lo), '合併 portfolio': stats(pC, cC, lo)}
    a = pB.reindex(idx, fill_value=0); b = pF.reindex(idx, fill_value=0)
    if lo: a, b = a[a.index >= lo], b[b.index >= lo]
    print(f'\n--- {per} ---  (兩書日損益 corr {a.corr(b):.3f})')
    print(pd.DataFrame(out).T.to_string())

# 公平對照：同資本放大 baseline
for lo, lab in [(None, '全樣本'), ('2026-04-01', 'holdout')]:
    pb = pB if lo is None else pB[pB.index >= lo]; cb = cB if lo is None else cB[cB.index >= lo]
    pc = pC if lo is None else pC[pC.index >= lo]; cc = cC if lo is None else cC[cC.index >= lo]
    k = cc.mean()/cb.mean()
    eq = pb.cumsum(); mdd_s = (eq - eq.cummax()).min()*k
    print(f'[{lab}] 合併 vs 同資本放大baseline(x{k:.2f}): pnl {pc.sum():.0f} vs {pb.sum()*k:.0f} | '
          f'MDD {(pc.cumsum()-pc.cumsum().cummax()).min():.0f} vs {mdd_s:.0f}')

# FH 書自己的月度 + 組成
print('\nFH15 書月度 pnl:', {k: round(v) for k, v in pF.groupby(pF.index.strftime('%Y%m')).sum().items()})
TB = {3:'0900-0930',2:'0930-1030',1:'1030-1130',0:'1130-1230'}
print('FH15 成交時段分佈:', {TB[k]: f'{v*100:.0f}%' for k, v in mF.tb4.value_counts(normalize=True).sort_index(ascending=False).items()})
