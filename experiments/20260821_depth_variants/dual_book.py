"""Two independent books: baseline + depth-NF, each with its own position caps."""
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

def daily(df, qn, qm):
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

pA, cA = daily(load(f'{S}/stage3_preds.parquet'), 0.3, 0.1)      # baseline
pB, cB = daily(load(f'{S}/stage7_depth_preds.parquet'), 0.2, 0.05)  # depth NF
idx = pA.index.union(pB.index)
pC = pA.reindex(idx, fill_value=0) + pB.reindex(idx, fill_value=0)
cC = cA.reindex(idx, fill_value=0) + cB.reindex(idx, fill_value=0)

def stats(p, c, lo=None):
    if lo: p, c = p[p.index >= lo], c[c.index >= lo]
    eq = p.cumsum(); mdd = (eq - eq.cummax()).min()
    return {'pnl萬': p.sum(), 'Sharpe': p.mean()/p.std()*np.sqrt(240), 'MDD萬': mdd,
            '部位萬': c.mean(), 'capw_bp': p.sum()/c.sum()*1e4, '勝率%': (p > 0).mean()*100,
            'MDD/部位%': -mdd/c.mean()*100}

for per, lo in [('全樣本', None), ('2026 Jan-Aug', '2026-01-01'), ('holdout Apr-Aug', '2026-04-01')]:
    out = {'baseline': stats(pA, cA, lo), '深度nf': stats(pB, cB, lo), '合併(兩本書)': stats(pC, cC, lo)}
    print(f'--- {per} ---')
    print(pd.DataFrame(out).T.round(2).to_string())
    a = pA.reindex(idx, fill_value=0); b = pB.reindex(idx, fill_value=0)
    if lo: a, b = a[a.index >= lo], b[b.index >= lo]
    print(f'兩本書日損益相關: {a.corr(b):.3f}\n')
