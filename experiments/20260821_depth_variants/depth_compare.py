"""Current negFill vs depth-augmented negFill — full pipeline comparison."""
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
    df['cell'] = df['tb4'] * 10 + df['rb4']
    df['kb'] = df['D'] + '_' + df['cell'].astype(str)
    return df.sort_values(['Date','QuoteCode','TransTime','ChannelSeq']).reset_index(drop=True)

def thr(df, col, q, n_days=3):
    d = df[['D','kb',col]].dropna()
    store = {k: g[col].to_numpy() for k, g in d.groupby('kb', observed=True)}
    DATES = sorted(df['D'].unique())
    buckets = sorted(df.cell.dropna().unique())
    m = {}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i - n_days):i]
        if not win: continue
        for b in buckets:
            pool = [store.get(f'{w}_{b}') for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if pool: m[(f'{dt}_{b}')] = np.quantile(np.concatenate(pool), q)
    return df['kb'].map(m)

def run(df, cond, mask, want_m=False):
    m = df[cond & mask].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    if len(m) < 200: return None, None
    pre = m.copy()
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]
    pnl = m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5
    daily = pnl.groupby(m.Date).sum()
    cap_d = (m.BidPrice1/10).groupby(m.Date).sum()
    eq = daily.cumsum(); mdd = (eq - eq.cummax()).min()
    st = {'pnl萬': pnl.sum(), 'Sharpe': daily.mean()/daily.std()*np.sqrt(240), 'MDD萬': mdd,
          '部位萬': cap_d.mean(), 'capw_bp': pnl.sum()/(m.BidPrice1/10).sum()*1e4,
          '筆/日': len(m)/m.Date.nunique(), '勝率%': (daily > 0).mean()*100}
    return st, (pre, m) if want_m else None

cur = load(f'{S}/stage3_preds.parquet')
dep = load(f'{S}/stage7_depth_preds.parquet')
print(f'rows: current {len(cur):,} | depth {len(dep):,} ({len(dep)/len(cur):.0%})')

def static(df):
    return (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
            & (df.ToRef > 0) & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))

configs = {}
for name, df in [('現行', cur), ('深度版', dep)]:
    c = (df.net > thr(df, 'net', 0.3)) & (df.netM > thr(df, 'netM', 0.1)) & static(df)
    configs[f'{name} q=0.3/0.1'] = (df, c)

# depth 版重調 q（TUNE 期選）
TUNEm = (dep.Date >= '2025-12-05') & (dep.Date <= '2026-03-31')
best, bp = None, -1e18
grids = {}
for qn in [0.2, 0.3, 0.4]:
    grids[('net', qn)] = thr(dep, 'net', qn)
for qm in [0.05, 0.1, 0.2]:
    grids[('netM', qm)] = thr(dep, 'netM', qm)
for qn in [0.2, 0.3, 0.4]:
    for qm in [0.05, 0.1, 0.2]:
        c = (dep.net > grids[('net', qn)]) & (dep.netM > grids[('netM', qm)]) & static(dep)
        st, _ = run(dep, c, TUNEm)
        if st and st['pnl萬'] > bp:
            best, bp = (qn, qm), st['pnl萬']
print('depth 版 TUNE 最佳 q:', best)
c = (dep.net > grids[('net', best[0])]) & (dep.netM > grids[('netM', best[1])]) & static(dep)
configs[f'深度版 q={best[0]}/{best[1]}(重調)'] = (dep, c)

print('\n===== 績效對比 =====')
for per, lo in [('全樣本', None), ('2026 Jan-Aug', '2026-01-01'), ('holdout Apr-Aug', '2026-04-01')]:
    print(f'--- {per} ---')
    out = {}
    for tag, (df, c) in configs.items():
        mask = df.Date >= (lo or '2000-01-01')
        st, _ = run(df, c, mask)
        if st: out[tag] = st
    print(pd.DataFrame(out).T.round(2).to_string())

print('\n===== 樣本分佈（holdout,倉位上限前/後）=====')
TB = {3: '09:00-09:30', 2: '09:30-10:30', 1: '10:30-11:30', 0: '11:30-12:30'}
RB = {0: '0-1%', 1: '1-2%', 2: '2-3%', 3: '3-5%'}
for tag, (df, c) in configs.items():
    _, mm = run(df, c, df.Date >= '2026-04-01', want_m=True)
    if mm is None: continue
    pre, post = mm
    tt = pd.DataFrame({'pre%': pre.tb4.value_counts(normalize=True)*100,
                       'post%': post.tb4.value_counts(normalize=True)*100}).rename(index=TB)
    rr = pd.DataFrame({'pre%': pre.rb4.value_counts(normalize=True)*100,
                       'post%': post.rb4.value_counts(normalize=True)*100}).rename(index=RB)
    print(f'\n[{tag}] 時段:'); print(tt.sort_index(ascending=False).round(1).to_string())
    print(f'[{tag}] ToRef:'); print(rr.sort_index().round(1).to_string())
