"""Drawdown comparison: current WITH-split config vs new bucketed q=0.5/0.2 config."""
import numpy as np, pandas as pd

S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
FEE = 19.3
df = pd.read_parquet(f'{S}/stage3_preds.parquet')
df = df.rename(columns={'TakerSell_CloseBP_net_pred': 'net_norm', 'TakerSell_CloseBP_netM_pred': 'netM_norm'})
df['D'] = df['Date'].astype(str).str.replace('-', '')
df['Date'] = pd.to_datetime(df['Date'])
df['bucket'] = pd.cut(df.RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False).fillna(-1).astype(int)
df['kb'] = df['D'] + '_' + df['bucket'].astype(str)
DATES = sorted(df['D'].unique())

def build_thr_col(col, qs, bucketed, n_days=3):
    d = df[['D','bucket','kb',col]].dropna(subset=[col])
    store = {k: g[col].to_numpy() for k, g in d.groupby('kb' if bucketed else 'D', observed=True)}
    buckets = sorted(df.bucket.unique()) if bucketed else [None]
    maps = {q: {} for q in qs}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i - n_days):i]
        if not win: continue
        for b in buckets:
            pool = [store.get(f'{w}_{b}' if bucketed else w) for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if not pool: continue
            qv = np.quantile(np.concatenate(pool), qs)
            key = f'{dt}_{b}' if bucketed else dt
            for q, v in zip(qs, qv): maps[q][key] = v
    kcol = df['kb'] if bucketed else df['D']
    return {q: kcol.map(maps[q]) for q in qs}

static = (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
          & (df.ToRef > 0) & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))

tn = build_thr_col('net_norm', [0.7], False)[0.7]
tm = build_thr_col('netM_norm', [0.3], False)[0.3]
ta = build_thr_col('TakerSell_CloseBP_net_ABpred', [0.9], False)[0.9]
tma = build_thr_col('TakerSell_CloseBP_netM_ABpred', [0.5], False)[0.5]
is_n, is_a = df.isAbnormalDate == 0, df.isAbnormalDate == 1
cond_cur = ((is_n & (df.net_norm > tn) & (df.netM_norm > tm)) |
            (is_a & (df.TakerSell_CloseBP_net_ABpred > ta) & (df.TakerSell_CloseBP_netM_ABpred > tma))) & static
tn2 = build_thr_col('net_norm', [0.5], True)[0.5]
tm2 = build_thr_col('netM_norm', [0.2], True)[0.2]
cond_new = (df.net_norm > tn2) & (df.netM_norm > tm2) & static

def daily_series(cond):
    m = df[cond].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]
    pnl = (m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5).groupby(m.Date).sum()
    cap = (m.BidPrice1 / 10).groupby(m.Date).sum()
    return pnl, cap

def stats(pnl, cap, lab):
    eq = pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    mdd = dd.min()
    # longest time to new high
    is_high = eq >= peak - 1e-9
    longest, cnt = 0, 0
    for h in is_high:
        cnt = 0 if h else cnt + 1
        longest = max(longest, cnt)
    avg_pos = cap.mean()
    total = pnl.sum()
    print(f"{lab:26s} pnl {total:8.0f}萬 | MDD {mdd:8.0f}萬 ({mdd/avg_pos*100:6.1f}% of avg部位 {avg_pos:6.0f}萬) | "
          f"風報比 {total/abs(mdd):5.2f} | 最長回撤 {longest:3d}天 | 最差日 {pnl.min():6.0f}萬 | "
          f"日勝率 {(pnl>0).mean():5.1%} | Sharpe {pnl.mean()/pnl.std()*np.sqrt(240):5.2f}")
    return dd

pn_cur, cap_cur = daily_series(cond_cur)
pn_new, cap_new = daily_series(cond_new)
for lo, lab in [(None, '全樣本 2509-2608'), ('2026-01-01', '2026 Jan-Aug'), ('2026-04-01', 'holdout Apr-Aug')]:
    print(f'\n--- {lab} ---')
    for name, (p, c) in [('原本作法', (pn_cur, cap_cur)), ('新配置', (pn_new, cap_new))]:
        pp = p if lo is None else p[p.index >= lo]
        cc = c if lo is None else c[c.index >= lo]
        stats(pp, cc, name)

# worst drawdown episodes for the new config (2026)
print('\n--- 新配置 2026 的回撤剖面 ---')
p26 = pn_new[pn_new.index >= '2026-01-01']
eq = p26.cumsum(); dd = eq - eq.cummax()
print('最深 5 個回撤日:', dd.nsmallest(5).round(0).to_dict())
