"""Monthly like-for-like: current WITH-split config vs new bucketed q=0.5/0.2 config."""
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

# current config (WITH split, pooled thresholds)
tn = build_thr_col('net_norm', [0.7], False)[0.7]
tm = build_thr_col('netM_norm', [0.3], False)[0.3]
ta = build_thr_col('TakerSell_CloseBP_net_ABpred', [0.9], False)[0.9]
tma = build_thr_col('TakerSell_CloseBP_netM_ABpred', [0.5], False)[0.5]
is_n, is_a = df.isAbnormalDate == 0, df.isAbnormalDate == 1
cond_cur = ((is_n & (df.net_norm > tn) & (df.netM_norm > tm)) |
            (is_a & (df.TakerSell_CloseBP_net_ABpred > ta) & (df.TakerSell_CloseBP_netM_ABpred > tma))) & static
# new config (no split, bucketed, 0.5/0.2)
tn2 = build_thr_col('net_norm', [0.5], True)[0.5]
tm2 = build_thr_col('netM_norm', [0.2], True)[0.2]
cond_new = (df.net_norm > tn2) & (df.netM_norm > tm2) & static

def monthly(cond, tag):
    m = df[cond].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]
    m['pnl'] = m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5
    g = m.groupby(m.Date.dt.strftime('%Y%m')).agg(pnl=('pnl','sum'), n=('pnl','size'), days=('Date','nunique'))
    g[f'{tag}_pnl'] = g.pnl.round(0)
    g[f'{tag}_tpd'] = (g.n / g.days).round(0)
    return g[[f'{tag}_pnl', f'{tag}_tpd']]

t = monthly(cond_cur, 'cur').join(monthly(cond_new, 'new'), how='outer')
t['diff'] = (t.new_pnl - t.cur_pnl).round(0)
print(t.to_string())
for lo, hi, lab in [('202601','202603','2026 Jan-Mar (新配置的調參窗內)'),
                    ('202604','202608','2026 Apr-Aug (純 holdout)'),
                    ('202601','202608','2026 全段'),
                    ('202509','202608','全樣本')]:
    s = t.loc[(t.index >= lo) & (t.index <= hi)]
    print(f'{lab:38s} cur {s.cur_pnl.sum():8.0f} 萬 | new {s.new_pnl.sum():8.0f} 萬')
