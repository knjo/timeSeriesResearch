"""Three generations: original(AB-split 0.7/0.3) vs time-bucket(0.5/0.2) vs 16-cell(0.3/0.1)."""
import numpy as np, pandas as pd

S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
FEE = 19.3
df = pd.read_parquet(f'{S}/stage3_preds.parquet')
df = df.rename(columns={'TakerSell_CloseBP_net_pred': 'net', 'TakerSell_CloseBP_netM_pred': 'netM'})
df['D'] = df['Date'].astype(str).str.replace('-', '')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Date','QuoteCode','TransTime','ChannelSeq']).reset_index(drop=True)
DATES = sorted(df['D'].unique())
df['tb4'] = pd.cut(df.RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False)
df['rb4'] = pd.cut(df.ToRef, [0, .01, .02, .03, .05], labels=False)
df['cell'] = df['tb4']*10 + df['rb4']
static = (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
          & (df.ToRef > 0) & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))

def thr(pred_col, bucket, q, n_days=3):
    b = bucket if bucket is not None else pd.Series(0, index=df.index)
    d = pd.DataFrame({'D': df['D'], 'b': b, 'v': df[pred_col]}).dropna()
    store = {k: g['v'].to_numpy() for k, g in d.groupby(['D','b'], observed=True)}
    buckets = sorted(d.b.unique()); m = {}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i-n_days):i]
        if not win: continue
        for bb in buckets:
            pool = [store.get((w, bb)) for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if pool: m[(dt, bb)] = np.quantile(np.concatenate(pool), q)
    kb = list(zip(df['D'], b))
    return pd.Series([m.get(k, np.nan) for k in kb], index=df.index)

is_n, is_a = df.isAbnormalDate == 0, df.isAbnormalDate == 1
cfg = {
    '原本 (AB分支 0.7/0.3)': ((is_n & (df.net > thr('net', None, 0.7)) & (df.netM > thr('netM', None, 0.3))) |
        (is_a & (df.TakerSell_CloseBP_net_ABpred > thr('TakerSell_CloseBP_net_ABpred', None, 0.9))
              & (df.TakerSell_CloseBP_netM_ABpred > thr('TakerSell_CloseBP_netM_ABpred', None, 0.5)))) & static,
    '時間桶 (0.5/0.2)': (df.net > thr('net', df['tb4'], 0.5)) & (df.netM > thr('netM', df['tb4'], 0.2)) & static,
    '16-cell (0.3/0.1)': (df.net > thr('net', df['cell'], 0.3)) & (df.netM > thr('netM', df['cell'], 0.1)) & static,
}

def stats(cond, mask):
    m = df[cond & mask].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]
    pnl = m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5
    daily = pnl.groupby(m.Date).sum()
    cap_d = (m.BidPrice1/10).groupby(m.Date).sum()
    eq = daily.cumsum(); mdd = (eq - eq.cummax()).min()
    return {'pnl萬': pnl.sum(), 'Sharpe': daily.mean()/daily.std()*np.sqrt(240),
            'MDD萬': mdd, '平均部位萬': cap_d.mean(), 'capw_bp': pnl.sum()/(m.BidPrice1/10).sum()*1e4,
            '筆/日': len(m)/m.Date.nunique(), '日勝率%': (daily > 0).mean()*100}

for lo, lab in [(None, '全樣本 2025-09 ~ 2026-08'), ('2026-01-01', '2026 Jan-Aug'), ('2026-04-01', '驗證期 2026 Apr-Aug')]:
    mask = df.Date >= (lo or '2000-01-01')
    print(f'\n--- {lab} ---')
    out = pd.DataFrame({k: stats(c, mask) for k, c in cfg.items()}).T
    print(out.round(2).to_string())
