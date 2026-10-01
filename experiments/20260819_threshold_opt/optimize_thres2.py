"""Threshold optimization v2 — full pipeline objective (position caps included),
string date keys, floors {off, fee}. Thresholds use ONLY prior-3-day data +
fixed clock rules: fully known before each trading day."""
import numpy as np, pandas as pd

S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
FEE = 19.3
df = pd.read_parquet(f'{S}/stage3_preds.parquet')
p5 = pd.read_parquet(f'{S}/stage5_pooled_preds.parquet').rename(
    columns={'TakerSell_CloseBP_net_pred': 'net_pooled', 'TakerSell_CloseBP_netM_pred': 'netM_pooled'})
keys = ['Date','QuoteCode','TransTime','ChannelSeq']
df = df.merge(p5[keys + ['net_pooled','netM_pooled']], on=keys, how='left', validate='1:1')
df = df.rename(columns={'TakerSell_CloseBP_net_pred': 'net_norm', 'TakerSell_CloseBP_netM_pred': 'netM_norm'})
df['D'] = df['Date'].astype(str).str.replace('-', '')
df['Date'] = pd.to_datetime(df['Date'])
df['bucket'] = pd.cut(df.RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False).fillna(-1).astype(int)
df['kb'] = df['D'] + '_' + df['bucket'].astype(str)
df = df.sort_values(keys).reset_index(drop=True)
print('rows', len(df))

DATES = sorted(df['D'].unique())
Q_NET, Q_NETM = [0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5]

def build_thr_col(col, qs, bucketed, n_days=3):
    """returns {q: pd.Series aligned to df} — quantile of prior-n_days preds (per bucket if bucketed)."""
    d = df[['D','bucket','kb',col]].dropna(subset=[col])
    if bucketed:
        store = {k: g[col].to_numpy() for k, g in d.groupby('kb', observed=True)}
        buckets = sorted(df.bucket.unique())
    else:
        store = {k: g[col].to_numpy() for k, g in d.groupby('D', observed=True)}
        buckets = [None]
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
            for q, v in zip(qs, qv):
                maps[q][key] = v
    src_key = df['kb'] if bucketed else df['D']
    return {q: src_key.map(maps[q]) for q in qs}

thr = {}
for src in ['norm', 'pooled']:
    for bk in [False, True]:
        thr[('net', src, bk)] = build_thr_col(f'net_{src}', Q_NET, bk)
        thr[('netM', src, bk)] = build_thr_col(f'netM_{src}', Q_NETM, bk)
# AB thresholds for reproducing user's current WITH-split config
thr_ab_net = build_thr_col('TakerSell_CloseBP_net_ABpred', [0.9], False)[0.9]
thr_ab_netM = build_thr_col('TakerSell_CloseBP_netM_ABpred', [0.5], False)[0.5]
print('thresholds built (sanity: baseline thr non-null share =',
      round(thr[('net','norm',False)][0.7].notna().mean(), 3), ')')

static = (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
          & (df.ToRef > 0) & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))
df['resid_bp'] = df['residual_TakerSell_CloseBP_pred'] * df['_volbp']
floors = {'off': pd.Series(True, index=df.index), 'fee': df.resid_bp > FEE}
TUNE = (df.Date >= '2025-12-05') & (df.Date <= '2026-03-31')
HOLD = (df.Date >= '2026-04-01')

def full_eval(cond, mask):
    m = df[cond & mask].sort_values(['Date','QuoteCode','TransTime'])
    if len(m) < 300: return None
    m = m.reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]
    days = m.Date.nunique()
    if days == 0: return None
    pnl = m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5
    daily = pnl.groupby(m.Date).sum()
    cap = (m.BidPrice1 / 10).sum()
    return {'trades_day': len(m)/days, 'capw_bp': pnl.sum()/cap*1e4, 'pnl_wan': pnl.sum(),
            'win_day': (daily > 0).mean(),
            'sharpe': daily.mean()/daily.std()*np.sqrt(240) if daily.std() > 0 else np.nan}

# sweep (no-split single model) on TUNE, then holdout for all
rows = []
for src in ['norm', 'pooled']:
    for bk in [False, True]:
        for qn in Q_NET:
            cn = df[f'net_{src}'] > thr[('net', src, bk)][qn]
            for qm in Q_NETM:
                cbase = cn & (df[f'netM_{src}'] > thr[('netM', src, bk)][qm]) & static
                for fl, fmask in floors.items():
                    cond = cbase & fmask
                    ev_t = full_eval(cond, TUNE)
                    if ev_t is None: continue
                    ev_h = full_eval(cond, HOLD) or {}
                    rows.append({'src': src, 'bk': bk, 'q_net': qn, 'q_netM': qm, 'floor': fl,
                                 **{f't_{k}': v for k, v in ev_t.items()},
                                 **{f'h_{k}': v for k, v in ev_h.items()}})
res = pd.DataFrame(rows)
res.to_csv(f'{S}/thres_sweep_v2.csv', index=False)

# user's current WITH-split config, for reference
is_norm, is_ab = df.isAbnormalDate == 0, df.isAbnormalDate == 1
cond_cur = ((is_norm & (df.net_norm > thr[('net','norm',False)][0.7]) & (df.netM_norm > thr[('netM','norm',False)][0.3])) |
            (is_ab & (df.TakerSell_CloseBP_net_ABpred > thr_ab_net) & (df.TakerSell_CloseBP_netM_ABpred > thr_ab_netM))) & static
cur_t, cur_h = full_eval(cond_cur, TUNE), full_eval(cond_cur, HOLD)

ok = res[(res.t_trades_day >= 30) & (res.t_trades_day <= 350)].copy()
print('\n===== sweep (TUNE-selected top 12, holdout shown but NOT used for selection) =====')
cols = ['src','bk','q_net','q_netM','floor','t_trades_day','t_capw_bp','t_pnl_wan','t_sharpe',
        'h_trades_day','h_capw_bp','h_pnl_wan','h_sharpe','h_win_day']
print(ok.sort_values('t_pnl_wan', ascending=False).head(12)[cols].round(2).to_string(index=False))

base = res[(res.src=='norm') & (~res.bk) & (res.q_net==0.7) & (res.q_netM==0.3) & (res.floor=='off')]
print('\n===== references =====')
print('baseline no-split q=0.7/0.3 no-floor:')
print(base[cols].round(2).to_string(index=False))
print(f"current WITH-split (q 0.7/0.3 + AB 0.9/0.5): TUNE {cur_t} | HOLDOUT {cur_h}")

# rank stability: does TUNE ranking survive into holdout?
ok2 = ok.dropna(subset=['h_pnl_wan'])
if len(ok2) > 10:
    from scipy.stats import spearmanr
    rho = spearmanr(ok2.t_pnl_wan, ok2.h_pnl_wan).statistic
    print(f'\nrank stability TUNE→HOLDOUT across {len(ok2)} configs: spearman {rho:.3f}')
print('\nSWEEP V2 DONE')
