"""Evaluate stage2 predictions: monthly IC, qcut inversion decomposition, threshold pipeline."""
import sys, json
import numpy as np
import pandas as pd

S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
TAG = sys.argv[1] if len(sys.argv) > 1 else 'stage2'
FEE = 19.3
df = pd.read_parquet(f'{S}/{TAG}_preds.parquet')
df['Date'] = pd.to_datetime(df['Date'])
df['ym'] = df.Date.dt.strftime('%Y%m')
print(f'{TAG}: rows={len(df)}, {df.Date.min().date()} .. {df.Date.max().date()}')

def spearman(x, y):
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    n = len(x)
    if n < 200: return np.nan
    rx = np.empty(n); ry = np.empty(n)
    rx[np.argsort(x, kind='stable')] = np.arange(n)
    ry[np.argsort(y, kind='stable')] = np.arange(n)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d > 0 else np.nan

models = {
    'net60':     'TakerSell_CloseBP_net_pred',
    'netM20':    'TakerSell_CloseBP_netM_pred',
    'rawnet60':  'TakerSell_CloseBP_rawnet_pred',
    'rawnetM20': 'TakerSell_CloseBP_rawnetM_pred',
    'resid60':   'residual_TakerSell_CloseBP_pred',
}
models = {k: v for k, v in models.items() if v in df.columns}

# ---------- T1: monthly per-day IC vs raw outcome and vs z outcome ----------
recs = []
for d, part in df.groupby('Date', sort=True):
    y_raw = part['TakerSell_CloseBP'].to_numpy(float)
    y_z = part['residual_TakerSell_CloseBP'].to_numpy(float)
    r = {'Date': d}
    for name, col in models.items():
        x = part[col].to_numpy(float)
        r[f'{name}|raw'] = spearman(x, y_raw)
        r[f'{name}|z'] = spearman(x, y_z)
    recs.append(r)
ic = pd.DataFrame(recs)
ic['ym'] = ic.Date.dt.strftime('%Y%m')
t1 = ic.groupby('ym')[[c for c in ic.columns if '|' in c]].mean()
print('\n========== T1: monthly mean per-day Spearman IC ==========')
print(t1.round(4).to_string())
t1.to_csv(f'{S}/{TAG}_t1_monthly_ic.csv')

# ---------- T2: pooled qcut8 decomposition for net60 ----------
print('\n========== T2: pooled qcut8 of net60 pred — per-bin decomposition ==========')
for lo, hi, lab in [('2025-09-01','2025-12-31','2025H2'), ('2026-01-01','2026-05-31','2026 Jan-May'), ('2026-06-01','2026-08-31','2026 Jun-Aug')]:
    sub = df[(df.Date >= lo) & (df.Date <= hi)].copy()
    col = models.get('net60')
    if col is None or sub[col].notna().sum() < 1000: continue
    sub = sub[sub[col].notna()]
    sub['bin'] = pd.qcut(sub[col], 8, labels=False, duplicates='drop')
    g = sub.groupby('bin').agg(
        n=('TakerSell_CloseBP','size'),
        raw_bp=('TakerSell_CloseBP','mean'),
        hedge_bp=('_hedgebp','mean'),
        idio_bp=('_idiobp','mean'),
        z=('residual_TakerSell_CloseBP','mean'),
        beta=('txf_beta_60d','mean'),
        vol_bp=('_volbp','mean'),
        toref=('ToRef','mean'),
    )
    print(f'\n--- {lab} ---   (raw = idio - hedge; hedge>0 means market rose after event)')
    print(g.round(2).to_string())

# ---------- T3: threshold pipeline (their cells 19-27) ----------
def prior_day_quantile(dfx, col, q=0.9, n_days=3, date_col='Date', min_days=1):
    vals = (dfx.dropna(subset=[col]).groupby(date_col, observed=True)[col]
              .apply(lambda s: s.to_numpy()))
    dates = np.sort(vals.index.to_numpy())
    thr = {}
    for i, d in enumerate(dates):
        window = dates[max(0, i - n_days):i]
        if len(window) < min_days: continue
        pool = np.concatenate([vals[w] for w in window])
        if pool.size: thr[d] = np.quantile(pool, q)
    return dfx[date_col].map(thr)

def run_threshold_pipeline(base, label_prefix, title):
    p, pM = f'{label_prefix}_pred', f'{label_prefix}M_pred'
    pA, pMA = f'{label_prefix}_ABpred', f'{label_prefix}M_ABpred'
    if not all(c in base.columns for c in [p, pM, pA, pMA]):
        print(f'[{title}] missing pred cols, skip'); return
    d = base.copy()
    d['thr_net'] = prior_day_quantile(d, p, q=0.7, n_days=3)
    d['thr_netM'] = prior_day_quantile(d, pM, q=0.3, n_days=3)
    d['thr_net_AB'] = prior_day_quantile(d, pA, q=0.9, n_days=3)
    d['thr_netM_AB'] = prior_day_quantile(d, pMA, q=0.5, n_days=3)
    is_norm = d.isAbnormalDate == 0
    is_ab = d.isAbnormalDate == 1
    condition = (
        (is_norm & (d[p] > d.thr_net) & (d[pM] > d.thr_netM)) |
        (is_ab & (d[pA] > d.thr_net_AB) & (d[pMA] > d.thr_netM_AB))
    )
    condition = condition & ((d.OTC == 1) | (d.MD_L1Rate_30_re > 0.25)) & (d.SpreadPairElapsed > 0.1) & (d.ToRef > 0) & ((d.AmountRank_canDayTrade <= 100) | (d.day_amount_rank <= 100))
    m = d[condition].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))].reset_index(drop=True)
    m['ym'] = m.Date.dt.strftime('%Y%m')
    m['pnl_wan'] = m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 10000 / 10
    m['cap_wan'] = m.BidPrice1 / 10
    g = m.groupby('ym').agg(trades=('BidPrice1','size'),
                            days=('Date','nunique'),
                            stocks_day=('QuoteCode', lambda s: np.nan),
                            raw_bp=('TakerSell_CloseBP','mean'),
                            hedge_bp=('_hedgebp','mean'),
                            idio_bp=('_idiobp','mean'),
                            pnl_wan=('pnl_wan','sum'),
                            cap_wan=('cap_wan','sum'))
    g['stocks_day'] = m.groupby(['ym','Date']).QuoteCode.nunique().groupby('ym').mean()
    g['trades_day'] = g.trades / g.days
    g['capw_bp_net'] = g.pnl_wan / g.cap_wan * 10000
    daily = m.groupby('Date').pnl_wan.sum()
    print(f'\n--- T3 [{title}] monthly (fee {FEE}bp) ---')
    print(g[['trades_day','stocks_day','raw_bp','hedge_bp','idio_bp','capw_bp_net','pnl_wan']].round(2).to_string())
    print(f'total pnl(萬): {daily.sum():.1f} | 2026+ pnl: {daily[daily.index >= "2026-01-01"].sum():.1f} | win-day rate 2026+: {(daily[daily.index >= "2026-01-01"] > 0).mean():.2%}')

run_threshold_pipeline(df, 'TakerSell_CloseBP_net', 'residual labels (their current)')
run_threshold_pipeline(df, 'TakerSell_CloseBP_rawnet', 'raw labels (old style)')

# ---------- T4: resid60 prediction level drift ----------
if 'residual_TakerSell_CloseBP_pred' in df.columns:
    q = df.groupby('ym')['residual_TakerSell_CloseBP_pred'].quantile([.5, .9]).unstack()
    q.columns = ['p50','p90']
    q['label_mean'] = df.groupby('ym')['residual_TakerSell_CloseBP'].mean()
    print('\n========== T4: resid60 pred level drift (why fixed thresholds die) ==========')
    print(q.round(4).to_string())
print('\nEVAL DONE')
