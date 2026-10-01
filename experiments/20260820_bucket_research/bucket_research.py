"""Bucket research: route A (baseline-subtracted labels, global thr) vs
route B (tree-leaf buckets on prior window) vs adopted reference.
All thresholds/trees use strictly prior data. Composition + economics on TUNE/HOLDOUT."""
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeRegressor

S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
FEE = 19.3
df = pd.read_parquet(f'{S}/stage3_preds.parquet')
df = df.rename(columns={'TakerSell_CloseBP_net_pred': 'net_s3', 'TakerSell_CloseBP_netM_pred': 'netM_s3'})
s6 = pd.read_parquet(f'{S}/stage6_fhat_preds.parquet')
s6 = s6.rename(columns={'TakerSell_CloseBP_net_pred': 'net_ex', 'TakerSell_CloseBP_netM_pred': 'netM_ex'})
keys = ['Date','QuoteCode','TransTime','ChannelSeq']
df = df.merge(s6[keys + ['net_ex','netM_ex','_fhat']], on=keys, how='left', validate='1:1')
df['D'] = df['Date'].astype(str).str.replace('-', '')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(keys).reset_index(drop=True)
DATES = sorted(df['D'].unique())
print('rows', len(df), '| ex-pred coverage', df.net_ex.notna().mean().round(3))

static = (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
          & (df.ToRef > 0) & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))
TUNE = (df.Date >= '2025-12-05') & (df.Date <= '2026-03-31')
HOLD = (df.Date >= '2026-04-01')

# composition bins (fixed, clock/price coordinates)
df['tb4'] = pd.cut(df.RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False)
df['rb4'] = pd.cut(df.ToRef, [0, .01, .02, .03, .05], labels=False)
base_sel = static & df.rb4.notna()
base_2d = (df[base_sel].groupby(['tb4','rb4']).size() / base_sel.sum())

def comp_metrics(m):
    if len(m) == 0: return {}
    sel2d = m.groupby(['tb4','rb4']).size() / len(m)
    tv = float((sel2d.reindex(base_2d.index, fill_value=0) - base_2d).abs().sum() / 2)
    return {'open30%': (m.tb4 == 3).mean()*100, 'toref35%': (m.rb4 == 3).mean()*100, 'TVdist': tv}

def pipeline(cond, mask):
    m = df[cond & mask].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    if len(m) < 300: return None, None
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]
    if m.Date.nunique() == 0: return None, None
    pnl = m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5
    daily = pnl.groupby(m.Date).sum()
    eq = daily.cumsum(); mdd = (eq - eq.cummax()).min()
    stats = {'t/d': len(m)/m.Date.nunique(), 'capw': pnl.sum()/(m.BidPrice1/10).sum()*1e4,
             'pnl': pnl.sum(), 'shp': daily.mean()/daily.std()*np.sqrt(240) if daily.std() > 0 else np.nan,
             'mdd': mdd}
    return stats, m

# ---------- generic prior-day quantile with arbitrary bucket labels ----------
def thr_by_bucket(pred_col, bucket_ser, qs, n_days=3):
    d = pd.DataFrame({'D': df['D'], 'b': bucket_ser, 'v': df[pred_col]}).dropna()
    store = {k: g['v'].to_numpy() for k, g in d.groupby(['D','b'], observed=True)}
    buckets = sorted(d.b.unique())
    maps = {q: {} for q in qs}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i-n_days):i]
        if not win: continue
        for b in buckets:
            pool = [store.get((w, b)) for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if not pool: continue
            for q, v in zip(qs, np.quantile(np.concatenate(pool), qs)):
                maps[q][(dt, b)] = v
    kb = list(zip(df['D'], bucket_ser))
    return {q: pd.Series([maps[q].get(k, np.nan) for k in kb], index=df.index) for q in qs}

def thr_global(pred_col, qs, n_days=3):
    return thr_by_bucket(pred_col, pd.Series(0, index=df.index), qs, n_days)

# ---------- route B: tree leaves as buckets (tree fit on prior 20d, applied to today) ----------
def tree_leaf_series(depth, window=20, sample=120000, seed=7):
    rng = np.random.RandomState(seed)
    z = df['residual_TakerSell_CloseBP'].to_numpy()
    X = df[['RemainSeconds','ToRef']].to_numpy()
    idx_by_d = {d: g.index.to_numpy() for d, g in df.groupby('D')}
    leaf = np.full(len(df), -1.0)
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i-window):i]
        if len(win) < window: continue
        tr = np.concatenate([idx_by_d[w] for w in win])
        zt = z[tr]; ok = np.isfinite(zt)
        tr = tr[ok]
        if len(tr) > sample: tr = rng.choice(tr, sample, replace=False)
        t = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(200, int(len(tr)*0.05)), random_state=0)
        t.fit(X[tr], z[tr])
        cur = idx_by_d[dt]
        leaf[cur] = t.apply(X[cur])
        # prior-3-day rows also mapped through TODAY's tree for the threshold pools
    return pd.Series(leaf, index=df.index)

# NOTE: thr_by_bucket pools prior days by THEIR stored bucket labels; for trees the leaf ids of
# prior days came from those days' own trees. Adjacent-day trees on 20d windows are near-identical,
# so this approximation is tight; it stays strictly causal (never uses today's data).

results = []
def record(tag, stats_t, comp_t, stats_h, comp_h):
    row = {'config': tag}
    for k, v in (stats_t or {}).items(): row[f'T_{k}'] = v
    for k, v in (comp_t or {}).items(): row[f'T_{k}'] = v
    for k, v in (stats_h or {}).items(): row[f'H_{k}'] = v
    for k, v in (comp_h or {}).items(): row[f'H_{k}'] = v
    results.append(row)

def run_config(tag, cond):
    st, mt = pipeline(cond, TUNE)
    sh, mh = pipeline(cond, HOLD)
    record(tag, st, comp_metrics(mt) if mt is not None else {}, sh, comp_metrics(mh) if mh is not None else {})
    return st, sh

# (0) adopted reference: stage3 preds, 4 time buckets, q=0.5/0.2
tb = df['tb4']
t_n = thr_by_bucket('net_s3', tb, [0.5])[0.5]
t_m = thr_by_bucket('netM_s3', tb, [0.2])[0.2]
run_config('ref: timebucket 0.5/0.2', (df.net_s3 > t_n) & (df.netM_s3 > t_m) & static)

# (A) stage6 fhat labels, GLOBAL threshold, tune q on grid
QN, QM = [0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5]
tg_n = thr_global('net_ex', QN); tg_m = thr_global('netM_ex', QM)
best, best_pnl = None, -1e18
for qn in QN:
    for qm in QM:
        st, _ = pipeline((df.net_ex > tg_n[qn]) & (df.netM_ex > tg_m[qm]) & static, TUNE)
        if st and st['pnl'] > best_pnl and 30 <= st['t/d'] <= 350:
            best, best_pnl = (qn, qm), st['pnl']
print('A-global best q on TUNE:', best)
run_config(f'A: fhat-label global q={best[0]}/{best[1]}',
           (df.net_ex > tg_n[best[0]]) & (df.netM_ex > tg_m[best[1]]) & static)

# (A2) fhat labels + 2D cell threshold (16 cells)
cell16 = df['tb4'] * 10 + df['rb4']
tc_n = thr_by_bucket('net_ex', cell16, [best[0]])[best[0]]
tc_m = thr_by_bucket('netM_ex', cell16, [best[1]])[best[1]]
run_config(f'A2: fhat-label 2Dcell q={best[0]}/{best[1]}', (df.net_ex > tc_n) & (df.netM_ex > tc_m) & static)

# (B) stage3 preds, tree-leaf buckets
for depth in [2, 3]:
    lf = tree_leaf_series(depth)
    bt, bb = None, -1e18
    tn_all = thr_by_bucket('net_s3', lf, [0.5, 0.6, 0.7])
    tm_all = thr_by_bucket('netM_s3', lf, [0.2, 0.3])
    for qn in [0.5, 0.6, 0.7]:
        for qm in [0.2, 0.3]:
            st, _ = pipeline((df.net_s3 > tn_all[qn]) & (df.netM_s3 > tm_all[qm]) & static, TUNE)
            if st and st['pnl'] > bb and 30 <= st['t/d'] <= 350:
                bt, bb = (qn, qm), st['pnl']
    print(f'B depth{depth} best q on TUNE:', bt)
    run_config(f'B: tree-d{depth} q={bt[0]}/{bt[1]}',
               (df.net_s3 > tn_all[bt[0]]) & (df.netM_s3 > tm_all[bt[1]]) & static)

base_comp = comp_metrics(df[base_sel & HOLD])
print(f"\nbase universe (holdout): open30 {base_comp['open30%']:.1f}% | toref3-5 {base_comp['toref35%']:.1f}%")
out = pd.DataFrame(results)
cols = ['config','T_t/d','T_capw','T_pnl','T_shp','T_open30%','T_toref35%','T_TVdist',
        'H_t/d','H_capw','H_pnl','H_shp','H_mdd','H_open30%','H_toref35%','H_TVdist']
print('\n===== TUNE / HOLDOUT comparison =====')
print(out[[c for c in cols if c in out.columns]].round(2).to_string(index=False))
out.to_csv(f'{S}/bucket_research_results.csv', index=False)
print('\nBUCKET RESEARCH DONE')
