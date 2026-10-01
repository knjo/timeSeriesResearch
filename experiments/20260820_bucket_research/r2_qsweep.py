"""Mini q-sweep for R2 (s3 preds + 16-cell 2D thresholds). Tune on TUNE, report holdout."""
import numpy as np, pandas as pd

S = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
FEE = 19.3
df = pd.read_parquet(f'{S}/stage3_preds.parquet')
df = df.rename(columns={'TakerSell_CloseBP_net_pred': 'net_s3', 'TakerSell_CloseBP_netM_pred': 'netM_s3'})
df['D'] = df['Date'].astype(str).str.replace('-', '')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Date','QuoteCode','TransTime','ChannelSeq']).reset_index(drop=True)
DATES = sorted(df['D'].unique())
df['tb4'] = pd.cut(df.RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False)
df['rb4'] = pd.cut(df.ToRef, [0, .01, .02, .03, .05], labels=False)
cell16 = df['tb4']*10 + df['rb4']
static = (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
          & (df.ToRef > 0) & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))
TUNE = (df.Date >= '2025-12-05') & (df.Date <= '2026-03-31')
HOLD = (df.Date >= '2026-04-01')
base_sel = static & df.rb4.notna() & HOLD
base_2d = df[base_sel].groupby(['tb4','rb4']).size() / base_sel.sum()

def thr_by_bucket(pred_col, bucket_ser, qs, n_days=3):
    d = pd.DataFrame({'D': df['D'], 'b': bucket_ser, 'v': df[pred_col]}).dropna()
    store = {k: g['v'].to_numpy() for k, g in d.groupby(['D','b'], observed=True)}
    buckets = sorted(d.b.unique()); maps = {q: {} for q in qs}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i-n_days):i]
        if not win: continue
        for b in buckets:
            pool = [store.get((w, b)) for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if not pool: continue
            for q, v in zip(qs, np.quantile(np.concatenate(pool), qs)): maps[q][(dt, b)] = v
    kb = list(zip(df['D'], bucket_ser))
    return {q: pd.Series([maps[q].get(k, np.nan) for k in kb], index=df.index) for q in qs}

QN, QM = [0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.3]
tn = thr_by_bucket('net_s3', cell16, QN)
tm = thr_by_bucket('netM_s3', cell16, QM)

def run(cond, mask):
    m = df[cond & mask].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    if len(m) < 300: return None, None
    pre = m.copy()
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]
    pnl = m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5
    daily = pnl.groupby(m.Date).sum()
    st = {'t/d': len(m)/m.Date.nunique(), 'capw': pnl.sum()/(m.BidPrice1/10).sum()*1e4, 'pnl': pnl.sum(),
          'shp': daily.mean()/daily.std()*np.sqrt(240) if daily.std() > 0 else np.nan}
    sel = pre.groupby(['tb4','rb4']).size() / len(pre)
    st['preTV'] = float((sel.reindex(base_2d.index, fill_value=0) - base_2d).abs().sum()/2)
    st['pre_open30'] = pre.tb4.eq(3).mean()*100
    st['pre_toref35'] = pre.rb4.eq(3).mean()*100
    return st, m

rows = []
for qn in QN:
    for qm in QM:
        cond = (df.net_s3 > tn[qn]) & (df.netM_s3 > tm[qm]) & static
        st, _ = run(cond, TUNE)
        sh, _ = run(cond, HOLD)
        if st and sh:
            rows.append({'q_net': qn, 'q_netM': qm, 'T_pnl': st['pnl'], 'T_shp': st['shp'], 'T_t/d': st['t/d'],
                         'H_pnl': sh['pnl'], 'H_shp': sh['shp'], 'H_capw': sh['capw'], 'H_t/d': sh['t/d'],
                         'H_preTV': sh['preTV'], 'H_open30': sh['pre_open30'], 'H_toref35': sh['pre_toref35']})
t = pd.DataFrame(rows).sort_values('T_pnl', ascending=False)
print(t.round(2).to_string(index=False))
