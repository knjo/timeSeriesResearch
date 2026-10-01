"""Q1: statistical comparison time-bucket vs 16-cell. Q2: extend cells to ToRef<0, hedged edge."""
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
df['rb'] = pd.cut(df.ToRef, [0, .01, .02, .03, .05], labels=False)
df['rb6'] = pd.cut(df.ToRef, [-.015, -.005, 0, .01, .02, .03, .05], labels=False)
static_common = (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
                 & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))
static = static_common & (df.ToRef > 0)
HOLD = df.Date >= '2026-04-01'
TUNE = (df.Date >= '2025-12-05') & (df.Date <= '2026-03-31')

def thr(pred_col, bucket, q, n_days=3):
    d = pd.DataFrame({'D': df['D'], 'b': bucket, 'v': df[pred_col]}).dropna()
    store = {k: g['v'].to_numpy() for k, g in d.groupby(['D','b'], observed=True)}
    buckets = sorted(d.b.unique()); m = {}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i-n_days):i]
        if not win: continue
        for bb in buckets:
            pool = [store.get((w, bb)) for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if pool: m[(dt, bb)] = np.quantile(np.concatenate(pool), q)
    kb = list(zip(df['D'], bucket))
    return pd.Series([m.get(k, np.nan) for k in kb], index=df.index)

def capped(cond):
    m = df[cond].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    return m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
             & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
             & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))].copy()

# ============ Q1: paired statistical comparison ============
cell16 = df['tb4']*10 + df['rb']
cA = (df.net > thr('net', df['tb4'], 0.5)) & (df.netM > thr('netM', df['tb4'], 0.2)) & static
cB = (df.net > thr('net', cell16, 0.3)) & (df.netM > thr('netM', cell16, 0.1)) & static
mA, mB = capped(cA), capped(cB)
for m in (mA, mB): m['pnl'] = m.BidPrice1 * (m.TakerSell_CloseBP - FEE) / 1e5
dA = mA.groupby('Date').pnl.sum(); dB = mB.groupby('Date').pnl.sum()
print('===== Q1: 時間桶(0.5/0.2) vs 16-cell(0.3/0.1) — 配對日損益檢定 =====')
for mask, lab in [(None, '全樣本'), ('2026-04-01', '驗證期')]:
    a = dA if mask is None else dA[dA.index >= mask]
    b = dB.reindex(a.index).fillna(0)
    diff = a - b
    t = diff.mean() / diff.std() * np.sqrt(len(diff))
    print(f'[{lab}] 日均差 {diff.mean():+.2f}萬 | t={t:+.2f} | 兩序列相關 {a.corr(b):.3f} | '
          f'月勝負(桶勝): {sum((a-b).resample("ME").sum() > 0)}/{len((a-b).resample("ME").sum())}')
# ToRef 集中度風險量化: holdout pnl 佔比 by ToRef bin
for m, lab in [(mA, '時間桶'), (mB, '16-cell')]:
    h = m[m.Date >= '2026-04-01']
    share = h.groupby('rb').pnl.sum() / h.pnl.sum() * 100
    npct = h.groupby('rb').size() / len(h) * 100
    print(f'[{lab}] holdout ToRef bin (0-1/1-2/2-3/3-5%) 損益佔比: {share.round(0).tolist()} | 筆數佔比: {npct.round(0).tolist()}')

# ============ Q2: 往 ToRef<0 延伸 ============
print('\n===== Q2a: 母體 edge by ToRef bin(基礎條件, 全樣本, gross bp)=====')
base = df[static_common & df.rb6.notna()]
t = base.groupby('rb6').agg(n=('TakerSell_CloseBP','size'), raw=('TakerSell_CloseBP','mean'),
                            hedged=('_idiobp','mean'))
t.index = ['-1.5~-0.5%','-0.5~0%','0~1%','1~2%','2~3%','3~5%']
print(t.round(1).to_string())

print('\n===== Q2b: 模型在 ToRef<=0 子集內還會排序嗎(per-day IC vs 對應label) =====')
def sp(x, y):
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    if len(x) < 200: return np.nan
    rx = np.empty(len(x)); ry = np.empty(len(x))
    rx[np.argsort(x)] = np.arange(len(x)); ry[np.argsort(y)] = np.arange(len(x))
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum()*(ry**2).sum())
    return (rx*ry).sum()/d if d > 0 else np.nan
neg = df[df.ToRef <= 0]
ic = neg.groupby('Date').apply(lambda p: pd.Series({
    'net': sp(p.net.values, p.TakerSell_CloseBP_net.values),
    'netM': sp(p.netM.values, p.TakerSell_CloseBP_netM.values)}), include_groups=False)
print(ic.groupby(ic.index >= '2026-01-01').mean().rename(index={False: '2025', True: '2026'}).round(4).to_string())

print('\n===== Q2c: 24-cell 延伸 pipeline(q=0.3/0.1, holdout)——新增 cell 的貢獻 =====')
cell24 = df['tb4']*10 + df['rb6']
cC = (df.net > thr('net', cell24, 0.3)) & (df.netM > thr('netM', cell24, 0.1)) & static_common & df.rb6.notna()
mC = capped(cC & HOLD.reindex(df.index))
mC['pnl_raw'] = mC.BidPrice1 * (mC.TakerSell_CloseBP - FEE) / 1e5
mC['pnl_hedged'] = mC.BidPrice1 * (mC._idiobp - FEE - 1.0) / 1e5   # 1bp 期貨腿成本
mC['grp'] = np.where(mC.ToRef > 0, 'ToRef>0', 'ToRef<=0')
g = mC.groupby('grp').agg(trades=('pnl_raw','size'), raw_mean=('TakerSell_CloseBP','mean'),
                          hedged_mean=('_idiobp','mean'), pnl_raw=('pnl_raw','sum'), pnl_hedged=('pnl_hedged','sum'))
print(g.round(1).to_string())
d_new = mC[mC.grp == 'ToRef<=0'].groupby('Date').pnl_hedged.sum()
print(f'ToRef<=0 新增段(避險後): 日均 {d_new.mean():.1f}萬, 日勝率 {(d_new>0).mean():.1%}, '
      f'{len(d_new)} 天, t={d_new.mean()/d_new.std()*np.sqrt(len(d_new)):.2f}')
