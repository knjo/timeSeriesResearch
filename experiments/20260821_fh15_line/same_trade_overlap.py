"""視同一筆重疊率：同日同檔 且 (|ΔToRef|<=1% 或 |Δt|<=60s) 即視為同一筆。"""
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
    DATES = sorted(df['D'].unique()); buckets = sorted(set(k.split('_')[1] for k in store)); m = {}
    for i, dt in enumerate(DATES):
        win = DATES[max(0, i-n_days):i]
        if not win: continue
        for b in buckets:
            pool = [store.get(f'{w}_{b}') for w in win]
            pool = [x for x in pool if x is not None and len(x)]
            if pool: m[f'{dt}_{b}'] = np.quantile(np.concatenate(pool), q)
    return df['kb'].map(m)

def book(df, qn, qm):
    static = (((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.25)) & (df.SpreadPairElapsed > 0.1)
              & (df.ToRef > 0) & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100)))
    c = (df.net > thr(df, 'net', qn)) & (df.netM > thr(df, 'netM', qm)) & static
    m = df[c].sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)
          & ((m.hft_strick_makerSpreadBP > -70) | (m.hft_strick_makerSpreadBP.isna()))
          & ((m.OTC == 1) | (m.MD_L1Rate_30_re > 0.25))]
    return m[['Date','QuoteCode','TransTime','ToRef','BidPrice1']].reset_index(drop=True)

A = book(load(f'{S}/stage3_preds.parquet'), 0.3, 0.1)        # negFill
B = book(load(f'{S}/stage_fh15_preds.parquet'), 0.3, 0.1)    # FH15
print(f'negFill {len(A):,} 筆 | FH15 {len(B):,} 筆')

def same_trade_share(X, Y, lab):
    """X 的每筆單，在 Y 中找同日同檔的最近鄰（時間、ToRef 各一次）"""
    x = X.sort_values('TransTime').reset_index(drop=True)
    y = Y.sort_values('TransTime').reset_index(drop=True)
    mt = pd.merge_asof(x, y[['Date','QuoteCode','TransTime']].assign(t2=lambda d: d.TransTime),
                       on='TransTime', by=['Date','QuoteCode'], direction='nearest')
    dt_ok = (mt.TransTime - mt.t2).abs().dt.total_seconds() <= 60
    x2 = X.sort_values('ToRef').reset_index(drop=True)
    y2 = Y.sort_values('ToRef').reset_index(drop=True)
    mr = pd.merge_asof(x2, y2[['Date','QuoteCode','ToRef']].rename(columns={'ToRef':'r2'}).assign(ToRef=y2.ToRef),
                       on='ToRef', by=['Date','QuoteCode'], direction='nearest')
    rr_ok = (mr.ToRef - mr.r2).abs() <= 0.01
    key = ['Date','QuoteCode','TransTime']
    okt = set(map(tuple, mt.loc[dt_ok.fillna(False), key].values))
    okr = set(map(tuple, mr.loc[rr_ok.fillna(False), key].values))
    hit = X.apply(lambda r: (r.Date, r.QuoteCode, r.TransTime) in okt or (r.Date, r.QuoteCode, r.TransTime) in okr, axis=1)
    print(f'{lab}: 視同一筆 {hit.mean():.1%} (筆數) | {X.BidPrice1[hit].sum()/X.BidPrice1.sum():.1%} (資本)')
    return hit

hB = same_trade_share(B, A, 'FH15 → negFill')
hA = same_trade_share(A, B, 'negFill → FH15')
