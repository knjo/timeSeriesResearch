"""Stage 2: exact replication of the model notebook (cells 2-27) + controlled comparisons.

Runs 5 rolling models on the identical sample:
  A. net60    : TakerSell_CloseBP_net   (residual z, stock-day demeaned), 60d   [their run]
  B. netM20   : TakerSell_CloseBP_netM  (residual z, day demeaned),       20d   [their run]
  C. rawnet60 : raw bp, stock-day demeaned, 60d                                 [old-style baseline]
  D. rawnetM20: raw bp, day demeaned, 20d                                       [old-style baseline]
  E. resid60  : residual z itself (no demean), 60d                              [their "no trades after June" run]
Then: monthly per-day IC tables, qcut decomposition of the 2026 inversion,
threshold-pipeline reproduction for residual vs raw variants.
"""
import sys, os, json, gc, datetime
import numpy as np
import pandas as pd
import polars as pl

sys.path.append('/home/kevin/Project/HFT')
from src.research.rolling_model import train_and_predict_ridge_rolling

DATA = '/home/kevin/Project/HFT/src/research/timeSeries/data'
OUT = '/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad'
FEE = 19.3

# ---------------- cell 1/2: file list ----------------
files = sorted(f for f in os.listdir(DATA) if f.endswith('.parquet'))
files = files[-230:]
print(f'files: {files[0]} .. {files[-1]}  n={len(files)}', flush=True)

schema_cols = pl.scan_parquet(f'{DATA}/{files[-1]}').collect_schema().names()
pre_existing_re = [c for c in schema_cols if ('_re' in c) and ('2330' not in c)]
print('pre-existing _re columns in merged schema:', pre_existing_re, flush=True)

cell10_re = ['LOB_BidVelocity_30_re','LOB_AskVelocity_30_re','FillLots_atLow_re','FillLots_atHigh_re',
             'L1_SellBiggestLots_30_re','L1_BuyBiggestLots_30_re','MD_ElaspeTime_30_re','MD_L1Rate_30_re',
             'IA_BuyImpact_30_re','IA_SellImpact_30_re','LOB_NetPressure_30_re','QL_Asymmetry_re']
explicit16 = ['ToLow','ToHigh','Low_High','RemainSeconds','ToOpen','ToRef','B1_A1B1','B1_B1B5','A1_A1A5',
              'B2_Last','A2_Last','B45_AB45','B1_B12','A1_A12','QL_BidHHI','QL_AskHHI']
# replicate cell 11 order: columns.str.contains('_re') scans signal_df column order =
# parquet order then creation order; pre-existing ones come first
dynamic_re = pre_existing_re + [c for c in cell10_re if c not in pre_existing_re]
feature_columns = explicit16 + dynamic_re
print(f'feature_columns n={len(feature_columns)} (notebook log said 31)', flush=True)

need = list(dict.fromkeys([
    'TransTime','QuoteCode','ChannelSeq','TrialMatch','BidPrice1','RefPrice','Close','Open','TickSize',
    'FutureHigh','day_amount_rank','AmountRank_canDayTrade','AmountRank',
    'txf_beta_60d','txf_to_1330_bp','txf_residual_vol_to_1330',
    'ToLow','ToHigh','Low_High','RemainSeconds','B1_A1B1','B1_B1B5','A1_A1A5','QL_BidHHI','QL_AskHHI',
    'BidLots1','BidLots2','BidLots4','BidLots5','AskLots1','AskLots2','AskLots4','AskLots5',
    'avg_bidLots1','avg_askLots1','TotalFillLots','last_fillLots',
    'LOB_BidVelocity_30','LOB_AskVelocity_30','FillLots_atLow','FillLots_atHigh',
    'L1_SellBiggestLots_30','L1_BuyBiggestLots_30','big_sell_lots','big_buy_lots',
    'MD_ElaspeTime_30','MD_L1Rate_30','IA_BuyImpact_30','IA_SellImpact_30','LOB_NetPressure_30',
    'QL_Asymmetry','midEdge_300sBP','Spread','OTC','investment_netLots','SpreadPairElapsed',
    'hft_strick_makerSpreadBP',
] + pre_existing_re))
need = [c for c in need if c in schema_cols]

# ---------------- cell 2/3: load + base filters ----------------
parts = []
for f in files:
    lf = (pl.scan_parquet(f'{DATA}/{f}')
            .select(need)
            .filter(pl.col('BidPrice1') > 0))
    lf = lf.with_columns(
        ((pl.col('BidPrice1') - pl.col('RefPrice')) / pl.col('RefPrice')).round(6).alias('ToRef'),
        ((pl.col('BidPrice1') - pl.col('Open')) / pl.col('Open')).round(6).alias('ToOpen'),
        pl.when(pl.col('FutureHigh') > pl.col('RefPrice') * 1.08)
          .then((pl.col('BidPrice1') - pl.col('RefPrice') * 1.08 - pl.col('TickSize') * 2) / pl.col('BidPrice1') * 10000)
          .otherwise((pl.col('BidPrice1') - pl.col('Close')) / pl.col('BidPrice1') * 10000)
          .alias('TakerSell_CloseBP'),
    ).filter(
        (pl.col('ToRef') > -0.015) & (pl.col('ToRef') < 0.05)
        & ((pl.col('day_amount_rank') <= 150) | (pl.col('AmountRank_canDayTrade') <= 150))
    ).with_columns(pl.lit(f.split('.')[0]).alias('Date'))
    parts.append(lf.collect())
big = pl.concat(parts, how='vertical_relaxed')
del parts; gc.collect()
print('after cell2 filters:', big.shape, flush=True)

# ---------------- cell 4/5 ----------------
big = big.with_columns(
    (pl.col('txf_beta_60d') * pl.col('txf_to_1330_bp')).alias('_hedgebp'),
    (pl.col('txf_residual_vol_to_1330') * 1e4).alias('_volbp'),
).with_columns(
    ((pl.col('TakerSell_CloseBP') + pl.col('_hedgebp')) / pl.col('_volbp')).alias('residual_TakerSell_CloseBP'),
    (pl.col('TakerSell_CloseBP') + pl.col('_hedgebp')).alias('_idiobp'),
)
big = big.filter(
    (pl.col('TransTime').dt.time() < datetime.time(12, 30)) &
    (pl.col('TransTime').dt.time() > datetime.time(9, 0, 30)) &
    (pl.col('TrialMatch') == 0) &
    (pl.col('TakerSell_CloseBP').abs() < 10000)
).sort(['Date','QuoteCode','TransTime'])
big = big.with_columns((pl.col('midEdge_300sBP') * -1).alias('midEdge_300sBP'))
print('after cell5 filters:', big.shape, flush=True)

signal_df = big.to_pandas()
del big; gc.collect()

# ---------------- cell 7: labels ----------------
g_sq = signal_df.groupby(['Date','QuoteCode'])
g_d = signal_df.groupby(['Date'])
signal_df['TakerSell_CloseBP_net'] = signal_df['residual_TakerSell_CloseBP'] - g_sq['residual_TakerSell_CloseBP'].transform('mean')
signal_df['TakerSell_CloseBP_netM'] = signal_df['residual_TakerSell_CloseBP'] - g_d['residual_TakerSell_CloseBP'].transform('mean')
# raw-label baselines (old style)
signal_df['TakerSell_CloseBP_rawnet'] = signal_df['TakerSell_CloseBP'] - g_sq['TakerSell_CloseBP'].transform('mean')
signal_df['TakerSell_CloseBP_rawnetM'] = signal_df['TakerSell_CloseBP'] - g_d['TakerSell_CloseBP'].transform('mean')

# ---------------- cell 9 ----------------
with open('/home/kevin/Project/HFT/src/research/timeSeries/abnormal_dates.json') as fh:
    ab_dates = json.load(fh)
signal_df['isAbnormalDate'] = signal_df['Date'].isin(ab_dates) * 1

# ---------------- cell 10 ----------------
sd = signal_df
sd['B2_Last'] = sd.BidLots2 / sd.avg_bidLots1
sd['A2_Last'] = sd.AskLots2 / sd.avg_askLots1
sd['B1_B12'] = sd.BidLots1 / (sd.BidLots1 + sd.BidLots2)
sd['A1_A12'] = sd.AskLots1 / (sd.AskLots1 + sd.AskLots2)
sd['B45_AB45'] = (sd.BidLots4 + sd.BidLots5) / (sd.AskLots4 + sd.AskLots5 + sd.BidLots4 + sd.BidLots5)
sd['Total_Last'] = sd.TotalFillLots / sd.last_fillLots
sd['TickBP'] = sd.TickSize / sd.BidPrice1 * 10000
sd['LOB_BidVelocity_30_re'] = sd.LOB_BidVelocity_30.abs() / sd.avg_bidLots1
sd['LOB_AskVelocity_30_re'] = sd.LOB_AskVelocity_30.abs() / sd.avg_askLots1
sd['FillLots_atLow_re'] = sd.FillLots_atLow / (sd.avg_bidLots1 + sd.avg_askLots1)
sd['FillLots_atHigh_re'] = sd.FillLots_atHigh / (sd.avg_bidLots1 + sd.avg_askLots1)
sd['L1_SellBiggestLots_30_re'] = sd.L1_SellBiggestLots_30.abs() / sd.big_sell_lots
sd['L1_BuyBiggestLots_30_re'] = sd.L1_BuyBiggestLots_30.abs() / sd.big_buy_lots
sd['MD_ElaspeTime_30_re'] = (sd.MD_ElaspeTime_30 + 1).apply(np.log)
sd['MD_L1Rate_30_re'] = sd.MD_L1Rate_30
sd['IA_BuyImpact_30_re'] = (sd.IA_BuyImpact_30.abs() + 1).apply(np.log)
sd['IA_SellImpact_30_re'] = (sd.IA_SellImpact_30.abs() + 1).apply(np.log)
sd['LOB_NetPressure_30_re'] = (sd.LOB_NetPressure_30.abs() + 1).apply(np.log)
sd['QL_Asymmetry_re'] = (sd.QL_Asymmetry.abs() + 1).apply(np.log)

# ---------------- cell 12 ----------------
missing_feats = [c for c in feature_columns if c not in sd.columns]
if missing_feats:
    print('!! features missing from build:', missing_feats, flush=True)
    feature_columns = [c for c in feature_columns if c in sd.columns]
for c in feature_columns:
    sd[c] = sd[c].round(6)
sd['OTC'] = sd['OTC'] * 1
sd['isInvestBuy'] = ((sd['investment_netLots'] > 1000) & (sd.OTC == True)) * 1
print('final feature count:', len(feature_columns), flush=True)

# ---------------- feature IC panel (pre-model) ----------------
def spearman(x, y):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 200: return np.nan
    rx = np.empty(n); ry = np.empty(n)
    rx[np.argsort(x, kind='stable')] = np.arange(n)
    ry[np.argsort(y, kind='stable')] = np.arange(n)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d > 0 else np.nan

rng = np.random.RandomState(7)
ic_rows = []
for d, part in sd.groupby('Date', sort=True):
    idx = part.index.to_numpy()
    if len(idx) > 25000:
        idx = rng.choice(idx, 25000, replace=False)
    p = sd.loc[idx]
    y_raw = p['TakerSell_CloseBP'].to_numpy(float)
    y_z = p['residual_TakerSell_CloseBP'].to_numpy(float)
    row_r = {'Date': d, 'y': 'raw'}
    row_z = {'Date': d, 'y': 'z'}
    for c in feature_columns:
        xv = p[c].to_numpy(float)
        row_r[c] = spearman(xv, y_raw)
        row_z[c] = spearman(xv, y_z)
    ic_rows += [row_r, row_z]
ic_panel = pd.DataFrame(ic_rows)
ic_panel.to_csv(f'{OUT}/feature_ic_daily.csv', index=False)
print('feature IC panel saved', flush=True)

# ---------------- cell 13 chain: five model runs ----------------
runs = [
    ('TakerSell_CloseBP_net', 60),
    ('TakerSell_CloseBP_netM', 20),
    ('TakerSell_CloseBP_rawnet', 60),
    ('TakerSell_CloseBP_rawnetM', 20),
    ('residual_TakerSell_CloseBP', 60),
]
for ycol, ndays in runs:
    res = train_and_predict_ridge_rolling(
        signal_df[:], feature_columns, ycol, n_training_days=ndays,
        day_features=[], feature_screening=True, alpha=0.1, max_position_per_stock=500)
    signal_df = res['df']
    del res; gc.collect()
    print(f'== done {ycol} ({ndays}d): rows now {len(signal_df)}', flush=True)

keep = ['Date','QuoteCode','TransTime','ChannelSeq','BidPrice1','TakerSell_CloseBP','_hedgebp','_idiobp','_volbp',
        'residual_TakerSell_CloseBP','TakerSell_CloseBP_net','TakerSell_CloseBP_netM',
        'TakerSell_CloseBP_rawnet','TakerSell_CloseBP_rawnetM','txf_beta_60d','txf_to_1330_bp',
        'isAbnormalDate','OTC','MD_L1Rate_30_re','SpreadPairElapsed','ToRef','AmountRank_canDayTrade',
        'day_amount_rank','avg_askLots1','avg_bidLots1','hft_strick_makerSpreadBP','RemainSeconds'] + \
       [c for c in signal_df.columns if c.endswith('_pred') or c.endswith('_ABpred')]
signal_df[keep].to_parquet(f'{OUT}/stage2_preds.parquet', index=False)
print('preds saved:', [c for c in keep if 'pred' in c], flush=True)
print('STAGE2 BUILD+RUNS COMPLETE', flush=True)
