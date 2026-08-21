"""Stage 1: daily decomposition of the negFill base sample.

Replicates notebook cells 2/4/5 exactly, then decomposes:
  raw label      = TakerSell_CloseBP                    (short PnL, bp)
  hedge term     = beta * txf_to_1330_bp                (long-futures hedge leg, bp)
  residual_bp    = raw + hedge                          (de-beta'd, bp)
  z              = residual_bp / (vol*1e4)              (cell 4's residual_TakerSell_CloseBP)
  x              = hedge / (vol*1e4)                    (hedge leg in z units -> leak regressor)
"""
import polars as pl
from pathlib import Path
import datetime as dt

DATA = Path('/home/kevin/Project/HFT/src/research/timeSeries/data')
OUT = Path('/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad')

files = sorted(DATA.glob('*.parquet'))
files = [f for f in files if '20250211' <= f.stem <= '20260817']
print(f'{len(files)} day files {files[0].stem}..{files[-1].stem}')

rows = []
for f in files:
    lf = pl.scan_parquet(f).filter(pl.col('BidPrice1') > 0)
    lf = lf.with_columns(
        ((pl.col('BidPrice1') - pl.col('RefPrice')) / pl.col('RefPrice')).round(6).alias('ToRef2'),
        pl.when(pl.col('FutureHigh') > pl.col('RefPrice') * 1.08)
          .then((pl.col('BidPrice1') - pl.col('RefPrice') * 1.08 - pl.col('TickSize') * 2) / pl.col('BidPrice1') * 10000)
          .otherwise((pl.col('BidPrice1') - pl.col('Close')) / pl.col('BidPrice1') * 10000)
          .alias('raw'),
        (pl.col('FutureHigh') > pl.col('RefPrice') * 1.08).alias('stopcap'),
    ).filter(
        (pl.col('ToRef2') > -0.015) & (pl.col('ToRef2') < 0.05)
        & ((pl.col('day_amount_rank') <= 150) | (pl.col('AmountRank_canDayTrade') <= 150))
        & (pl.col('TransTime').dt.time() < dt.time(12, 30))
        & (pl.col('TransTime').dt.time() > dt.time(9, 0, 30))
        & (pl.col('TrialMatch') == 0)
        & (pl.col('raw').abs() < 10000)
    ).with_columns(
        (pl.col('txf_beta_60d') * pl.col('txf_to_1330_bp')).alias('hedge'),
        (pl.col('txf_residual_vol_to_1330') * 1e4).alias('vol_bp'),
    ).with_columns(
        (pl.col('raw') + pl.col('hedge')).alias('res_bp'),
        ((pl.col('raw') + pl.col('hedge')) / pl.col('vol_bp')).alias('z'),
        (pl.col('hedge') / pl.col('vol_bp')).alias('x'),
    )
    # static non-model trade filters from cell 24 (approx "condition" pre-model)
    cond = (
        ((pl.col('OTC') == 1) | (pl.col('MD_L1Rate_30') > 0.25))
        & (pl.col('SpreadPairElapsed') > 0.1)
        & (pl.col('ToRef2') > 0)
        & ((pl.col('AmountRank_canDayTrade') <= 100) | (pl.col('day_amount_rank') <= 100))
    )
    agg = lf.select(
        pl.len().alias('n'),
        pl.col('z').is_finite().sum().alias('n_z'),
        pl.col('stopcap').sum().alias('n_stop'),
        pl.col('raw').sum().alias('s_raw'),
        pl.col('hedge').sum().alias('s_hedge'),
        pl.col('res_bp').sum().alias('s_res'),
        pl.col('z').filter(pl.col('z').is_finite()).sum().alias('s_z'),
        (pl.col('z').filter(pl.col('z').is_finite()) ** 2).sum().alias('s_z2'),
        pl.col('x').filter(pl.col('x').is_finite()).sum().alias('s_x'),
        pl.col('raw').filter(pl.col('z').is_finite()).sum().alias('s_raw_zok'),
        pl.col('hedge').filter(pl.col('z').is_finite()).sum().alias('s_hedge_zok'),
        pl.col('vol_bp').filter(pl.col('vol_bp').is_finite()).mean().alias('m_vol'),
        pl.col('txf_beta_60d').mean().alias('m_beta'),
        pl.col('txf_to_1330_bp').mean().alias('m_txf'),
        # conditioned subset
        cond.sum().alias('c_n'),
        pl.col('raw').filter(cond).sum().alias('c_s_raw'),
        pl.col('hedge').filter(cond).sum().alias('c_s_hedge'),
        pl.col('res_bp').filter(cond).sum().alias('c_s_res'),
        pl.col('z').filter(cond & pl.col('z').is_finite()).sum().alias('c_s_z'),
        (cond & pl.col('z').is_finite()).sum().alias('c_n_z'),
    ).collect()
    r = agg.to_dicts()[0]
    r['Date'] = f.stem
    rows.append(r)

daily = pl.DataFrame(rows)
daily.write_parquet(OUT / 'stage1_daily.parquet')
print('saved', daily.shape)

# ---- monthly pooled table ----
d = daily.with_columns(pl.col('Date').str.slice(0, 6).alias('ym'))
m = d.group_by('ym').agg(
    pl.col('n').sum(), pl.col('n_z').sum(), pl.col('n_stop').sum(), pl.col('c_n').sum(), pl.col('c_n_z').sum(),
    (pl.col('s_raw').sum() / pl.col('n').sum()).alias('raw_bp'),
    (pl.col('s_raw_zok').sum() / pl.col('n_z').sum()).alias('raw_bp_zok'),
    (pl.col('s_hedge_zok').sum() / pl.col('n_z').sum()).alias('hedge_bp'),
    ((pl.col('s_raw_zok').sum() + pl.col('s_hedge_zok').sum()) / pl.col('n_z').sum()).alias('res_bp'),
    (pl.col('s_z').sum() / pl.col('n_z').sum()).alias('z_mean'),
    ((pl.col('s_z2').sum() / pl.col('n_z').sum() - (pl.col('s_z').sum() / pl.col('n_z').sum()) ** 2) ** 0.5).alias('z_std'),
    (pl.col('s_x').sum() / pl.col('n_z').sum()).alias('x_mean'),
    pl.col('m_vol').mean().alias('vol_bp'),
    pl.col('m_txf').mean().alias('txf_bp_dayavg'),
    (pl.col('c_s_raw').sum() / pl.col('c_n').sum()).alias('c_raw_bp'),
    ((pl.col('c_s_raw').sum() + pl.col('c_s_hedge').sum()) / pl.col('c_n').sum()).alias('c_res_bp'),
    (pl.col('c_s_z').sum() / pl.col('c_n_z').sum()).alias('c_z_mean'),
).sort('ym')
with pl.Config(tbl_rows=25, tbl_cols=25, float_precision=3, tbl_width_chars=250):
    print(m)

# ---- leak regression: daily mean z  vs  daily mean x ----
import numpy as np
dd = daily.filter(pl.col('n_z') > 500).with_columns(
    (pl.col('s_z') / pl.col('n_z')).alias('mz'),
    (pl.col('s_x') / pl.col('n_z')).alias('mx'),
)
for lab, sub in [('ALL', dd), ('2025', dd.filter(pl.col('Date') < '20260101')), ('2026', dd.filter(pl.col('Date') >= '20260101'))]:
    mz = sub['mz'].to_numpy(); mx = sub['mx'].to_numpy()
    ok = np.isfinite(mz) & np.isfinite(mx)
    mz, mx = mz[ok], mx[ok]
    slope, icpt = np.polyfit(mx, mz, 1)
    r = np.corrcoef(mx, mz)[0, 1]
    print(f'[{lab}] days={len(mz)}  slope(mean_z ~ mean_hedge_z)={slope:.3f}  corr={r:.3f}  intercept={icpt:.4f}  -> implied k = {1-slope:.3f}')
