"""Does adding depth (FillLots_atLow ratio < -2) to the real negFill condition lift its edge?"""
import datetime as dt
import polars as pl
from pathlib import Path

MERGED = Path('/home/kevin/Project/HFT/src/research/timeSeries/data')
files = sorted(f for f in MERGED.glob('*.parquet') if '20250211' <= f.stem <= '20260817')
rows = []
for f in files:
    lf = (pl.scan_parquet(f)
          .select('TransTime','BidPrice1','Close','FutureHigh','RefPrice','TickSize','TrialMatch',
                  'day_amount_rank','AmountRank_canDayTrade','FillLots_atLow','avg_bidLots1','avg_askLots1',
                  'txf_beta_60d','txf_to_1330_bp','txf_residual_vol_to_1330')
          .filter((pl.col('BidPrice1') > 0) & (pl.col('TrialMatch') == 0)
                  & (pl.col('TransTime').dt.time() > dt.time(9, 0, 30))
                  & (pl.col('TransTime').dt.time() < dt.time(12, 30))
                  & ((pl.col('day_amount_rank') <= 150) | (pl.col('AmountRank_canDayTrade') <= 150)))
          .with_columns(((pl.col('BidPrice1') - pl.col('RefPrice')) / pl.col('RefPrice')).round(6).alias('ToRef2'))
          .filter((pl.col('ToRef2') > -0.015) & (pl.col('ToRef2') < 0.05))
          .with_columns(
              pl.when(pl.col('FutureHigh') > pl.col('RefPrice') * 1.08)
                .then((pl.col('BidPrice1') - pl.col('RefPrice') * 1.08 - pl.col('TickSize') * 2) / pl.col('BidPrice1') * 1e4)
                .otherwise((pl.col('BidPrice1') - pl.col('Close')) / pl.col('BidPrice1') * 1e4).alias('raw'),
              (pl.col('FillLots_atLow') / (pl.col('avg_bidLots1') + pl.col('avg_askLots1'))).alias('ratio'))
          .filter(pl.col('raw').abs() < 10000)
          .with_columns(((pl.col('raw') + pl.col('txf_beta_60d') * pl.col('txf_to_1330_bp'))
                         / (pl.col('txf_residual_vol_to_1330') * 1e4)).alias('z'))
          .filter(pl.col('z').is_finite()))
    deep = pl.col('ratio') < -2
    agg = lf.select(
        pl.len().alias('n_all'), pl.col('z').sum().alias('z_all'),
        deep.sum().alias('n_deep'), pl.col('z').filter(deep).sum().alias('z_deep'),
        pl.col('z').filter(~deep).sum().alias('z_shallow'),
    ).collect().to_dicts()[0]
    agg['Date'] = f.stem
    rows.append(agg)

import pandas as pd, numpy as np
df = pd.DataFrame(rows)
df['ym'] = df.Date.str[:6]; df['yr'] = df.Date.str[:4]
for per, m in [('2025', df.yr == '2025'), ('2026', df.yr == '2026')]:
    s = df[m]
    z_all = s.z_all.sum()/s.n_all.sum()
    z_deep = s.z_deep.sum()/s.n_deep.sum()
    z_sh = s.z_shallow.sum()/(s.n_all.sum()-s.n_deep.sum())
    share = s.n_deep.sum()/s.n_all.sum()
    d = (s.z_deep/s.n_deep - s.z_all/s.n_all).dropna()
    print(f"[{per}] negFill全體 z={z_all:+.3f} | ∩深度 z={z_deep:+.3f} (佔 {share:.0%}, n/日 {s.n_deep.sum()/len(s):,.0f}) "
          f"| 淺層 z={z_sh:+.3f} | 深-全 日級 t={d.mean()/d.std()*np.sqrt(len(d)):.1f}")
mm = (df.z_deep/df.n_deep - df.z_all/df.n_all).groupby(df.ym).mean()
print('月度 (深-全) Δz:', ' '.join(f'{k}:{v:+.2f}' for k, v in mm.items() if k >= '202601'))
