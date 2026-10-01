"""FHL scaffold-fix quick pass: no model, extended ToRef band, take-profit variants, intensity cuts."""
import datetime as dt
import numpy as np, pandas as pd, polars as pl
from pathlib import Path

DATA = Path('/home/kevin/Project/HFT/src/research/intradaySignal/data_fhL')
FEE = 19.3
files = sorted(f for f in DATA.glob('*.parquet') if '20250903' <= f.stem <= '20260817')
parts = []
for f in files:
    lf = (pl.scan_parquet(f)
          .select('TransTime','QuoteCode','BidPrice1','AskPrice1','Close','FutureHigh','RefPrice','TickSize',
                  'TrialMatch','day_amount_rank','AmountRank_canDayTrade','OTC','MD_L1Rate_30','SpreadPairElapsed',
                  'hft_strick_makerSpreadBP','avg_bidLots1','avg_askLots1','RecordHigh','RecordLow',
                  'FillLots_atHigh','txf_beta_60d','txf_to_1330_bp','txf_residual_vol_to_1330')
          .filter((pl.col('BidPrice1') > 0) & (pl.col('TrialMatch') == 0)
                  & (pl.col('TransTime').dt.time() > dt.time(9, 0, 30))
                  & (pl.col('TransTime').dt.time() < dt.time(12, 30))
                  & ((pl.col('day_amount_rank') <= 150) | (pl.col('AmountRank_canDayTrade') <= 150)))
          .with_columns(((pl.col('BidPrice1') - pl.col('RefPrice')) / pl.col('RefPrice')).round(6).alias('ToRef'))
          .filter((pl.col('ToRef') > 0.0) & (pl.col('ToRef') < 0.09))     # 帶寬放開到 +9%
          .with_columns(
              ((pl.col('BidPrice1') - pl.col('RecordLow'))
               / (pl.col('RecordHigh') - pl.col('RecordLow')).clip(lower_bound=1e-9)).alias('posrng'),
              (pl.col('FillLots_atHigh') / (pl.col('avg_bidLots1') + pl.col('avg_askLots1'))).alias('fhr'),
              pl.when(pl.col('FutureHigh') > pl.col('RefPrice') * 1.09)
                .then(pl.col('RefPrice') * 1.09).otherwise(pl.col('Close')).alias('exl_tp'),
              (pl.col('txf_beta_60d') * pl.col('txf_to_1330_bp')).alias('hedge'),
          ).with_columns(
              ((pl.col('exl_tp') - pl.col('BidPrice1')) / pl.col('BidPrice1') * 1e4).alias('raw_tp'),   # maker + 1.09停利
              ((pl.col('Close') - pl.col('BidPrice1')) / pl.col('BidPrice1') * 1e4).alias('raw_nc'),    # maker + 抱到收盤
          ).filter((pl.col('raw_tp').abs() < 10000) & (pl.col('raw_nc').abs() < 10000))
          .with_columns(pl.lit(f.stem).alias('D')))
    parts.append(lf.collect())
df = pl.concat(parts, how='vertical_relaxed').to_pandas()
df['Date'] = pd.to_datetime(df['D'])
print('rows', len(df))

def run_book(m, raw_col):
    m = m.sort_values(['Date','QuoteCode','TransTime']).reset_index(drop=True)
    m['accLots'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumcount') + 1
    m['Position'] = m.groupby(['Date','QuoteCode']).BidPrice1.transform('cumsum') / 10
    m = m[(m.accLots < m.avg_askLots1 + m.avg_bidLots1) & (m.Position < 200) & (m.BidPrice1 <= 1000)]
    pnl = (m.BidPrice1 * (m[raw_col] - FEE) / 1e5).groupby(m.Date).sum()
    cap = (m.BidPrice1 / 10).groupby(m.Date).sum()
    if len(pnl) < 50: return None
    sh = pnl.mean()/pnl.std()*np.sqrt(240)
    out = {'pnl萬': round(pnl.sum()), 'Sharpe': round(sh, 2),
           'capw': round(pnl.sum()/cap.sum()*1e4, 1), '部位萬': round(cap.mean()), '筆/日': round(len(m)/m.Date.nunique())}
    h = pnl[pnl.index >= '2026-04-01']
    out['holdout_capw'] = round(h.sum()/cap[cap.index >= '2026-04-01'].sum()*1e4, 1)
    out['holdout_Sharpe'] = round(h.mean()/h.std()*np.sqrt(240), 2)
    return out, pnl

variants = {
    '全事件+停利':        (df, 'raw_tp'),
    '全事件+抱收盤':      (df, 'raw_nc'),
    'pos>=0.85+停利':     (df[df.posrng >= 0.85], 'raw_tp'),
    'pos>=0.85+抱收盤':   (df[df.posrng >= 0.85], 'raw_nc'),
    '吸收(fhr<0)+抱收盤': (df[(df.fhr < 0)], 'raw_nc'),
    'pos>=0.85+ToRef>5%+抱收盤': (df[(df.posrng >= 0.85) & (df.ToRef > 0.05)], 'raw_nc'),
}
b = pd.read_parquet('/home/kevin/Project/HFT/src/research/intradaySignal/baseline/timeseries_points.parquet')
b['Date'] = pd.to_datetime(b['Date'])
pB = (b.BidPrice1*(b.TakerSell_CloseBP-FEE)/1e5).groupby(b.Date).sum()
for tag, (sub, col) in variants.items():
    r = run_book(sub.copy(), col)
    if r is None:
        print(f'{tag}: 樣本不足'); continue
    st, pnl = r
    rho = pnl.reindex(pB.index, fill_value=0).corr(pB)
    print(f'{tag:26s} {st} ρ={rho:+.2f}')
