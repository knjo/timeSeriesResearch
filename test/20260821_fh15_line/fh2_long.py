"""Long side at the high zone: taker (A1 entry) vs maker-optimistic (B1 entry), take-profit 1.09."""
import sys, datetime as dt, pickle
import polars as pl
from pathlib import Path

sys.path.append('/home/kevin/Project/HFT')
from src.dataloader.txfDataLoader import TxfDataLoader, attach_txf_to_1330_return
from src.features.market_beta import add_txf_event_residual_vol

ROOT = Path('/home/kevin/Project/HFT')
OUT = Path('/tmp/claude-1000/-home-kevin-Project-HFT/a5215848-bd0c-4df7-9d88-9bbbed1c6d1e/scratchpad')
loader = TxfDataLoader(ROOT / 'data/txfTickData')
MERGED = ROOT / 'src/research/timeSeries/data'
days = sorted(f.stem for f in MERGED.glob('*.parquet') if '20250211' <= f.stem <= '20260817')[::5]
print(len(days), 'days', flush=True)

def tick_size(p):
    return (pl.when(p < 10).then(0.01).when(p < 50).then(0.05).when(p < 100).then(0.1)
              .when(p < 500).then(0.5).when(p < 1000).then(1.0).otherwise(5.0))

rows = []
for i, d in enumerate(days):
    try:
        pm = pl.scan_parquet(ROOT / f'data/preMarket/{d}_preMarketData.parquet').select(
            'QuoteCode','day_amount_rank','nextday_allow_day_trade_mark',
            'txf_beta_60d','txf_residual_vol_0900_1330_60d','avg_bidLots1','avg_askLots1')
        tf = pl.scan_parquet(ROOT / f'data/tickFeature/{d}_tickFeature.parquet').select(
            'QuoteCode','ChannelSeq','FillLots_atHigh')
        base = (pl.scan_parquet(ROOT / f'data/tickData/{d}_StockTick.parquet').select(
                    'TransTime','QuoteCode','ChannelSeq','BidPrice1','AskPrice1','Close','FutureHigh',
                    'RefPrice','TrialMatch','RecordHigh','RecordLow')
                .join(tf, on=['QuoteCode','ChannelSeq'], how='inner')
                .join(pm, on='QuoteCode', how='inner')
                .filter(
                    (pl.col('BidPrice1') > 0) & (pl.col('AskPrice1') > 0) & (pl.col('TrialMatch') == 0)
                    & (pl.col('TransTime').dt.time() > dt.time(9, 0, 30))
                    & (pl.col('TransTime').dt.time() < dt.time(12, 30))
                    & (pl.col('nextday_allow_day_trade_mark') == 'X') & (pl.col('day_amount_rank') <= 150))
                .with_columns(((pl.col('BidPrice1') - pl.col('RefPrice')) / pl.col('RefPrice')).round(6).alias('ToRef2'))
                .filter((pl.col('ToRef2') > -0.015) & (pl.col('ToRef2') < 0.05))
                .collect())
        if len(base) == 0: continue
        base = base.sort(['QuoteCode','ChannelSeq']).with_columns(
            (pl.col('FillLots_atHigh') - pl.col('FillLots_atHigh').shift(1)).over('QuoteCode').alias('dfill'),
            ((pl.col('BidPrice1') - pl.col('RecordLow'))
             / (pl.col('RecordHigh') - pl.col('RecordLow')).clip(lower_bound=1e-9)).alias('posrng'),
            (pl.col('RecordHigh') / pl.col('RefPrice') - 1).alias('rally'),
        )
        rly5, rly6 = pl.col('rally') >= 0.05, pl.col('rally') >= 0.06
        upH, upT = pl.col('posrng') >= 0.5, pl.col('posrng') >= 0.7
        sell2, sell5 = pl.col('dfill') <= -2, pl.col('dfill') <= -5
        buy2 = pl.col('dfill') >= 2
        conds = {
            'L_absorb':   rly5 & upH & sell2,          # 高檔吸收賣壓（空方反訊號的正主）
            'L_absorb5':  rly5 & upH & sell5,          # 更大賣單被吸收
            'L_absorbT':  rly5 & upT & sell2,          # 更貼近高點
            'L_absorb_r6': rly6 & upH & sell2,
            'L_buy2':     rly5 & upH & buy2,           # 對照：高檔追買
        }
        any_cond = conds['L_absorb'] | conds['L_buy2']
        ev = base.filter(any_cond)
        samp = base.sample(n=min(20000, len(base)), seed=int(d))
        uni = pl.concat([ev.with_columns(pl.lit('ev').alias('src')),
                         samp.with_columns(pl.lit('bl').alias('src'))], how='vertical_relaxed')
        uni = uni.with_columns(tick_size(pl.col('BidPrice1')).alias('TickSize')).sort('TransTime')
        uni = add_txf_event_residual_vol(attach_txf_to_1330_return(uni, d, loader))
        uni = uni.with_columns(
            pl.when(pl.col('FutureHigh') > pl.col('RefPrice') * 1.09)
              .then(pl.col('RefPrice') * 1.09).otherwise(pl.col('Close')).alias('_exl'),
            (pl.col('txf_beta_60d') * pl.col('txf_to_1330_bp')).alias('hedge'),
            (pl.col('txf_residual_vol_to_1330') * 1e4).alias('volbp'),
        ).with_columns(
            ((pl.col('_exl') - pl.col('AskPrice1')) / pl.col('AskPrice1') * 1e4).alias('raw_lt'),
            ((pl.col('_exl') - pl.col('BidPrice1')) / pl.col('BidPrice1') * 1e4).alias('raw_lm'),
        ).filter((pl.col('raw_lt').abs() < 10000) & (pl.col('raw_lm').abs() < 10000)
        ).with_columns(
            ((pl.col('raw_lt') - pl.col('hedge')) / pl.col('volbp')).alias('z_lt'),
            ((pl.col('raw_lm') - pl.col('hedge')) / pl.col('volbp')).alias('z_lm'),
        )
        tags = {'baseline': pl.col('src') == 'bl'}
        tags.update({k: (pl.col('src') == 'ev') & v for k, v in conds.items()})
        for tag, c in tags.items():
            sub = uni.filter(c & pl.col('z_lt').is_finite() & pl.col('z_lm').is_finite())
            if len(sub) == 0: continue
            rows.append({'Date': d, 'tag': tag, 'n': len(sub),
                         's_lt': sub['z_lt'].sum(), 's_lt2': (sub['z_lt']**2).sum(),
                         's_lm': sub['z_lm'].sum(),
                         's_rawlt': sub['raw_lt'].sum(), 's_rawlm': sub['raw_lm'].sum()})
        if i % 15 == 0:
            print(f'[{i+1}/{len(days)}] {d} ev={len(ev):,}', flush=True)
    except Exception as e:
        print(f'{d}: ERROR {e}', flush=True)

pickle.dump(rows, open(OUT / 'fh2_long_daily.pkl', 'wb'))
print('FH2 LONG SCAN DONE', flush=True)
