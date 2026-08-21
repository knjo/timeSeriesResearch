"""Build the long line (FHL): rally>=5% & posrng>=0.7 & |dfill|>=2 -> own index + merged data_fhL/."""
import sys
from pathlib import Path
import polars as pl

sys.path.append('/home/kevin/Project/HFT')
from src.research.timeSeries.data_loader import _merge_one_day
from src.dataloader.txfDataLoader import TxfDataLoader

ROOT = Path('/home/kevin/Project/HFT')
OUT = ROOT / 'src/research/timeSeries/data_fhL'
OUT.mkdir(exist_ok=True)
txf_loader = TxfDataLoader(ROOT / 'data/txfTickData')
days = sorted(f.stem.split('_')[0] for f in (ROOT / 'data/tickFeature').glob('*_tickFeature.parquet'))
days = [d for d in days if '20250211' <= d <= '20260817'
        and (ROOT / f'data/preMarket/{d}_preMarketData.parquet').exists()]
print(len(days), 'days', flush=True)

idx_parts = []
for i, d in enumerate(days):
    out_path = OUT / f'{d}.parquet'
    try:
        keys = (pl.scan_parquet(ROOT / f'data/tickData/{d}_StockTick.parquet')
                .select('QuoteCode', 'ChannelSeq', 'BidPrice1', 'RefPrice', 'RecordHigh', 'RecordLow')
                .join(pl.scan_parquet(ROOT / f'data/tickFeature/{d}_tickFeature.parquet')
                        .select('QuoteCode', 'ChannelSeq', 'FillLots_atHigh'),
                      on=['QuoteCode', 'ChannelSeq'], how='inner')
                .join(pl.scan_parquet(ROOT / f'data/preMarket/{d}_preMarketData.parquet')
                        .select('QuoteCode', 'day_amount_rank', 'nextday_allow_day_trade_mark'),
                      on='QuoteCode', how='inner')
                .filter((pl.col('nextday_allow_day_trade_mark') == 'X') & (pl.col('day_amount_rank') <= 150)
                        & (pl.col('BidPrice1') > 0))
                .sort('QuoteCode', 'ChannelSeq')
                .with_columns(
                    (pl.col('FillLots_atHigh') - pl.col('FillLots_atHigh').shift(1)).over('QuoteCode').alias('dfill'),
                    ((pl.col('BidPrice1') - pl.col('RecordLow'))
                     / (pl.col('RecordHigh') - pl.col('RecordLow')).clip(lower_bound=1e-9)).alias('posrng'),
                    (pl.col('RecordHigh') / pl.col('RefPrice') - 1).alias('rally'))
                .filter((pl.col('rally') >= 0.05) & (pl.col('posrng') >= 0.7) & (pl.col('dfill').abs() >= 2))
                .select('QuoteCode', 'ChannelSeq')
                .collect())
        idx_parts.append(keys.with_columns(pl.lit(d).alias('Date')))
        if out_path.exists():
            continue
        result = _merge_one_day(d, keys, txf_loader)
        if result is not None and len(result) > 0:
            result.write_parquet(out_path, compression='zstd')
        if i % 20 == 0:
            print(f'[{i+1}/{len(days)}] {d}: keys={len(keys):,}', flush=True)
    except Exception as e:
        print(f'{d}: ERROR {e}', flush=True)

pl.concat(idx_parts).write_parquet(ROOT / 'src/research/timeSeries/fhL_index.parquet')
print('FHL BUILD COMPLETE', flush=True)
