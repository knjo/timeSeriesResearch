"""Build the FH_neg15 condition line: own index + own merged per-day data (data_fh15/).
Does NOT touch the negFill index/data. Resumable (skips existing outputs)."""
import sys, datetime as _dt
from pathlib import Path
import polars as pl

sys.path.append('/home/kevin/Project/HFT')
from src.research.timeSeries.data_loader import _merge_one_day
from src.dataloader.txfDataLoader import TxfDataLoader

ROOT = Path('/home/kevin/Project/HFT')
OUT = ROOT / 'src/research/timeSeries/data_fh15'
OUT.mkdir(exist_ok=True)
IDX_OUT = ROOT / 'src/research/timeSeries/fh15_index.parquet'
txf_loader = TxfDataLoader(ROOT / 'data/txfTickData')

days = sorted(f.stem.split('_')[0] for f in (ROOT / 'data/tickFeature').glob('*_tickFeature.parquet'))
days = [d for d in days if '20250211' <= d <= '20260817'
        and (ROOT / f'data/preMarket/{d}_preMarketData.parquet').exists()]
print(f'{len(days)} days', flush=True)

idx_parts = []
for i, d in enumerate(days):
    out_path = OUT / f'{d}.parquet'
    try:
        keys = (pl.scan_parquet(ROOT / f'data/tickFeature/{d}_tickFeature.parquet')
                .select('QuoteCode', 'ChannelSeq', 'FillLots_atHigh')
                .join(pl.scan_parquet(ROOT / f'data/preMarket/{d}_preMarketData.parquet')
                        .select('QuoteCode', 'day_amount_rank', 'nextday_allow_day_trade_mark',
                                'avg_bidLots1', 'avg_askLots1'),
                      on='QuoteCode', how='inner')
                .filter((pl.col('nextday_allow_day_trade_mark') == 'X') & (pl.col('day_amount_rank') <= 150))
                .sort('QuoteCode', 'ChannelSeq')
                .with_columns(
                    (pl.col('FillLots_atHigh') != pl.col('FillLots_atHigh').shift(1))
                    .over('QuoteCode').fill_null(True).alias('chg'),
                    (pl.col('FillLots_atHigh') / (pl.col('avg_bidLots1') + pl.col('avg_askLots1'))).alias('ratio'))
                .filter(pl.col('chg') & (pl.col('ratio') < -15))
                .select('QuoteCode', 'ChannelSeq')
                .collect())
        idx_parts.append(keys.with_columns(pl.lit(d).alias('Date')))
        if out_path.exists():
            if i % 20 == 0: print(f'[{i+1}/{len(days)}] {d} exists, skip merge', flush=True)
            continue
        result = _merge_one_day(d, keys, txf_loader)
        if result is not None and len(result) > 0:
            result.write_parquet(out_path, compression='zstd')
        if i % 10 == 0:
            print(f'[{i+1}/{len(days)}] {d}: keys={len(keys):,} merged={len(result) if result is not None else 0:,}', flush=True)
    except Exception as e:
        print(f'{d}: ERROR {e}', flush=True)

pl.concat(idx_parts).write_parquet(IDX_OUT)
print('FH15 BUILD COMPLETE', flush=True)
