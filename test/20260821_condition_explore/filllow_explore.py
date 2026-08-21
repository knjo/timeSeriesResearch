"""Blind exploration: does FillLots_atLow contain usable events? (methodology recovery test)"""
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
            'QuoteCode','ChannelSeq','FillLots_atLow')
        base = (pl.scan_parquet(ROOT / f'data/tickData/{d}_StockTick.parquet').select(
                    'TransTime','QuoteCode','ChannelSeq','BidPrice1','AskPrice1','Close','FutureHigh','RefPrice','TrialMatch')
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
        n_pool = len(base)
        if n_pool == 0: continue
        base = base.sort(['QuoteCode','ChannelSeq']).with_columns(
            (pl.col('FillLots_atLow') != pl.col('FillLots_atLow').shift(1)).over('QuoteCode').fill_null(True).alias('chg'),
            (pl.col('FillLots_atLow').shift(1).over('QuoteCode') >= 0).fill_null(False).alias('prev_nonneg'),
            (pl.col('FillLots_atLow') / (pl.col('avg_bidLots1') + pl.col('avg_askLots1'))).alias('ratio'),
        )
        fl, ch, rt = pl.col('FillLots_atLow'), pl.col('chg'), pl.col('ratio')
        conds = {
            'FL_neg_chg':   (fl < 0) & ch,
            'FL_neg2_chg':  (rt < -2) & ch,
            'FL_neg8_chg':  (rt < -8) & ch,
            'FL_onset':     (fl < 0) & pl.col('prev_nonneg'),
            'FL_pos_chg':   (fl > 0) & ch,
            'FL_pos8_chg':  (rt > 8) & ch,
        }
        any_cond = conds['FL_neg_chg'] | conds['FL_pos_chg'] | conds['FL_onset']
        ev = base.filter(any_cond)
        samp = base.sample(n=min(20000, n_pool), seed=int(d))
        uni = pl.concat([ev.with_columns(pl.lit('ev').alias('src')),
                         samp.with_columns(pl.lit('bl').alias('src'))], how='vertical_relaxed')
        uni = uni.with_columns(tick_size(pl.col('BidPrice1')).alias('TickSize')).sort('TransTime')
        uni = add_txf_event_residual_vol(attach_txf_to_1330_return(uni, d, loader))
        uni = uni.with_columns(
            pl.when(pl.col('FutureHigh') > pl.col('RefPrice') * 1.08)
              .then(pl.col('RefPrice') * 1.08 + pl.col('TickSize') * 2).otherwise(pl.col('Close')).alias('_exs'),
            pl.when(pl.col('FutureHigh') > pl.col('RefPrice') * 1.09)
              .then(pl.col('RefPrice') * 1.09).otherwise(pl.col('Close')).alias('_exl'),
        ).with_columns(
            ((pl.col('BidPrice1') - pl.col('_exs')) / pl.col('BidPrice1') * 1e4).alias('raw_s'),
            ((pl.col('_exl') - pl.col('AskPrice1')) / pl.col('AskPrice1') * 1e4).alias('raw_l'),
            (pl.col('txf_beta_60d') * pl.col('txf_to_1330_bp')).alias('hedge'),
            (pl.col('txf_residual_vol_to_1330') * 1e4).alias('volbp'),
        ).filter((pl.col('raw_s').abs() < 10000) & (pl.col('raw_l').abs() < 10000)
        ).with_columns(
            ((pl.col('raw_s') + pl.col('hedge')) / pl.col('volbp')).alias('z_s'),
            ((pl.col('raw_l') - pl.col('hedge')) / pl.col('volbp')).alias('z_l'),
        )
        tags = {'baseline': pl.col('src') == 'bl'}
        tags.update({k: (pl.col('src') == 'ev') & v for k, v in conds.items()})
        for tag, c in tags.items():
            sub = uni.filter(c & pl.col('z_s').is_finite() & pl.col('z_l').is_finite())
            if len(sub) == 0: continue
            rows.append({'Date': d, 'tag': tag, 'n_pool': n_pool, 'n': len(sub),
                         's_zs': sub['z_s'].sum(), 's_zs2': (sub['z_s']**2).sum(),
                         's_zl': sub['z_l'].sum(), 's_zl2': (sub['z_l']**2).sum(),
                         's_raws': sub['raw_s'].sum(), 's_rawl': sub['raw_l'].sum(),
                         's_hedge': sub['hedge'].sum()})
        if i % 15 == 0:
            print(f'[{i+1}/{len(days)}] {d} pool={n_pool:,} ev={len(ev):,}', flush=True)
    except Exception as e:
        print(f'{d}: ERROR {e}', flush=True)

pickle.dump(rows, open(OUT / 'filllow_daily.pkl', 'wb'))
print('FILLLOW SCAN DONE', flush=True)
