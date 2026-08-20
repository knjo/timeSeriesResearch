"""
condition_benchmark.py
======================
Condition 驗收儀表板：評估「事件條件」作為樣本過濾器的水平。

比較對象：條件事件樣本（--events-dir 的 merged per-day parquet）
          vs 同一組靜態濾網下的全 tick 流均勻抽樣（baseline）。
兩邊套同一組濾網（可當沖、day_amount_rank<=150、09:00:30-12:30、
TrialMatch==0、ToRef in (-1.5%, 5%)、|label|<10000），差異即條件本身的切割力。

指標（negFill 2025-02~2026-08 基準值見 SKILL.md §0.1）：
  A. 選擇率、樣本/日、cv
  B. 切割力 Δz（必須用 residual z 口徑；raw 口徑會把市場方向誤計成條件功勞）
     + 日級 t-stat、正 Δz 日比例、月度序列、市場搭車成分（Δhedge）
  C. 均勻度：對 tick 流的 TV 距離（時段 / ToRef / 聯合）

Usage:
    uv run python src/research/timeSeries/condition_benchmark.py \
        --start 20250211 --end 20260817 --step 3 --sample 25000
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from src.dataloader.txfDataLoader import TxfDataLoader, attach_txf_to_1330_return  # noqa: E402
from src.features.market_beta import add_txf_event_residual_vol  # noqa: E402

T_EDGES = [3300.0, 6900.0, 10500.0, 14100.0]
T_LABEL = {'1': '11:30-12:30', '2': '10:30-11:30', '3': '09:30-10:30', '4': '09:00-09:30'}
R_EDGES = [-0.005, 0.0, 0.01, 0.02, 0.03]
R_LABEL = {'0': '-1.5~-0.5', '1': '-0.5~0', '2': '0~1', '3': '1~2', '4': '2~3', '5': '3~5'}


def tick_size(p):
    return (pl.when(p < 10).then(0.01).when(p < 50).then(0.05).when(p < 100).then(0.1)
              .when(p < 500).then(0.5).when(p < 1000).then(1.0).otherwise(5.0))


def enrich(df):
    df = df.with_columns(
        pl.when(pl.col('FutureHigh') > pl.col('RefPrice') * 1.08)
          .then((pl.col('BidPrice1') - pl.col('RefPrice') * 1.08 - pl.col('TickSize') * 2) / pl.col('BidPrice1') * 1e4)
          .otherwise((pl.col('BidPrice1') - pl.col('Close')) / pl.col('BidPrice1') * 1e4)
          .alias('raw'),
        ((pl.lit(dt.time(13, 25)) - pl.col('TransTime').dt.time()).dt.total_seconds()).alias('sec1325'),
    ).filter(pl.col('raw').abs() < 10000)
    return df.with_columns(
        (pl.col('txf_beta_60d') * pl.col('txf_to_1330_bp')).alias('hedge'),
        (pl.col('txf_residual_vol_to_1330') * 1e4).alias('volbp'),
    ).with_columns(
        ((pl.col('raw') + pl.col('hedge')) / pl.col('volbp')).alias('z'),
        (pl.col('raw') + pl.col('hedge')).alias('resbp'),
        pl.col('sec1325').cut(T_EDGES, labels=[str(i) for i in range(5)]).alias('tb'),
        pl.col('ToRef2').cut(R_EDGES, labels=[str(i) for i in range(6)]).alias('rb'),
    )


def aggregate(df, side, date, n_pool):
    zok = df.filter(pl.col('z').is_finite())
    grid = df.group_by('tb', 'rb').len().to_dicts()
    return {
        'Date': date, 'side': side, 'n_pool': n_pool, 'n': len(df), 'n_z': len(zok),
        's_raw': df['raw'].sum(), 's_hedge': zok['hedge'].sum(), 's_res': zok['resbp'].sum(),
        's_z': zok['z'].sum(), 's_z2': (zok['z'] ** 2).sum(),
        'grid': {f"{g['tb']}_{g['rb']}": g['len'] for g in grid},
    }


def scan(events_dir, start, end, step, sample):
    loader = TxfDataLoader(PROJECT_ROOT / 'data/txfTickData')
    days = sorted(f.stem for f in Path(events_dir).glob('*.parquet') if start <= f.stem <= end)[::step]
    rows = []
    for i, d in enumerate(days):
        try:
            ev = pl.scan_parquet(Path(events_dir) / f'{d}.parquet').select(
                'TransTime', 'QuoteCode', 'BidPrice1', 'Close', 'FutureHigh', 'RefPrice', 'TickSize',
                'TrialMatch', 'day_amount_rank', 'txf_beta_60d', 'txf_to_1330_bp', 'txf_residual_vol_to_1330',
            ).filter(
                (pl.col('BidPrice1') > 0) & (pl.col('TrialMatch') == 0)
                & (pl.col('TransTime').dt.time() > dt.time(9, 0, 30))
                & (pl.col('TransTime').dt.time() < dt.time(12, 30))
                & (pl.col('day_amount_rank') <= 150)
            ).with_columns(((pl.col('BidPrice1') - pl.col('RefPrice')) / pl.col('RefPrice')).round(6).alias('ToRef2')
            ).filter((pl.col('ToRef2') > -0.015) & (pl.col('ToRef2') < 0.05)).collect()
            rows.append(aggregate(enrich(ev), 'condition', d, len(ev)))

            pm = pl.scan_parquet(PROJECT_ROOT / f'data/preMarket/{d}_preMarketData.parquet').select(
                'QuoteCode', 'day_amount_rank', 'nextday_allow_day_trade_mark',
                'txf_beta_60d', 'txf_residual_vol_0900_1330_60d')
            base = pl.scan_parquet(PROJECT_ROOT / f'data/tickData/{d}_StockTick.parquet').select(
                'TransTime', 'QuoteCode', 'BidPrice1', 'Close', 'FutureHigh', 'RefPrice', 'TrialMatch'
            ).filter(
                (pl.col('BidPrice1') > 0) & (pl.col('TrialMatch') == 0)
                & (pl.col('TransTime').dt.time() > dt.time(9, 0, 30))
                & (pl.col('TransTime').dt.time() < dt.time(12, 30))
            ).join(pm, on='QuoteCode', how='inner').filter(
                (pl.col('nextday_allow_day_trade_mark') == 'X') & (pl.col('day_amount_rank') <= 150)
            ).with_columns(((pl.col('BidPrice1') - pl.col('RefPrice')) / pl.col('RefPrice')).round(6).alias('ToRef2')
            ).filter((pl.col('ToRef2') > -0.015) & (pl.col('ToRef2') < 0.05)).collect()
            if base.is_empty():
                continue
            n_pool = len(base)
            base = base.sample(n=min(sample, n_pool), seed=int(d))
            base = base.with_columns(tick_size(pl.col('BidPrice1')).alias('TickSize')).sort('TransTime')
            base = add_txf_event_residual_vol(attach_txf_to_1330_return(base, d, loader))
            rows.append(aggregate(enrich(base), 'baseline', d, n_pool))
            if i % 20 == 0:
                print(f'[{i + 1}/{len(days)}] {d}', flush=True)
        except Exception as e:  # noqa: BLE001
            print(f'{d}: ERROR {e}', flush=True)
    return rows


def report(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    df['ym'] = df.Date.str[:6]
    piv = df.pivot_table(index='Date', columns='side', values='n_pool')
    sel = piv['condition'] / piv['baseline'] * 100
    nn = df[df.side == 'condition'].n_pool
    print(f"\n[A] 選擇率 median {sel.median():.2f}% (p10-p90 {sel.quantile(.1):.2f}~{sel.quantile(.9):.2f}%) | "
          f"樣本/日 mean {nn.mean():,.0f} cv {nn.std() / nn.mean():.2f}")

    def pooled(sub):
        zm = sub.s_z.sum() / sub.n_z.sum()
        return {'raw_bp': sub.s_raw.sum() / sub.n.sum(), 'res_bp': sub.s_res.sum() / sub.n_z.sum(),
                'z_mean': zm, 'z_std': float(np.sqrt(sub.s_z2.sum() / sub.n_z.sum() - zm ** 2)),
                'hedge_bp': sub.s_hedge.sum() / sub.n_z.sum()}
    c, b = pooled(df[df.side == 'condition']), pooled(df[df.side == 'baseline'])
    mz = df.pivot_table(index='Date', columns='side', values='s_z')
    nz = df.pivot_table(index='Date', columns='side', values='n_z')
    dz = (mz['condition'] / nz['condition']) - (mz['baseline'] / nz['baseline'])
    ymm = dz.groupby(df.groupby('Date').ym.first()).mean()
    print(f"[B] Δz {c['z_mean'] - b['z_mean']:+.3f}σ | Δres {c['res_bp'] - b['res_bp']:+.1f}bp | "
          f"Δraw {c['raw_bp'] - b['raw_bp']:+.1f}bp | 市場搭車Δhedge {c['hedge_bp'] - b['hedge_bp']:+.1f}bp")
    print(f"    日級 t={dz.mean() / dz.std() * np.sqrt(len(dz)):.1f} | 正Δz日 {(dz > 0).mean():.0%} | "
          f"正Δz月 {(ymm > 0).sum()}/{len(ymm)} | baseline z_std {b['z_std']:.2f} (應≈1)")
    print('    月度Δz:', {k: round(v, 3) for k, v in ymm.items()})

    def share(sub, axis):
        tot = {}
        for g in sub.grid:
            for k, v in g.items():
                tot[k] = tot.get(k, 0) + v
        s = pd.Series(tot)
        s = s.groupby([k.split('_')[axis] for k in s.index]).sum()
        return s / s.sum()
    for axis, lab, names in [(0, '時段', T_LABEL), (1, 'ToRef', R_LABEL)]:
        sc = share(df[df.side == 'condition'], axis)
        sb = share(df[df.side == 'baseline'], axis)
        tv = (sc.reindex(sb.index.union(sc.index), fill_value=0)
              - sb.reindex(sb.index.union(sc.index), fill_value=0)).abs().sum() / 2
        tbl = pd.DataFrame({'condition%': sc * 100, 'tick流%': sb * 100}).rename(index=names)
        print(f"[C] {lab} TV={tv:.3f}\n{tbl.round(1).to_string()}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--events-dir', default=str(PROJECT_ROOT / 'src/research/timeSeries/data'))
    ap.add_argument('--start', default='20250211')
    ap.add_argument('--end', default='20991231')
    ap.add_argument('--step', type=int, default=3)
    ap.add_argument('--sample', type=int, default=25000)
    args = ap.parse_args()
    report(scan(args.events_dir, args.start, args.end, args.step, args.sample))
