"""Rebuild the timeSeries negFill selection book (notebook time_series_research_model.ipynb cells 2-27) over 2023-.

Three overlapping chunks; each keeps only dates with >= WARMUP prior trading days inside the chunk (60-day A model,
20-day M model, 3-day thresholds).  Output: <out_dir>/book_chunk{i}.parquet = signal_df_model rows (cell 27) for the kept dates.
Usage: uv run python build_book.py <out_dir>
"""
import sys, os, datetime, gc
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/kevin/Project/HFT"); sys.path.append(str(ROOT))
DATA = ROOT / "src/research/timeSeries/data"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
WARMUP = 65
FEATURES = ['ToLow', 'ToHigh', 'Low_High', 'ToRef', 'ToOpen', 'B1_A1B1', 'B1_B1B5', 'A1_A1A5', 'B45_AB45', 'B1_B12', 'A1_A12', 'RemainSeconds',
            'MD_ElaspeTime_30_re', 'MD_L1Rate_30']
LABEL = 'TakerSell_CloseBP_net'


def prior_day_quantile(df, col, q=0.9, n_days=3, date_col="Date", min_days=1, group=None):   # cell 23, verbatim
    if group is not None:
        out = pd.Series(np.nan, index=df.index)
        for g, part in df.groupby(group, observed=True):
            out.loc[part.index] = prior_day_quantile(part, col, q=q, n_days=n_days, date_col=date_col, min_days=min_days)
        return out
    vals = df.dropna(subset=[col]).groupby(date_col, observed=True)[col].apply(lambda s: s.to_numpy())
    dates = np.sort(vals.index.to_numpy())
    thr = {}
    for i, d in enumerate(dates):
        window = dates[max(0, i - n_days):i]
        if len(window) < min_days:
            continue
        pool = np.concatenate([vals[w] for w in window])
        if pool.size:
            thr[d] = np.quantile(pool, q)
    return df[date_col].map(thr)


def load_day(f):   # cell 2
    s = pd.read_parquet(DATA / f)[lambda x: x['BidPrice1'] * x.BidPrice1 > 0]
    s['ToRef'] = ((s['BidPrice1'] - s.RefPrice) / s.RefPrice).round(6)
    s['ToOpen'] = ((s['BidPrice1'] - s.Open) / s.Open).round(6)
    s['TakerSell_CloseBP'] = (s['BidPrice1'] - s.Close) / s['BidPrice1'] * 10000
    s.loc[s.FutureHigh > s.RefPrice * 1.08, 'TakerSell_CloseBP'] = (s['BidPrice1'] - s.RefPrice * 1.08 - s.TickSize * 2) / s['BidPrice1'] * 10000
    t = s[((s.ToRef > -0.015) & (s.ToRef < 0.05)) & ((s.day_amount_rank <= 150) | (s.AmountRank_canDayTrade <= 150))].reset_index(drop=True)
    t['Date'] = f.split('.')[0]
    return t


def build(files, keep_from):
    from src.research.rolling_model import train_and_predict_ridge_rolling
    parts = []
    for f in files:
        try:
            parts.append(load_day(f))
        except Exception as e:   # the notebook's try/except: skip files missing columns
            print(f"  skip {f}: {type(e).__name__}: {e}", flush=True)
    df = pd.concat(parts, ignore_index=True, copy=False); del parts                                    # cell 3
    df['residual_TakerSell_CloseBP'] = (df.TakerSell_CloseBP + df.txf_beta_60d * df.txf_to_1330_bp) / (df.txf_residual_vol_to_1330 * 1e4)   # cell 4
    df = df[(df.TransTime.dt.time < datetime.time(12, 30, 0)) & (df.TransTime.dt.time > datetime.time(9, 0, 30)) & (df.TrialMatch == 0)].reset_index(drop=True)   # cell 6
    df = df[(df.TakerSell_CloseBP.abs() < 10000)].sort_values(['Date', 'QuoteCode', 'TransTime']).reset_index(drop=True)
    df['TakerSell_CloseBP_net'] = df.residual_TakerSell_CloseBP - df.groupby(['Date', 'QuoteCode']).residual_TakerSell_CloseBP.transform('mean')   # cell 8
    df['TakerSell_CloseBP_netM'] = df.residual_TakerSell_CloseBP - df.groupby(['Date']).residual_TakerSell_CloseBP.transform('mean')
    df = df.drop(columns=['isAbnormalDate'], errors='ignore')                                          # cell 10
    df['B1_B12'] = df.BidLots1 / (df.BidLots1 + df.BidLots2)                                           # cell 11 (features used)
    df['A1_A12'] = df.AskLots1 / (df.AskLots1 + df.AskLots2)
    df['B45_AB45'] = (df.BidLots4 + df.BidLots5) / (df.AskLots4 + df.AskLots5 + df.BidLots4 + df.BidLots5)
    df['TickBP'] = df.TickSize / df.BidPrice1 * 10000
    df['MD_ElaspeTime_30_re'] = (df.MD_ElaspeTime_30 + 1).apply(np.log)
    df['MD_L1Rate_30_re'] = df.MD_L1Rate_30
    for c in FEATURES:                                                                                # cell 13
        df[c] = df[c].round(6)
    df['OTC'] = df['OTC'] * 1
    res = train_and_predict_ridge_rolling(df[:], FEATURES, LABEL, n_training_days=60, day_features=[], feature_screening=True, alpha=10, max_position_per_stock=500)   # cell 14
    df = res['df']; del res; gc.collect()
    res = train_and_predict_ridge_rolling(df[:], FEATURES, LABEL + 'M', n_training_days=20, day_features=[], feature_screening=True, alpha=10, max_position_per_stock=500)
    df = res['df'].reset_index(drop=True); del res; gc.collect()
    df['TimeBucket'] = pd.cut(df.RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False)     # cell 24
    df['ToRefBucket'] = pd.cut(df.ToRef, [0, 0.01, 0.02, 0.03, 0.05], labels=False)
    df['Cell2D'] = df['TimeBucket'] * 10 + df['ToRefBucket']
    df['thr_net'] = prior_day_quantile(df, 'TakerSell_CloseBP_net_pred', q=0.3, n_days=3, group='Cell2D')
    df['thr_netM'] = prior_day_quantile(df, 'TakerSell_CloseBP_netM_pred', q=0.1, n_days=3, group='Cell2D')
    cond = (df.TakerSell_CloseBP_net_pred > df.thr_net) & (df.TakerSell_CloseBP_netM_pred > df.thr_netM)   # cell 25
    model = df[cond & ((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.2)) & (df.SpreadPairElapsed > 0.1) & (df.ToRef > 0.0)
               & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100))].sort_values(['Date', 'QuoteCode', 'TransTime']).reset_index(drop=True)   # cell 27
    model = model[pd.to_datetime(model.Date) >= pd.to_datetime(keep_from)].reset_index(drop=True)
    return model


if __name__ == "__main__":
    files = np.array(sorted(f for f in os.listdir(DATA) if f.endswith('.parquet') and f[:8].isdigit() and (DATA / f).is_file()))
    n = len(files)
    bounds = [(0, 365, WARMUP), (300, 665, 365), (600, n, 665)]
    for i, (a, b, k) in enumerate(bounds):
        out = OUT / f"book_chunk{i}.parquet"
        if out.exists():
            print(f"chunk {i} exists, skip", flush=True); continue
        keep_from = files[k][:8]
        print(f"chunk {i}: files {files[a][:8]}..{files[b - 1][:8]} ({b - a} days) keep from {keep_from}", flush=True)
        m = build(list(files[a:b]), keep_from)
        m.to_parquet(out, index=False)
        print(f"chunk {i}: kept rows {len(m):,} dates {m.Date.nunique()} ({m.Date.min()}..{m.Date.max()})", flush=True)
        del m; gc.collect()
