"""negFill notebook replica (cells 12-34): two rolling Ridge models, 16-cell prior thresholds, gates, tp/sl, BackTest.

Compares feature sets (baseline 14 vs baseline + vwap additions). Study-local outputs under artifacts/<tag>/.
Usage: uv run python src/research/timeSeries/test/20260924_vwap_feature/run_model_backtest.py \
          --signal artifacts/signal_20260101_20260923.parquet --set baseline --set vwap_a --eval_start 20260401
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))
from src.research.backTest import BackTest  # noqa: E402
from src.research.rolling_model import train_and_predict_ridge_rolling  # noqa: E402
from src.research.tool import first_touch_times  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
BASE14 = ["ToLow", "ToHigh", "Low_High", "ToRef", "ToOpen", "B1_A1B1", "B1_B1B5", "A1_A1A5",
          "B45_AB45", "B1_B12", "A1_A12", "RemainSeconds", "MD_ElaspeTime_30_re", "MD_L1Rate_30"]
FEATURE_SETS: dict[str, list[str]] = {
    "baseline": BASE14,
}


def register_sets(extra: dict[str, list[str]]) -> None:
    for k, v in extra.items():
        FEATURE_SETS[k] = BASE14 + [c for c in v if c not in BASE14]


def prior_day_quantile(df, col, q=0.9, n_days=3, date_col="Date", min_days=1, group=None):
    """notebook cell 20: per-day threshold = quantile of the prior n_days (strictly before) within group."""
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


def daily_ic(df: pd.DataFrame, pred: str, label: str) -> pd.Series:
    return df.groupby("Date").apply(lambda p: p[pred].corr(p[label], method="spearman"))


def run_set(signal_df: pd.DataFrame, feature_columns: list[str], tag: str, out_dir: Path, *, n_a: int, n_m: int,
            alpha: float, eval_start: str, std_span: float = 2.2, pos_limit: float = 500, cap: float = 10000) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = signal_df.copy()
    for c in feature_columns:
        df[c] = df[c].astype(float).round(6)
    res = train_and_predict_ridge_rolling(df, feature_columns, "TakerSell_CloseBP_net", n_training_days=n_a, day_features=[],
                                          feature_screening=True, alpha=alpha, max_position_per_stock=500)
    df = res["df"]
    coeff_a = res["coeff_log"]
    res = train_and_predict_ridge_rolling(df, feature_columns, "TakerSell_CloseBP_netM", n_training_days=n_m, day_features=[],
                                          feature_screening=True, alpha=alpha, max_position_per_stock=500)
    df = res["df"]
    coeff_m = res["coeff_log"]

    ic = pd.DataFrame({
        "net|net": daily_ic(df, "TakerSell_CloseBP_net_pred", "TakerSell_CloseBP_net"),
        "netM|netM": daily_ic(df, "TakerSell_CloseBP_netM_pred", "TakerSell_CloseBP_netM"),
    })
    ic_month = ic.groupby(pd.to_datetime(ic.index).strftime("%Y%m")).agg(["mean", "count"])
    ic.to_csv(out_dir / "model_daily_ic.csv")
    ic_month.to_csv(out_dir / "model_monthly_ic.csv")

    # cells 21-22: 16-cell prior-3-day thresholds
    df["TimeBucket"] = pd.cut(df.RemainSeconds, [3300, 6900, 10500, 14100, 99999], labels=False)
    df["ToRefBucket"] = pd.cut(df.ToRef, [0, 0.01, 0.02, 0.03, 0.05], labels=False)
    df["Cell2D"] = df["TimeBucket"] * 10 + df["ToRefBucket"]
    df["thr_net"] = prior_day_quantile(df, "TakerSell_CloseBP_net_pred", q=0.3, n_days=3, group="Cell2D")
    df["thr_netM"] = prior_day_quantile(df, "TakerSell_CloseBP_netM_pred", q=0.1, n_days=3, group="Cell2D")
    condition = (df.TakerSell_CloseBP_net_pred > df.thr_net) & (df.TakerSell_CloseBP_netM_pred > df.thr_netM)

    # cell 23 static gates
    model = df[condition & ((df.OTC == 1) | (df.MD_L1Rate_30_re > 0.2)) & (df.SpreadPairElapsed > 0.1) & (df.ToRef > 0.0)
               & ((df.AmountRank_canDayTrade <= 100) | (df.day_amount_rank <= 100))].sort_values(["Date", "QuoteCode", "TransTime"]).reset_index(drop=True)
    # cell 25
    model["Position"] = model.groupby(["Date", "QuoteCode"]).BidPrice1.transform("cumsum") / 10
    model["accLots"] = model.groupby(["Date", "QuoteCode"]).BidPrice1.transform("cumcount") + 1
    # cell 30 masks
    r1_mask = ((model["d_intraday_return"] > 0.04) & (model["d_intraday_return"] <= 0.1) & (model["dt_major_count"] <= 0)) | (model.MarketCap >= 8 * (10 ** 11))
    brench_mask = ((model["sell_branch_breadth_rank"] >= 0.98) | (model["net_buy_gini"] >= 0.92)
                   | (model["top1_net_buy_amount"] / model["trading_turnover_bf"] > 0.1))
    r1_mask = r1_mask.fillna(False).astype(bool)
    brench_mask = brench_mask.fillna(False).astype(bool)
    swr = model[(model.accLots < model.avg_askLots1 + model.avg_bidLots1) & (model.Position < pos_limit) & (model.RefPrice <= 3000)
                & ~r1_mask & ~brench_mask & (model.RemainSeconds >= 3600 + 60 * 25)
                & ((model.hft_strick_makerSpreadBP > -70) | (model.hft_strick_makerSpreadBP.isna()))
                & ((model.OTC == 1) | (model.MD_L1Rate_30_re > 0.2))].reset_index(drop=True)
    swr["Date"] = pd.to_datetime(swr.Date).dt.strftime("%Y%m%d")
    vol = "txf_residual_vol_0900_1330_60d"
    swr["tpPrice"] = swr["BidPrice1"] * (1 - swr[vol] * std_span)
    swr["slPrice"] = np.minimum(swr["BidPrice1"] * (1 + swr[vol] * std_span), swr.RefPrice * 1.08)
    t = first_touch_times(swr, low_col="tpPrice", high_col="slPrice")
    tp_first = t.low_time.notna() & (t.high_time.isna() | (t.low_time < t.high_time))
    sl_first = t.high_time.notna() & ~tp_first
    swr.loc[tp_first, "TakerSell_CloseBP"] = (swr.BidPrice1 - swr.tpPrice) / swr.BidPrice1 * 10000
    swr.loc[sl_first, "TakerSell_CloseBP"] = (swr.BidPrice1 - swr.slPrice - swr.TickSize * 2) / swr.BidPrice1 * 10000
    swr = swr.sort_values(["Date", "TransTime"]).reset_index(drop=True)
    swr = swr[(swr.groupby("Date").BidPrice1.cumsum() / 10 < cap)].reset_index(drop=True)
    swr = swr[swr.Date.astype(int) >= int(eval_start)].reset_index(drop=True)
    swr.to_parquet(out_dir / "signal_with_risk.parquet", index=False)

    bt = BackTest(swr)
    stats, daily = bt.run_analysis(return_daily=True, plot=False)
    daily = daily.reset_index()
    daily.to_csv(out_dir / "daily_pnl.csv", index=False)
    pnl = daily["PnL"] / 10000.0
    cum = pnl.cumsum()
    mdd = float((cum - cum.cummax()).min())
    summary = {
        "tag": tag, "n_features": len(feature_columns), "eval_start": eval_start, "days": int(len(daily)),
        "trades": int(len(swr)), "PnL_10k": float(pnl.sum()), "DailyVol_10k": float(pnl.std()),
        "Sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else None, "MDD_10k": mdd,
        "win_days": float((pnl > 0).mean()), "avg_bp": float(swr.TakerSell_CloseBP.mean()),
        "avg_products_per_day": float(swr.groupby("Date").QuoteCode.nunique().mean()),
        "ic_net_mean": float(ic["net|net"].mean()), "ic_netM_mean": float(ic["netM|netM"].mean()),
    }
    monthly = daily.assign(m=pd.to_datetime(daily.Date).dt.strftime("%Y%m")).groupby("m").agg(PnL_10k=("PnL", lambda s: s.sum() / 1e4), days=("PnL", "size"))
    monthly.to_csv(out_dir / "monthly_pnl.csv")
    summary["monthly_PnL_10k"] = {k: round(float(v), 1) for k, v in monthly["PnL_10k"].items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    # last-day coefficient snapshot for the new features
    for name, log in [("A", coeff_a), ("M", coeff_m)]:
        try:
            last = sorted(log.keys())[-1]
            (out_dir / f"coeff_last_{name}.json").write_text(json.dumps({str(k): (v if not hasattr(v, "tolist") else v.tolist()) for k, v in log[last].items()}, default=str, indent=1))
        except Exception:
            pass
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", action="append", required=True, help="signal parquet(s); several are concatenated")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--extra", default="", help="json {set_name: [cols]} to register additional feature sets")
    ap.add_argument("--eval_start", default="20260401")
    ap.add_argument("--n_a", type=int, default=60)
    ap.add_argument("--n_m", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.extra:
        register_sets(json.loads(a.extra))
    signal_df = pd.concat([pd.read_parquet(p) for p in a.signal], ignore_index=True)
    signal_df = signal_df.drop_duplicates(["Date", "QuoteCode", "ChannelSeq"]).sort_values(["Date", "QuoteCode", "TransTime"]).reset_index(drop=True)
    print("signal rows", len(signal_df), "days", signal_df.Date.nunique(), signal_df.Date.min(), signal_df.Date.max())
    out_root = ART / (a.out or "runs")
    results = []
    for s in a.set or ["baseline"]:
        results.append(run_set(signal_df, FEATURE_SETS[s], s, out_root / s, n_a=a.n_a, n_m=a.n_m, alpha=a.alpha, eval_start=a.eval_start))
    pd.DataFrame(results).drop(columns=["monthly_PnL_10k"]).to_csv(out_root / "compare.csv", index=False)
    print(pd.DataFrame(results).drop(columns=["monthly_PnL_10k"]).to_string())


if __name__ == "__main__":
    main()
