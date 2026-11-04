"""negFill 空單「停手日」閘門：行情狀態合成 z（研究版，未接進 daily_pipeline）。

comp(D) = ( z(近5日全市場收漲停比例) + z(近5日開高股 09:04→收盤均報酬) − z(近5日 negFill 觸發商品日均 return) ) / 3
三條序列都只用 D−5..D−1；z 用擴張視窗（≥120 天、到 D−1 為止）。comp(D) < THR → D 日整天不開新空單。

用法（收盤後、timeSeries data/ 當日點位產出後執行）：
  uv run python src/research/timeSeries/test/20261005_dd_market_state_gate/market_state_gate.py --rebuild           # 2023-01-03 起全部重算
  uv run python src/research/timeSeries/test/20261005_dd_market_state_gate/market_state_gate.py --update            # 補到最新一天並印出下一交易日判定
  uv run python src/research/timeSeries/test/20261005_dd_market_state_gate/market_state_gate.py --show 20260901     # 看某日起的逐日 comp

輸出：artifacts/state_daily.parquet（逐日原始三序列）、artifacts/gate_daily.csv（逐日 z、comp、stop）。
資料來源：marketData（參考價/漲停價/股票類別）、NAS Klines stock_round（09:04 價、收盤、最高、量）、timeSeries data/（negFill 點位）。
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[4]
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline_storage import load_pipeline_storage  # noqa: E402

STORAGE = load_pipeline_storage(project_root=PROJECT_ROOT)
MARKET_DIR = STORAGE.market_dir
KLINE_ROOT = Path("/mnt/NAS/Parquet/Klines")
POINT_DIR = PROJECT_ROOT / "src/research/timeSeries/data"
ART = HERE / "artifacts"
STATE_PATH = ART / "state_daily.parquet"
GATE_PATH = ART / "gate_daily.csv"

WINDOW = 5          # 近 N 日
MIN_HIST = 120      # z 需要的最少歷史天數
THR = -0.7          # comp 低於此值停手
ACTIVE_LOTS = 500   # 「活躍股」門檻（張）


# ----------------------------------------------------------------------------- 單日三序列
def market_state_for_day(day: dt.date) -> dict | None:
    """全市場收漲停比例、開高股 09:04→收盤均報酬（活躍普通股）。"""
    mf = MARKET_DIR / f"{day:%Y%m%d}_marketData.parquet"
    kf = KLINE_ROOT / f"{day:%Y}/{day:%m}/{day:%d}/stock_round.parquet"
    if not mf.exists() or not kf.exists():
        return None
    md = (pl.read_parquet(mf, columns=["quote_code", "opening_ref_price", "limit_up_price", "ins_type"])
            .rename({"quote_code": "stock_id"})
            .with_columns(pl.col("stock_id").cast(pl.Utf8),
                          pl.col("opening_ref_price").cast(pl.Float64).alias("ref"),
                          pl.col("limit_up_price").cast(pl.Float64).alias("lu"))
            .filter(pl.col("stock_id").str.contains(r"^[1-9]\d{3}$") & (pl.col("ins_type") == "stock") & (pl.col("ref") > 0)))
    kk = (pl.scan_parquet(kf).sort("stock_id", "minute").group_by("stock_id")
            .agg(pe=pl.col("close").filter(pl.col("minute") <= "09:04:00").last(),
                 cl=pl.col("close").last(), vol=pl.col("volume").sum()).collect())
    j = (md.join(kk, on="stock_id", how="inner").filter(pl.col("vol") >= ACTIVE_LOTS)
           .with_columns(tr=pl.col("pe") / pl.col("ref") - 1, oc=pl.col("cl") / pl.col("pe") - 1,
                         close_lu=pl.col("cl") >= pl.col("lu") - 1e-9))
    if j.height == 0:
        return None
    oh = j.filter(pl.col("tr") > 0.005)["oc"].drop_nulls()
    return {"date": day, "n_active": j.height,
            "lu_close": float(j["close_lu"].mean()),
            "oh_oc": float(oh.mean()) if oh.len() else np.nan}


def negfill_for_day(day: dt.date) -> dict | None:
    """negFill 點位（timeSeries data/）：每商品日第一個觸發點持有到收盤的 TakerSell_CloseBP，取全部商品平均。
    篩選與 notebook cell 2/6 一致：BidPrice1>0、ToRef∈(−1.5%,5%)、成交額排名≤150、09:00:30<t<12:30、TrialMatch==0、|bp|<10000。"""
    f = POINT_DIR / f"{day:%Y%m%d}.parquet"
    if not f.exists():
        return None
    need = ["QuoteCode", "TransTime", "TrialMatch", "BidPrice1", "RefPrice", "Close", "FutureHigh", "TickSize",
            "day_amount_rank", "AmountRank_canDayTrade"]
    avail = pl.scan_parquet(f).collect_schema().names()
    if any(c not in avail for c in need):
        return None
    b1 = pl.col("BidPrice1")
    lf = (pl.scan_parquet(f).select(need).filter(b1 > 0)
            .with_columns(ToRef=((b1 - pl.col("RefPrice")) / pl.col("RefPrice")).round(6))
            .with_columns(bp=pl.when(pl.col("FutureHigh") > pl.col("RefPrice") * 1.08)
                               .then((b1 - pl.col("RefPrice") * 1.08 - pl.col("TickSize") * 2) / b1 * 1e4)
                               .otherwise((b1 - pl.col("Close")) / b1 * 1e4))
            .filter((pl.col("ToRef") > -0.015) & (pl.col("ToRef") < 0.05)
                    & ((pl.col("day_amount_rank") <= 150) | (pl.col("AmountRank_canDayTrade") <= 150))
                    & (pl.col("TrialMatch") == 0) & (pl.col("bp").abs() < 10000)
                    & (pl.col("TransTime").dt.time() > dt.time(9, 0, 30)) & (pl.col("TransTime").dt.time() < dt.time(12, 30)))
            .sort("QuoteCode", "TransTime").group_by("QuoteCode").agg(pl.col("bp").first()))
    df = lf.collect()
    if df.height == 0:
        return None
    return {"date": day, "nf_n": df.height, "nf_first_bp": float(df["bp"].mean())}


def trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    out = []
    for f in sorted(MARKET_DIR.glob("*_marketData.parquet")):
        d = dt.datetime.strptime(f.name[:8], "%Y%m%d").date()
        if start <= d <= end:
            out.append(d)
    return out


# ----------------------------------------------------------------------------- 歷史表維護
def build_state(days: list[dt.date], existing: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for d in days:
        m = market_state_for_day(d)
        n = negfill_for_day(d)
        if m is None and n is None:
            continue
        rows.append({**({"date": d} | (m or {})), **(n or {})})
    new = pd.DataFrame(rows)
    if new.empty:
        return existing if existing is not None else new
    new["date"] = pd.to_datetime(new["date"])
    if existing is not None and not existing.empty:
        new = pd.concat([existing[~existing["date"].isin(new["date"])], new], ignore_index=True)
    return new.sort_values("date").reset_index(drop=True)


def compute_gate(state: pd.DataFrame, window: int = WINDOW, thr: float = THR, min_hist: int = MIN_HIST) -> pd.DataFrame:
    """回傳逐日 z、comp、stop。stop(D) 用的是 D−window..D−1 的資料，所以 D 收盤前就能算。"""
    s = state.set_index("date").sort_index()
    out = pd.DataFrame(index=s.index)

    def zexp(x: pd.Series) -> pd.Series:
        r = x.rolling(window, min_periods=max(2, window // 2)).mean().shift(1)
        mu = r.expanding(min_hist).mean().shift(1)
        sd = r.expanding(min_hist).std().shift(1)
        return (r - mu) / sd

    out["z_lu"] = zexp(s["lu_close"])
    out["z_ohoc"] = zexp(s["oh_oc"])
    out["z_nf"] = zexp(s["nf_first_bp"])
    out["comp"] = (out["z_lu"] + out["z_ohoc"] - out["z_nf"]) / 3
    out["stop"] = out["comp"] < thr
    # 下一交易日的判定：把 shift 拿掉再算一次（= 用到今天為止的資料），只取最後一列
    r = {c: s[c].rolling(window, min_periods=max(2, window // 2)).mean() for c in ("lu_close", "oh_oc", "nf_first_bp")}
    nxt = {}
    for k, c in (("z_lu", "lu_close"), ("z_ohoc", "oh_oc"), ("z_nf", "nf_first_bp")):
        hist = r[c].shift(1)  # 歷史分佈仍只用到 D−1 的 rolling 值，與逐日版一致
        nxt[k] = float((r[c].iloc[-1] - hist.expanding(min_hist).mean().iloc[-1]) / hist.expanding(min_hist).std().iloc[-1])
    nxt["comp"] = (nxt["z_lu"] + nxt["z_ohoc"] - nxt["z_nf"]) / 3
    nxt["stop"] = nxt["comp"] < thr
    out.attrs["next"] = nxt
    out.attrs["last_date"] = s.index[-1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="從 2023-01-03 重算全部歷史")
    ap.add_argument("--update", action="store_true", help="只補 state 表沒有的日期")
    ap.add_argument("--start", default="20230103")
    ap.add_argument("--end", default=dt.date.today().strftime("%Y%m%d"))
    ap.add_argument("--show", default=None, help="YYYYMMDD：印出此日起逐日 comp")
    ap.add_argument("--thr", type=float, default=THR)
    a = ap.parse_args()
    ART.mkdir(exist_ok=True)
    existing = pd.read_parquet(STATE_PATH) if STATE_PATH.exists() and not a.rebuild else None
    start = dt.datetime.strptime(a.start, "%Y%m%d").date()
    end = dt.datetime.strptime(a.end, "%Y%m%d").date()
    if a.rebuild or a.update or existing is None:
        days = trading_days(start, end)
        if existing is not None:
            have = set(pd.to_datetime(existing["date"]).dt.date)
            days = [d for d in days if d not in have]
        state = build_state(days, existing)
        state.to_parquet(STATE_PATH, index=False)
    else:
        state = existing
    gate = compute_gate(state, thr=a.thr)
    gate.to_csv(GATE_PATH)
    last = gate.attrs["last_date"].date()
    nxt = gate.attrs["next"]
    print(f"state 表：{state['date'].min().date()} → {last}，{len(state)} 天")
    if a.show:
        print(gate.loc[a.show:].round(2).to_string())
    print(f"\n最新一天 {last} 的 comp（用到 {last} 前一日）= {gate['comp'].iloc[-1]:.2f}  stop={bool(gate['stop'].iloc[-1])}")
    print(f"下一交易日判定（用到 {last} 收盤）：z_lu {nxt['z_lu']:+.2f}  z_ohoc {nxt['z_ohoc']:+.2f}  z_nf {nxt['z_nf']:+.2f}  → comp {nxt['comp']:+.2f}  "
          f"{'【停手】' if nxt['stop'] else '交易'}（門檻 {a.thr}）")


if __name__ == "__main__":
    main()
