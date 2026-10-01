"""ETF 週內持有夜盤 + 台指期鎖倉。

情境：週換手 (回看 20 日、upcapture 指標、beta 名目) 選出的 ETF，週一開盤買進後整週持有到週五收盤，
週內 4 個夜盤 (d 收 13:30 → d+1 開 09:00) 用台指期空單鎖住，台指期在 13:30 中價放空、隔日 09:00 中價回補。
日盤 ETF 曝險維持跟前面一樣 (避險日盤空單)，本檔只評估「夜盤 ETF + 台指期」是否接近不賺不賠。

ETF 夜盤報酬 = 隔日開盤價 / 隔日參考價 − 1  (參考價已扣除息，等同含息報酬；分割亦同)
TXF 夜盤報酬 = 同一合約 09:00 中價(d+1) / 13:30 中價(d) − 1，合約取 d+1 早盤最活躍者，
             其 d 日 13:30 中價由逐日 tick 檔重算 (避免換月夜跨合約)。
口數 = round(名目 × β / (13:30 中價 × 200))，β 為過去 60 個夜盤 ETF 對 TXF 的 OLS 斜率。
"""
from __future__ import annotations
import sys
from datetime import datetime, time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_etf_hedge_analysis as R  # noqa: E402
import run_rotation as RO  # noqa: E402
from src.dataloader.txfDataLoader import _detect_price_scale  # noqa: E402

OUT = R.OUT
C = R.C
BETA_WINDOW = 60
MULT = 200.0            # 大台每點
TXF_COST_PER_LOT = 300.0  # 來回：期交稅 2×~96 + 手續費 2×~50
L, FREQ = 20, "weekly"


# --------------------------------------------------------------------------- #
def txf_contract_mids(dates) -> pd.DataFrame:
    """每日、每合約的 09:00 / 13:30 中價與早盤成交量。快取在 artifacts。"""
    cache = OUT / "txf_contract_mids.parquet"
    have = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
    done = set(have["date"].astype(str)) if not have.empty else set()
    tick_dir = R.load_pipeline_storage(project_root=R.PROJECT_ROOT).txf_tick_dir
    rows = []
    for i, d in enumerate(dates):
        ds = d.strftime("%Y%m%d")
        if ds in done:
            continue
        p = tick_dir / f"{ds}_TxfTick.parquet"
        if not p.exists():
            continue
        df = pl.read_parquet(p, columns=["QuoteCode", "TransTime", "TrialMatch", "BidPrice1", "AskPrice1", "FillLots"])
        df = df.filter(pl.col("QuoteCode").str.starts_with("TXF") & (pl.col("TrialMatch") == 0)).with_columns(pl.col("TransTime").dt.time().alias("t"))
        if df.is_empty():
            continue
        # 主力合約分數：08:45–09:00 成交口數 (在買賣價過濾之前算，同 loader)
        pre = df.filter((pl.col("t") >= time(8, 45)) & (pl.col("t") < time(9, 0))).group_by("QuoteCode").agg(
            pl.col("FillLots").cast(pl.Int64).fill_null(0).sum().alias("pre_lots"), pl.len().alias("pre_rows"))
        df = df.filter((pl.col("BidPrice1") > 0) & (pl.col("AskPrice1") > 0) & (pl.col("AskPrice1") >= pl.col("BidPrice1")))
        if df.is_empty():
            continue
        scale = _detect_price_scale(df)
        df = df.with_columns(((pl.col("BidPrice1") + pl.col("AskPrice1")).cast(pl.Float64) / 2 / scale).alias("mid"))
        o = df.filter(pl.col("t") >= time(9, 0)).sort("TransTime").group_by("QuoteCode").first().select("QuoteCode", pl.col("mid").alias("open_mid"), pl.col("TransTime").alias("open_t"))
        c = df.filter(pl.col("t") <= time(13, 30)).sort("TransTime").group_by("QuoteCode").last().select("QuoteCode", pl.col("mid").alias("close_mid"), pl.col("TransTime").alias("close_t"))
        m = o.join(c, on="QuoteCode", how="full", coalesce=True).join(pre, on="QuoteCode", how="left").to_pandas()
        m["date"] = ds
        rows.append(m)
        if (i + 1) % 40 == 0:
            print(f"[txf] {i+1}/{len(dates)}", flush=True)
    if rows:
        have = pd.concat([have, pd.concat(rows, ignore_index=True)], ignore_index=True)
        have.to_parquet(cache)
    have["date"] = pd.to_datetime(have["date"].astype(str))
    # 剔除離 09:00 / 13:30 超過 60 秒的
    day0 = have["date"]
    have.loc[(have["open_t"] - (day0 + pd.Timedelta(hours=9))).dt.total_seconds().abs() > 60, "open_mid"] = np.nan
    have.loc[((day0 + pd.Timedelta(hours=13, minutes=30)) - have["close_t"]).dt.total_seconds().abs() > 60, "close_mid"] = np.nan
    return have


def etf_overnight(dates) -> pd.DataFrame:
    """回傳 index=持有夜的起始日 d, columns=ETF, 值=open(d+1)/ref(d+1)−1 (bp)。只含同一週內連續交易日。"""
    market_dir = R.load_pipeline_storage(project_root=R.PROJECT_ROOT).market_dir
    fr = []
    for d in dates:
        p = market_dir / f"{d:%Y%m%d}_marketData.parquet"
        if p.exists() and "open_price" in pl.read_parquet_schema(p):
            fr.append(pl.read_parquet(p, columns=["date", "quote_code", "open_price", "close_price", "opening_ref_price"]).filter(pl.col("quote_code").is_in(RO.UNIVERSE)))
    df = pl.concat(fr).to_pandas(); df["date"] = pd.to_datetime(df["date"])
    op = df.pivot(index="date", columns="quote_code", values="open_price")
    ref = df.pivot(index="date", columns="quote_code", values="opening_ref_price")
    cl = df.pivot(index="date", columns="quote_code", values="close_price")
    idx = op.index
    nxt = pd.Series(idx[1:], index=idx[:-1])          # d → d+1 (資料裡的下一個交易日)
    ov = (op.shift(-1) / ref.shift(-1) - 1) * 1e4
    ov = ov.iloc[:-1]
    same_week = (idx[:-1].to_period("W-FRI") == idx[1:].to_period("W-FRI"))
    return ov, pd.Series(same_week, index=idx[:-1]), nxt, cl


def main():
    daily = R.build_daily_pnl()
    panel = RO.load_panel(daily.index)
    RO.METRIC = "upcapture"
    _, legs = RO.run(daily, panel, FREQ, L)
    legs = legs[(legs.sizing == "beta")]
    pick = legs[legs.tag == "best"].set_index("Date")[["code", "notional"]]
    pick0050 = legs[legs.tag == "0050"].set_index("Date")[["code", "notional"]]

    ov, same_week, nxt, cl = etf_overnight(daily.index)
    # TXF：夜盤 d→d+1 用 d+1 早盤最活躍合約
    all_dates = pd.DatetimeIndex(sorted(set(daily.index) | set(ov.index) | set(nxt.values)))
    tm = txf_contract_mids(all_dates)
    active = tm.sort_values(["date", "pre_lots", "pre_rows"], ascending=[True, False, False]).groupby("date").first()["QuoteCode"]
    key = tm.set_index(["date", "QuoteCode"])
    txf_rows = []
    for d, d1 in nxt.items():
        c1 = active.get(d1)
        if c1 is None:
            continue
        try:
            close_d = key.at[(d, c1), "close_mid"]
        except KeyError:
            close_d = np.nan
        open_d1 = key.at[(d1, c1), "open_mid"] if (d1, c1) in key.index else np.nan
        c0 = active.get(d)
        txf_rows.append({"d": d, "d1": d1, "contract": c1, "close_mid": close_d, "open_mid": open_d1,
                         "roll": (c0 is not None and c0 != c1)})
    txf = pd.DataFrame(txf_rows).set_index("d")
    txf["ret_bp"] = (txf["open_mid"] / txf["close_mid"] - 1) * 1e4
    print(f"[txf] 夜盤 {len(txf)} 個, 換月夜 {int(txf.roll.sum())} 個, 缺中價 {int(txf.ret_bp.isna().sum())} 個")

    # β：過去 60 個夜盤 (含跨週) ETF 對 TXF 的 OLS
    def beta_asof(code, d):
        hist = pd.concat([ov[code], txf["ret_bp"]], axis=1).dropna().loc[:d - pd.Timedelta(days=1)].tail(BETA_WINDOW)
        if len(hist) < 30:
            return np.nan, np.nan, len(hist)
        b, a = np.polyfit(hist.iloc[:, 1], hist.iloc[:, 0], 1)
        resid = hist.iloc[:, 0] - (a + b * hist.iloc[:, 1])
        return b, 1 - resid.var() / hist.iloc[:, 0].var(), len(hist)

    def build(pick_df, label):
        rows = []
        week_of = pick_df.index.to_period("W-FRI")
        for d in ov.index:
            if d not in pick_df.index or not same_week.get(d, False) or d not in txf.index:
                continue
            code, N = pick_df.at[d, "code"], pick_df.at[d, "notional"]
            wk_start = pick_df.index[week_of == d.to_period("W-FRI")].min()
            b, r2, n = beta_asof(code, wk_start)
            if pd.isna(b):
                b = 1.0
            e_ret, t = ov.at[d, code], txf.loc[d]
            if pd.isna(e_ret) or pd.isna(t["ret_bp"]):
                continue
            lots_f = N * b / (t["close_mid"] * MULT)
            lots = round(lots_f)
            etf_pnl = N * e_ret / 1e4
            txf_pnl_f = -lots_f * MULT * (t["open_mid"] - t["close_mid"])
            txf_pnl = -lots * MULT * (t["open_mid"] - t["close_mid"]) - lots * TXF_COST_PER_LOT
            rows.append({"d": d, "d1": t["d1"], "code": code, "notional": N, "beta": b, "beta_r2": r2, "lots_f": lots_f, "lots": lots,
                         "etf_bp": e_ret, "txf_bp": t["ret_bp"], "etf_pnl": etf_pnl, "txf_pnl_frac": txf_pnl_f, "txf_pnl": txf_pnl,
                         "net_frac": etf_pnl + txf_pnl_f, "net": etf_pnl + txf_pnl, "roll": t["roll"]})
        out = pd.DataFrame(rows).set_index("d")
        out.to_csv(OUT / f"overnight_txf_{label}.csv")
        return out

    res = build(pick, "best")
    res50 = build(pick0050, "0050")

    def stats(df):
        s = {}
        for k in ["etf_pnl", "txf_pnl", "net_frac", "net"]:
            s[k] = {"總和(萬)": df[k].sum() / 1e4, "每夜std(萬)": df[k].std() / 1e4, "最差夜(萬)": df[k].min() / 1e4, "最好夜(萬)": df[k].max() / 1e4}
        return pd.DataFrame(s).T.round(1)
    for label, df in [("最互補 ETF", res), ("固定 0050", res50)]:
        print(f"\n== {label}: {len(df)} 個夜盤, 平均名目 {df.notional.mean()/1e4:,.0f} 萬, 平均口數 {df.lots.mean():.1f}, β 平均 {df.beta.mean():.2f} (R² {df.beta_r2.mean():.2f}) ==")
        print(stats(df).to_string())
        print("  夜盤 ETF bp vs TXF bp corr:", round(df.etf_bp.corr(df.txf_bp), 2), " 殘差 std (bp, 名目基準):", round((df.net_frac / df.notional * 1e4).std(), 1))
        print("  各 ETF:", df.groupby("code").agg(n=("net", "size"), beta=("beta", "mean"), r2=("beta_r2", "mean"), etf=("etf_pnl", lambda s: s.sum() / 1e4), net=("net", lambda s: s.sum() / 1e4)).round(2).to_string().replace("\n", "\n        "))
        print("  換月夜 net 合計(萬):", round(df[df.roll].net.sum() / 1e4, 1), " 非換月夜:", round(df[~df.roll].net.sum() / 1e4, 1))

    # 對照：日盤避險腿
    day_leg = legs[legs.tag == "best"].set_index("Date")["leg"]
    day_leg50 = legs[legs.tag == "0050"].set_index("Date")["leg"]

    # ---- 圖 ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1.6]})
    ax = axes[0]
    for k, lab, col, ls in [("etf_pnl", "ETF 夜盤 (含息)", C["orange"], "-"), ("txf_pnl", "台指期空單 (整數口, 含成本)", C["aqua"], "-"),
                            ("net", "淨: ETF 夜盤 + 台指期", C["blue"], "-"), ("net_frac", "淨 (口數不取整、無成本)", C["blue"], "--")]:
        s = res[k].cumsum() / 1e4
        ax.plot(s.index, s, color=col, lw=2 if k == "net" else 1.5, ls=ls, label=lab)
    s = day_leg.loc[res.index.min():].cumsum() / 1e4
    ax.plot(s.index, s, color=C["muted"], lw=1.2, ls=":", label="對照: 同 ETF 日盤避險腿 (開買收賣)")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_title(f"最互補 ETF 週內持有夜盤 + 台指期鎖倉：累積損益 (萬)  {len(res)} 個夜盤, 平均名目 {res.notional.mean()/1e4:,.0f} 萬, 平均 {res.lots.mean():.0f} 口大台")
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[1]
    ax.bar(res.index, res["net"] / 1e4, color=np.where(res["net"] >= 0, C["blue"], C["orange"]), width=1.0)
    for d in res[res.roll].index:
        ax.axvline(d, color=C["violet"], lw=0.8, alpha=0.5)
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_title("每夜淨損益 (萬)  紫線=台指期換月夜")
    ax = axes[2]
    for k, lab, col in [("net", "最互補 ETF 淨", C["blue"]), ]:
        pass
    ax.plot(res50.index, res50["net"].cumsum() / 1e4, color=C["aqua"], lw=2, label="固定 0050: 淨 (夜盤+台指期)")
    ax.plot(res50.index, res50["etf_pnl"].cumsum() / 1e4, color=C["aqua"], lw=1.2, ls="--", label="固定 0050: ETF 夜盤未鎖")
    ax.plot(res.index, res["net"].cumsum() / 1e4, color=C["blue"], lw=2, label="最互補 ETF: 淨")
    ax.axhline(0, color=C["muted"], lw=0.8)
    ax.set_title("對照：固定 0050 版")
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    fig.tight_layout()
    fig.savefig(OUT / "overnight_fig1_txf_lock.png", dpi=130)
    plt.close(fig)

    # 散點：ETF 夜盤 bp vs TXF 夜盤 bp
    fig, ax = plt.subplots(figsize=(7, 6))
    for code, g in res.groupby("code"):
        ax.scatter(g.txf_bp, g.etf_bp, s=22, label=f"{code} (β={g.beta.mean():.2f})", alpha=0.8)
    lim = max(abs(res.txf_bp).max(), abs(res.etf_bp).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=C["muted"], lw=0.8, ls="--")
    ax.set_xlabel("TXF 夜盤 13:30→09:00 (bp)"); ax.set_ylabel("ETF 夜盤 收→隔日開 (bp, 含息)")
    ax.set_title("夜盤：被選 ETF vs 台指期"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "overnight_fig2_scatter.png", dpi=130); plt.close(fig)
    print("\n輸出:", OUT)


if __name__ == "__main__":
    main()
