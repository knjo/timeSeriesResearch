"""400 × 7,000 壓力測試：參數擾動（comp 門檻/縮額、台積電門檻/比例、額外成本）× 日損益 stationary block bootstrap 一年路徑。
每次抽一組參數 → 該組逐日損益（cap × scale 真實重放）→ 抽 242 天路徑（平均區塊 10 天）→ 年獲利、一年內最大回撤、最差單日。
"""
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append("/home/kevin/Project/HFT")
from src.strategy.negFill.stopTradingGate import scale_for_comp, apply_cooldown
from src.research.tool import add_stop_loss_prod

OUT = Path(__file__).resolve().parent
INP = OUT / "inputs"   # 11/03 快照：build_rows.py 400 keep_stop 成交列、停損時間 cache、book 交易日曆、gate_daily.csv
FEE, CAP, DPY, N_PATHS, BLOCK = 19.3, 7000, 242, 20000, 10
rng = np.random.default_rng(20261103)

sw = pd.read_parquet(INP / "rows_pos400_keepstop.parquet")
sw["stopLossProd"] = add_stop_loss_prod(sw, cache_path=INP / "stop_times_cache.parquet")
sw = sw.sort_values(["Date", "TransTime"]).reset_index(drop=True)
sw = sw[~((sw.stopLossProd >= 3) & (sw.RemainSeconds < 12300))].reset_index(drop=True)
sw["cum"] = sw.groupby("Date").BidPrice1.cumsum() / 10
sw["gross"] = sw.BidPrice1 * sw.TakerSell_CloseBP / 10 / 1e4; sw["notional"] = sw.BidPrice1 / 10
cal = pd.read_csv(INP / "calendar.csv", dtype=str).Date.tolist()
ci = {d: i for i, d in enumerate(cal)}

SC = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]); SIDX = {float(s): i for i, s in enumerate(SC)}
G = np.zeros((len(cal), len(SC))); V = np.zeros_like(G)          # 萬：毛損益、名目
for d, g in sw.groupby("Date"):
    cum, cg, cv = g.cum.to_numpy(), np.cumsum(g.gross.to_numpy()), np.cumsum(g.notional.to_numpy())
    n = np.searchsorted(cum, CAP * SC, side="left")
    G[ci[d]] = np.where(n > 0, cg[np.maximum(n - 1, 0)], 0.0); V[ci[d]] = np.where(n > 0, cv[np.maximum(n - 1, 0)], 0.0)

gd = pd.read_csv(INP / "gate_daily_20261103.csv", dtype={"asof_date": str, "next_trade_day": str}, float_precision="round_trip").set_index("next_trade_day")
gate_days, comp_all = gd.index.tolist(), gd.comp.to_numpy(); gap = gd.tsmc_gap.reindex(cal).to_numpy()

GRID = {"stop_thr": [-1.8, -1.5, -1.2], "reduce_thr": [-0.9, -0.7, -0.5], "reduce_scale": [0.05, 0.1, 0.2],
        "tsmc_thr": [0.0075, 0.01, 0.0125], "tsmc_scale": [0.3, 0.5, 0.7], "extra_cost_bp": [0.0, 2.0, 4.0]}
combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
series, det = [], []
for c in combos:
    base = [scale_for_comp(x, c["stop_thr"], c["reduce_thr"], c["reduce_scale"]) for x in comp_all]
    cs = pd.Series(apply_cooldown(base, c["reduce_scale"], 2)[0], index=gate_days).reindex(cal).fillna(1.0).to_numpy()
    sc = np.minimum(cs, np.where(gap > c["tsmc_thr"], c["tsmc_scale"], 1.0))
    ix = np.array([SIDX[round(float(x), 2)] for x in sc])
    p = G[np.arange(len(cal)), ix] - (FEE + c["extra_cost_bp"]) / 1e4 * V[np.arange(len(cal)), ix]
    series.append(p); cc = np.cumsum(p)
    det.append({**c, "total": p.sum(), "annual": p.mean() * DPY, "mdd": (cc - np.maximum.accumulate(cc)).min(), "worst": p.min()})
series = np.array(series); det = pd.DataFrame(det)
base_i = det.index[(det.stop_thr == -1.5) & (det.reduce_thr == -0.7) & (det.reduce_scale == 0.1) & (det.tsmc_thr == 0.01) & (det.tsmc_scale == 0.5) & (det.extra_cost_bp == 0)][0]
print("base combo (report version):", det.loc[base_i, ["total", "annual", "mdd", "worst"]].round(1).to_dict())
print("\n729 組全期（不抽樣）分位：")
print(det[["annual", "mdd", "worst"]].quantile([0, 0.05, 0.5, 0.95, 1]).round(0).to_string())

# stationary bootstrap, 1-year paths
T = len(cal)
pick = rng.integers(0, len(combos), N_PATHS)
idx = np.empty((N_PATHS, DPY), dtype=np.int64); idx[:, 0] = rng.integers(0, T, N_PATHS)
jump = rng.random((N_PATHS, DPY)) < 1 / BLOCK; fresh = rng.integers(0, T, (N_PATHS, DPY))
for t in range(1, DPY):
    idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % T)
paths = series[pick[:, None], idx]
cum = np.cumsum(paths, axis=1); mdd = (cum - np.maximum.accumulate(np.maximum(cum, 0), axis=1)).min(axis=1)
ann, worst = cum[:, -1], paths.min(axis=1)
q = lambda x, ps: {f"P{p}": float(np.percentile(x, p)) for p in ps}
mc = {"n_paths": N_PATHS, "n_combos": len(combos), "block": BLOCK, "grid": GRID,
      "annual": q(ann, [1, 5, 10, 50, 90]), "mdd": q(mdd, [1, 5, 10, 50]), "worst_day": q(worst, [1, 5, 50]),
      "p_loss_year": float((ann < 0).mean()), "p_mdd_gt_500": float((mdd < -500).mean()), "p_mdd_gt_400": float((mdd < -400).mean()),
      "det_mdd_range": [float(det.mdd.min()), float(det.mdd.max())], "det_annual_range": [float(det.annual.min()), float(det.annual.max())]}
# same bootstrap on the report version only (parameter part off) for comparison
pb = series[base_i][idx]; cb = np.cumsum(pb, axis=1); mb = (cb - np.maximum.accumulate(np.maximum(cb, 0), axis=1)).min(axis=1)
mc["base_only"] = {"annual": q(cb[:, -1], [5, 50]), "mdd": q(mb, [1, 5, 10, 50])}
json.dump(mc, open(OUT / "mc_summary.json", "w"), ensure_ascii=False, indent=1)
det.to_csv(OUT / "mc_param_grid.csv", index=False)
print(json.dumps({k: v for k, v in mc.items() if k != "grid"}, ensure_ascii=False, indent=1))
