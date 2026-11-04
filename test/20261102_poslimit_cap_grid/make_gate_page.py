"""Render the posLimit 400 x cap 7000 gate study (gate_tune.py + gate_combo.py outputs) as a static page.
Usage: uv run python make_gate_page.py <gate_dir> <findings.html> <out.html>
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd

D, findings, OUT = Path(sys.argv[1]), Path(sys.argv[2]).read_text(encoding="utf-8"), Path(sys.argv[3])
combo = pd.read_csv(D / "combo_grid.csv"); comp = pd.read_csv(D / "comp_grid.csv"); tsmc = pd.read_csv(D / "tsmc_grid.csv")
bc = pd.read_csv(D / "bucket_comp.csv"); bg = pd.read_csv(D / "bucket_tsmc_gap.csv")
CAP = 7000
MINUS = lambda s: s.replace("-", "−")


def label(comp_key, tsmc_key):
    a = "comp 關" if comp_key == "comp off" else MINUS(re.sub(r"comp (-?[\d.]+)/x([\d.]+)/cd(\d)", lambda m: f"comp −1.5／{m[1]} ×{m[2]}" + ("＋冷卻 2 天" if m[3] == "2" else ""), comp_key))
    b = "台積電關" if tsmc_key == "tsmc off" else "台積電 " + tsmc_key.replace("1-2%", "1–2%").replace(" / ", "／").replace("gap", "").replace(" x", " ×")
    return f"{a}；{b}"


def heat(pv, fmt, higher, rfmt, cfmt, axis, cur=(), rec=()):
    vals = pv.to_numpy(dtype=float); lo, hi = np.nanmin(vals), np.nanmax(vals)
    h = [f'<div class="scroll"><table class="heat"><thead><tr><th class="axis">{axis}</th>' + "".join(f"<th>{cfmt(c)}</th>" for c in pv.columns) + "</tr></thead><tbody>"]
    for r in pv.index:
        h.append(f'<tr><th scope="row">{rfmt(r)}</th>')
        for c in pv.columns:
            x = pv.loc[r, c]
            if pd.isna(x):
                h.append('<td class="na">–</td>'); continue
            t = (x - lo) / (hi - lo) if hi > lo else 0.5
            t = 0.08 + 0.92 * (t if higher else 1 - t)
            cls = " ".join(k for k in ("ink" if t > 0.62 else "", "cur" if (r, c) in cur else "", "rec" if (r, c) in rec else "") if k)
            h.append(f'<td class="{cls}" style="--t:{t:.3f}">{fmt(x)}</td>')
        h.append("</tr>")
    return "".join(h) + "</tbody></table></div>"


def table(df, cols, highlight=None):
    h = ['<div class="scroll"><table class="plain"><thead><tr>' + "".join(f"<th>{name}</th>" for name, _, _ in cols) + "</tr></thead><tbody>"]
    for i, r in df.iterrows():
        cls = ' class="pick"' if highlight is not None and highlight(r) else ""
        h.append(f"<tr{cls}>" + "".join(f"<td>{fmt(r) if k is None else fmt(r[k])}</td>" for _, k, fmt in cols) + "</tr>")
    return "".join(h) + "</tbody></table></div>"


f0 = lambda v: MINUS(f"{v:,.0f}"); f1 = lambda v: MINUS(f"{v:,.1f}"); f2 = lambda v: MINUS(f"{v:.2f}"); txt = lambda v: v
sc_lab = lambda s: f"×{s:g}<br><span class='sub'>{CAP * s:,.0f} 萬</span>"
gap_lab = lambda g: f"&gt;{g * 100:g}%"

# candidates
CANDS = [("無閘門", "comp off", "tsmc off"), ("只 comp（−1.5／−0.7 ×0.1）", "comp -0.7/x0.1/cd0", "tsmc off"),
         ("只台積電（1–2% ×0.1／&gt;2% ×0.4）", "comp off", "1-2% x0.1 / >2% x0.4"),
         ("現行正式（comp ×0.1＋冷卻 2 天＋台積電）", "comp -0.7/x0.1/cd2", "1-2% x0.1 / >2% x0.4"),
         ("建議（comp ×0.15、無冷卻＋台積電）", "comp -0.7/x0.15/cd0", "1-2% x0.1 / >2% x0.4"),
         ("偏 DD（台積電 1–2% 全停）", "comp -0.7/x0.15/cd0", "1-2% x0 / >2% x0.4"),
         ("最低 DD（台積電 &gt;0.75% ×0.1）", "comp -0.7/x0.15/cd0", "gap>0.75% x0.1")]
cand = pd.DataFrame([{"name": n, **combo[(combo.comp == a) & (combo.tsmc == b)].iloc[0].to_dict()} for n, a, b in CANDS])
CAND_COLS = [("設定", "name", txt), ("總獲利（萬）", "total_pnl_wan", f0), ("MDD（萬）", "mdd_wan", f0), ("最差單日（萬）", "worst_day_wan", f0),
             ("Sharpe", "sharpe", f2), ("日損益 P10／平均／P90（萬）", None, lambda r: f"{f1(r.pnl_p10_wan)}／{f1(r.pnl_mean_wan)}／{f1(r.pnl_p90_wan)}"),
             ("2023–25 Sharpe／MDD（萬）", None, lambda r: f"{f2(r.sharpe_23_25)}／{f0(r.mdd_wan_23_25)}"),
             ("2026 Sharpe／MDD（萬）", None, lambda r: f"{f2(r.sharpe_26)}／{f0(r.mdd_wan_26)}"), ("受限天數（天）", "restricted_days", f0)]

# comp layer (TSMC off)
cg = comp[(comp.stop_thr == -1.5) & (comp.cooldown_days == 0) & np.isfinite(comp.reduce_thr)]
comp_sh = cg.pivot(index="reduce_thr", columns="reduce_scale", values="sharpe")
comp_pnl = cg.pivot(index="reduce_thr", columns="reduce_scale", values="total_pnl_wan")
stop_t = comp[(comp.reduce_thr == -0.7) & (comp.reduce_scale == 0.15) & (comp.cooldown_days == 0)].copy()
stop_t["stop_lab"] = [("不設" if not np.isfinite(s) else MINUS(f"{s:g}")) for s in stop_t.stop_thr]
cd_t = comp[(comp.stop_thr == -1.5) & (comp.reduce_thr == -0.7) & (comp.reduce_scale.isin([0.1, 0.15]))].copy()
cd_t["scale_lab"] = [f"×{s:g}（{CAP * s:,.0f} 萬）" for s in cd_t.reduce_scale]
BUCKET_COLS = lambda first: [(first, "bucket", txt), ("天數（天）", "days", f0), ("日均（萬）", "mean_wan", f1), ("t 值", "t_stat", f2),
                             ("2023 日均（萬）", "mean_wan_2023", f1), ("2024 日均（萬）", "mean_wan_2024", f1), ("2025 日均（萬）", "mean_wan_2025", f1), ("2026 日均（萬）", "mean_wan_2026", f1)]
bracket = lambda s: MINUS(s.replace("-9.0", "−∞").replace("9.0]", "∞)").replace(", ", "～"))
bc["bucket"] = bc.bucket.map(bracket); bg["bucket"] = bg.bucket.map(bracket)

# TSMC layer (comp off), one tier; then two tiers on top of comp
one = tsmc[tsmc.kind == "one"]
ts_sh = one.pivot(index="t1", columns="s1", values="sharpe"); ts_dd = one.pivot(index="t1", columns="s1", values="mdd_wan")
two = combo[(combo.comp == "comp -0.7/x0.15/cd0") & combo.tsmc.str.startswith("1-2%")].copy()
two["s1"] = two.tsmc.str.extract(r"1-2% x([\d.]+)")[0].astype(float); two["s2"] = two.tsmc.str.extract(r">2% x([\d.]+)")[0].astype(float)
two_sh, two_dd, two_pnl = (two.pivot(index="s1", columns="s2", values=v) for v in ("sharpe", "mdd_wan", "total_pnl_wan"))

# Pareto over the combined grid (no cooldown)
pc = combo[~combo.comp.str.endswith("cd2")].sort_values(["mdd_wan", "sharpe"], ascending=[False, False])
keep, best = [], -9
for i, r in pc.iterrows():
    if r.sharpe > best:
        keep.append(i); best = r.sharpe
par = pc.loc[keep].copy(); par["name"] = [label(a, b) for a, b in zip(par.comp, par.tsmc)]

css = Path(__file__).with_name("make_hi_page.py").read_text(encoding="utf-8").split("<style>")[1].split("</style>")[0].replace("{{", "{").replace("}}", "}")
css += """
table.plain { border-spacing: 0; font-size: 0.8rem; }
table.plain th { text-align: right; border-bottom: 1px solid var(--line); padding: 6px 8px; vertical-align: bottom; }
table.plain th:first-child, table.plain td:first-child { text-align: left; font-family: var(--font-body); white-space: normal; min-width: 9em; }
table.plain th { white-space: normal; }
table.plain td { background: none; border-radius: 0; border-bottom: 1px solid var(--line); padding: 6px 8px; }
table.plain tr.pick td { background: color-mix(in oklab, var(--mark) 12%, transparent); font-weight: 500; }
td.na { background: none; color: var(--muted); text-align: center; }
td.cur { outline: 2px dashed var(--muted); outline-offset: -2px; }
td.rec { outline: 2px solid var(--mark); outline-offset: -2px; font-weight: 500; }
.sub { font-size: 0.68rem; color: var(--muted); }
.pair { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 16px 24px; }
.pair > * { min-width: 0; }
section.block, .metric { display: grid; grid-template-columns: minmax(0, 1fr); gap: 14px; }
.metric { gap: 8px; min-width: 0; }
section.block > h2 { font-family: var(--font-display); font-weight: 600; font-size: 1.3rem; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
h3 { font-size: 0.95rem; font-weight: 700; margin: 0; }
.note { margin: 0; font-size: 0.85rem; color: var(--muted); max-width: 80ch; }
.sw-cur { outline: 2px dashed var(--muted); outline-offset: -2px; background: var(--heat-lo); }
"""
LEG = ('<div class="legend"><span><span class="swatch sw-scale"></span>同一張表內由差到好</span>'
       '<span><span class="swatch sw-cur"></span>現行設定</span><span><span class="swatch sw-hot"></span>建議設定</span></div>')


def metric(title, note, body):
    return f'<div class="metric"><header><h3>{title}</h3><p>{note}</p></header>{body}</div>'


# restricted-day posLimit x cap (group_poscap.py); optional section
poscap_html = ""
if (D / "group_poscap_grid.csv").exists():
    ext = (D / "group_poscap_grid_ext.csv").exists()   # posLimit 50-1000 (group_poscap_hi.py) when available
    pg = pd.read_csv(D / ("group_poscap_grid_ext.csv" if ext else "group_poscap_grid.csv")); uni = pd.read_csv(D / ("group_poscap_uniform_ext.csv" if ext else "group_poscap_uniform.csv"))
    PBASE = {"G1 comp −1.5～−0.7": (400, 1000), "G2 台積電 1–2%": (400, 700), "G3 台積電 >2%": (400, 2800)}
    cap_lab = lambda c: f"{c:,}"
    parts = []
    PREC = {"G1 comp −1.5～−0.7": {(400, 1000)}, "G2 台積電 1–2%": {(400, 700)}, "G3 台積電 >2%": {(400, 3500)}}   # 11/03 recommendation
    for g, (bp, bc_) in PBASE.items():
        x = pg[pg.group == g]; days = int((pd.read_csv(D / "group_poscap_grid.csv").query("group == @g").grp_days.iloc[0]))
        sh = x.pivot(index="posLimit_wan", columns="cap_wan", values="sharpe"); dd = x.pivot(index="posLimit_wan", columns="cap_wan", values="mdd_wan")
        parts.append(f'<h3>{g}（{days} 天，不含 comp 停手日；其他兩組固定在現行）</h3>'
                     + metric("年化 Sharpe", "列 = 這組日子的單檔上限（萬），欄 = 這組日子的 cap（萬）", heat(sh, f2, True, lambda r: f"{r:,}", cap_lab, "單檔＼cap", cur={(bp, bc_)}, rec=PREC[g]))
                     + metric("MDD（萬）", "同上", heat(dd, f0, True, lambda r: f"{r:,}", cap_lab, "單檔＼cap", cur={(bp, bc_)}, rec=PREC[g])))
    ush = uni.pivot(index="posLimit_wan", columns="cap_wan", values="sharpe"); udd = uni.pivot(index="posLimit_wan", columns="cap_wan", values="mdd_wan")
    parts.append('<h3>三組用同一組（單檔、cap）</h3>'
                 + metric("年化 Sharpe", "三組日子都用這格的單檔上限與 cap", heat(ush, f2, True, lambda r: f"{r:,}", cap_lab, "單檔＼cap"))
                 + metric("MDD（萬）", "同上", heat(udd, f0, True, lambda r: f"{r:,}", cap_lab, "單檔＼cap")))
    boot_html = ""
    if (D / "verify_bootstrap.csv").exists():
        bt = pd.read_csv(D / "verify_bootstrap.csv")
        boot_html = metric("只動單檔上限：成對區塊 bootstrap（95% 區間）", "三組 cap 都固定 1,000，和單檔 400 比；區塊長 10 天、5,000 次", table(bt, [
            ("設定", "label", txt), ("Sharpe 差", "sharpe_diff", f2), ("Sharpe 95% 區間", None, lambda r: f"{f2(r.sharpe_ci_lo)}～{f2(r.sharpe_ci_hi)}"),
            ("MDD 差（萬）", "mdd_diff_wan", f0), ("MDD 95% 區間（萬）", None, lambda r: f"{f0(r.mdd_ci_lo)}～{f0(r.mdd_ci_hi)}")]))
    g3_html = ""
    if (D / "g3_fine.csv").exists():   # g3_fine.py: G3 cap in 350 萬 steps, posLimit 400, G1/G2 = 1,000/700
        g3 = pd.read_csv(D / "g3_fine.csv").query("G1_G2_cap_wan == '1000/700'")
        g3 = g3[g3.G3_cap_wan.isin([0, 700, 1050, 1400, 2100, 2800, 3150, 3500, 3850, 4200, 4550, 5250, 7000])]
        g3_html = metric("G3（台積電 &gt;2%）cap 細網格：每 350 萬一格", "單檔 400；G1／G2 固定 1,000／700；65 天是 G3 裡沒和 comp 縮額重疊的日子，組內損益用全部 78 天", table(g3, [
            ("G3 cap（萬）", "G3_cap_wan", f0), ("總獲利（萬）", "total_pnl_wan", f0), ("MDD（萬）", "mdd_wan", f0), ("Sharpe", "sharpe", f2),
            ("G3 組內（萬）", "g3_pnl_wan", f0), ("2023（萬）", "g3_pnl_wan_2023", f0), ("2024（萬）", "g3_pnl_wan_2024", f0), ("2025（萬）", "g3_pnl_wan_2025", f0), ("2026（萬）", "g3_pnl_wan_2026", f0),
            ("G3 最差日（萬）", "g3_worst_wan", f0)], highlight=lambda r: r["G3_cap_wan"] == 3500))
    poscap_html = (f'<section class="block"><h2>受限日的單檔上限 × cap</h2><p class="note">正常日固定單檔 400、cap 7,000；comp &lt; −1.5 照停。'
                   f'每次只動一組，其他兩組固定在單檔 400、cap 1,000／700／2,800。日子同時屬於兩組時取較小的單檔與 cap。單檔上限改變時停損商品數也跟著重算。'
                   f'MDD 的差異幾乎都來自 2025-07-23→08-18 回撤裡的兩個受限日（07-31 屬 G2、08-08 屬 G3），別拿 MDD 挑這裡的設定。</p>'
                   f'{LEG}' + "".join(parts) + g3_html + boot_html + '</section>')

page = f"""<title>negFill 400×7000 停手閘門</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Noto+Serif+TC:wght@600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}</style>
<div class="wrap">
  <div>
    <h1>單檔 400 萬 × cap 7,000 萬：停手日怎麼設</h1>
    <p class="lede">timeSeries negFill notebook 流程重建 2023-04-21～2026-10-30（856 個交易日），單檔累計名目 400 萬、當日累計新倉名目 7,000 萬。
    閘門的 scale 一律當成「當日總額上限 = 7,000 萬 × scale」，每個設定都是逐筆重跑：停手日的成交也留在資料裡，所以放寬門檻時能直接重放。
    comp 與台積電兩層先各自調（另一層關掉），再疊在一起看。成本 19.3 bp，日損益統計含停手日與無交易日（記 0）。</p>
  </div>
  <section class="findings"><h2>重點</h2>{findings}</section>

  <section class="block"><h2>候選設定比較</h2>
    {table(cand, CAND_COLS, highlight=lambda r: r["name"].startswith("建議"))}
    <p class="note">受限天數 = scale &lt; 1 的天數（856 天中）。comp 在前 58 天（2023-04～07）歷史不足、不會亮燈。</p>
  </section>

  <section class="block"><h2>comp 層（台積電層關閉）</h2>
    {metric("comp 區間 → 隔天損益（不設閘門、滿額 7,000 萬）", "comp = 前一交易日的市場狀態分數；日均含無交易日", table(bc, BUCKET_COLS("comp 區間")))}
    {LEG}
    <div class="pair">
      {metric("年化 Sharpe", "停手門檻 −1.5、無冷卻；列 = 縮額門檻，欄 = 縮額日 scale", heat(comp_sh, f2, True, lambda r: MINUS(f"{r:g}"), sc_lab, "縮額門檻＼scale", cur={(-0.7, 0.1)}, rec={(-0.7, 0.15)}))}
      {metric("總獲利（萬）", "同左；MDD 除縮額門檻 ≥ −0.2（−383）外都是 −432", heat(comp_pnl, f0, True, lambda r: MINUS(f"{r:g}"), sc_lab, "縮額門檻＼scale", cur={(-0.7, 0.1)}, rec={(-0.7, 0.15)}))}
    </div>
    <div class="pair">
      {metric("停手門檻（comp 低於此值全停）", "縮額 −0.7 ×0.15、無冷卻", table(stop_t, [("停手門檻", "stop_lab", txt), ("全停天數（天）", "stop_days", f0), ("總獲利（萬）", "total_pnl_wan", f0), ("MDD（萬）", "mdd_wan", f0), ("Sharpe", "sharpe", f2)], highlight=lambda r: r["stop_thr"] == -1.5))}
      {metric("出關冷卻天數", "停手 −1.5、縮額 −0.7；MDD 全是 −432", table(cd_t, [("縮額 scale", "scale_lab", txt), ("冷卻（天）", "cooldown_days", f0), ("總獲利（萬）", "total_pnl_wan", f0), ("Sharpe", "sharpe", f2), ("2023–25／2026 Sharpe", None, lambda r: f"{f2(r.sharpe_23_25)}／{f2(r.sharpe_26)}")], highlight=lambda r: r["reduce_scale"] == 0.15 and r["cooldown_days"] == 0))}
    </div>
  </section>

  <section class="block"><h2>台積電層（comp 層關閉）</h2>
    {metric("昨日 2330 開盤跳空 → 隔天損益（不設閘門、滿額 7,000 萬）", "跳空 = 09:00 開盤 ÷ 參考價 − 1（%）", table(bg, BUCKET_COLS("跳空區間（%）")))}
    {LEG}
    <div class="pair">
      {metric("年化 Sharpe（單一門檻）", "跳空大於門檻 → 隔天 scale", heat(ts_sh, f2, True, gap_lab, sc_lab, "跳空＼scale", rec={(0.01, 0.1)}))}
      {metric("MDD（萬）（單一門檻）", "同左", heat(ts_dd, f0, True, gap_lab, sc_lab, "跳空＼scale", rec={(0.01, 0.1)}))}
    </div>
    <h3>兩段式疊在 comp（−1.5／−0.7 ×0.15、無冷卻）上</h3>
    <p class="note">列 = 跳空 1–2% 的 scale，欄 = 跳空 &gt;2% 的 scale。comp 不改 MDD，所以這裡的 MDD 全由台積電層決定。</p>
    <div class="pair">
      {metric("年化 Sharpe", "", heat(two_sh, f2, True, sc_lab, sc_lab, "1–2%＼&gt;2%", cur={(0.1, 0.4)}, rec={(0.1, 0.4)}))}
      {metric("MDD（萬）", "", heat(two_dd, f0, True, sc_lab, sc_lab, "1–2%＼&gt;2%", cur={(0.1, 0.4)}, rec={(0.1, 0.4)}))}
      {metric("總獲利（萬）", "", heat(two_pnl, f0, True, sc_lab, sc_lab, "1–2%＼&gt;2%", cur={(0.1, 0.4)}, rec={(0.1, 0.4)}))}
    </div>
  </section>

  <section class="block"><h2>合併後的 MDD–Sharpe 前緣</h2>
    <p class="note">comp（停手 −1.5；縮額門檻 −1.0～−0.5、scale 0.1／0.15／0.2）× 台積電（單一門檻 0.75／1／1.5／2%、兩段式 1%／2%）共 1,073 組（不含冷卻）。每一列都是「MDD 不比它差的設定裡 Sharpe 最高」。</p>
    {table(par, [("設定", "name", txt), ("MDD（萬）", "mdd_wan", f0), ("Sharpe", "sharpe", f2), ("總獲利（萬）", "total_pnl_wan", f0), ("2023–25 Sharpe", "sharpe_23_25", f2), ("2026 Sharpe", "sharpe_26", f2), ("受限天數（天）", "restricted_days", f0)],
             highlight=lambda r: r["comp"] == "comp -0.7/x0.15/cd0" and r["tsmc"] == "1-2% x0.1 / >2% x0.4")}
  </section>

  {poscap_html}
  <footer><span>scale 以當日累計新倉名目上限實作（7,000 萬 × scale），不是逐筆等比縮。comp、台積電跳空取自 stopTradingGate 的 gate_daily.csv（asof 日 → 下一交易日）；現行正式設定逐日重現 scale_by_date，數字與先前網格一致。</span>
  <span>腳本：src/research/timeSeries/test/20261102_poslimit_cap_grid/（build_rows.py … keep_stop → gate_tune.py → gate_combo.py → make_gate_page.py）。</span></footer>
</div>
"""
OUT.write_text(page, encoding="utf-8"); print(OUT, OUT.stat().st_size)
