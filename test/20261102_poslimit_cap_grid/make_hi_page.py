"""Render hi_grid.csv (comp-only gate, posLimit 300-1000 x cap 5000-10000) as a static heatmap page; cells with normal-day usage >= 85% are outlined.
Usage: uv run python make_hi_page.py <hi_grid.csv> <findings.html> <out.html>
"""
import sys
from pathlib import Path
import pandas as pd

g = pd.read_csv(sys.argv[1]); findings = Path(sys.argv[2]).read_text(encoding="utf-8"); OUT = Path(sys.argv[3])
POS = sorted(g.posLimit_wan.unique()); CAPS = sorted(g.cap_wan.unique()); TH = 85.0
METRICS = [("usage_normal_pct", "資金效率（%）", "正常日（comp 未縮額）平均每日累計名目 ÷ cap", lambda v: f"{v:.1f}", True),
           ("total_pnl_wan", "總獲利（萬）", "2023-04-21～2026-10-30 累計淨損益，扣 19.3 bp", lambda v: f"{v:,.0f}", True),
           ("mdd_wan", "累積最大回撤 MDD（萬）", "每日損益累計的最大回撤", lambda v: f"{v:,.0f}", True),
           ("worst_day_wan", "最差單日（萬）", "單日收盤損益最小值", lambda v: f"{v:,.0f}", True),
           ("pnl_p10_wan", "日損益 P10（萬）", "所有交易日（停手／無交易日記 0）", lambda v: f"{v:.1f}", True),
           ("pnl_mean_wan", "日損益平均（萬）", "所有交易日（停手／無交易日記 0）", lambda v: f"{v:.1f}", True),
           ("pnl_p90_wan", "日損益 P90（萬）", "所有交易日（停手／無交易日記 0）", lambda v: f"{v:.1f}", True),
           ("sharpe", "年化 Sharpe", "日損益平均 ÷ 標準差 × √252", lambda v: f"{v:.2f}", True)]
val = {(r.posLimit_wan, r.cap_wan): r for r in g.itertuples()}


def heat(key, fmt, higher):
    vals = [getattr(val[(p, c)], key) for p in POS for c in CAPS]; lo, hi = min(vals), max(vals)
    h = ['<div class="scroll"><table><thead><tr><th class="axis">單檔＼cap（萬）</th>' + "".join(f"<th>{c:,}</th>" for c in CAPS) + "</tr></thead><tbody>"]
    for p in POS:
        h.append(f'<tr><th scope="row">{p:,}</th>')
        for c in CAPS:
            r = val[(p, c)]; x = getattr(r, key)
            t = (x - lo) / (hi - lo) if hi > lo else 0.5
            t = 0.08 + 0.92 * (t if higher else 1 - t)
            cls = " ".join(k for k in ("ink" if t > 0.62 else "", "hot" if r.usage_normal_pct >= TH else "") if k)
            h.append(f'<td class="{cls}" style="--t:{t:.3f}" title="單檔 {p} 萬 × cap {c:,} 萬｜資金效率 {r.usage_normal_pct:.0f}%">{fmt(x)}</td>')
        h.append("</tr>")
    return "".join(h) + "</tbody></table></div>"


page = f"""<title>negFill 高效率網格</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Noto+Serif+TC:wght@600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --bg: #f3f5f4; --surface: #ffffff; --fg: #17201e; --muted: #5a6864; --line: #d3dbd8;
  --heat-lo: #eef2f0; --heat-hi: #1d6b58; --heat-ink: #ffffff; --mark: #c2410c;
  --font-display: "Noto Serif TC", "Songti TC", serif;
  --font-body: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", system-ui, sans-serif;
  --font-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #111715; --surface: #18201e; --fg: #e2eae7; --muted: #94a39e; --line: #2c3734;
  --heat-lo: #1b2321; --heat-hi: #3fae8d; --heat-ink: #0d1412; --mark: #f59e5b; color-scheme: dark }} }}
:root[data-theme="dark"] {{
  --bg: #111715; --surface: #18201e; --fg: #e2eae7; --muted: #94a39e; --line: #2c3734;
  --heat-lo: #1b2321; --heat-hi: #3fae8d; --heat-ink: #0d1412; --mark: #f59e5b; color-scheme: dark }}
body {{ background: var(--bg); color: var(--fg); font-family: var(--font-body); line-height: 1.6; }}
.wrap {{ max-width: 1120px; margin: 0 auto; padding-block: 28px 56px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; grid-template-columns: minmax(0, 1fr); gap: 28px; }}
h1 {{ font-family: var(--font-display); font-weight: 600; font-size: 1.75rem; line-height: 1.3; margin: 0; text-wrap: balance; }}
h2 {{ font-size: 1.05rem; font-weight: 700; margin: 0; text-wrap: balance; }}
.lede {{ color: var(--muted); max-width: 72ch; margin: 6px 0 0; }}
.findings {{ background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 18px 20px; }}
.findings ul {{ margin: 10px 0 0; padding-left: 1.2em; display: grid; gap: 8px; max-width: 82ch; }}
.num {{ font-family: var(--font-data); font-variant-numeric: tabular-nums; font-size: 0.92em; }}
.metric {{ display: grid; gap: 8px; }}
.metric header {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; }}
.metric header p {{ margin: 0; font-size: 0.85rem; color: var(--muted); }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: separate; border-spacing: 2px; font-family: var(--font-data); font-variant-numeric: tabular-nums; font-size: 0.8rem; width: 100%; }}
th {{ font-weight: 500; color: var(--muted); padding: 4px 6px; white-space: nowrap; font-family: var(--font-body); }}
thead th {{ text-align: right; }} tbody th {{ text-align: left; }}
td {{ text-align: right; padding: 5px 5px; border-radius: 3px; white-space: nowrap;
  background: color-mix(in oklab, var(--heat-hi) calc(var(--t) * 100%), var(--heat-lo)); color: var(--fg); }}
td.ink {{ color: var(--heat-ink); }}
td.hot {{ outline: 2px solid var(--mark); outline-offset: -2px; font-weight: 500; }}
.axis {{ font-size: 0.72rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.8rem; color: var(--muted); }}
.swatch {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; vertical-align: -2px; margin-right: 4px; }}
.sw-scale {{ background: linear-gradient(90deg, var(--heat-lo), var(--heat-hi)); width: 48px; }}
.sw-hot {{ outline: 2px solid var(--mark); outline-offset: -2px; background: var(--heat-lo); }}
footer {{ font-size: 0.8rem; color: var(--muted); border-top: 1px solid var(--line); padding-top: 14px; display: grid; gap: 4px; }}
</style>
<div class="wrap">
  <div>
    <h1>negFill 單檔 × 總額：資金效率拉高會怎樣</h1>
    <p class="lede">timeSeries negFill notebook 流程重建 2023-04-21～2026-10-30（856 個交易日），停手閘門只保留 comp（&lt;−1.5 停、−1.5～−0.7 cap × 0.1），不含出關冷卻與台積電跳空。
    每格都逐筆重跑：單檔累計名目 300–1,000 萬（每 100 萬）、當日累計新倉名目 cap 5,000 萬–1 億（每 500 萬），停損商品數依單檔各自重算。成本 19.3 bp，未避險。</p>
  </div>
  <section class="findings"><h2>重點</h2>{findings}</section>
  <div class="legend"><span><span class="swatch sw-scale"></span>同一張網格內由差到好</span><span><span class="swatch sw-hot"></span>資金效率 ≥ {TH:.0f}%（正常日）</span></div>
"""
for key, name, desc, fmt, higher in METRICS:
    page += f'  <section class="metric"><header><h2>{name}</h2><p>{desc}</p></header>{heat(key, fmt, higher)}</section>\n'
page += """  <footer><span>資金效率 = 正常日（comp 未縮額，含無交易日）平均每日累計名目 ÷ cap。日損益分位與平均含停手日與無交易日（記 0）。MDD／最差單日以每日收盤損益計。</span>
  <span>腳本：src/research/timeSeries/test/20261102_poslimit_cap_grid/（build_rows.py → hi_grid.py → make_hi_page.py）。</span></footer>
</div>
"""
OUT.write_text(page, encoding="utf-8"); print(OUT, OUT.stat().st_size)
