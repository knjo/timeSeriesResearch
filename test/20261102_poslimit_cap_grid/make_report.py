"""Render grid.csv into a standalone, print-ready HTML (no JS) and a PDF via headless Chrome.
Usage: uv run python make_report.py
"""
import html, subprocess
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
g = pd.read_csv(HERE / "grid.csv")
POS = [250, 300, 350, 400, 450, 500]
CAPS = list(range(4000, 7001, 250))
PERIODS = [("2023-2025", "2023–2025（637 天）"), ("2026", "2026（190 天，至 10/30）")]
METRICS = [("sharpe", "年化 Sharpe", "日損益平均 ÷ 標準差 × √252", lambda v: f"{v:.2f}"),
           ("pnl_wan", "總獲利（萬）", "期間累計淨損益，扣 19.3 bp", lambda v: f"{v:,.0f}"),
           ("unit_bp", "資金獲利（bp）", "總損益 ÷ 每日累計進場名目總和", lambda v: f"{v:.1f}"),
           ("mdd_wan", "最大回撤 MDD（萬）", "期間內累計損益的最大回撤", lambda v: f"{v:,.0f}"),
           ("usage_pct", "資金使用率（%）", "平均每日累計名目 ÷ cap", lambda v: f"{v:.0f}")]
VARIANTS = [("graded", "正式分級閘門（cap × scale）",
             "正式 RD 語意：停手日 cap × 0、縮額日 cap × 0.1、台積電跳空日 cap × 0.4（取小）。使用率以基準 cap 為分母。"),
            ("notebook", "notebook 現行（只排除停手日）",
             "notebook cell 34 現行寫法：只排除 stop_dates（28 天），縮額日與台積電跳空日照基準 cap 交易。")]
PICKS = [(250, 5000), (350, 5000), (350, 5500), (350, 6500), (350, 7000), (400, 7000), (500, 6000), (500, 7000)]
MARKED = {(350, 5000), (350, 7000)}


def val(v, p, pos, cap, m):
    return float(g[(g.variant == v) & (g.period == p) & (g.posLimit_wan == pos) & (g.cap_wan == cap)][m].iloc[0])


def heat(v, p, m, fmt):
    vals = [val(v, p, pos, cap, m) for pos in POS for cap in CAPS]
    lo, hi = min(vals), max(vals)
    best = max(((val(v, p, pos, cap, "sharpe"), pos, cap) for pos in POS for cap in CAPS))
    out = ['<table class="heat"><thead><tr><th class="axis">單檔＼cap</th>' + "".join(f"<th>{c:,}</th>" for c in CAPS) + "</tr></thead><tbody>"]
    for pos in POS:
        out.append(f"<tr><th>{pos}</th>")
        for cap in CAPS:
            x = val(v, p, pos, cap, m)
            t = 0.08 + 0.92 * ((x - lo) / (hi - lo) if hi > lo else 0.5)
            cls = " ".join(c for c in ("ink" if t > 0.62 else "", "best" if (pos, cap) == best[1:] else "", "pick" if (pos, cap) in MARKED else "") if c)
            out.append(f'<td class="{cls}" style="--t:{t:.3f}">{fmt(x)}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def picks_table(v):
    rows = ['<table class="plain"><thead><tr><th rowspan="2">單檔×cap（萬）</th><th colspan="5">2023–2025</th><th colspan="5">2026</th></tr><tr>'
            + "<th>Sharpe</th><th>獲利（萬）</th><th>bp</th><th>MDD（萬）</th><th>使用率（%）</th>" * 2 + "</tr></thead><tbody>"]
    for pos, cap in PICKS:
        cells = []
        for p, _ in PERIODS:
            cells += [f"{val(v, p, pos, cap, 'sharpe'):.2f}", f"{val(v, p, pos, cap, 'pnl_wan'):,.0f}", f"{val(v, p, pos, cap, 'unit_bp'):.1f}",
                      f"{val(v, p, pos, cap, 'mdd_wan'):,.0f}", f"{val(v, p, pos, cap, 'usage_pct'):.0f}"]
        rows.append(f"<tr><th>{pos}×{cap:,}</th>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "".join(rows) + "</tbody></table>"


CSS = """
@page { size: A4 landscape; margin: 10mm 10mm 12mm; }
:root { --bg:#ffffff; --fg:#17201e; --muted:#5a6864; --line:#d3dbd8; --heat-lo:#eef2f0; --heat-hi:#1d6b58; --heat-ink:#ffffff; --mark:#c2410c; }
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font-family: "Noto Sans CJK TC", "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif; font-size: 11.5px; line-height: 1.55; }
.page { max-width: 1060px; margin: 0 auto; padding: 18px 16px; }
h1 { font-family: "Noto Serif CJK TC", "Noto Serif TC", serif; font-weight: 600; font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 15px; margin: 14px 0 6px; }
h3 { font-size: 12.5px; margin: 0 0 4px; }
p { margin: 4px 0; max-width: 95ch; }
.muted { color: var(--muted); }
ul { margin: 6px 0; padding-left: 1.3em; } li { margin: 3px 0; max-width: 100ch; }
.box { border: 1px solid var(--line); border-radius: 5px; padding: 10px 14px; margin: 10px 0; }
table { border-collapse: separate; border-spacing: 2px; font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace; font-variant-numeric: tabular-nums; }
th { font-weight: 500; color: var(--muted); padding: 2px 4px; white-space: nowrap; font-family: "Noto Sans CJK TC", sans-serif; }
.heat { font-size: 9px; border-spacing: 1px; }
@media print { .page { max-width: none; padding: 0; } }
.heat thead th { text-align: right; } .heat tbody th { text-align: left; }
.heat td { text-align: right; padding: 1.5px 2.5px; border-radius: 2px; background: color-mix(in oklab, var(--heat-hi) calc(var(--t) * 100%), var(--heat-lo)); }
.heat td.ink { color: var(--heat-ink); }
.heat td.best { outline: 2px solid var(--mark); outline-offset: -2px; font-weight: 600; }
.heat td.pick { box-shadow: inset 0 -3px 0 var(--mark); }
.axis { font-size: 9.5px; }
.plain { font-size: 11px; margin-top: 6px; }
.plain td { text-align: right; padding: 3px 8px; border-bottom: 1px solid var(--line); }
.plain thead th { border-bottom: 1px solid var(--line); text-align: center; }
.plain tbody th { text-align: left; }
.metric { break-inside: avoid; margin: 6px 0 2px; }
.keep { break-inside: avoid; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; } .pair > div { min-width: 0; }
.variant { break-before: page; }
.legend { display: flex; gap: 16px; font-size: 10.5px; color: var(--muted); margin: 4px 0 8px; }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: -2px; margin-right: 4px; background: var(--heat-lo); }
.sw.best { outline: 2px solid var(--mark); outline-offset: -2px; } .sw.pick { box-shadow: inset 0 -3px 0 var(--mark); }
.sw.scale { width: 40px; background: linear-gradient(90deg, var(--heat-lo), var(--heat-hi)); }
footer { margin-top: 14px; font-size: 10px; color: var(--muted); border-top: 1px solid var(--line); padding-top: 6px; }
"""

parts = [f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>negFill 部位網格</title><style>{CSS}</style></head><body><div class="page">
<h1>negFill 單檔上限 × 每日總額網格</h1>
<p class="muted">timeSeries negFill 現行 notebook 流程（模型、16-cell 門檻、cell 34 風控、2σ 停利停損、停損商品數閘門）重建 2023-04-21 到 2026-10-30，
只改單檔累計名目上限 250–500 萬（每 50 萬）與當日累計新倉名目上限 4,000–7,000 萬（每 250 萬）。成本 19.3 bp，未避險。2026-11-02 產出。</p>
<div class="box"><h3>結論</h3><ul>
<li><b>單檔上限決定回撤，總額決定獲利。</b>正式閘門、cap 5,000 萬時，2026 MDD 單檔 250／350／500 萬為 −182／−262／−407 萬。cap 從 6,000 拉到 7,000 萬，MDD 幾乎不動（500 萬 −378→−372、350 萬 −272→−275）。</li>
<li><b>大單檔配小總額最差。</b>早盤的大單檔部位先把當日額度吃完，後段較好的成交被擋：2026 正式閘門 500 萬 × 4,000 萬 Sharpe 3.91、71.7 bp；350 萬 × 4,000 萬為 4.52、80.6 bp。</li>
<li><b>cap 6,000 萬以上獲利還在增加，但邊際遞減，Sharpe 持平。</b>2026 正式閘門 cap 6,000→7,000 萬：單檔 500 萬 +436 萬（4,645→5,081）、400 萬 +257 萬、350 萬 +174 萬，Sharpe 都在 4.6–4.7。cap 7,000 萬時資金使用率只剩 39–45%（單檔 350–500 萬），只有重的日子會被擋。</li>
<li><b>兩段結論不同。</b>2026 單看仍是 350 萬 × 5,000 萬 Sharpe 最高（4.74、MDD −262 萬）。2023–2025 的 Sharpe 則隨 cap 一路升到網格邊界：350 萬 × 7,000 萬 4.01、MDD −288 萬，對比 350 萬 × 5,000 萬 3.80、−322 萬，7,000 萬還沒到頂。
兩段一起看，350 萬 × 6,500–7,000 萬優於 350 萬 × 5,000 萬：2023–2025 Sharpe 高 0.15–0.21、MDD 較小；2026 Sharpe 低 0.09–0.10，但獲利多 357–428 萬，MDD −275 對 −262 萬。</li>
<li><b>小單檔很快飽和。</b>單檔 250 萬時，cap 超過約 4,750 萬後獲利幾乎不再增加：2026 cap 6,000→7,000 萬只多 100 萬，資金使用率 32%。</li>
<li><b>最大獲利在 500 萬 × 7,000 萬。</b>2026 為 5,081 萬、Sharpe 4.64、MDD −372 萬；2023–2025 為 9,586 萬、Sharpe 3.82、MDD −356 萬。</li>
<li><b>閘門語意差很多。</b>notebook 只排除 28 個停手日，縮額日照全額做；正式 RD 在縮額日把 cap 乘上 0.1／0.4。後者 2023–2025 Sharpe 高 0.6–0.8，選參數請看正式閘門。</li>
</ul></div>
<div class="keep"><h2>候選組合（正式分級閘門）</h2>{picks_table("graded")}</div>
<div class="keep"><h2>候選組合（notebook 現行）</h2>{picks_table("notebook")}</div>
<footer>交易日只計有交易的日子，與 BackTest 相同；資金獲利 bp = 總損益 ÷ 每日累計進場名目總和；MDD 以各期間自身累計損益計；資金使用率 = 平均每日累計名目 ÷ cap。
單格（500 萬 × 6,000 萬、notebook、2026）已用 BackTest 驗算一致。腳本：src/research/timeSeries/test/20261102_poslimit_cap_grid/。</footer>
"""]
for v, vname, vnote in VARIANTS:
    parts.append(f'<section class="variant"><h2>完整網格：{html.escape(vname)}</h2><p class="muted">{html.escape(vnote)}</p>'
                 '<div class="legend"><span><span class="sw scale"></span>同一張網格內由差到好</span><span><span class="sw best"></span>該網格 Sharpe 最高</span>'
                 '<span><span class="sw pick"></span>候選格 350 萬 × 5,000 萬、350 萬 × 7,000 萬</span><span>列 = 單檔上限（萬），欄 = 每日 cap（萬）</span></div>')
    for key, name, desc, fmt in METRICS:
        parts.append(f'<div class="metric"><h3>{name} <span class="muted" style="font-weight:400">— {desc}</span></h3><div class="pair">')
        for p, plabel in PERIODS:
            parts.append(f'<div><div class="muted" style="font-size:10.5px">{plabel}</div>{heat(v, p, key, fmt)}</div>')
        parts.append("</div></div>")
    parts.append("</section>")
parts.append("</div></body></html>")
out_html = HERE / "negfill_poslimit_cap_grid.html"
out_html.write_text("".join(parts), encoding="utf-8")
out_pdf = HERE / "negfill_poslimit_cap_grid.pdf"
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer", "--virtual-time-budget=5000",
                f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, capture_output=True)
print(out_html, out_html.stat().st_size); print(out_pdf, out_pdf.stat().st_size)
