"""主管版 PDF v4（加蒙地卡羅壓力測試、拿掉控管對照）：negFill 400 萬 × 7,000 萬、comp 分級＋冷卻、台積電跳空 0.5 × cap。讀 build_series_v3.py 輸出；不寫停手判斷方法與天數、無附註，手寫 SVG，Chrome 印 PDF。"""
import json, subprocess
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
d = pd.read_parquet(HERE / "daily_400x7000_tsmc05.parquet")
s = json.load(open(HERE / "summary_tsmc05.json"))
mc = json.load(open(HERE / "mc_summary.json"))
traded = d[d.notional > 0]

BLUE, BLUE_SOFT, INK, INK2, MUTED, LINE, SURF = "#2a78d6", "rgba(42,120,214,0.16)", "#1d1d1b", "#55554f", "#8a8a84", "#e4e4df", "#ffffff"
md = lambda x: x[5:].replace("-", "/")          # 2026-10-14 -> 10/14
ymd = lambda x: x.replace("-", "/")


def wan(x, sign=False):
    v = f"{abs(x):,.0f}"
    return ("−" if round(x) < 0 else ("+" if sign and round(x) > 0 else "")) + v


def chart_cum():
    W, L, R = 680, 56, 64
    T1, H1, GAP, H2, B = 14, 230, 34, 92, 26
    Ht = T1 + H1 + GAP + H2 + B
    t0, t1 = d.index[0], d.index[-1]
    xs = lambda t: L + (t - t0).days / (t1 - t0).days * (W - L - R)
    cum = d.pnl.cumsum().to_numpy(); dd = cum - np.maximum.accumulate(np.maximum(cum, 0))
    ymax = 15000; y1 = lambda v: T1 + H1 - v / ymax * H1
    dmin = -420; y2 = lambda v: T1 + H1 + GAP + (v / dmin) * H2
    X = [xs(t) for t in d.index]
    o = [f'<svg viewBox="0 0 {W} {Ht}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans CJK TC, sans-serif">']
    for v in range(0, ymax + 1, 3000):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{y1(v):.1f}" y2="{y1(v):.1f}" stroke="{LINE}"/>')
        o.append(f'<text x="{L-8}" y="{y1(v)+4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">{v:,}</text>')
    for v in (0, -200, -400):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{y2(v):.1f}" y2="{y2(v):.1f}" stroke="{LINE}"/>')
        o.append(f'<text x="{L-8}" y="{y2(v)+4:.1f}" text-anchor="end" font-size="11" fill="{MUTED}">{wan(v)}</text>')
    for y in (2024, 2025, 2026):
        x = xs(pd.Timestamp(f"{y}-01-01"))
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T1}" y2="{T1+H1+GAP+H2}" stroke="{LINE}"/>')
    for y, t in ((2023, "2023-08-15"), (2024, "2024-07-01"), (2025, "2025-07-01"), (2026, "2026-06-01")):
        o.append(f'<text x="{xs(pd.Timestamp(t)):.1f}" y="{Ht-6}" text-anchor="middle" font-size="11.5" fill="{INK2}">{y}</text>')
    pts = " ".join(f"{x:.1f},{y1(v):.1f}" for x, v in zip(X, cum))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2" stroke-linejoin="round"/>')
    o.append(f'<circle cx="{X[-1]:.1f}" cy="{y1(cum[-1]):.1f}" r="4" fill="{BLUE}" stroke="{SURF}" stroke-width="2"/>')
    o.append(f'<text x="{X[-1]+8:.1f}" y="{y1(cum[-1])+4:.1f}" font-size="12" font-weight="600" fill="{INK}">{cum[-1]:,.0f}</text>')
    base = y2(0)
    o.append(f'<path d="M{X[0]:.1f},{base:.1f} ' + " ".join(f"L{x:.1f},{y2(v):.1f}" for x, v in zip(X, dd)) + f' L{X[-1]:.1f},{base:.1f} Z" fill="{BLUE_SOFT}"/>')
    o.append(f'<polyline points="{" ".join(f"{x:.1f},{y2(v):.1f}" for x, v in zip(X, dd))}" fill="none" stroke="{BLUE}" stroke-width="1.2"/>')
    for date, lab in ((s["mdd_trough"], f'最大回撤 {wan(s["mdd"])}'), (s["recent_trough"], f'近期 {wan(s["recent_mdd"])}')):
        i = d.index.get_loc(pd.Timestamp(date))
        o.append(f'<circle cx="{X[i]:.1f}" cy="{y2(dd[i]):.1f}" r="4" fill="{BLUE}" stroke="{SURF}" stroke-width="2"/>')
        o.append(f'<text x="{X[i]-8:.1f}" y="{y2(dd[i])+4:.1f}" text-anchor="end" font-size="11" fill="{INK}">{lab}</text>')
    o.append(f'<text x="{L}" y="{T1+H1+GAP-8}" font-size="12" font-weight="600" fill="{INK}">回撤：離歷史最高累計獲利差多少（萬元）</text>')
    o.append("</svg>")
    return "".join(o)


def chart_hist():
    W, L, R, T, H, B = 680, 44, 16, 22, 150, 30
    edges = np.arange(-220, 481, 20); cnt, _ = np.histogram(traded.pnl.clip(-219.9, 479.9), edges)
    xs = lambda v: L + (v - edges[0]) / (edges[-1] - edges[0]) * (W - L - R)
    cmax = int(np.ceil(cnt.max() / 50) * 50); ys = lambda c: T + H - c / cmax * H
    o = [f'<svg viewBox="0 0 {W} {T+H+B}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Noto Sans CJK TC, sans-serif">']
    for c in range(0, cmax + 1, 50):
        o.append(f'<line x1="{L}" x2="{W-R}" y1="{ys(c):.1f}" y2="{ys(c):.1f}" stroke="{LINE}"/>')
        o.append(f'<text x="{L-6}" y="{ys(c)+4:.1f}" text-anchor="end" font-size="10.5" fill="{MUTED}">{c}</text>')
    o.append(f'<text x="{L}" y="{T-8}" font-size="10.5" fill="{MUTED}">天數</text>')
    for a, b, c in zip(edges[:-1], edges[1:], cnt):
        if c == 0: continue
        x0, x1 = xs(a) + 1, xs(b) - 1; y = ys(c); h = T + H - y; r = min(3, h / 2, (x1 - x0) / 2)
        o.append(f'<path d="M{x0:.1f},{T+H:.1f} V{y+r:.1f} Q{x0:.1f},{y:.1f} {x0+r:.1f},{y:.1f} H{x1-r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y+r:.1f} V{T+H:.1f} Z" fill="{BLUE}"/>')
    for v in range(-200, 481, 100):
        o.append(f'<text x="{xs(v):.1f}" y="{T+H+16}" text-anchor="middle" font-size="10.5" fill="{MUTED}">{wan(v, True) if v else 0}</text>')
    o.append(f'<text x="{W-R}" y="{T+H+28}" text-anchor="end" font-size="10.5" fill="{MUTED}">單日損益（萬元，每 20 萬一格；最右格含 &gt;+460）</text>')
    for k, lab in (("p10", "P10"), ("p50", "P50"), ("p90", "P90")):
        x = xs(s[k])
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{T-4}" y2="{T+H}" stroke="{INK}" stroke-width="1.2" stroke-dasharray="4 3"/>')
        o.append(f'<text x="{x+5:.1f}" y="{T+6+(16 if k == "p50" else 0)}" font-size="11" fill="{INK}">{lab} {wan(s[k], True)}</text>')
    o.append("</svg>")
    return "".join(o)


yrs = s["years"]
lab_year = {2023: "2023（4/21 起）", 2026: "2026（至 10/30）"}
rows = []
for y in yrs:
    rows.append(f'<tr><td class="l">{lab_year.get(y["year"], y["year"])}</td><td>{y["traded_days"]}</td>'
                f'<td><b>{wan(y["pnl"])}</b></td><td>{wan(y["annualized"])}</td><td>{wan(y["mdd"])}</td><td>{wan(y["worst_day"])}</td>'
                f'<td>{wan(y["p10"])} / {wan(y["p50"], True)} / {wan(y["p90"], True)}</td><td>{y["win_rate"]*100:.0f}%</td><td>{wan(y["avg_notional"])}</td></tr>')
rows.append(f'<tr class="tot"><td class="l">全期</td><td>{s["traded_days"]}</td>'
            f'<td><b>{wan(s["total_pnl"])}</b></td><td>{wan(s["annual_pnl"])}</td><td>{wan(s["mdd"])}</td><td>{wan(s["worst_day"])}</td>'
            f'<td>{wan(s["p10"])} / {wan(s["p50"], True)} / {wan(s["p90"], True)}</td><td>{s["win_rate"]*100:.0f}%</td><td>{wan(s["avg_notional"])}</td></tr>')

y25 = next(y for y in yrs if y["year"] == 2025); y26 = next(y for y in yrs if y["year"] == 2026)
months_mdd = -s["mdd"] / (s["annual_pnl"] / 12)
ymdd = [y["mdd"] for y in yrs]

html = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>negFill 策略績效摘要</title>
<style>
@page {{ size: A4; margin: 14mm 15mm 13mm; }}
:root {{ --ink:{INK}; --ink2:{INK2}; --muted:{MUTED}; --line:{LINE}; --blue:{BLUE}; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:#fff; color:var(--ink); font-family:"Noto Sans CJK TC",sans-serif; font-size:12.5px; line-height:1.6; }}
.kicker {{ color:var(--muted); font-size:11.5px; letter-spacing:.04em; }}
h1 {{ font-family:"Noto Serif CJK TC",serif; font-weight:600; font-size:25px; margin:2px 0 4px; }}
h2 {{ font-size:15px; margin:18px 0 8px; padding-bottom:4px; border-bottom:1px solid var(--line); }}
.version {{ background:#f4f6f9; border-left:3px solid var(--blue); padding:8px 12px; margin:10px 0 14px; color:var(--ink2); }}
.version b {{ color:var(--ink); }}
.tiles {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
.tile {{ border:1px solid var(--line); border-radius:6px; padding:10px 12px 9px; }}
.tile .lab {{ color:var(--ink2); font-size:11.5px; }}
.tile .val {{ font-size:24px; font-weight:600; font-variant-numeric:tabular-nums; line-height:1.3; margin:2px 0; white-space:nowrap; }}
.tile .val small {{ font-size:13px; font-weight:400; color:var(--ink2); margin-left:2px; }}
.tile .note {{ color:var(--muted); font-size:10.8px; line-height:1.5; }}
.read {{ margin:12px 0 0; padding-left:18px; color:var(--ink2); }}
.read li {{ margin:2px 0; }} .read b {{ color:var(--ink); }}
.chart-title {{ font-weight:600; font-size:12.5px; margin:14px 0 2px; }}
.chart-sub {{ color:var(--muted); font-size:11px; margin:0 0 4px; }}
.page2 {{ break-before:page; }}
ol.logic {{ padding-left:20px; margin:6px 0; }} ol.logic li {{ margin:4px 0; }}
table.rc {{ width:100%; border-collapse:collapse; margin:4px 0; }}
table.rc td {{ padding:4px 6px; border-bottom:1px solid var(--line); vertical-align:top; }}
table.rc td:first-child {{ white-space:nowrap; font-weight:600; width:118px; }}
table.yr {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; font-size:10.6px; }}
table.yr th {{ font-weight:500; color:var(--ink2); text-align:right; padding:4px 4px; border-bottom:1px solid var(--ink2); line-height:1.35; vertical-align:bottom; white-space:nowrap; }}
table.yr td {{ text-align:right; padding:4px 4px; border-bottom:1px solid var(--line); white-space:nowrap; }}
table.yr .l {{ text-align:left; }}
table.yr tr.tot td {{ border-top:1px solid var(--ink2); border-bottom:none; }}
.notes {{ color:var(--ink2); font-size:11px; padding-left:18px; margin:4px 0; }} .notes li {{ margin:2px 0; }}
</style></head><body>

<div class="kicker">策略績效摘要（回測）｜2026-11-03</div>
<h1>negFill 日內放空策略</h1>
<div class="version">本報告版本：<b>單檔上限 400 萬、每日總額上限 7,000 萬</b>；模型會<b>動態挑出信心分數較低的日子，降低部位或暫停交易</b>。
回測期間 2023/04/21～2026/10/30，共 {s["calendar_days"]} 個交易日。金額單位：新台幣萬元，已扣交易成本。</div>

<div class="tiles">
 <div class="tile"><div class="lab">預期年獲利</div><div class="val">約 {round(s["annual_pnl"], -2):,.0f}<small>萬/年</small></div>
  <div class="note">歷史平均。最弱的 2025 年 {wan(y25["pnl"])} 萬；2026 年至 10/30 已 {wan(y26["pnl"])} 萬（年化 {wan(y26["annualized"])} 萬）</div></div>
 <div class="tile"><div class="lab">累計最大回撤（MDD）</div><div class="val">{wan(s["mdd"])}<small>萬</small></div>
  <div class="note">{ymd(s["mdd_peak"])} 高點跌到 {md(s["mdd_trough"])}，{md(s["mdd_recover"])} 回到新高；約等於 {months_mdd:.1f} 個月的平均獲利</div></div>
 <div class="tile"><div class="lab">單日最大虧損</div><div class="val">{wan(s["worst_day"])}<small>萬</small></div>
  <div class="note">發生在 {ymd(s["worst_day_date"])}；單日最大獲利 +{s["best_day"]:,.0f} 萬</div></div>
 <div class="tile"><div class="lab">每日損益分佈 P10 / P50 / P90</div><div class="val">{wan(s["p10"])} / {wan(s["p50"], True)} / {wan(s["p90"], True)}<small>萬</small></div>
  <div class="note">10% 的日子虧超過 {abs(s["p10"]):.0f} 萬；一半的日子低於 +{s["p50"]:.0f} 萬；10% 的日子賺超過 {s["p90"]:.0f} 萬</div></div>
 <div class="tile"><div class="lab">平均每日交易金額</div><div class="val">{round(s["avg_notional"], -2):,.0f}<small>萬</small></div>
  <div class="note">當天累計放空的金額（上限 7,000 萬）；平均每天約 {s["avg_names"]:.0f} 檔股票</div></div>
 <div class="tile"><div class="lab">賺錢天數比例</div><div class="val">{s["win_rate"]*100:.0f}%</div>
  <div class="note">賠錢日多半小賠；總獲利主要來自最好的 10% 交易日</div></div>
</div>

<ul class="read">
 <li><b>獲利模式是「小賠多、大賺少」</b>：一般日子損益在小幅正負之間，總獲利主要來自少數大賺日。所以判斷好壞要看月、季，不要看單日。</li>
 <li><b>近期表現</b>：2026/09～10 的回撤是 {wan(s["recent_mdd"])} 萬，比 2025 年那次（{wan(s["mdd"])} 萬）小，{md(s["recent_recover"])} 已回到新高。歷年最大回撤在 {wan(max(ymdd))}～{wan(min(ymdd))} 萬之間。</li>
 <li><b>壓力測試</b>：用多組參數做蒙地卡羅模擬（{mc["n_paths"]:,} 個一年情境），一年內最大回撤一般約 {wan(mc["mdd"]["P50"])} 萬，較差情況可能到 {wan(mc["mdd"]["P10"])} 萬、極端情況約 {wan(mc["mdd"]["P5"])} 萬；
   年獲利在 90% 的情境高於 {round(mc["annual"]["P10"], -2):,.0f} 萬。</li>
</ul>

<div class="chart-title">累計損益（萬元）</div>
<div class="chart-sub">每天損益加總</div>
{chart_cum()}

<div class="page2">
<h2>策略怎麼運作</h2>
<ol class="logic">
 <li><b>只做空，當天結清。</b>盤中「先賣後買」（當沖放空），收盤前全部買回，不留倉過夜，沒有隔夜跳空的風險。</li>
 <li><b>由機器學習（ML）模型決定何時放空。</b>模型用過去幾年的盤中逐筆成交資料訓練，即時判斷每檔股票「從現在到收盤前下跌」的機率與幅度；分數夠高才下單。放空的對象主要是當天已經上漲的股票。</li>
 <li><b>多層風險控管</b>，限制單檔、單日與壞盤勢的損失：</li>
</ol>
<table class="rc">
 <tr><td>單檔上限 400 萬</td><td>同一檔股票一天最多放空 400 萬，避免單一股票失控。</td></tr>
 <tr><td>每日總額 7,000 萬</td><td>全部股票當天累計放空到 7,000 萬，就不再開新倉。</td></tr>
 <tr><td>個股停損／停利</td><td>依每檔股票的波動設出場價：漲過頭就認賠買回，跌夠了就先獲利了結；其餘收盤買回。</td></tr>
 <tr><td>進場時段</td><td>只在 09:00～12:05 開新倉；當天已有 3 檔停損，10:05 之後就不再開新倉。</td></tr>
 <tr><td>信心分數控管</td><td>模型每天動態評估隔天的信心分數；分數低的日子降低部位或暫停交易，避開對策略不利的盤勢。</td></tr>
</table>

<h2>逐年績效</h2>
<table class="yr">
<tr><th class="l">年度</th><th>交易天數<br>（天）</th><th>獲利<br>（萬）</th><th>年化獲利<br>（萬）</th><th>最大回撤<br>（萬）</th><th>單日最大<br>虧損（萬）</th><th>日損益<br>P10 / P50 / P90（萬）</th><th>賺錢天數<br>（%）</th><th>平均每日<br>交易金額（萬）</th></tr>
{''.join(rows)}
</table>

<div class="chart-title" style="margin-top:16px">每日損益分佈（{s["traded_days"]} 個交易日）</div>
<div class="chart-sub">大部分日子落在 {wan(s["p10"])}～{wan(s["p90"], True)} 萬之間，右邊（大賺日）的尾巴比左邊長</div>
{chart_hist()}

<h2>壓力測試（蒙地卡羅模擬）</h2>
<p style="margin:4px 0 6px;color:var(--ink2)">把風控參數（信心分數門檻、降額比例）和交易成本（每筆多加 0～0.04%）各取多個值，共 {mc["n_combos"]} 組；再把歷史每日損益以約 {mc["block"]} 天為一段重新隨機排列，模擬 {mc["n_paths"]:,} 個「一年」。用來估計參數或盤勢跟歷史不完全一樣時，可能出現的結果。</p>
<table class="yr" style="width:auto;min-width:62%">
<tr><th class="l">一年情境</th><th>一般情況<br>（中位數）</th><th>較差情況<br>（最差 10%）</th><th>極端情況<br>（最差 5%）</th><th>非常極端<br>（最差 1%）</th></tr>
<tr><td class="l">年獲利（萬）</td><td><b>{wan(mc["annual"]["P50"])}</b></td><td>{wan(mc["annual"]["P10"])}</td><td>{wan(mc["annual"]["P5"])}</td><td>{wan(mc["annual"]["P1"])}</td></tr>
<tr><td class="l">一年內最大回撤（萬）</td><td><b>{wan(mc["mdd"]["P50"])}</b></td><td>{wan(mc["mdd"]["P10"])}</td><td>{wan(mc["mdd"]["P5"])}</td><td>{wan(mc["mdd"]["P1"])}</td></tr>
</table>
<p style="margin:6px 0 0;color:var(--ink2)">管理上建議以 <b>−500 萬左右</b>作為一年內回撤的預期上緣（約 {(1-mc["p_mdd_gt_500"])*100:.0f}% 的情境不會超過）；{mc["n_paths"]:,} 個情境中沒有任何一個整年虧損。</p>

</div>
</body></html>"""

out_html = HERE / "negfill_400x7000_summary.html"; out_pdf = HERE.parents[2] / "Doc" / "negfill_400x7000_summary.pdf"
out_html.write_text(html, encoding="utf-8")
subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer", "--virtual-time-budget=5000",
                f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, capture_output=True)
print(out_pdf, out_pdf.stat().st_size, "| mdd months", round(months_mdd, 2))
