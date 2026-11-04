from pathlib import Path
import json, hashlib, sys
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('/home/kevin/Project/HFT')
base = Path('/tmp/negfill_history_20261014')
recent = Path('/tmp/negfill_recent_20261014')
native = pl.read_parquet(recent / 'native_exit_factor.parquet').with_columns(
    (pl.col('BidPrice1') * (pl.col('TakerSell_CloseBP') - 19.3) / 10).alias('net'),
    (pl.col('BidPrice1') * pl.col('market_native_bp') / 10).alias('market_native'),
    (pl.col('BidPrice1') * 1000).alias('notional'),
).with_columns((pl.col('net') - pl.col('market_native')).alias('residual_net'))
native_daily = native.group_by('Date').agg(pl.len().alias('trades'), *[
    pl.col(c).sum() for c in ['net', 'market_native', 'residual_net', 'notional']
]).sort('Date')
native_daily.write_csv(base / 'native_attribution_daily.csv')
periods = [
    ('DD 高點後至谷底', '20260910', '20261007'),
    ('DD 高點後至截止日', '20260910', '20261013'),
    ('9/15 起', '20260915', '20261013'),
    ('9 月', '20260901', '20260930'),
    ('10 月截至 13 日', '20261001', '20261013'),
]
nr = []
for label, start, end in periods:
    z = native_daily.filter(pl.col('Date').is_between(pl.lit(start), pl.lit(end)))
    v = {'period': label, 'start': start, 'end': end, 'days': z.height, 'trades': z['trades'].sum()}
    for c in ['net', 'market_native', 'residual_net']:
        v[c] = z[c].sum()
        v[c + '_bp'] = v[c] / z['notional'].sum() * 10000
    nr.append(v)
pl.DataFrame(nr).write_csv(base / 'native_attribution_periods.csv')

fh = pl.read_parquet(root / 'src/research/timeSeries/test/20260921_fh_neg_rejection_execution_gate_ablation/artifacts/gate_factorial_20261013_v1/accepted_signals.parquet')
peers = []
for label, start, end in [periods[2], periods[4]]:
    for variant in ['T1L1S1', 'T1L0S0']:
        z = fh.filter(pl.col('Date').is_between(pl.lit(start), pl.lit(end)) & (pl.col('variant') == variant))
        cash = z.select((pl.col('BidPrice1') * (pl.col('label_raw_bp') - 19.3) / 10).sum()).item()
        peers.append({'period': label, 'variant': variant, 'trades': z.height, 'net': cash, 'net_bp': cash / (z['BidPrice1'].sum() * 1000) * 10000})
pl.DataFrame(peers).write_csv(base / 'fh_recent_comparison.csv')

ic = pl.read_csv(base / 'model_ic_daily.csv', schema_overrides={'Date': pl.String})
icp = json.loads((base / 'model_ic_periods.json').read_text())
ps = pl.read_csv(base / 'period_summary.csv', schema_overrides={'start': pl.String, 'end': pl.String})
for r in icp:
    p = ps.filter(pl.col('period') == r['period']).row(0, named=True)
    z = ic.filter(pl.col('Date').is_between(pl.lit(p['start']), pl.lit(p['end'])))
    for c in ['timing_IC', 'selection_IC', 'selection_raw_IC']:
        r[c + 'IR_daily_unannualized'] = z[c].mean() / z[c].std()
(base / 'model_ic_periods.json').write_text(json.dumps(icp, indent=2, ensure_ascii=False))

# Retain the exact source cells and module fingerprints used in this replay.
nb = root / 'src/research/timeSeries/time_series_research_model.ipynb'
book = json.loads(nb.read_text())
cells = {str(i): ''.join(book['cells'][i].get('source', [])) for i in [2,4,5,6,8,10,11,12,13,14,23,24,25,27,34]}
(base / 'notebook_cells_snapshot.json').write_text(json.dumps(cells, indent=2, ensure_ascii=False))
sources = [nb, root/'src/research/rolling_model.py', root/'src/research/tool.py', root/'src/research/backTest.py', root/'src/dataloader/txfDataLoader.py']
(base / 'source_fingerprints.json').write_text(json.dumps({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2))
files = sorted((root / 'src/research/timeSeries/data').glob('*.parquet'))
pl.DataFrame([{'file': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in files if '20230901' <= p.stem <= '20261013']).write_csv(base/'input_inventory.csv')

# Actual holding-period factor chart: distinct from the full-history close-horizon chart.
z = native_daily.filter(pl.col('Date') >= '20260910')
from datetime import datetime
dates = [datetime.strptime(d, '%Y%m%d') for d in z['Date']]
fig, ax = plt.subplots(figsize=(11, 4.5))
for c, name in [('net', 'negFill net'), ('market_native', 'TXF factor during actual holding'), ('residual_net', 'remainder after factor, net of costs')]:
    ax.plot(dates, np.cumsum(z[c].to_numpy()) / 10000, label=name)
ax.axhline(0, color='grey', linewidth=.7)
ax.set(title='Since 2026-09-09 equity peak | actual entry-to-exit attribution', ylabel='TWD 10,000')
ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(base/'recent_native_attribution.png', dpi=160); plt.close(fig)

monthly = pl.read_csv(base/'monthly_summary.csv')
eps = pl.read_csv(base/'drawdown_episodes.csv', infer_schema_length=None)
def money(v): return f'{v/10000:+,.2f}'
def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])
annual = table(['年度','交易日／活躍日','筆數','淨損益（萬）','Net BP','日曆 Sharpe','年度內 MDD（萬）'], [[p['period'],f"{p['days']}／{p['active_days']}",f"{p['trades']:,}",money(p['net']),f"{p['net_bp']:.2f}",f"{p['sharpe']:.2f}",money(p['MDD'])] for p in ps.head(3).to_dicts()])
native_table = table(['期間','交易日','淨損益（萬）','持倉期間市場因子（萬）','扣成本後剩餘損益（萬）'], [[r['period']+f" {r['start']}–{r['end']}",r['days'],money(r['net']),money(r['market_native']),money(r['residual_net'])] for r in nr])
month_table = table(['月份','交易日／活躍日','正／負損益日','交易筆數','淨損益（萬）','Net BP','月內 MDD（萬）'], [[str(p['period']),f"{p['days']}／{p['active_days']}",f"{p['positive_days']}／{p['negative_days']}",f"{p['trades']:,}",money(p['net']),f"{p['net_bp']:.2f}",money(p['MDD'])] for p in monthly.to_dicts()])
episode_table = table(['權益高點','谷底','回復前高','回撤（萬）'],[[p['peak_date'],p['trough'],p['recovered'] or '截至 10/13 尚未回復',money(p['dd'])] for p in eps.head(5).to_dicts()])
ic_table = table(['期間','Timing IC','Timing ICIR','Selection IC','Selection ICIR'],[[r['period'],f"{r['timing_IC']:.4f}",f"{r['timing_ICIR_daily_unannualized']:.3f}",f"{r['selection_IC']:.4f}",f"{r['selection_ICIR_daily_unannualized']:.3f}"] for r in icp[:6]])
peer_table = table(['期間','FH 版本','筆數','淨損益（萬）','Net BP'], [[p['period'],p['variant'],p['trades'],money(p['net']),f"{p['net_bp']:.2f}"] for p in peers])
rel = '../../../test/20261014_negfill_market_attribution/artifacts/v1'
report = f'''# negFill：2024 起績效與 2026 年 9–10 月回撤診斷

範圍：`scoped`；狀態：`RESEARCH_DIAGNOSTIC_ONLY`。研究問題：本波 DD 主要是共同市場逆風，還是 negFill 相對轉弱？主要估計量為原 notebook 淨損益與實際持倉期間台指期因子之差。

執行日期：2026-10-14；統一資料截止 **2026-10-13 收盤**。對象是使用者指定的 **研究／notebook 回測**。沿用當前 notebook 的模型、門檻、容量、停利停損與 19.3 bp 成本，重新滾動訓練並重播；沒有變更正式策略或替換模型。這是**現行規則的歷史重播，不是當年實盤或乾淨的歷史 OOS 紀錄**。

## 判斷

**整波 DD 不能主要歸因於「市場不利放空」；9 月的 negFill 個股／選擇效果轉弱，10 月則有明顯共同市場逆風。**

- 2024、2025、2026 截至 10/13 的成本後損益分別為 +3,861.73、+2,487.85、+5,357.01 萬。全期仍獲利，不代表近期 DD 可以忽略。
- 本波從 9/9 權益高點到 10/7 谷底 **-407.62 萬**，是 2024 起連續權益曲線的**第二深回撤事件**；至 10/13 尚距前高 **-358.06 萬**。
- 實際進出場期間的台指期因子，在谷底前反而貢獻 **+10.70 萬**，扣除它後剩餘 **-418.32 萬**。至 10/13，市場因子累計僅 -3.64 萬，剩餘 -354.42 萬。這個剩餘項包含個股、選擇、執行／出場與成本，不能全稱為純模型 alpha。
- 單看 **10/1–10/13**，策略 -54.91 萬、市場因子 -86.19 萬、剩餘 +31.28 萬。近期市場逆風是真實現象，但不是 9 月以來整波回撤的充分解釋。
- 在相同日期、時段及 ToRef 區間，對照其他 negFill 候選股票，DD 谷底前所選部位的到收盤 residual 平均每日 **落後候選池 30.93 bp**；10 月反而領先 18.05 bp。方向上支持「9 月選擇效果轉弱、10 月有所回復」，不支持直接認定整套模型永久失效。

## 「控制版」是什麼

之前文件的 `T1L1S1` 是 **fh_neg_rejection 的固定參考版本**，不是 negFill，也不是大盤控制組。兩個 FH 版本使用相同 D/G/B 事件 selector、short stop108-or-close、每筆一張與成本。

- `T=1`：保留正 ToRef，配合母池上界即 `0 < ToRef < 0.05`。
- `L=1`：上市股 `MD_L1Rate_30_re > 0.25`；上櫃豁免。
- `S=1`：`SpreadPairElapsed > 0.1`。
- `T1L0S0`：先前選定的擴量候選，只保留 T，移除 L／S。不是根據本次 10 月結果重新挑選。

## 2024 起完整績效

金額為新台幣萬元；每個接受事件一張。Net BP = 淨損益／累計進場名目金額 × 10,000，不是帳戶報酬率。Sharpe = 每日日損益平均／樣本標準差 × √252，包含已確認來源完整但沒有接受訊號的交易日。年度內 MDD 以年初權益零點重設；跨年連續回撤另見下表。2026 是截至 10/13 的部分年度。

{annual}

![完整歷史績效與回撤]({rel}/history_attribution.png)

歷史圖橘線採「進場至 13:30」因子，綠線含原停利停損相對收盤的差額；它與下一節的**實際持倉期間因子**是不同分解，不能互換。全期會計恆等式：原始策略淨損益 = 進場至收盤市場因子 + 個股到收盤毛 residual + 原 TP/SL 相對收盤損益差 − 原成本。

## 這波的市場與策略拆解

每筆市場因子 = `−歷史 beta60d × (出場時 TXF mid／進場 TXF mid − 1) × 進場名目金額`。出場時間採 notebook 真正首次觸及 TP／SL 的時間；沒有觸及則 13:30。TXF 使用既有 loader 的選約／報價標準化，backward as-of 容忍 10 秒；4,236 筆均配對成功，出場報價最老 699 ms。使用來源既有歷史 beta，沒有用這段績效重估 beta。

{native_table}

這是 beta 因子的**事後損益歸因**，沒有模擬可成交的期貨對沖張數、基差與期貨成本，不是對沖策略回測。剩餘值也會包含 beta 估計誤差、產業／風格共同波動，不能把所有剩餘損失都因果歸咎於模型。

![本波實際持倉因子拆解]({rel}/recent_native_attribution.png)

市場廣度提供另一個角度：上市＋上櫃一般四碼股票、當日量至少 50 萬股、有效開收參考價，股票等權的平均每日**多方開盤到收盤**報酬，2024／2025／2026 分別 -13.34／-11.16／-13.69 bp；9/15–10/13 改為 **+18.22 bp**，10 月為 +7.51 bp。即廣泛日內放空環境確實變差。這是使用事後成交量篩選的市場診斷，不能視為可交易基準；它也不是 negFill 的實際進場時間、持倉股票與權重，所以不能拿它取代上表。不能用指數昨收到今收的漲幅直接解釋盤中空單損益。

## 與過去 DD 比較

依整條 2024 起權益曲線，從前高開始到再次回到前高算一個回撤事件，按最大金額排序：

{episode_table}

2025/7–8 月回撤的到收盤市場因子約 -276.67 萬，2025/10 月約 -202.28 萬；本波同口徑僅 -0.55 萬。與歷史上明顯由市場帶動的 DD 相比，本波性質不同。這些是同一個**收盤持有假設**下的比較，沒有宣稱 2024–2025 也已完成逐筆實際出場時間的市場歸因。

9/16、9/18 分別損失約 **222.86、200.06 萬**，是需要解釋的集中虧損日。9/15 起較大虧損股票包括 6672（-71.63 萬）、3441（-70.65 萬）、8358（-48.81 萬）、2426（-38.32 萬）。這是事後貢獻排序，不是建議直接排除這些股票；也不是把事後壞日篩掉的績效。

## 選股與進場時機是否一起失效

使用原母池與原訓練 label，逐日計算 Spearman IC；ICIR 為日 IC 平均／樣本標準差，**未年化**。Timing 與 Selection label 的去平均方式不同，兩者 IC 絕對大小不可互比；應看各自歷史變化。日 IC 的 5 日 block-bootstrap 90% 區間另存 JSON；沒有用它重新選策略。

{ic_table}

9 月 Selection IC 約 -0.008，較 8 月 +0.086 轉弱；10 月 +0.055，僅 8 日。Timing IC 仍約 +0.47。

兩種候選池對照補充模型 IC：

1. **同股票進場比較**：排除已接受事件後，與同日、同股票、同時段及同 ToRef cell 的未選 negFill 事件比較；每 cell 至少 10 筆，覆蓋 91.20% 接受交易。9/15 起選中進場的到收盤 residual 仍平均每日多 **9.44 bp**（90% block-bootstrap 區間 +5.66 至 +11.75）。故同一股票內的進場選擇仍有正貢獻。
2. **跨股票候選比較**：同日、同時段及同 ToRef cell，對未選母池先取每股票平均，再股票等權；至少 5 檔，覆蓋 98.85% 接受交易。所選交易按進場名目金額加權，再取每日差額平均。2024／2025／2026 全年分別領先 **20.30／22.24／46.52 bp**；DD 谷底前落後 **30.93 bp**，9/15 起落後 3.40 bp，10 月領先 18.05 bp。

兩種比較均使用「持有到收盤 residual」統一 label，成本相同；不是原生 TP/SL 的可成交替代組合，也未對未選池重跑相同容量。差額包含模型、條件、容量、股票與事件組合，不能稱為隔離某一模型的受控 A/B。時段切點 RemainSeconds = 3,300／6,900／10,500／14,100；ToRef 切點 = 0／1%／2%／3%／5%。

## FH 能否解決最近的問題

{peer_table}

同段 negFill 的 Net BP：9/15 起 **-32.37**，10 月 **-14.82**。FH 控制版最近較抗跌；但擴量候選並沒有穩定勝過控制版，9/15 起甚至已轉負。FH 與 negFill 的進場、用量、選股及退出規則不同，BP 也不能完全消除組成差異；這不是換策略的受控測試。本次沒有證據支持直接拿擴量 FH 取代 negFill。

FH 沒有本次重建的 2024 對照；2025 舊 frozen 結果早於 OTC 參考價資料修正，不能與本次重新讀取資料的 negFill 當成乾淨同資料歷史 A/B。近期 FH 延用原研究，仍為 `RESEARCH_SHADOW_ONLY`。

## 逐月結果

正／負日數只按淨損益符號；無交易日另由交易日／活躍日差額揭露。10 月僅至 13 日。

{month_table}

## 可重現範圍與限制

- 原 notebook：`src/research/timeSeries/time_series_research_model.ipynb`。讀取 2023-09-01–2026-10-13 的 755 個來源日，2024 起評估 672 日、207,425 筆；兩個 Ridge 模型沿用 60／20 日 rolling、alpha=10、原 feature screening 與門檻，沒有為本次 DD 重選參數。
- 原 prior 3 日分格門檻、上市 L1 > 0.2／上櫃豁免、SpreadPairElapsed > 0.1、正 ToRef、金額排名、產業／分點等條件及容量不變。原進場截點、每股 500 萬／總計 10,000 萬等限制沿用 notebook。原 `stopTradingDay` mask 仍是註解，沒有自行啟用。
- native 出場沿用前期 residual 波動的 2 倍 TP／SL、RefPrice 1.08 上限、原 tick rounding 與首次觸及規則；非 FH 的固定 stop108。研究母池／ToRef 仍沿用 notebook 既有定義，沒有因本次分析修正來源或改正式 feature。
- 當前 rolling model 的 `legacy_v1` 缺值處理仍包含全資料 median fallback；它構成歷史重播的非純 prior-only 限制。加上今天的 feature／規則可能經過歷史研究挑選，**不能將這些數字稱為乾淨 OOS 或當年線上績效**。本次保留 notebook 口徑，將問題揭露，沒有默改模型再混稱同一條曲線。
- 未使用過期的根目錄 `signalwithRisk_df.parquet` 或不同版本的 `daily_shortPL_460d.csv` 拼接。2024 起延長訓練歷史與原 260 日 notebook 重播的近 6,644 筆事件鍵、接受筆數、出場 label 完全一致。
- 以原 `BackTest` 再核對全期，每日現金損益最大差 **2.33e-10 元**。成本為總計 19.3 bp × 進場名目金額，沒有重複扣除。原 BackTest Sharpe 使用活躍日，本報表另明示完整交易日日曆；損益與 DD 的逐日數列一致。4 個零交易日均有來源資料。
- DD 起訖、最近月份與股票排名都是事後診斷，短期樣本不能推出下一段一定回復，也不構成新進場／停機條件的驗證。資料固定截至 10/13，沒有把 10/14 盤中或尚未納入的來源補零。

## 產出

- [逐日損益與收盤口徑拆解]({rel}/attribution_daily.csv)、[逐月績效]({rel}/monthly_summary.csv)、[回撤事件]({rel}/drawdown_episodes.csv)
- [實際持倉因子逐日]({rel}/native_attribution_daily.csv)、[期間拆解]({rel}/native_attribution_periods.csv)
- [模型 IC／ICIR 與區間]({rel}/model_ic_periods.json)、[同股票對照]({rel}/matched_selection_summary.json)、[跨股票對照]({rel}/matched_across_stock_summary.json)
- [接受交易]({rel}/notebook_trades.parquet)、[近期實際出場因子]({rel}/native_exit_factor.parquet)
- [驗證]({rel}/verification.json)、[重播 manifest]({rel}/replay_manifest.json)、[程式指紋]({rel}/source_fingerprints.json)、[原 notebook cells 快照]({rel}/notebook_cells_snapshot.json)

重現腳本存於本研究 test 目錄的 `audit_scripts/`，保留本次實際執行程式，使用 `uv run python`。先執行 recent／history replay 及 market，再 native factor、history analysis、across-stock diagnostic、verify、finalize。腳本固定使用 `/tmp/negfill_recent_20261014`、`/tmp/negfill_history_20261014` 工作目錄；大型母池可重新產生，沒有保留進正式 artifact。來源 notebook 必須對回 manifest 指紋；若來源行情或程式更新，不能宣稱完全重現本次快照。這些是本次研究核對腳本，不是新 production pipeline。
'''
(base/'REPORT.md').write_text(report)
print('Wrote final tables, report and provenance:', base)
