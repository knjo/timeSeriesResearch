# Baseline：確定採用的 timeSeries 點位

`timeseries_points.parquet` = 現行採用配置在全樣本(2025-09-03~2026-08-17)產出的成交點
（倉位上限後，57,830 筆 / 227 交易日）。欄位含 unikey、BidPrice1、TakerSell_CloseBP、
兩個模型分數、TimeBucket/ToRefBucket 與深度旗標(deep = FillLots_atLow ratio < -2)。

## 產生配置（凍結）
- 樣本：negFill 事件（cell 2/5 濾網），無深度過濾
- 特徵：無漏 30 特徵（txf_to_1330_return 排除）
- 模型：rolling Ridge net(60d)/netM(20d)、alpha 0.1、normal-only 訓練
- 門檻：16-cell(TimeBucket×ToRefBucket) prior-3-日分位數 q=0.3/0.1、單一分支（無 AB）
- 靜態濾網 + 倉位：OTC|L1Rate>0.25、SpreadPairElapsed>0.1、ToRef>0、rank<=100、
  accLots<avg 掛量、Position<200 萬/檔、價<=1000、hft_strick>-70、費用 19.3bp

## 驗證數字（點位重算基準，容差 ~1%）
| 區間 | pnl | Sharpe | MDD | 平均部位 | capw | 筆/日 |
|---|---|---|---|---|---|---|
| 全樣本 | 4,904 萬 | 5.00 | -341 萬 | 3,693 萬 | 58.5bp | 255 |
| 2026 Jan-Aug | 1,743 萬 | 2.43 | -341 萬 | 3,580 萬 | 32.7bp | 230 |
| holdout(2026-04+) | 1,111 萬 | 2.22 | -341 萬 | 3,599 萬 | 32.8bp | 202 |

實驗對照一律放 `test/`，與本 baseline 同窗比較。
