# vwap 窗格特徵對 negFill 點位的 IC 與模型增量

- 狀態：`RESEARCH_DIAGNOSTIC_ONLY`（scoped；不部署、不改正式 feature list）
- 日期：研究執行 2026-09-24 ～ 2026-09-29；資料窗 2026-01-02 ～ 2026-09-23（176 個交易日）
- 程式：[`test/20260924_vwap_feature/`](../../../test/20260924_vwap_feature/)；artifacts 只在該目錄 `artifacts/`（不進版控）
- 特徵來源：`src/features/definitions/vwap.py`（group `vwap`）與 `script/run_vwapFeature.py` 產出的
  `/media/kevin/SSD2/Data/vwapFeature/YYYYMMDD_vwapFeature.parquet`；契約見 `.agents/pages/features_registry.md`

## 問題與事前口徑

問題：把時間時鐘（1s/30s/60s/300s）與前日成交量時鐘（0.1/0.5/1/3/5%）的 VWAP 窗格特徵
接到正式 negFill 點位後，(1) 對訓練前 label 是否有跨月穩定的 IC（特別看 4 月後與 8 月）；
(2) 若有，放進現行 14 特徵 Ridge 模型後，用 notebook 同一套門檻／閘門／風控／BackTest
是否有增量。

- 點位與 label：完全依 `time_series_research_model.ipynb` cell 2–8 與
  `src/strategy/negFill/modelUpdate.py`：`BidPrice1>0`、ToRef∈(−1.5%,5%)、
  `day_amount_rank≤150 or AmountRank_canDayTrade≤150`、09:00:30<TransTime<12:30、
  `TrialMatch==0`、|TakerSell_CloseBP|<1e4；label = stop108 出場的 `TakerSell_CloseBP`
  → 去 `txf_beta_60d×txf_to_1330_bp` → 除以 `txf_residual_vol_to_1330` → `net`（Date×股票去均值，
  擇時）與 `netM`（Date 去均值，選股）。
- IC 口徑：每日 Spearman（跨該日全部點位），t = mean/std×√days；「穩定」看逐月 t 同號。
- 新特徵：檔案內 `VWR_*`／`VWZ_*` 加上點位層衍生的 `VWN_*`
  （`Imb=NetLots/Lots`、`Mid=(mid/VWAP−1)bp`、`BuySell`、`HftShare`、`LotsShare=Lots/PrevDayLots`、
  `logAvgLots`、`logElapsed`），共 9 個窗格；原始價位 `VW_Px_*` 不當特徵。
- 模型／回測：`test/20260924_vwap_feature/run_model_backtest.py` 重現 notebook cell 12–34
  （60 日 `net` 模型 + 20 日 `netM` 模型、alpha 10、feature_screening、16-cell prior-3-day
  門檻 q=0.3/0.1、cell 23 靜態閘門、cell 30 風控與 2.2σ tp/sl、`BackTest(fee=19.3)`）。
  訓練資料 2026-01-02 起，評估 2026-04-01 ～ 2026-09-23（121 日）。事前指定比較 4 組，
  paired 日損益差的 t 為判定。

## 1. 訓練前 IC（1–9 月，176 日）

`net`（擇時，同股同日內排序）：

| 特徵 | IC | t(4–7月) | t(8–9月) | 逐月 t 1..9 月 |
|---|---|---|---|---|
| ToRef（baseline 最強） | +0.43 | +42.8 | +36.2 | 全正 15.7～40.2 |
| Low_High（baseline） | +0.31 | +31.9 | +25.6 | 全正 13.0～29.2 |
| VWN_Imb_30s | +0.13 | +33.5 | +19.2 | 全正 9.5～19.2 |
| VWN_Imb_60s | +0.15 | +31.9 | +19.3 | 全正 9.4～19.7 |
| VWN_Imb_v3 / v5 | +0.20 | +30～31 | +16.3～16.5 | 全正 9.3～19.0 |
| VWN_Mid_v5 | +0.15 | +19.2 | +12.6 | 全正 6.0～12.5 |
| VWR_Px_v01_v1 | +0.08 | +13.9 | +12.5 | 全正 |

`netM`（選股，跨股排序）：

| 特徵 | IC | t(4–7月) | t(8–9月) | 逐月 t 4..9 月 |
|---|---|---|---|---|
| B1_A1B1（baseline 最強） | −0.02 | −5.7 | −3.2 | −3.4 −1.4 −3.7 −2.7 −2.9 −1.5 |
| VWN_Imb_60s | +0.03 | +7.2 | +1.3 | +2.4 +4.6 +5.3 +2.6 +0.3 +1.5 |
| VWR_LotsShare_1s_30s | −0.02 | −3.9 | −4.2 | −2.7 −0.4 −1.4 −3.3 −2.5 −3.5 |
| VWN_Imb_v1 / v05 | +0.02 | +5.7～6.0 | ≈0 | 8–9 月歸零 |

讀法：Imb 家族對 `net` 的 IC 每月同號且很大，但這是「同股同日內」的結構訊號，
與 ToRef／Low_High 屬同一類（模型已有）；對 `netM` 只有 `Imb_60s`（8–9 月轉弱）與
`LotsShare_1s_30s`（8–9 月最穩、方向為負）有可辨識但很小的 IC（|IC|≈0.02–0.03）。

## 2. 放進模型與 BackTest（評估 2026-04-01 ～ 09-23）

| 特徵組 | 特徵數 | 成交 | PnL（萬） | 日波動 | Sharpe | MDD | 模型 IC net / netM | vs baseline 日損益差 t |
|---|---|---|---|---|---|---|---|---|
| baseline（現行 14） | 14 | 25,967 | 4,019.7 | 133.8 | 3.94 | −350.0 | 0.481 / 0.014 | — |
| vwap_imb：+Imb_60s, Imb_v3, Mid_v5 | 17 | 26,299 | 4,070.7 | 135.2 | 3.95 | −349.6 | 0.479 / 0.017 | 0.69 |
| vwap_six：imb + Px_v01_v1, LotsShare_1s_30s, HftShare_v1 | 20 | 26,228 | 4,049.3 | 134.9 | 3.94 | −338.3 | 0.479 / 0.022 | 0.39 |
| vwap_px：+Px_30s_300s, Mid_300s, Imb_60s | 17 | 26,258 | 4,133.1 | 134.7 | 4.03 | −331.7 | 0.480 / 0.019 | 1.66 |

逐月 PnL（萬）：

| 月 | baseline | vwap_imb | vwap_six | vwap_px |
|---|---|---|---|---|
| 04 | 861.9 | 900.2 | 893.3 | 880.2 |
| 05 | 366.4 | 334.5 | 334.6 | 368.1 |
| 06 | 941.5 | 988.3 | 971.9 | 1,022.3 |
| 07 | 1,318.7 | 1,299.6 | 1,318.6 | 1,303.4 |
| 08 | 261.2 | 304.3 | 272.1 | 288.6 |
| 09 | 270.0 | 243.7 | 258.8 | 270.4 |

模型層：`net|net` 日 IC 全部 ≈0.48 不變；`netM|netM` 由 0.014 升到 0.017–0.022。
選單幾乎相同（成交數差 1%），PnL 差 +0.7%～+2.8%，最好的一組 paired t 也只有 1.66，
且是四組中挑出來的（selection bias 未校正）。

後置診斷（baseline 已選成交內，是否還能再切）：`VWR_Px_30s_300s` 五分位平均 bp
24.9 → 64.1（日 IC 0.089、t 4.1），`VWN_Mid_300s` 26.7 → 55.8；但 4 月的 IC 為負
（−0.06），且最低分位仍有 +25 bp。用靜態門檻砍掉最低 20% 是 in-sample 的，六個月裡四個月
反而少賺（4 月 862→683、6 月 942→664、7 月 1319→1103、9 月 270→255），不能當 gate。

## 結論與限制

- IC：對 `net` 有非常穩定的大 IC，對 `netM` 只有小 IC；「穩定」的部分不是新資訊。
- 模型／回測：無可辨識增量（Sharpe 3.94 → 3.94～4.03，日損益差 t ≤ 1.66）。
  不建議加入正式 14 特徵，也不建議當 gate。
- 可能的下一步（未驗證）：把 `VWR_Px_30s_300s`／`VWN_Mid_300s` 當排序或倉位權重軸而非
  硬門檻；或依 DISCOVERY_PROTOCOL 把 `Imb`／`LotsShare` 當新 condition 做 shared-pool bias
  研究（需先過 positive control）。
- 限制：notebook replica 用 `legacy_v1` 補值（與 notebook 相同、非嚴格因果）；資料只有
  2026；vwapFeature 只從 2026-01-02 起有檔，模型 1–3 月暖身；四組特徵組合都看過再報告，
  最佳組的 t 需打折；未跑正式 `modelUpdate.py`。

## 重現

```bash
uv run script/run_vwapFeature.py --start 2026-01-01 --end 2026-09-23
uv run python src/research/timeSeries/test/20260924_vwap_feature/build_signal.py --start 20260101 --end 20260331
uv run python src/research/timeSeries/test/20260924_vwap_feature/build_signal.py --start 20260401 --end 20260923
uv run python src/research/timeSeries/test/20260924_vwap_feature/run_model_backtest.py \
  --signal src/research/timeSeries/test/20260924_vwap_feature/artifacts/signal_20260101_20260331.parquet \
  --signal src/research/timeSeries/test/20260924_vwap_feature/artifacts/signal_20260401_20260923.parquet \
  --set baseline --set vwap_imb --set vwap_six --set vwap_px \
  --extra '{"vwap_imb": ["VWN_Imb_60s","VWN_Imb_v3","VWN_Mid_v5"], "vwap_six": ["VWN_Imb_60s","VWN_Imb_v3","VWN_Mid_v5","VWR_Px_v01_v1","VWR_LotsShare_1s_30s","VWN_HftShare_v1"], "vwap_px": ["VWR_Px_30s_300s","VWN_Mid_300s","VWN_Imb_60s"]}' \
  --eval_start 20260401 --out full2026_eval0401
```

訓練前 IC 的逐日計算腳本在 `test/20260924_vwap_feature/ic/`（`negfill_ic.py`、`maker_ic.py`、`vwap_ic_lib.py`、
`compact.py`；輸出目錄由各腳本的 `OUT_DIR` 決定），邏輯即上文口徑；`build_signal.py` 產出的訊號表可
直接重算。
