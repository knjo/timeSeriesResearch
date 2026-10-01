# `fh_neg_rejection`：全下 vs model gate、Ridge vs Elastic Net

Status: `RESEARCH_DIAGNOSTIC_ONLY`  
Classification: `scoped model-after-condition ablation`  
Source condition: `fh_neg_rejection` short  
Source data: `20250211–20260825` frozen model input  
Development: `20250513–20251231`  
Evaluation: `20260102–20260825`  
Cost: `19.3bp` per accepted event

## 直接答案

1. **如果「直接全下」是 condition 通過 static execution/capacity 後，不再用模型
   score gate 篩事件，2026 evaluation 的確比較好。** Static-only 是 24,685 筆、
   `+1,076.12萬`、net `+31.728bp`、daily Sharpe `2.550`；2025 development 選出的
   Ridge timing gate 是 17,739 筆、`+653.91萬`、net `+24.258bp`、Sharpe `2.149`。
   全下多賺 `+422.21萬`，net 多 `+7.470bp`，Sharpe 多 `+0.401`；代價是交易多
   39.2%，MDD 從 `-136.94萬` 變差到 `-158.29萬`。
2. **把 Ridge 直接換成 Elastic Net 沒有實質差異，也沒有改善。** 同一個 2025
   selection rule 仍選 timing-only；2026 Elastic Net 是 17,726 筆、`+652.61萬`、
   `+24.223bp`、Sharpe `2.144`，相對 Ridge 少 `1.30萬`、少 `0.035bp`、Sharpe
   少 `0.005`。這些差距遠小於策略本身的日間波動。

這裡的「feature screening」必須和「event/model score gate」分開：原 Ridge
本來就是 `feature_screening=False`，38 個 manifest features 每個 rolling fit 全部進模型；
不是先挑 feature 才訓練。少下單來自 score quantile gate。若使用者說的「不做提前篩選
feature」只是關閉 feature screening，現況早已如此；若意思是「不經模型直接全下」，
才是上面的 static-only 對照。

![2026 cumulative comparison](../../../test/20260921_fh_neg_rejection_model_ablation/artifacts/model_family_v1/holdout_cumulative_comparison.png)

## 2026 evaluation 結果

所有列使用同一個 selector、barrier-or-close short label、static execution、逐股容量與
19.3bp 成本。Static-only 沒有參加 2025 model-gate 選型；它是共同 baseline。

| Model | Gate | Trades | Net PnL（萬元） | Net BP | Daily Sharpe | Daily win | MDD（萬元） |
|---|---|---:|---:|---:|---:|---:|---:|
| none | **static-only／全下** | 24,685 | **+1,076.12** | **+31.728** | 2.550 | 49.68% | -158.29 |
| Ridge | timing-only（2025 selected） | 17,739 | +653.91 | +24.258 | 2.149 | 47.10% | -136.94 |
| Elastic Net | timing-only（2025 selected） | 17,726 | +652.61 | +24.223 | 2.144 | 47.10% | -137.07 |
| Ridge | selection-only（post-hoc diagnostic） | 21,998 | +972.70 | +31.542 | 2.590 | 49.03% | -148.33 |
| Elastic Net | selection-only（post-hoc diagnostic） | 21,998 | +972.35 | +31.531 | 2.589 | 49.03% | -148.10 |
| Ridge | dual（post-hoc diagnostic） | 16,437 | +636.42 | +25.139 | 2.195 | 47.10% | -130.49 |
| Elastic Net | dual（post-hoc diagnostic） | 16,423 | +641.88 | +25.375 | 2.213 | 47.10% | -130.32 |

`selection-only` 在 2026 看起來比 2025 已選定的 timing-only 好，甚至 Sharpe 略高於
static-only；但它沒有贏得事前凍結的 2025 selection，不能用 2026 結果事後換 gate。
它能說明的只是：問題不在 Ridge 還是 Elastic Net，而在原本選到的 timing score gate
沒有帶來 holdout lift。

## Elastic Net contract 與實際差異

Elastic Net 沒有另調參找 winner：固定 `l1_ratio=0.5`、原 requested alpha `0.1`、
`feature_screening=False`、同一 38-feature manifest、60/20 prior event-active day windows、
相同兩個 targets、prior-3-calendar-session thresholds、warmup、execution 與 capacity。

sklearn Ridge 與 ElasticNet 的 objective 對 alpha 使用不同 normalization，不能直接把
兩者都填 `0.1` 就稱為公平。這次每個 rolling fit 用
`alpha_enet = 0.1 / (n_train × (1 - l1_ratio))`，讓 Elastic Net 的 L2 項先對齊
Ridge alpha 0.1，再加入 L1。實際 effective alpha 範圍為 timing
`2.09e-6–1.31e-4`、selection `5.85e-6–1.31e-4`。

它幾乎沒有改變排序：

- 2026 timing score：Pearson `0.999990`、Spearman `0.999988`；
- 2026 selection score：Pearson `0.999979`、Spearman `0.999970`；
- 兩個 development-selected timing ledgers：17,696 筆共同事件，Jaccard `99.589%`；
- Elastic Net 每個 fit 的 38 係數，timing 平均只有 `0.93%` 精確為零、selection
  `0.59%`，兩者中位數都是 **0 個零係數**。

因此這次 L1 並沒有形成有意義的 feature selection；threshold 又只看 score rank，
近乎相同的排序自然產生近乎相同的交易。

### 收斂限制

為避免把未收斂誤認為「兩模型相同」，max iterations 從 10,000 提高到 50,000
重物化。兩版的 selected trade count、PnL 與上述相關係數在報告精度內不變；但 375
個 daily fits 中，timing 仍有 54 個、selection 45 個碰到 50,000 上限。這反映目前
未標準化的 feature geometry 不適合用 coordinate-descent Elastic Net 做乾淨的稀疏
估計。它不影響「直接替換沒有觀察到改善」的結論，但不足以宣稱所有妥善標準化、
另行 tune 的 Elastic Net 都不可能不同。

## 判讀與建議

目前最合理的結論不是「換模型」，而是：

- 若目標是這條 condition 在目前長持倉 label 下的 PnL／Sharpe，保留
  **no-model-gate static baseline** 比原本 development-selected timing gate 更合理；
- Ridge 換成同契約 Elastic Net 沒有理由，因為交易與績效幾乎完全相同，還增加
  coordinate-descent 收斂問題；
- 若目標真的是 feature reduction／可解讀性，應另立版本，加入 strictly-prior scaler，
  只在 development 選 alpha／l1-ratio，再用新的 outer first-live period 評估；不能把
  這次 2026 看過的結果拿來 tune。

「全下較好」仍不等於可直接部署。Static baseline 的 MDD 較大，而且這是逐 event
一張、Bid taker short、stop108-or-close 的理想化 replay，沒有 borrow、impact、latency、
同時曝險與 portfolio-wide capital cap。另有 selector-only static extension 到 9/17，
但只有 15 個 first-live sessions、利潤集中於少數日期；本 model-family 比較刻意停在
凍結的 8/25 model input，避免把不同資料窗混成模型效果。

## Lineage、artifacts 與重現

Source feature manifest：`filllots_at_high_short_context@1`，hash
`09a29ca300424620a1cbf6331e585072d91a9a6fa2c8dd7ea4e4f1d6e6559468`。
Static-only 2026 重建與凍結 source summary 在 trades、active days、PnL、net BP、Sharpe
逐項 parity。新 study 不改寫原 Ridge predictions、source model input 或 production。

- [study contract](../../../test/20260921_fh_neg_rejection_model_ablation/study_spec.json)
- [run manifest](../../../test/20260921_fh_neg_rejection_model_ablation/artifacts/model_family_v1/run_manifest.json)
- [model-family summary](../../../test/20260921_fh_neg_rejection_model_ablation/artifacts/model_family_v1/model_family_summary.parquet)
- [comparison diagnostics](../../../test/20260921_fh_neg_rejection_model_ablation/artifacts/model_family_v1/comparison_diagnostics.json)
- [2026 daily comparison](../../../test/20260921_fh_neg_rejection_model_ablation/artifacts/model_family_v1/holdout_daily_comparison.parquet)
- [Elastic Net selected ledger](../../../test/20260921_fh_neg_rejection_model_ablation/artifacts/model_family_v1/elastic_net_selected_signals.parquet)

```bash
UV_CACHE_DIR=/tmp/hft-uv-cache MPLCONFIGDIR=/tmp/matplotlib-hft \
uv run --no-sync python \
  src/research/intradaySignal/experiments/20260921_fh_neg_rejection_model_ablation/run_comparison.py \
  --n-jobs 8

UV_CACHE_DIR=/tmp/hft-uv-cache \
uv run --no-sync python -m unittest \
  src/research/intradaySignal/experiments/20260921_fh_neg_rejection_model_ablation/test_model_family.py \
  src/research/intradaySignal/tests/test_causal_preprocessing.py
```

本次新增 3 個 Elastic Net model-family tests，加上 10 個 causal preprocessing/model
integration tests 均通過；原 signal study 的 4 個契約測試亦單獨通過。
