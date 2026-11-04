# 20261005_dd_market_state_gate — negFill 空單停手日閘門（研究版）

**問題**：三年來幾次回撤能否事前看出來。**結論**：模型預測力（升降冪水位/速度/IC）看不出來；
行情狀態可以看出前四大回撤。詳見 [report](../../Doc/studies/20261005_dd_market_state_gate/REPORT.md)。

## 規則
```
comp(D) = ( z(近5日全市場收漲停比例) + z(近5日開高股 09:04→收盤均報酬) − z(近5日 negFill 觸發商品日均 return) ) / 3
comp(D) < −0.7  →  D 日不開新空單
```
- 三條序列只用 D−5..D−1；z 用擴張視窗（≥120 天、到 D−1）。
- 「開高股」= 09:04 比參考價高 0.5% 以上的活躍普通股（成交 ≥500 張）。
- 「negFill 觸發商品日均」= timeSeries `data/` 點位，每商品日第一個觸發點持有到收盤的 TakerSell_CloseBP（含 8% 停損口徑），
  篩選同 notebook cell 2/6，不經模型、門檻、風控。三個輸入都與策略參數脫鉤。

## 正式版
正式每日判定已搬到 `src/strategy/negFill/stopTradingGate.py`（daily_pipeline stage 7 與夜間重跑都會執行，
輸出 `src/strategy/negFill/stopTrading/stop_trading_days.json`，modelParams 多 `stopTrading` 鍵）。本目錄保留研究腳本與驗證。

## 每天怎麼用（研究版）
```
uv run python src/research/timeSeries/test/20261005_dd_market_state_gate/market_state_gate.py --update
```
在 daily_pipeline 的 timeSeries 階段（`data/YYYYMMDD.parquet` 產出）之後執行；NAS 分鐘 K 當日 14:12 後就有。
最後一行會印「下一交易日判定：… → comp ±x.xx 【停手】/ 交易」。
`--show 20260901` 看逐日；`--rebuild` 從 2023 重算；`--thr` 換門檻。
輸出 `artifacts/state_daily.parquet`（三條原始序列）與 `artifacts/gate_daily.csv`（逐日 z / comp / stop）。

## 狀態
- 未接進 `daily_pipeline.sh`，手動跑。
- 規則型態與門檻是看完單變數表後選的（樣本內）；門檻 −0.3～−1.5 皆優於基準，但 −0.7 是峰值，預期效果應打折。
- 停手日策略本身日均 −12.6 萬（2023-05～2026-09 複製書），停掉的是「期望值為零但波動照常」的日子，救不到所有 DD。
