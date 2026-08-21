# test/：實驗對照區

慣例：任何 condition / 門檻 / 模型變體，以 `baseline/`（確定採用的點位與凍結配置）為對照，
同窗（全樣本 / 2026 / holdout 2026-04+）報 pnl、Sharpe、MDD、capw、筆/日 + 樣本分佈。

## 已完成實驗
| 日期 | 實驗 | holdout 結果 vs baseline(1,111萬/2.22/32.8bp) | 結論 |
|---|---|---|---|
| 2026-08-21 | negFill∩深度(ratio<-2) 重訓全 pipeline | 776萬/2.44/39.9bp、部位 57%、MDD 近半 | 規模換品質，容量天花板 |
| 2026-08-21 | 兩層額度(深200/淺100、深200/淺50) | 740萬/2.13/31.2 與 472萬/1.96/29.5 | **否定**，屍檢三因：(1) 深淺品質差在成交層已被模型/門檻吸收(capw 全樣本 shallow 59.3 vs deep 56.6)；(2) capw 隨 Position 遞增(100-200 段 67bp 全場最高、資本占比 41%)——訊號持續性=品質，砍淺層 100-200 段等於砍主菜；(3) 共用 cumsum 語義下省下的資本無去處。條件層優勢不會自動平移到成交層，先查成交層分層 capw 再設計配置 |
| 2026-08-21 | 雙書合併(baseline + 深度nf,各自獨立 200萬/檔) | holdout 1,887萬/2.34/35.4bp @ 5,667萬部位；對同資本放大 baseline：全樣本僅 +1.9%、holdout +7.8%/capw +8%/MDD 淺 3~5% | corr 0.94 無分散效果＝1.6x 槓桿且增量偏向深度切片；有效方向與兩層相反（向上加碼順品質梯度）。代價：重疊點位實質 400萬/檔、同 bid 吃兩倍量，實盤衝擊高於回測，重疊第二張建議折扣 |
| 2026-08-21 | FillLots_atHigh 探勘（正向家族×4、負向×2，內建相似度前篩） | **FH_neg15_chg（高點後深度倒貨 ratio<−15）：全期 Δz_short +0.107/t3.3/正月18/19，2026 +0.097/8月全正**——首個 regime-穩健新候選；正向家族（買盤攻高）雙向全滅 | 前篩：tick 重疊 4.9%、disjoint Δz +0.108（切割力獨立）、但 stock-day 重疊 99.2%/proxy corr 0.79 ⇒ 獨立第二本書=再一次槓桿；正確用法是 union 進 negFill index 提高同股票日容量利用率（73% 股票日額度未滿），待驗 |
| 2026-08-21 | FH15 獨立成線（own index/data_fh15/模型/門檻）+ 損益疊加 | FH15 書 holdout 660萬/1.47/capw 18.7；合併 1,771萬 < 同資本放大 baseline 2,271萬（corr 0.90） | **照現狀不加**。層級定位：condition 2026 8/8 正✓、模型層 netM IC 0.054 尚且優於 negFill 0.031✓、缺口在 z→BP 轉換層（capw 19 vs 33，vol 組成/費用占比未診斷）。**結案（twin_split 解剖）**：FH15 獲利 84% 來自與 negFill 視同一筆的攣生單（holdout capw 28.7≈negFill）；獨有段 46% 資本 2026 capw 僅 2.7bp，且獨有段 corr 仍 0.715、Sharpe 0.55 < 0.715×2.22 ⇒ 溢流書路線亦死，全線關閉。z→BP 缺口=獨有段稀釋，非 vol 組成。機制教訓：**分散的單位是股票日分佈，不是觸發器**——同股票日換觸發器只是同一本書的變奏 |
| 2026-08-21 | FH2 重設計（拉高≥5-6% × 仍在區間上半場 × 賣單≥2/5張觸發） | 上半場家族全面**反訊號**（Δz_short −0.09~−0.17，2026 最差）；對照組「已摔到下半場」+0.28(2025)但 94% stock-day 重疊 negFill 且 2026 同死 | **高點錨線徹底結案**：台股空方 alpha 需要弱勢確認，確認時刻=negFill 域；高檔早攔截=被吸收/被軋（2026 強者恆強尤甚）。多方版實測（taker A1 vs maker B1 兩種進場）：baseline maker z +0.017（結構坑≈spread）；L_absorbT（貼近高點 posrng≥0.7）taker +0.030 唯一浮出水面、2026 maker raw +22.3bp；L_buy2（高檔追買）Δz t=2.49 最穩。但扣 19bp 費用+adverse selection 折扣後僅貼平，Δz t 1.4-2.5 弱，**未成書**。精修方向：posrng 0.85 劑量、1.09 停利 cap 與強勢股互斥問題、maker 真實成交率（→followMaker 技術棧為正確載體）|


## 儲存架構（四層）

| 層 | 位置 | 進版控 | 內容 |
|---|---|---|---|
| 結論 | `SKILL.md` + 本檔實驗總表 | ✓ | 原則、及格線、每個實驗的判定 |
| 方法 | `test/<日期>_<主題>/*.py` | ✓ | 實驗腳本（docstring 即文件），可重現 |
| 點位 | `baseline/*.parquet` | ✓（gitignore 例外） | 確定採用的成交點 |
| cache | session scratchpad（/tmp） | ✗ | preds parquet、pkl、log——可由腳本重生，session 結束自動消失 |

腳本內的 cache 路徑指向產生當下的 scratchpad，重跑時把 `S = /tmp/...` 改成任一工作目錄即可。

## 腳本歸檔索引

| 資料夾 | 對應實驗 |
|---|---|
| `20260819_leak_audit/` | notebook 完整復刻、_re 洩漏發現與拔除對照（stage2/3）、特徵 IC 月度面板、殘差漂移分解 |
| `20260819_threshold_opt/` | AB 分支移除 + q 掃描（128 配置、tune/holdout 分窗）、三代配置對比、DD 對比 |
| `20260820_bucket_research/` | 路線 A（f̂ 條件基線）vs B（樹切桶）vs 16-cell、混訓 pooled 對照、R2 q 掃描、AB 日價值檢定 |
| `20260821_condition_explore/` | Overshoot pilot、FillLots_atLow 盲挖回收測試、2026 加密復掃、negFill∩深度 |
| `20260821_depth_variants/` | 深度版全 pipeline、兩層額度（否定）、雙書合併 |
