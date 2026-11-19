# Round 1 合併審查意見（Fidelity F、Pedagogy/Build P；已去重）

1. **P1** η=0 與 Gated DeltaNet「等同」說法過強（F1、P5 部分）：本頁推得 $\mathcal M((1-\alpha)I-\theta kk^\top)+\theta vk^\top$；Gated DeltaNet 原論文的閘門可能同時乘在修正項上，兩者未必可互相重參數化。LO11、步驟 10、R13、C1 chip 改為「論文寫兩者等價；本頁只說更新規則形式相近」。
2. **P1** C2 讀者應觀察到 2 自相矛盾（F2）：0.00235 > 1e-3。改為「L=2 約 2×10⁻³，L=3、4 低於 1.5×10⁻⁴」。
3. **P1** $t'$ 的定義（F3）：論文 Eq. 16 的 $t'=t-\mathrm{mod}(t,b)$ 在 $t=b$ 得 $b$（文字卻說 0），$b=1$ 時梯度點變成 $\mathcal M_t$。改採 chunk 起點 $t'=t-1-\mathrm{mod}(t-1,b)$，列入 R1，$b=1$ 等價只在此讀法下成立。
4. **P2** Eq. 16 第二個等號的適用範圍（F4）：只在單一 chunk 內成立；跨 chunk 時各 chunk 用各自的 $\mathcal M_{i'}$。
5. **P1** 動量效果與有效步長混淆、單一 seed（F5、P5）：常數 η 使有效步長變為 θ/(1−η)=0.75。加 θ=0.75、η=0 對照列與 seed 11–30 的中位數；內文數字標 seed 11；圖版顯示目前 seed。
6. **P2** C1 損失的 ½ 係數（F6）：Eq. 12 無 ½；註明 C1 的 θ=0.3 相當於 Eq. 12 寫法下的 θ=0.15。
7. **P1** Table 1 的作者宣稱與表格不完全一致（F7、P1）：Wiki ppl 上 MAL 優於 MAC／MAG（340M、400M）；760M MAC Wiki 19.93 高於 GDN-H2 19.88；760M MAL Avg 50.97 低於 Samba 51.08 與 H2 51.49；760M MAL SIQA 30.98 疑似異常。標為作者宣稱並逐條列出例外（S9）。
8. **P2** Table 3 Traffic（F8）：MAE 0.289 不是最佳（iTransformer 0.282；審查原文寫 RLinear，核對 Table 3 欄位後為 iTransformer）。加註腳。
9. **P2** 消融排序的指標（F9）：weight decay > 動量 > … 只在 ppl 與 Reasoning 成立；Long Context 最低的是 Linear Memory 85.34。LO12 寫明指標。
10. **P2** LO10「MAC 長上下文最好」（F10）：Table 2 16K 的 N 欄 MAG 98.6 > MAC 97.4。改為「Table 5 Long Context 最好；Table 2 並非每格最好」。
11. **P2** 驗收輸出無法由腳本重現（F11）：`accept_node.txt` 有 10 行不是 `node widgets.js` 產生；θ0/η0/α0.01 → 0.4389 與 L=2、α=0.01 第 2082 步發散未寫進文件。把極端值區塊加入 `widgets.js` 並重新產生。
12. **P2** Python 驗算的執行指令（F12）：需 `PYTHONPATH=verify`。文件寫明指令。
13. **P1** S1 前置知識不足（P2）：注意力、softmax、q/k/v、φ 未先教；Q5 早於 S4。S1 加兩段前置說明、φ 可收合、Q1–Q5 移到 S2 末尾。
14. **P1** 含動量 chunk 公式索引不一致（P3）：$w_l$、$c_S$ 為 1-based，$q_l=-\theta\eta^{n-1-l}$ 為 0-based；η=0 需 $0^0=1$。統一 1-based，selfTest 加 η=0。
15. **P1** S3 過密（P4）：分成 3a 動量、3b 遺忘；C1 讀數預設只顯示 3 個，其餘收合。
16. **P1** C1 chip 使用未教過的名稱（P5）：chip 改以參數命名並註「對應 S8」。
17. **P1** S5 使用未教的 LTI、associative scan、全域卷積（P6）：以兩句教 scan；LTI／卷積收合為「進階」。
18. **P1** C4 矩陣在 390px 不可用、grid 無障礙規格不足（P7）：格寬 ≥14px、`overflow-x:auto` + `tabindex=0`、roving tabindex 只聚焦列、非目前列 `aria-hidden`；MAG／MAL 停用並隱藏 $N_l$。
19. **P1** 圖表沒有逐點資料的非 hover 取得方式（P8）：每個圖版加「顯示資料表」切換。
20. **P1** 對數縱軸低於下限或為 0 沒有規則（P9）：`scaleLog` 一律 `max(v, floor)`，下緣畫向下箭頭。
21. **P1** Hero 用到未定義術語（P10）：perplexity 加定義、「測試時」定義放 hero、token 軸加圖例並連到 S6、S7。
22. **P2** TC⁰ 與 ‖·‖_F 未定義（P11）：S7 加一句說明並標「論文宣稱，本頁不展開」；符號表補 Frobenius 範數。
23. **P2** 寬公式的窄螢幕處理不完整（P12）：所有 `.eq` 預設 `overflow-x:auto; tabindex=0`，手動拆行最寬四條。
24. **P2** 發散訊息被去抖延後（P13）：發散訊息用 `role="alert"`，立即且一次性。
25. **P2** C2 拖動時連續重設、分頁隱藏行為未定義（P14）：重設只在 `change` 觸發；`visibilitychange` 時自動暫停。
26. **P2** 深色模式的帶狀色與非文字對比未規定（P15）：指定深色帶狀不透明度，6.8 加 3:1 對比檢查。
