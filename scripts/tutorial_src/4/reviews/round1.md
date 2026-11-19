# Round 1 審查合併清單（tut/4/design.md）

來源：F = Fidelity review（15 點，無 P0）、P = Pedagogy/build review（15 點）。已去重；「處理」欄記錄 design.md 的修訂位置。

| # | 級別 | 來源 | 位置 | 問題 | 處理 |
| --- | --- | --- | --- | --- | --- |
| 1 | P0 | P1 | C1b、C2 的 SVG 標籤、讀數 | SVG `<text>` 與動態 DOM 無法跑 KaTeX；live region 若含 KaTeX 會重複朗讀 MathML | 4.0 新增「標籤與數學排版」規則：SVG 一律 Unicode 純文字（x¹new、δ s,t），`renderMath()` 只跑靜態節點，讀數純文字；C1b、C2 標籤改寫 |
| 2 | P1 | F1、P8 | 步驟 7、C3、R2 | $P_e$ 閉式是 interpolant 耦合下的讀法，Eq. 24 用 $x_t$、論文沒給公式；窗 0.71–0.85 與「2/17 vs 15」不能說是重現 Fig. 9。閉式在讀者沒直覺前出現 | 步驟 7、C3、R2 改標「〔本頁推導〕interpolant 耦合讀法」，不稱重現 Fig. 9；S4 順序改為先 C3 觀察再 `<details>` 公式，加一句「Φ 是標準常態 CDF」 |
| 3 | P1 | F2 | 步驟 6、LO5、C1b 觀察 3 | 「相關由共同 $x_0$ 經 ODE 帶出」不準；相關來自後驗條件在整個 $I_t$ 上、ODE 耦合各位置，$x_0$ 只提供隨機性 | 三處改寫，保留〔本頁推導〕標籤 |
| 4 | P1 | F3 | 步驟 12 Table 25 | Self-BLEU 不是越低越好；MDLM 基線 0.026/0.036、0.023/0.031 低於資料 | 改寫為「目標是接近真實資料」，列全表，不把 FMLM 當成勝過 MDLM；R4 的 Self-BLEU 出入改寫 |
| 5 | P1 | F4 | 1.1、LO11、S9 | OWT entropy FLM 5.33 對資料 5.44、Duo 5.55 | 1.1、LO11、LO12 補上 entropy 差距 |
| 6 | P1 | P2 | S3 | 600–700 字塞 LO3–LO5，$W_2$、$\xi$、Gronwall 未教 | S3 拆成 S3a denoiser（LO3、LO4）與 S3b 交叉熵與離散對照（LO5）；Eq. 13 正文一句結果，證明鏈入 `<details>` |
| 7 | P1 | P3 | S6、C2 | 3 滑桿 + 勾選 + 2 按鈕 + 3 面板 + 6 標記 + 約 20 個讀數 | 預設只顯示 $X_{s,t}$、$\delta_{s,t}$、simplex；$u$、教師、Eq. 82 面板收進「進階」；讀數縮為 γ、三個座標和、教師差 |
| 8 | P1 | P4、F13 | S1、C1a | ancestral sampling、$t\to s$、遮罩未定義；C1a 不能暗示 Fig. 3 用 masked diffusion 與此排程 | S1 加三個一句定義；C1a 與步驟 1 改為「只有 N=1 的 25% 與 Fig. 3 一致；Fig. 3 未說明模型與排程」 |
| 9 | P1 | P5 | S2 | 條件期望未解釋；符號表收合不能當定義 | S2 加一句條件期望說明，$p^l_{1\mid t}$ 首次出現於正文定義 |
| 10 | P1 | P6 | S8 | master equation、$f_\#\mu$、CTMC 未定義 | 先一句定義 pushforward，先用 C4 給直覺，再命名 CTMC／master equation |
| 11 | P1 | P7 | 390px 寬式 | Eq. 23、82、24–25 太寬 | 6.2 新增：`.katex-display{overflow-x:auto}` + `tabindex=0` + `aria-label`；Eq. 23 用 `aligned` 拆 3 行；Table 13 sticky 首欄 |
| 12 | P1 | P9 | C1a | 16 個隨機 chip 與閉式橫條不符會誤導 | chip 區加標「16 個隨機樣本，與上方閉式值不必相符」 |
| 13 | P1 | P10 | C3 效能 | 冷啟動 1,001 點 LUT 約 1.6M 次 erfc，手機未測；六條背景曲線在 init 算 | 以 `requestIdleCallback`（退回 `setTimeout`）每批 50 個 t 分段計算，顯示「計算中」，分段按鈕可用；六條背景曲線快取 |
| 14 | P2 | F5、R9 | 步驟 12 Table 18 | One-hot 130.42 是 300k 步、learned entropic time，不能和 96.91 比 | 子表標「300k 步、learned entropic time」 |
| 15 | P2 | F6 | 步驟 11、LO10 | $\mu_{\min}$ 是最小正質量；需 $\lvert S\rvert\ge2$ | 兩處補上 |
| 16 | P2 | F7 | 步驟 10 | Jacobian 不穩定出自 App. F；主文只推測 Eulerian 需正則化以留在 simplex | 改為分別歸屬 |
| 17 | P2 | F8 | 步驟 10、R8 | OWT 的 MSE 一階段 129.32 但 entropy 4.53（CE 168.30 / 5.17） | 補 OWT 列，寫「只有 LM1B 的比較 entropy 相當」 |
| 18 | P2 | F9 | 步驟 7 | App. E 只說以 Gauss–Hermite 近似「(25) 的 CDF」，未給被積函數 | 改為「App. E 描述」 |
| 19 | P2 | F10 | 步驟 13 第 10 條、R5 | 論文沒有說明 γ-sampling 與 Alg. 8 的關係 | 補一句「論文未調和兩者」 |
| 20 | P2 | F11 | 步驟 10 | Alg. 7 對角項是 $\tau(s)\sim U[0,1]$、輸入 $I_s$ | 步驟 10 註明 |
| 21 | P2 | F12 | 步驟 12 Sudoku | 7.6× 的推導另含「每次前向成本相同」假設（AR 有 KV cache） | 改為「〔本頁推導〕近似」並寫明兩個假設 |
| 22 | P2 | F14 | 步驟 5 | $\xi\to0$ 發散的只是上界 | 改寫為「上界發散，不代表真誤差發散」 |
| 23 | P2 | F15 | 步驟 7 Table 28 | 不要加步數 | 維持；R10 已記錄 |
| 24 | P2 | P11 | C2 觀察 3、S5 標題 | 比喻／擬人（「預告方向」「一次跨越」） | 改為字面敘述 |
| 25 | P2 | P12 | `flowU` | 超過 20,000 步靜默回傳 0 | 回傳 `{u, failed:true}`，讀數顯示「數值不收斂」，selfTest 加一項 |
| 26 | P2 | P13 | C1b | 「N≥8 已近乎精確」削弱 LO7；免責聲明位置 | 先寫結論「Euler 一步 = 兩模態平均」，Fig. 11 連到 S9；免責放在控制列旁 |
| 27 | P2 | P14 | 全域 | 未規定 reduced motion 與深色主題的紋理色 | 6.5 新增：`prefers-reduced-motion` 關閉過場；斜線紋理用 `currentColor` 低不透明度；兩主題檢查對比 |
| 28 | P2 | P15 | Hero 卡 3、S9 | 卡 3 用未定義術語；S9 過長 | 卡 3 改寫；guidance、Sudoku、Table 27 收進「其他結果」`<details>`（無 LO） |
| 29 | — | notes/ | 步驟 7、S11 | 研究筆記：$P_e$ 指標沿用自 Duo、CANDI（論文 [19, 20]，已核對 paper.txt 第 549 行）；[40] 為 Li & He 2511.13720；Analog Bits 為 one-hot 類先例 | 步驟 4、7 引用；S11 文字納入；timeline 仍 14 筆 |
