# Round 1 審查合併（No.129）

來源：`fidelity.md`（F1–F13）、`pedagogy.md`（P1–P15）。合併後 27 項：P0 0、P1 15、P2 12（F13 與 P13 合併）。處理狀態：採用 24、部分採用 2、不改 1。設計文件變更見 `design.md` §8。

| # | 來源 | 等級 | 問題 | 處理 |
| --- | --- | --- | --- | --- |
| 1 | F1 | P1 | C2「G 越大越接近 ±1」錯：極限是 2(h − prompt 平均)，π=0.8 時 0.24 vs 0.60 | 採用。步驟 8 與 C2 改寫；check.py 加 G = 10⁶ 極限斷言（0.24／0.60） |
| 2 | F2 | P1 | C3 是固定偏差的線性累積，t½ ∝ 1/ε 來自假設，不是論文的 compounding | 採用。C3 改名、圖版首行範圍聲明、t½ 圖下聲明「不是論文主張」；驗算 λ·t½ = 14.0（四個 λ） |
| 3 | F3 | P1 | C3 中 SC 降熵連到 §5.3 staleness 不成立（偏差固定，非時變） | 採用。刪除連結；SC 曲線只作 Cov_q ≠ Cov_p 示意；R3 改寫 |
| 4 | F4 | P1 | §1「drift = 0 iff p = q」為寬鬆說法 | 採用。步驟 6 加〔本頁說明〕（E_q[R] = 0；Jᵀ(q − p) 可為 0），S4b 正文寫出，Hero 不用 |
| 5 | F5 | P2 | C1b 的「逐 token IS = on-policy」未說明 Q_v 固定 | 採用。C1b 圖版註 |
| 6 | F6 | P2 | Eq. 6/11 文字抽取無括號 | 不改。PDF 第 6 頁 Eq. 6 印有括號 $-R(\log p_{y_t}-\sum\cdots)$，原設計正確；步驟 11 記錄此查證 |
| 7 | F7 | P2 | 殘餘 drift「只有」尾部成比例時為 0 → 應為充分條件 | 採用。步驟 12 改寫（logit 座標下才是必要） |
| 8 | F8 | P2 | LO3 漏「二元 reward」 | 採用 |
| 9 | F9 | P2 | Table 1 P/V 上標遺失 | 採用。步驟 3 恢復上標 |
| 10 | F10 | P2 | Fig. 2 group-centered 基準：B.1 預設 ±1 | 採用。步驟 4、S3、R7 改為「依預設應為 ±1，Fig. 2 未另述」 |
| 11 | F11 | P2 | top-k 疊加版 q̂ = q 的限定未進 LO11 與 C4 | 採用。LO11 答案與 C4 圖版註 |
| 12 | F12 | P2 | LO12 未修正結論句「唯一穩定」 | 採用 |
| 13 | F13＋P13 | P2 | Hero 卡用未定義術語；卡 3 應寫「每個 prefix 減去確定的修正項」 | 採用。三張卡重寫，不含 SC/TIS/量化縮寫；首次出現寫「推論引擎（sampler，q）」 |
| 14 | P1 | P1 | 先備知識缺：log-derivative trick、advantage、value、共變異數、熵/CE/KL | 採用。§3 開頭列出各自放置位置 |
| 15 | P2 | P1 | S8 過密、順序 11→7→8→6 混亂 | 採用。拆 S8a（Eq. 11）／S8b（Eq. 7→8→6，疊加進 details，但 α 留在正文） |
| 16 | P3 | P1 | S4 過密；E_q[R\|prefix] 無白話；需數值例 | 部分採用。拆 S4a／S4b、加白話；數值例改用 C1 預設值（4 token）逐步計算，不另設 3-token 例子，避免第二組數字 |
| 17 | P4 | P1 | 前向引用：SC 在 S5/S6、advantage 在 S3 | 採用。advantage／group centering 在 S3 首段定義；C2 移除 SC 列；C3 SC 曲線預設隱藏、S7 開啟 |
| 18 | P5 | P1 | C3 線性橫軸看不到早期崩潰；t½ 未達到 | 採用。橫軸對數；所有曲線 t½，未達到顯示「未達到」；+1/0 模式 t½ 不具診斷性，改看 Ent@3000 與熵 ≤ 0.35 步數 |
| 19 | P6 | P1 | C4 σ = 0 相對殘餘 0/0 | 採用。‖q − p‖ < 1e-12 顯示「—」，selfTest 斷言 |
| 20 | P7 | P1 | 小誤差顯示 0.0000 | 採用。`toExponential(1)`＋「✓」 |
| 21 | P8 | P1 | 符號衝突 | 採用。λ、Ent、N、m_q^T／m_p^T、token 1–4、甲／乙與 h；L_i 加註（保留論文記號） |
| 22 | P9 | P1 | 390px 版面 | 採用。C4 窄螢幕只標分界、`.pg two` 單欄、katex 橫捲、selfTest 逐圖版檢查 scrollWidth |
| 23 | P10 | P1 | 鍵盤無法取得長條數值 | 採用。roving tabindex＋左右鍵、固定讀數框、「以表格顯示」，C1／C2／C4 皆適用 |
| 24 | P11 | P1 | SVG 內 KaTeX 未規範；lazy render | 部分採用。SVG 純 Unicode、KaTeX 只在 HTML；不做 IntersectionObserver lazy render（公式量小，載入後一次渲染即可） |
| 25 | P12 | P2 | 字面風格：標題擬人、過度概括、論證寫成事實 | 採用。S3、S5、S9 標題改寫；「兩條軌道」改「trainer 與 sampler 對照圖」；「拉向／推離」保留，因為是 §4 原文 pulled toward／pushed away 的直譯，並註明 |
| 26 | P14 | P2 | S2 放全 Table 1 太早 | 採用。S2 只放 TIS／MIS，全表進 details |
| 27 | P15 | P2 | ‖z‖ < 10 保證只在 η = 1 驗過；reduced-motion；pattern 顏色 | 採用。拿掉保證；reduced-motion 規則；pattern 用 currentColor |

協調者指定事項對應：C3 範圍（#2、#3）、C2 極限（#1）、iff 寬鬆（#4）、S4/S8 拆分與前向引用（#15–17）、本批筆記 `ppo-grpo-dapo-llm-rl-objectives`／`importance-sampling-corrections-llm-rl` 直接連結（R6 已註明，只有四篇非本批筆記需建置時檢查）。
