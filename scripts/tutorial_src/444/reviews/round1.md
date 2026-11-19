# Round 1 合併審查清單（fidelity F1–F15 + pedagogy P1–P15，去重）

| # | 優先 | 來源 | 項目 | 處理 |
| --- | --- | --- | --- | --- |
| 1 | P0 | P1 | C1 quantile 圖的 h 未定義 | h = 0.02（網格步長 0.0025 × 8）；「→ ∞」標在 a = V 的縱軸頂端；寫入 C1 |
| 2 | P0 | P2 | C4 時間索引 1-based／0-based 混用 | 改為 `beta[1..T]`、`alpha[1..T]`、`abar[0..T]`（abar[0]=1），迴圈 k = T..1，k > 1 才加噪；c4.py 依此改寫，輸出與舊版逐位相同 |
| 3 | P1 | F1、F5、P12 | S9 標題與「其他方法下降較多」被 Table 1 反駁（antmaze：IDQL −35.3、IQL −9.6）；「一組超參數」= 每個領域一個 | 標題改「每個領域只調一個超參數時，IDQL 的總分最高」；列出四個方法 -A→-1 的變化（tables.py）；寫明 antmaze 上 CQL、DQL 掉得多、IQL 只掉 9.6，IDQL 對 IQL 的差距由 96.6 縮為 70.9；Hero 與各處改「每個領域一組 τ」 |
| 4 | P1 | F2 | C2 的 TV 混合了單峰限制與權重不一致 | c2.py 新增「以動差擬合 π_imp 的最佳單峰高斯」：τ = 0.9 時 TV 0.4928、低密度質量 0.2128；C2 顯示這條參考線；AWR 的 TV 改稱「AWR 目標與 π_imp 的差距」 |
| 5 | P2 | F3 | DQL-1 被標為不一致 | 部分不接受：tables.py 改用容許誤差後，DQL-1 的 locomotion 與 antmaze 列和相符，但印出的 total 957.4 ≠ 741.2 + 216.4 = 957.6，確實不一致；文件改寫成「DQL-1 total」 |
| 6 | P2 | F4 | IDQL-A 與 IDQL-1 的 τ 相同，差在步數與 N；-1 的基線是作者重跑 | 步驟 10、S9 寫明（App. E、D.2 原句「may not fairly represent each algorithm」） |
| 7 | P2 | F6 | f′ 的符號 | 頁面定義 f′ 為對引數 u = Q − V 的導數；註明論文寫 ∂f/∂V(s)（差一個負號），證明實際用對引數的導數 |
| 8 | P2 | F7 | Theorem 4.1 的隱含假設 | 加「隱含假設」callout：f 可微、V* 存在、Q = V* 處權重有定義（或該處機率為 0）、0 < Z_imp < ∞ |
| 9 | P2 | F8 | App. C.2 的 KL 方向 | 讀數改標 KL(μ‖π_awr)；新增〔本頁推導〕V_exp = E_awr[Q] − KL(π_awr‖μ)/α（c1.py 數值核對），說明它與 E_awr[Q] > V_exp 一致 |
| 10 | P2 | F9 | argmax 選到的尾端樣本是合法的 μ 樣本 | C2 觀察 (3) 改「分布尾端（不是離群樣本）」，刪除連到 S7 的句子 |
| 11 | P2 | F10 | 「介於 μ 與 π_imp 之間」未證明 | 改為「N = 1 為 μ、N → ∞ 趨近 π_imp」；寫明 Alg. 2「給出正確隱含 actor」只在 N → ∞ 成立 |
| 12 | P2 | F11 | quantile 權重在 Alg. 2 中發散 | S5 加一句（連回 S3 的點質量說明） |
| 13 | P2 | F12 | 程式碼的截斷是每步 | 寫「clip_sampler 時每步截斷，最終輸出再截斷一次」 |
| 14 | P2 | F13 | Table 3「相當」 | 照原文「remains competitive」；註明 IDQL 總分低於 RLPD 571.0、IDQL-Imp umaze-diverse 78.4 → 73.0、RLPD 的 0.0 是沒有預訓練 |
| 15 | P2 | F14 | 注意事項 (c) 出處 | 改「Eq. 8 之後」 |
| 16 | P2 | F15 | R3 的 IQL 正負號比較 | 已對照 No. 569 design R3（IQL 原文 Eq. 7 寫成最大化式、Alg. 1 寫梯度下降）；改寫成「IQL 教學 R3 記錄了 IQL 原文同類問題」 |
| 17 | P1 | P3 | S5 先用 μ_φ、Eq. 4、擴散 | S5 開頭一句把 μ_φ 當成可抽樣的黑盒、細節在第 06 章；Alg. 1 的 φ 列寫「擴散行為複製（第 06 章）」；S4 的「AWR 加權 DDPM」改「以 AWR 權重訓練擴散模型（第 06 章介紹擴散模型）」 |
| 18 | P1 | P4 | C2 在 S4 使用 S5 才教的 Alg. 2、Eq. 13、N | C2 移到 S5 末尾；S4 保留文字（LO7）；Sec. 6.3 的實驗結論與 Fig. 2 移到 S9，LO6 改屬 S9 |
| 19 | P1 | P5 | S3、S6 過密 | S3 主文只講 expectile 與 exponential，quantile 的兩點推導放 `details.more`；S6 的三個排程公式與 cosine 截斷放 `details.more` |
| 20 | P1 | P6 | 缺 V^π = E_π[Q^π] 的前置 | S1 加兩句與本章符號 Q^π、V^π |
| 21 | P1 | P7 | 兩版公式重複朗讀；窄螢幕表格 | 非作用版本 `display:none` + `aria-hidden="true"`，KaTeX 照常渲染但不進無障礙樹；selfTest 只數可見公式；損失表 < 600px 改堆疊卡片，Table 1 第一欄 sticky |
| 22 | P1 | P8 | ᾱ（U+1FB1）字型缺字；selfTest regex 寫法 | 動態文字改寫「abar_T」；SVG 用 `a` + `text-decoration: overline` 的 tspan 呈現 ᾱ；regex 寫成跳脫形式 `/[̀-ͯ]/`，只掃讀數、SVG `<text>`、aria-label |
| 23 | P1 | P9 | 分段 MC 沒有取消規則 | 每次啟動遞增 run token，舊 token 的分段直接丟棄；計算中停用該元件全部控制項；累積權重沒超過 u 時選最後一個；只在完成時 `announce()` 一次 |
| 24 | P1 | P10 | LO9「T = 5」與 C4「T 大較好」看似矛盾 | R9 的說明寫進 C4 旁的正文；S6 標題改「D4RL 設定以五步去噪產生動作」；LO9 答案加這個限定 |
| 25 | P2 | P11 | 控制項細節 | C3 ε 用索引滑桿 0–8 + `aria-valuetext`；C1 α 滑桿的 aria-valuetext 用「α = 1」；C3 N 按鈕 < 600px 兩列；1 − (1 − ε)^N 只畫點不畫線 |
| 26 | P2 | P12 | 標題過度宣稱與比喻動詞 | S2 標題改「凸且 f′(0) = 0 的損失都對應一個隱含 actor」；改寫「把質量放在」「推向」等句子 |
| 27 | P2 | P13 | C1 效能、MC 容許誤差 | C1 曲線以 (loss, p_h) 快取並實測；selfTest 的 MC 斷言 ±0.02 |
| 28 | P2 | P14 | 新元件的深色模式與 reduced-motion | 6.2 寫明斜線底紋、長條用 tokens；本頁無動畫 |
| 29 | P2 | P15 | 連結與筆記重複 | Diffusion-QL 頁 slug 由協調者確認為 `diffusion-ql`；`diffusion-ql-lineage-hub` 加入 S11、S12 與「擴散 × RL」時間線卡片；444 的 `diffusion-behavior-model-policy-extraction`（SfBC／ARQ／EMaQ／IDQL）與 443 的 `diffusion-in-rl-concurrent-work`（Diffuser／DT／軌跡擴散）範圍不同，不重複 |
