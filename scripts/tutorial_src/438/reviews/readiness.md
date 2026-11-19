NO-GO

## 解釋回述

本頁以 Flow-GRPO（arXiv 2505.05470）為例。流匹配模型用確定性 ODE 取樣，沒有隨機性，策略機率也難算，所以線上 RL 無法探索。論文的解法有兩個：(1) ODE 轉 SDE，使每個時刻的邊際分布不變；離散化後每一步是高斯策略，平均為 x + [v + σ²/(2t)(x+(1−t)v)]Δt，標準差 σ√|Δt|，σ = a√(t/(1−t))，因此機率比與 KL 有閉式；(2) Denoising Reduction，訓練取樣 10 步、推論 40 步。目標為 GRPO：同一提示詞取 G=24 張影像，優勢為獎勵減組平均再除以組標準差；替代目標用裁剪的機率比 ρ，KL 的對象是 π_ref。數字取自 Table 1–2。互動元件：C1 組相對優勢（G 與獎勵類型、RMSE 曲線）；C2 裁剪替代目標（Â、ε、ρ 三個滑桿，畫 L(ρ) 與梯度）；C3 一維玩具的 ODE/SDE 粒子直方圖與 W₁（取樣器、a、T、顯示時刻）；C4 一步高斯策略的密度、c_j 長條圖、KL 與 log ρ（a、T、j、Δv、z）。各部分我都能解釋；歧義只在 S5 的 σ_t（見阻擋項 1）。

## 阻擋項

1. **符號先用後定義：σ_t、a。** S5 的正文保留 Eq. 7 與 Eq. 8，兩式都含 σ_t（design.md 步驟 3 的 `dx_t=[…]dt+σ_t dw`，以及 S5 行的「正文保留 Eq. 7、Eq. 27、Eq. 8」）。但 σ_t = a√(t/(1−t)) 與噪聲等級 a 首次定義在 S6（design.md 步驟 6 與 S6 行）。S5 的規格只定義 Wiener 過程、分數與 Fokker–Planck，沒有 σ_t。需在 S5 內 Eq. 7 之前加上 σ_t 與 a 的定義，或明確寫成前向引用。

## 非阻擋項

1. C3 讀者觀察 (2) 寫「a≤2.5 的 SDE（0.017–0.041）」，但 design.md 第 5 節 T=40 只列 a=0.7 與 a=2.5。範圍的端點來自 verify/check_out.txt 第 26 行（a=0.1，0.0170）與第 30 行（a=1.5，0.0410）。建議把這兩列加入驗收表。
2. S9 的「Table 1 摘錄（Flow Matching 區＋GPT-4o 列）」沒有列出實際要摘錄的列。建議逐列寫出。
3. S3 的「先假設每一步已經是隨機的（S5 交代）」與 S4 的「ρ 在 S6 說明」是前向引用。可以接受，但頁面文字需明寫，讀者才知道要跳過。
4. Hero 的連結 `../index.html#paper-438` 本次未核對錨點是否存在，需於建置時確認。
5. 6.6 的截圖流程依賴 `scratchpad/wt-paper-master/…` 之外的 `scratchpad/katex/0.16.9` 與 `tut/447/build/index.html`。兩者本次都存在，但建置時應改為專案內路徑，避免依賴本機暫存目錄。
6. verify/check_out.txt 沒有 FAIL 或 mismatch。本次抽查的數值（C1 RMSE 與二元序列、C3 表與多種子 W₁、C3 精確樣本 W₁ 下限 0.0087–0.0428、C4 七列、c_j 總和、C2 的 L 與梯度）都與 design.md 第 5 節一致。
7. 符號與連結檢查結果：design.md 與 7 篇 notes/ 中的所有連結 slug 都存在於 notes/ 或 notes_split/；7 篇 notes/ 的 frontmatter 都有 slug、title、cluster、kind、tags、arxiv、sources、related 八欄；13 個 timeline.json arXiv ID 與 6.4 節的 TL_GROUP 一一對應。
8. Eq. 9 的 Δt 符號（取絕對值）與 t=1 的 σ 處理都已在 design.md 標為〔本頁推導〕或推定，頁面需保留這些標記。
