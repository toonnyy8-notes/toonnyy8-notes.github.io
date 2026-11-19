---
slug: score-geometry-score-matching-sde
title: 分數學習與擴散模型的基礎：從分數匹配到 VE／VP-SDE
cluster: Diffusion & Generative Models
kind: topic
tags: [score-matching, denoising-score-matching, diffusion, langevin, sde]
arxiv: ["1907.05600", "2006.11239", "2011.13456", "2206.00364", "2509.24912"]
sources: [2509.24912 introduction, 2509.24912 section 2.1, 1907.05600 abstract, 2006.11239 abstract, 2011.13456 abstract, 2206.00364 abstract, user_terms_note.md]
related: ["score-geometry-lineage-hub", "eps-prediction-ddpm-score", "score-geometry-tempered-langevin"]
---
# 分數學習與擴散模型的基礎

回到 [總覽](score-geometry-lineage-hub.html)。本篇整理讀懂 arXiv 2509.24912 所需的分數式生成模型名詞。DDPM、ε 參數化與 SDE 的細節見 [ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)，此處不重寫。

## 發展順序

- **分數匹配**：Hyvärinen（2005）提出不知道歸一化常數、只用樣本學 ∇ log p 的方法。**去噪分數匹配（DSM）**：Vincent（2011）對樣本加噪音 x = x₀ + σz，訓練 s(x, σ) 逼近 −z/σ，等價於最小化 Fisher 散度。這兩篇不在 arXiv，內容來自本論文引用與 user_terms_note.md。
- **Song & Ermon 2019（1907.05600）**：用分數加 Langevin 動力學生成樣本，對應 SMLD / VE（本論文第 2 節的說法）。
- **Ho et al. 2020（2006.11239）**：DDPM，對應 VP。
- **Song et al. 2021（2011.13456）**：以 SDE 統一兩者，並提出 Predictor–Corrector（PC）採樣器：每個噪音程度先做一步反向擴散（predictor），再做幾步 Langevin（corrector）。
- **Karras et al. 2022（2206.00364）**：設計空間分析；本論文引它說明小 σ 的分數難以準確估計。

## 本論文用到的定義

高斯平滑分佈（式 (1)）：

$$\mu_\sigma = \mathrm{law}(\gamma(\sigma)X + \sigma Z),\quad X\sim\mu_{\text{data}},\ Z\sim\mathcal N(0,I)$$

- VE：γ(σ) = 1，對應 SMLD / VE-SDE。
- VP：γ(σ) = √(1−σ²)，對應 DDPM / VP-SDE；展開式多一個 −½⟨P_M(x), x − P_M(x)⟩ 項。
- 真實分數 s*(x, σ) = ∇ log p_σ(x)；模型分數 s(x, σ) 稱為 score oracle。
- DSM 損失（附錄 C）：

$$\mathcal L_{\text{DSM}}=\tfrac12\mathbb E_{u\sim p_{\text{data}}}\mathbb E_{x\sim\mathcal N(\Phi(u),\sigma^2I)}\Big[\tfrac1{\sigma^2}\|\hat s(x,\sigma)+x-\Phi(u)\|^2\Big]$$

- **噪音排程**：反向過程中 σ 隨時間變化的方式。作者指出排程常強調小 σ，而小 σ 的分數最難學。
- **Langevin 動力學**：dX = s(X) dt + √2 dW；若 s = ∇ log p，平穩分佈是 p。Euler–Maruyama 離散化：x_{k+1} = x_k + δ s(x_k) + √(2δ) ξ_k，ξ_k ~ N(0, I)；橢圓實驗步長 0.1、跑 10,000 步。
- **低溫極限**：σ → 0。把 p_σ 寫成 exp(−f/θ)，θ 稱為溫度；論文的結論都是這個極限下的漸近結果。

## 與本論文的關係

論文只修改 PC 採樣器的 corrector：把分數乘上 σ^α，見 [Tempered Score Langevin](score-geometry-tempered-langevin.html)。
