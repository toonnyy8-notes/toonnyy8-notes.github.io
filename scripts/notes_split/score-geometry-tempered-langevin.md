---
slug: score-geometry-tempered-langevin
title: Tempered Score Langevin 與流形上的均勻採樣（定理 5.1）
cluster: Diffusion & Generative Models
kind: topic
tags: [tempered-langevin, uniform-sampling, maximum-entropy, diffusion-maps]
arxiv: ["2506.15385", "2501.00429", "2509.24912"]
sources: [2509.24912 section 1.1, 2509.24912 section 5, 2506.15385 abstract, 2501.00429 abstract, user_terms_note.md]
related: ["score-geometry-lineage-hub", "score-geometry-sde-pde-wkb"]
---
# Tempered Score（TS）Langevin

回到 [總覽](score-geometry-lineage-hub.html)。

## 為什麼要均勻採樣

論文引用 Belkin & Niyogi 2008、Coifman & Lafon 2006（Diffusion Maps）：用均勻樣本估計 Laplace–Beltrami 算子、熱核、測地距離與擴散距離，不會受 p_data 偏差影響。這些古典方法以圖正規化抵消取樣密度（另見 Hein et al. 2007），但難以擴充到高維大型生成模型。De Santi et al. 2025（2506.15385）用微調讓擴散模型產生均勻樣本；本論文只改推論，不需微調。

## 方法

標準：dX = s dt + √2 dW。TS Langevin（式 (8)，0 < α < 2）：

$$dX_t=\sigma^{\alpha}s(X_t,\sigma)\,dt+\sqrt2\,dW_t$$

α = 0 即標準 Langevin。名稱中的 tempered 是字面意思：若 s = −∇f_σ，平穩分佈為 π̃_σ ∝ exp(−σ^α f_σ) = p_σ^(σ^α)，等於把溫度調高為 σ^(−α) 倍。

## 定理

- **定理 5.1**（梯度場）：若 ‖s_πσ − s*‖_{L∞(K)} = o(σ^β)，β > −2，則 π̃_σ 弱收斂到流形上的均勻分佈，極限 π̃(u) ∝ √det g(u)（即 dM/du）。分數誤差 o(σ^(−2)) 即可，比還原 p_data 的 o(1) 寬鬆。
- **定理 5.2**：去掉「s 是梯度場」的假設，另設 p_data ∈ C²，且平穩分佈存在唯一，結論同 5.1。證明見 [SDE、Fokker–Planck 與 WKB 展開](score-geometry-sde-pde-wkb.html)。
- 均勻分佈在緊緻流形上熵最大，所以也稱最大熵先驗（第 6 節用）。

## 實作細節

- 附錄 C.1 訓練 ŝ = σ² s，因主導項為 σ^(−2) 階，乘 σ² 後目標為 O(1)。採樣用 dx = ŝ dt + √(2σ²) dW，與 dx = s dt + √2 dW 平穩分佈相同；TS 版本擴散係數改為 √(2σ^(2−α))。
- 合成實驗（單位圓、橢圓，α = 1）：TS 對弧長均勻、對角度不均勻。

## 收斂速度（附錄 D）

Poincaré 常數 C_PI：成立時 χ²(p_t, p_∞) ≤ e^(−C_PI t) χ²(p₀, p_∞)。論文用 Holley–Stroock 擾動原理、d_M 局部滿足 Polyak–Łojasiewicz（PL）條件，以及 Gong, He, Shen（2501.00429）的結果，得到與溫度無關的 C_PI；Menz–Schlichting（2014）的 Lyapunov 方法把下界由 Ω(c) 改進到 Ω(1)。標準 Langevin 在多峰誤差下混合時間可能指數級增長（Eyring–Kramers 定律，約 exp(Δf/θ)）。
