---
slug: score-geometry-inverse-problems-guidance
title: 貝氏逆問題、Plug-and-Play 先驗與 classifier-free guidance
cluster: Diffusion & Generative Models
kind: topic
tags: [bayesian-inverse-problems, plug-and-play, classifier-free-guidance, stable-diffusion]
arxiv: ["2103.04715", "2007.13640", "2507.15397", "2207.12598", "2112.10752", "2509.24912"]
sources: [2509.24912 section 2.2, 2509.24912 section 6, 2509.24912 section 7, 2103.04715 abstract, 2507.15397 abstract, 2207.12598 abstract, 2112.10752 abstract, user_terms_note.md]
related: ["score-geometry-lineage-hub", "score-geometry-eval-metrics"]
---
# 貝氏逆問題與 guidance

回到 [總覽](score-geometry-lineage-hub.html)。

## 設定

從觀測 y = A(x) + ξ 推回 x（去模糊、補全、CT 重建）。高斯噪音 ξ ~ N(0, s²I) 時，似然 p(y|x) ∝ exp(−v(x; y))，v = ‖A(x) − y‖²/(2s²)。後驗 p(x|y) ∝ p(y|x) p_prior(x)。**Plug-and-Play**（Venkatakrishnan et al. 2013）把現成去噪器或分數模型當先驗。後驗 Langevin（式 (4)）：

$$dX_t=-\nabla v(X_t;y)\,dt+\hat s(X_t)\,dt+\sqrt2\,dW_t,\quad \hat s\approx\nabla\log p_{\text{prior}}$$

**Tweedie 公式**：E[X | X + σZ = x] = x + σ²∇ log p_σ(x)，聯繫去噪器與分數。

## 先行研究（依本論文描述）

- Laumont et al. 2022（2103.04715）：Langevin 加 Tweedie 的 PnP 貝氏成像；理論需要 ‖ŝ − ∇ log p_σ‖_{L∞} = o(1)（其 Proposition 3.3 與 H2）。
- Kadkhodaie & Simoncelli 2020（2007.13640）：以去噪器隱含的先驗解線性逆問題。
- Pesme et al. 2025（2507.15397）：以去噪器做 MAP 估計；本論文指出其保證需精確分數。

## 本論文的結果（定理 6.1）

以流形上均勻分佈為先驗，分數只需 o(σ^(−2)) 精度；以 p_σ 為先驗則需 o(1) 或精確分數。tempered 後驗動力學（式 (10)）的平穩分佈：

$$dx_t=-\nabla v\,dt-\sigma^{\alpha}\nabla f_\sigma\,dt+\sqrt2\,dW_t\ \Rightarrow\ \tilde\pi(u)\propto e^{-v(\Phi(u))}\frac{dM}{du}(u)$$

v 不被洗掉的原因：WKB 中 θ = σ^(2−α)、f₀ = d_M、f₁ = v；v 為 O(1)，落在 θ¹ 階而保留在極限分佈；分數誤差乘 σ^α 後是 o(θ)，消失。

## Classifier-free guidance（CFG）

Ho & Salimans 2022（2207.12598）：有效分數 = 無條件分數 + w ×（條件分數 − 無條件分數）。本論文把引導項 w(s_cond − s_uncond) 視為 −∇v，無條件分數視為先驗分數，所以 Stable Diffusion 實驗只把無條件分數乘 σ^α。Stable Diffusion 1.5 出自 Rombach et al. 2022（2112.10752）的潛在空間擴散；它用 ε 參數化，分數 = −ε(x, t)/σ_t，σ_t = √(1 − ᾱ_t)。指標見 [評估指標](score-geometry-eval-metrics.html)。
