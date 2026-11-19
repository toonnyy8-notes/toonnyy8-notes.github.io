---
slug: score-geometry-laplace-expansion
title: 小噪音下的 log p_σ 展開：Laplace 方法與尺度分離（定理 3.1）
cluster: Diffusion & Generative Models
kind: topic
tags: [laplace-method, asymptotic-expansion, rate-separation, gibbs-measure]
arxiv: ["2509.24912"]
sources: [2509.24912 section 3, 2509.24912 appendix B, user_terms_note.md]
related: ["score-geometry-lineage-hub"]
---
# 定理 3.1：尺度分離的來源

回到 [總覽](score-geometry-lineage-hub.html)。

## 結果

對 x ∈ T_M(ε)，式 (6)：

$$\log p_\sigma(x)=-\frac{d_M(x)}{\sigma^2}+\log p_{\text{data}}(\Phi^{-1}(P_M(x)))-\frac{d-n}{2}\log(2\pi\sigma^2)+H(x)+o(1)$$

σ² log p_σ → −d_M 對 T_M(ε) 一致成立（式 (7)）。幾何資訊在 Θ(σ^(−2)) 階，p_data 在 Θ(1) 階；任何 d_M 的誤差會被 σ^(−2) 放大，所以要先還原支撐才可能學到密度。

## 所需的機率工具

- **奇異測度**：μ_data 集中在 Lebesgue 測度為 0 的 M 上，在 R^d 沒有密度，只在 U 上有 p_data(u)。加噪後 p_σ = μ_data * N(0, σ²I) 才有 R^d 密度（式 (3)）；因此 σ → 0 時 log p_σ 在流形外趨向 −∞。
- **弱收斂**：對所有有界連續函數 φ，∫φ dμ_σ → ∫φ dμ。論文所有「收斂到某分佈」都是此意義；密度本身在 R^d 上不收斂。引理 B.2 用 Portmanteau 定理證明支撐在 M 內。
- **Laplace 方法**（推論 B.1，m 為積分維度，x* 為 F 的唯一極小點）：

$$\int_\Omega g\,e^{-F/\theta}dx=e^{-F(x^*)/\theta}\frac{(2\pi\theta)^{m/2}}{\sqrt{|\nabla^2F(x^*)|}}\big(g(x^*)+O(\sqrt\theta)\big)$$

用法：對 p_σ(x) = ∫ N(x; Φ(u), σ²I) p_data(u) du 的積分變數 u 取 F(u) = ½‖x − Φ(u)‖²、g = p_data、θ = σ²。定理 5.2 需 O(θ) 精度的版本（Majerski 2015，arXiv 1511.00302），代價是 F 為 C⁴、g 為 C²，所以定理 5.2 多假設 p_data ∈ C²。
- **Gibbs（Boltzmann）測度**：密度 ∝ exp(−f/θ)。θ = σ²（第 4 節）或 σ^(2−α)（第 5 節）。Hwang（1980）：θ → 0 時集中在 f 的極小值集；若極小值集為流形，極限由法向 Hessian 與次階項決定。定理 B.1 沿用其做法。
- 其他：Tonelli 定理（引理 B.3）、高斯集中（p_σ 集中在 T_M(ε)）、Weyl 不等式（Ĥ 最小特徵值的下界）。
