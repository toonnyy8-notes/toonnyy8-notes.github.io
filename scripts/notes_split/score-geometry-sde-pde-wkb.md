---
slug: score-geometry-sde-pde-wkb
title: 非梯度漂移的平穩分佈：Fokker–Planck 與 WKB 展開（定理 5.2 的證明工具）
cluster: Diffusion & Generative Models
kind: topic
tags: [fokker-planck, wkb, non-reversible-langevin, maximum-principle]
arxiv: ["2509.24912"]
sources: [2509.24912 section 2.2, 2509.24912 appendix B.4, user_terms_note.md]
related: ["score-geometry-lineage-hub"]
---
# Fokker–Planck 與 WKB 展開

回到 [總覽](score-geometry-lineage-hub.html)。

## 問題

神經網路輸出的分數不一定是某函數的梯度，Langevin 因此不可逆，平穩分佈沒有一般公式（論文引用 Graham & Tél 1984、Maes et al. 2009、Rey-Bellet & Spiliopoulos 2015 等）。既有方法以 WKB 假設式處理，但多只考慮單一穩定點的漂移；本論文的漂移穩定在整個流形上。

## 名詞

- **SDE** dX = b(X) dt + √(2θ) dW；**Fokker–Planck**：∂_t p = −div(b p) + θΔp，平穩時 ∂_t p = 0。
- **可逆**：漂移為 −∇f 時，平穩分佈 ∝ exp(−f/θ)（定理 5.1）；否則一般無公式（定理 5.2）。
- **WKB 假設式**：π_θ ∝ exp(−V/θ) c_θ(x)，c_θ = c₀ + c₁θ + …（式 (5)）；代入方程後按 θ 的冪次令各階係數為 0（匹配漸近展開）。
- **橢圓型算子**：二階項正定，如 Δ_M。**強極大值原理**（Gilbarg–Trudinger Thm 3.5）：無零階項的橢圓方程，解在內部取極大值則為常數；緊緻無邊界流形必有極大值點。

## 證明順序（附錄 B.4，定理 B.3）

1. 平穩方程 0 = div(−b_θ π_θ + θ∇π_θ)，b_θ = −∇f₀ − θ∇f₁ + o(θ)。
2. 代入假設式得式 (14)。
3. θ^(−1) 階（Hamilton–Jacobi）：⟨∇f₀, ∇V⟩ = ‖∇V‖²，得 V = f₀ = d_M。
4. θ⁰ 階（傳輸方程）：令 c̃₀ = e^(f₁)c₀，⟨∇c̃₀, ∇f₀⟩ = 0；c̃₀ 沿梯度流 dψ/dt = −∇f₀(ψ) 不變，只需知道它在 M 上的值。
5. θ¹ 階只在 M 上取值：因 ∇f₀ = 0，高階未知項消失。
6. 換到 (u, r) 座標（引理 B.5），用第 4 步消去對 r 的導數（引理 B.6），得到以 Δ_M 為二階部分、無零階項的方程。
7. 強極大值原理：c̃₀ 為常數，故 c₀ = C e^(−f₁)。

$$\pi_\theta\propto\exp(-V/\theta)\,c_\theta,\quad V=f_0,\ c_0=Ce^{-f_1}$$

**用語提醒**：正文第 5 節寫「c₀ 滿足拋物型（parabolic）PDE」，但附錄推出的是不含時間的二階方程，引用的 Thm 3.5 也是橢圓型結果。理解證明時以附錄為準。
