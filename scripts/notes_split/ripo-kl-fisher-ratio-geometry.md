---
slug: ripo-kl-fisher-ratio-geometry
title: KL 散度、Fisher 矩陣與重要性比值的二次式
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [KL-divergence, Fisher-information, TRPO, Riemannian-geometry, importance-ratio]
arxiv: ["2607.10169", "1502.05477"]
sources: ["arXiv:2607.10169 §2.1, §3.2, Eq. 4–8", "arXiv abstract 1502.05477"]
related: ["ripo-lineage-hub", "trpo-monotonic-improvement-tv-bound", "f-divergence-geometry-proximal-regions"]
---
# KL 散度、Fisher 矩陣與重要性比值的二次式

屬於 [RIPO 的研究脈絡總覽](ripo-lineage-hub.html)。本篇整理 RIPO（arXiv 2607.10169）§3.2 如何從 TRPO 的 KL 約束推出「幾何距離」。

**TRPO 的約束。** TRPO（Schulman et al., 2015，arXiv 1502.05477）在新舊策略的 KL 散度不超過 δ 的條件下最大化 surrogate 目標，本文（§2.1）稱此約束可保證單調改進，但約束式最佳化的計算量大，難以擴展。TRPO 的下界推導見 [TRPO：trust region 與單調改進下界](trpo-monotonic-improvement-tv-bound.html)。

**二階展開。** 本文將 Φ(θ) = D_KL(π_old(·|s) ‖ π_θ(·|s)) 在 θ_old 展開到二階。零階與一階項為 0，得到 Φ ≈ ½ Δθᵀ F Δθ，其中 F 是 Fisher information matrix，本文稱它定義了一個 Riemannian manifold。

**映射到機率空間。** 對 π_θ 做一階 Taylor 展開，π_θ(a|s) ≈ π_old(a|s) + ∇π_θ(a|s)ᵀΔθ，代入 Fisher 矩陣的定義（Eq. 5）後得到

D_KL(π_old ‖ π_θ) ≈ ½ Σ_a (π_θ − π_old)² / π_old = ½ Σ_a π_old(a|s) (r(θ) − 1)²　（Eq. 7）

其中 r = π_θ/π_old 是重要性比值。因此單一動作的幾何距離正比於 π_old(a|s)·(r−1)²（Eq. 8）。

**要點。** 距離同時取決於比值偏離 (r−1) 與該動作在舊策略下的機率 π_old。同樣的 |r−1|，高機率動作對應的距離大，低機率動作對應的距離小，所以本文稱此幾何是非均勻的。

f-散度在 r = 1 附近的局部等價（皆近似 f''(1)/2·E[(r−1)²]）另見 [f-散度幾何與 proximal region](f-divergence-geometry-proximal-regions.html)；本文只使用 KL 這一種。
