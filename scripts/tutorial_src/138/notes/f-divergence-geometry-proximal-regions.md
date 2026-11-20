---
slug: f-divergence-geometry-proximal-regions
title: f-散度幾何與 proximal region：Vajda 不等式與 Theorem 3
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [f-divergence, Vajda, Pinsker, curvature, KL, chi-square, Hellinger, TV]
arxiv: ["2610.13015", "1502.05477"]
sources: ["arXiv:2610.13015 §3.1, §3.2, Appendix A"]
related: ["trpo-monotonic-improvement-tv-bound", "divergence-estimator-topologies", "gppo-design-space-unification", "gppo-lineage-hub"]
---
# f-散度幾何與 proximal region

屬於 [GPPO 的研究脈絡總覽](gppo-lineage-hub.html)。本篇內容皆為 GPPO 論文（arXiv:2610.13015）§3.1–3.2 的整理。

**目標。** 把 [TRPO 單調改進下界](trpo-monotonic-improvement-tv-bound.html)中的 D_TV² 換成任意 f-散度 D_f(P‖Q) = ∫ q f(p/q)，f 為凸函數且 f(1) = 0。

**Theorem 2（Vajda 緊下界）。** 令 V = D_TV(P‖Q)。則 D_f ≥ L_f(V)，其中 L_f(V) 是對輔助質量 β ∈ [0, 1−V] 取 β f(1 + V/β) + (1−β) f(1 − V/(1−β)) 的下確界。附錄以 Hahn 分解把空間化為二元分割，並以 Jensen 不等式導出，且以二元分佈說明此界緊。

**最佳常數。** C_f^Vajda = inf_{0<V≤1} L_f(V)/V²，因此 D_TV² ≤ D_f / C_f^Vajda。

**Theorem 3。** 對 C_f^Vajda > 0 的 f-散度，η(π) − η(µ) ≥ L_µ(π)/(1−γ) − 4ξγ/((1−γ)² C_f^Vajda) · D_f^max(π‖µ)。因此約束任何 f-散度於足夠小的水準集，即保證單調改進；TV、Pearson χ²、KL、Hellinger 皆適用。

**局部等價與有限偏移。** 在 r = 1 附近 Taylor 展開，D_f ≈ f''(1)/2 · E[(r−1)²]，所以 f''(1) > 0 的光滑 f-散度共享同一個局部 Fisher–Rao 幾何，差異出現在 r 偏離 1 的有限偏移區。

**四種幾何（論文式 (14)–(17)）。**

| 散度 | f(r) | f''(r) | 論文描述的性質 |
|---|---|---|---|
| TV | ½\|r−1\| | 0（r≠1） | 分段線性、零曲率，邊際懲罰與偏離量無關 |
| Pearson χ² | (r−1)² | 2 | 橢球（Mahalanobis）幾何，懲罰隨偏離平方成長 |
| KL | −log r | 1/r² | 不對稱、對數扭曲；r→0 的收縮受更強懲罰，r 變大時邊際阻力遞減，有利提高低機率動作的機率 |
| Hellinger | (√r−1)² | 1/(2r^{3/2}) | 平方根嵌入下的球面幾何，曲率在 r→0 增大、r 變大時減小 |

**Proximal region。** 以 D̂_f 表示估計器，區域為 R̂_f(δ) = { r > 0 : D̂_f(r) ≤ δ }，目標為 min(r·Â, clip_{R̂_f(δ)}(r)·Â)。估計器的選擇見 [散度估計器拓撲](divergence-estimator-topologies.html)；(f, T) 如何涵蓋既有方法見 [GPPO 的設計空間](gppo-design-space-unification.html)。
