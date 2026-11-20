---
slug: divergence-estimator-topologies
title: 散度估計器拓撲：Single-Sample MC、Atomic 與 Binary
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [estimator, Monte-Carlo, Atomic, Binary, DPPO, RIPO, proximal-region]
arxiv: ["2610.13015", "2602.04879", "2607.10169"]
sources: ["arXiv:2610.13015 §3.3, §3.4, footnote 3", "arXiv abstract 2602.04879", "arXiv abstract 2607.10169"]
related: ["ppo-clip-tv-single-sample-proxy", "f-divergence-geometry-proximal-regions", "gppo-design-space-unification", "gppo-lineage-hub"]
---
# 散度估計器拓撲

屬於 [GPPO 的研究脈絡總覽](gppo-lineage-hub.html)。

**前驅工作。**

- DPPO（Qi et al., 2026，Rethinking the Trust Region in LLM Reinforcement Learning，arXiv:2602.04879）：摘要稱 PPO 的比值 clipping 對低機率 token 過度懲罰、對高機率 token 約束不足；提出以散度的直接估計取代 clipping，並引入 Binary 與 Top-K 近似。
- RIPO（Cai et al., 2026，Beyond Euclidean Clipping，arXiv:2607.10169）：摘要稱 PPO-Clip 隱含以歐氏度量衡量策略差異，與策略 Riemannian 流形的內在幾何不一致，造成低機率區過度保守、高機率區過度激進並使探索崩潰；提出保證等距更新的 RIPO。

**GPPO 論文 §3.3 定義的三種估計器**（對 a ~ µ(·|s)，r = π/µ）：

| 估計器 | D̂_f | 論文說明 |
|---|---|---|
| Single-Sample MC | f(r) | 無偏，但 clipping 決定完全依賴單一樣本，受取樣變異影響 |
| Atomic（GPPO 新提出） | µ(a\|s)·f(r) | 對離散動作展開 D_f = Σ µ f(r) 後，約束每個原子質量的貢獻；低機率動作的大比值偏移不一定是大散度貢獻，可避免過早 clip 稀有的探索動作；附錄證明偏差—變異權衡優於 MC |
| Binary（引自 Qi et al., 2026） | µ f(r) + (1−µ) f((1−µ r)/(1−µ)) | 把動作空間折成「取樣動作」與「其餘」，納入互補質量，反映分佈層級的偏移 |

**Top-K。** 論文腳註 3 指出已試過 Top-K，記憶體開銷 O(K)、候選動作間變異高，使閾值 δ 難以調整，因此不採用。

**δ 的校準（§4.1）。** 為隔離幾何與尺度，各散度的收縮側下限對齊 PPO-Clip 的 r_min = 0.8，即 δ = f(0.8)：δ_TV = 0.2、δ_χ² = 0.04、δ_H ≈ 0.01、δ_KL ≈ 0.22；同一散度的各估計器共用 δ，實際邊界隨 µ(a|s) 調整。

散度端的比較見 [f-散度幾何與 proximal region](f-divergence-geometry-proximal-regions.html)。
