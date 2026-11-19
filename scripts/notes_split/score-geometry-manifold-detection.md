---
slug: score-geometry-manifold-detection
title: 流形假設與「擴散模型學到流形」的先行研究
cluster: Diffusion & Generative Models
kind: topic
tags: [manifold-hypothesis, score-singularity, intrinsic-dimension, differential-geometry]
arxiv: ["2206.01018", "2212.12611", "2301.07882", "2404.02954", "2505.09922", "2505.02508", "2506.07841", "2509.24912"]
sources: [2509.24912 section 1.1, 2509.24912 section 2, 2206.01018 abstract, 2212.12611 abstract, 2301.07882 abstract, 2404.02954 abstract, user_terms_note.md]
related: ["score-geometry-lineage-hub", "x-prediction-manifold-view"]
---
# 流形假設與擴散模型偵測流形

回到 [總覽](score-geometry-lineage-hub.html)。近期以流形假設分析擴散的另一條線見 [x-prediction 與流形觀點](x-prediction-manifold-view.html)。

## 流形假設

假設 2.1：μ_data 支撐在 R^d 中緊緻、無邊界、C⁴ 的嵌入子流形 M，dim M = n < d。綜述見 Loaiza-Ganem et al. 2024（2404.02954）。緊緻使各種常數有一致界，無邊界是強極大值原理所需，C⁴ 讓距離函數夠光滑以使用 O(θ) 精度的 Laplace 方法。

## 先行研究（依本論文 1.1 節的描述）

- Pidstrigach 2022（2206.01018）：給出高精度分數使質量集中在流形附近的充分正則條件，但未分析誤差如何隨 σ 變化。
- Stanczuk et al. 2024（2212.12611，arXiv 標題為 Your diffusion model secretly knows the dimension of the data manifold）：估計資料流形的內在維度。
- Pavlova & Wei 2025（2506.07841）：主要提供實驗證據。
- Ventura et al. 2024：只分析線性流形（arXiv 編號未查證）。
- Lu, Wang, Bal 2023（2301.07882）、Liu et al. 2025（2505.09922）、Lyu et al. 2025（2505.02508）：小噪音下分數在垂直於流形的方向發散，作用像投影到流形。這對應本論文展開式 (6) 的首項，但他們把分佈資訊放進有界餘項。

## 微分幾何名詞

- **座標圖** Φ: U → M，局部座標 u；Jacobian J = ∂Φ/∂u；黎曼度量 g(u) = JᵀJ。
- **體積測度** dM：

$$\int_M f\,dM=\int_U f(\Phi(u))\sqrt{\det g(u)}\,du$$

- 多張座標圖以**單位分解**拼合；論文為簡化只寫一張。
- **均勻分佈**＝與 dM 成正比，不是對 u 均勻。橢圓上對角度均勻不等於對弧長均勻。
- **管狀鄰域** T_M(ε) = {x : dist(x, M) < ε}；ε 夠小時每個 x 有唯一最近點 P_M(x)。座標 Φ(u, r) = Φ(u) + N(u) r，N(u) 的各行是單位法向量。
- **平方距離** d_M(x) = ½ dist²(x, M)（式 (2)）：∇d_M = x − P_M(x)；法向座標下 d_M = ½‖r‖²，Hessian 為單位矩陣，對每個 u 相同。
- **曲率項** Ĥ_ij = ⟨∂²Φ/∂u_i∂u_j, Φ(u) − x⟩ + ⟨∂Φ/∂u_i, ∂Φ/∂u_j⟩，在流形上退化為 g；來自第二基本形式。
- **Laplace–Beltrami 算子**：Δ_M f = |g|^(−1/2) ∂_i(|g|^(1/2) (g⁻¹)_ij ∂_j f)。
- **推前測度** (Φ_*μ)(S) = μ(Φ(S))，用來定義 p_data(u)。
- reach（觸及半徑）一詞論文未用，以「ε 夠小」代替。
