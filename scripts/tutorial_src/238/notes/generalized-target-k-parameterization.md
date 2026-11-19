---
slug: generalized-target-k-parameterization
title: 廣義預測目標 u = kx − (1 − k)n 與 k-Diff
cluster: Diffusion & Generative Models
kind: topic
tags: [k-diff, optimal-target, intrinsic-dimension, linear-diffusion]
arxiv: ["2601.21419"]
sources: [2601.21419]
related: ["prediction-targets-overview", "x-prediction-manifold-view", "loss-weighting-and-elbo", "diffusion-dynamics-theory-background", "flow-matching-velocity-target"]
---
# 廣義預測目標 u = kx − (1 − k)n 與 k-Diff

回到 [總覽](prediction-targets-overview.html)。以下皆出自 arXiv:2601.21419。

## 廣義式
u = kx − (1 − k)n，0 ≤ k ≤ 1。k = 0、0.5、1 在差一個常數倍下分別為 ε-、v-、x-prediction。

## 理論
- 分析對象：單層線性擴散模型，資料為內在維度 d、嵌入環境維度 D 的低維資料。
- 學習動態分成沿資料流形的平行模式與垂直於流形的垂直模式；前者受資料還原主導，後者受噪聲消除主導。
- 定理 3.2（flow matching 設定、t ~ U[0,1]、κ = 1）：k* = D/(D + d)。D ≈ d 時 k* ≈ 0.5（v-prediction）；D ≫ d 時 k* → 1（x-prediction）；其間最佳 k 落在 0.5 與 1 之間。
- 3.4 節：最佳目標也受 denoiser、time sampler 與 loss 影響。

## k-Diff
內在維度難以估計，因此把 k 當成可學的純量，與 denoiser 一起以反向傳播訓練。摘要稱在 latent-space 與 pixel-space 影像生成上，k-Diff 一致優於固定目標基線。

相關背景：[x-prediction 與流形觀點](x-prediction-manifold-view.html)、[Loss weighting 與 ELBO](loss-weighting-and-elbo.html)、[學習動態的理論背景](diffusion-dynamics-theory-background.html)、[Flow matching 的速度目標](flow-matching-velocity-target.html)。
