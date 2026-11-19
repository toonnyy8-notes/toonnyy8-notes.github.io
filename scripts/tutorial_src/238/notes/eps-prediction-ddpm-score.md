---
slug: eps-prediction-ddpm-score
title: ε-prediction 與 DDPM、score-based 模型
cluster: Diffusion & Generative Models
kind: topic
tags: [ddpm, epsilon-prediction, score-based, sde]
arxiv: ["1503.03585", "2006.11239", "2011.13456", "2601.21419"]
sources: [2601.21419]
related: ["prediction-targets-overview", "v-prediction-and-preconditioning"]
---
# ε-prediction 與 DDPM、score-based 模型

回到 [總覽](prediction-targets-overview.html)。

## 文獻
- Sohl-Dickstein 等人 2015，*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*（arXiv:1503.03585）：本論文以 [1] 引用，作為擴散模型的基礎。
- Ho、Jain、Abbeel 2020，*Denoising Diffusion Probabilistic Models*（arXiv:2006.11239）：本論文 related work 指出它展示了 ε-prediction 目標在影像合成上的效果。
- Song 等人 2020，*Score-Based Generative Modeling through Stochastic Differential Equations*（arXiv:2011.13456）：與 DDPM 一同被本論文歸為早期 ε-prediction 路線 [2, 3]。

## 與預測目標的關係
本論文把這個階段描述為「主要處理低解析度或低維資料」的時期，網路輸出是噪聲 n。在廣義式 u = kx − (1 − k)n 中，ε-prediction 是 k = 0。後續的 [v-prediction 與 preconditioning](v-prediction-and-preconditioning.html) 是下一步。
