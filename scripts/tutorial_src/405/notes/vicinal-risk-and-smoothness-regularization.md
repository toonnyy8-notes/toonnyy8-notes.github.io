---
slug: vicinal-risk-and-smoothness-regularization
title: Vicinal risk、Mixup 與預測函數的平滑性正則化
cluster: Diffusion & Generative Models
kind: topic
tags: [regularization, vicinal-risk-minimization, mixup, lipschitz]
arxiv: ["1710.09412", "2301.11706"]
sources: ["paper.txt §1, §5.1, §5.4", "arXiv API metadata"]
related: ["input-perturbation-ddpm-ip", "diffusion-exposure-bias"]
---
# Vicinal risk、Mixup 與預測函數的平滑性正則化

本篇整理 Ning 等人（2301.11706）用來解釋 DDPM-IP 的正則化概念。內容依該論文的描述。

## Vicinal Risk Minimization

Chapelle 等人（2000，NIPS，無 arXiv 版本，未另行查證原文）。論文的描述：在每個訓練樣本周圍定義鄰域，用鄰域內的點擾動該樣本，同時保持其目標標籤不變。

## Mixup

Zhang 等人（2018），*mixup: Beyond Empirical Risk Minimization*（arXiv 1710.09412，經查證）。論文將 DDPM-IP 與 Mixup、VRM 並列為相似的做法。

## 與 DDPM-IP 的對應

DDPM-IP 在網路輸入加入 ξ，但目標仍是 ε：輸入被擾動、目標不變，與 VRM 的結構相同。論文因此說它迫使預測函數變平滑：兩個相近的輸入 x_1、x_2 應得到相近的 ε_θ(x_1, t)、ε_θ(x_2, t)。細節見 [DDPM-IP](input-perturbation-ddpm-ip.html)。

## Lipschitz 正則化

論文 Sec. 5.4 提出兩個替代方案，明確讓 ε_θ 為 Lipschitz 連續：Lipschitz 連續的函數在相鄰點產生小的預測差異，因此對推論時的預測誤差較穩健。論文的結果是 input perturbation 比明確最小化 Lipschitz 常數更有效率也更有效（Sec. 6.1）。論文列出的相關文獻有 Rifai 等人（2011）、Yoshida & Miyato（2017）、Liu 等人（2022）等，此處未逐一查證 arXiv ID。

問題背景見 [擴散模型的 exposure bias](diffusion-exposure-bias.html)。
