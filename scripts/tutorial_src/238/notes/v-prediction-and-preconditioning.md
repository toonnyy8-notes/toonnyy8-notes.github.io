---
slug: v-prediction-and-preconditioning
title: v-prediction 與 preconditioning：Progressive Distillation 與 EDM
cluster: Diffusion & Generative Models
kind: topic
tags: [v-prediction, preconditioning, edm, progressive-distillation]
arxiv: ["2202.00512", "2206.00364", "2601.21419"]
sources: [2601.21419, abstract]
related: ["prediction-targets-overview", "eps-prediction-ddpm-score", "flow-matching-velocity-target"]
---
# v-prediction 與 preconditioning

回到 [總覽](prediction-targets-overview.html)。前一篇：[ε-prediction](eps-prediction-ddpm-score.html)；下一篇：[Flow matching 的速度目標](flow-matching-velocity-target.html)。

## Progressive Distillation（Salimans & Ho 2022，arXiv:2202.00512）
摘要寫明該文提出「新的擴散模型參數化，在少步取樣時提供更高穩定性」，並提出逐次把取樣步數減半的蒸餾方法。本論文把 v-prediction 歸到這一系（引用 [4]）。

## EDM（Karras 等人 2022，arXiv:2206.00364）
摘要指出該文把設計選擇分離成清楚的設計空間，並提出對 score network 的 preconditioning，在 CIFAR-10 達到 FID 1.79（class-conditional）。本論文 related work 的說法是：EDM 透過精心設計的參數化，使網路輸出的變異數在整個 SNR 範圍內維持固定，進而降低訓練難度。

## 與本論文的關係
本論文的附錄 A 把任一目標 w = ξx + ηn 的 MSE 與原始 u-loss 用係數 κ 連起來；在該附錄中 v-loss 對應 w = x − n。本論文指出，這些目標的選擇「與資料的幾何性質大致無關」，這是它要補的缺口。
