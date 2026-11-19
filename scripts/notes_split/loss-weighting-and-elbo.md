---
slug: loss-weighting-and-elbo
title: Loss weighting 與 ELBO：目標選擇等於加權選擇
cluster: Diffusion & Generative Models
kind: topic
tags: [loss-weighting, elbo, snr]
arxiv: ["2303.00848", "2410.19324", "2601.21419"]
sources: [2601.21419, abstract]
related: ["prediction-targets-overview", "x-prediction-manifold-view", "generalized-target-k-parameterization"]
---
# Loss weighting 與 ELBO

回到 [總覽](prediction-targets-overview.html)。

## Kingma & Gao 2023（arXiv:2303.00848）
摘要：常用的擴散目標都等於「不同噪聲層級上 ELBO 的加權積分」，加權方式取決於所用目標；加權單調時，目標等於 ELBO 加上簡單的高斯噪聲資料擴增。本論文附錄 A 的推導「與 [17] 一致」，用 κ 係數把不同目標的 loss 互相轉換。

## SiD2（Hoogeboom 等人 2024，arXiv:2410.19324）
摘要：用 Kingma & Gao 的 sigmoid loss weighting、簡化架構與高解析度為主的縮放，使 pixel-space 擴散在 ImageNet512 達 1.5 FID。本論文 related work 把它與 JiT 等歸為採用 x-prediction 或「重度 SNR 加權 loss」以維持穩定的 pixel-space 模型。

## 結論
換目標與換加權在 MSE 下可互相轉換（κ），所以本論文在 3.4 節分析 time conditioning 與 loss 變化如何改變最佳目標。見[廣義目標](generalized-target-k-parameterization.html)；pixel-space 路線見[x-prediction 與流形觀點](x-prediction-manifold-view.html)。
