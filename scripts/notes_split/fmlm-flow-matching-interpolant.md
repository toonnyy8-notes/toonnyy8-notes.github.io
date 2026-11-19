---
slug: fmlm-flow-matching-interpolant
title: Flow matching、interpolant 與速度場／denoiser 的關係
cluster: Diffusion Language Models
kind: topic
tags: [flow-matching, stochastic-interpolant, rectified-flow, denoiser]
arxiv: ["2602.16813", "2210.02747", "2209.03003", "2511.13720", "2209.15571"]
sources: [paper-2602.16813-section-3, arxiv-abstracts]
related: [fmlm-lineage-overview, fmlm-flow-map-distillation-lineage, fmlm-onehot-denoiser-two-time, flow-matching-velocity-target, x-prediction-manifold-view]
---
# Flow matching、interpolant 與速度場／denoiser 的關係

屬於 [Flow Map Language Models 的研究脈絡總覽](fmlm-lineage-overview.html)。

## 前置工作

- Lipman 等人 *Flow Matching for Generative Modeling*（2210.02747，2022）與 Liu 等人 *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*（2209.03003，2022）提出以回歸速度場訓練連續生成模型。
- Albergo、Boffi、Vanden-Eijnden 的 stochastic interpolants（JMLR 2025，本文 [15]）提供統一框架。本文第 3.2 節說明沿用 Lipman 等人與該框架。（該框架的 arXiv ID 未在此驗證；Lafa 論文引用處對應的 arXiv 2209.15571 為其前作 *Building Normalizing Flows with Stochastic Interpolants*。）
- Li 與 He 的 *Back to Basics: Let Denoising Generative Models Denoise*（2511.13720，2025）指出速度預測在 |V| 遠大於網路特徵維度 d 時會欠擬合，本文以此作為改用 denoiser 的理由。

## 本文使用的式子

線性 interpolant I_t = (1 − t) x0 + t x1，x0 ~ N(0, I)。機率流速度 b_t(x) = E[x1 − x0 | I_t = x]，可用平方損失回歸。denoiser D_t(x) = E[x1 | I_t = x]，與速度的關係為 b_t(x) = (D_t(x) − x) / (1 − t)。

預測 x1 比預測 x1 − x0 好的原因：x0 帶有滿秩高斯雜訊，x1 是堆疊的 one-hot，結構化且低熵。

速度目標的另一篇整理見 [Flow matching 的速度目標與 rectified flow 的大規模使用](flow-matching-velocity-target.html)；x 預測的同類觀點見 [x-prediction 與流形觀點](x-prediction-manifold-view.html)。
