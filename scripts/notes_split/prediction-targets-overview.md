---
slug: prediction-targets-overview
title: 擴散模型預測目標的演進總覽：從 ε 到 v 到 x
cluster: Diffusion & Generative Models
kind: hub
tags: [prediction-target, epsilon-prediction, v-prediction, x-prediction, flow-matching, manifold]
arxiv: ["2601.21419", "1503.03585", "2006.11239", "2011.13456", "2202.00512", "2206.00364", "2210.02747", "2303.00848", "2403.03206", "2410.19324", "2511.13720", "2412.20292", "2402.18491", "2505.17638"]
sources: [2601.21419]
related: ["eps-prediction-ddpm-score", "v-prediction-and-preconditioning", "flow-matching-velocity-target", "loss-weighting-and-elbo", "x-prediction-manifold-view", "generalized-target-k-parameterization", "diffusion-dynamics-theory-background", "pixel-space-diffusion", "cdlm-training-objective", "velocity-irreducible-variance"]
---
# 擴散模型預測目標的演進總覽：從 ε 到 v 到 x

「預測目標」指網路輸出要逼近的量：噪聲 n、速度（x − n）或乾淨資料 x。*Understanding and Learning Prediction Targets in Diffusion Models*（arXiv:2601.21419）的引言把這條脈絡寫成：早期以 ε-prediction 為主，後來被 v-prediction 取代，近期在高解析度像素空間又出現 x-prediction 較佳的結果。本文依閱讀順序列出各主題筆記。

## 閱讀順序
1. [ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)：擴散模型起點與噪聲預測。
2. [v-prediction 與 preconditioning](v-prediction-and-preconditioning.html)：Progressive Distillation 與 EDM 的參數化。
3. [Flow matching 的速度目標](flow-matching-velocity-target.html)：回歸向量場，與 rectified flow 的大規模使用。
4. [Loss weighting 與 ELBO](loss-weighting-and-elbo.html)：不同目標等價於不同的 loss 加權。
5. [x-prediction 與流形觀點](x-prediction-manifold-view.html)：JiT 與 pixel-space 的近期結果；pixel-space 的完整陣營見[Pixel-space diffusion](pixel-space-diffusion.html)。
6. [廣義目標 u = kx − (1 − k)n 與 k-Diff](generalized-target-k-parameterization.html)：本論文的理論與方法。
7. [擴散模型學習動態的理論背景](diffusion-dynamics-theory-background.html)：本論文所接續的分析傳統。

## 相關的既有筆記
- [連續擴散語言模型的訓練目標與參數化](cdlm-training-objective.html)：同一個問題在語言模型上的版本。
- [Velocity Irreducible Variance](velocity-irreducible-variance.html)：以速度目標的歧義衡量 latent 品質。

## 一句話對照
ε、v、x 在本論文的廣義式中分別對應 k = 0、0.5、1；理論結果 k* = D/(D+d) 說明最佳 k 依環境維度 D 與內在維度 d 而定。細節見第 6 篇。
