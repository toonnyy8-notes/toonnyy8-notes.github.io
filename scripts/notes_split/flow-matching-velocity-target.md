---
slug: flow-matching-velocity-target
title: Flow matching 的速度目標與 rectified flow 的大規模使用
cluster: Diffusion & Generative Models
kind: topic
tags: [flow-matching, velocity, rectified-flow]
arxiv: ["2210.02747", "2403.03206", "2601.21419"]
sources: [2601.21419, abstract]
related: ["prediction-targets-overview", "v-prediction-and-preconditioning", "generalized-target-k-parameterization", "cdlm-training-objective"]
---
# Flow matching 的速度目標

回到 [總覽](prediction-targets-overview.html)。

## Flow Matching（Lipman 等人 2022，arXiv:2210.02747）
摘要：以對「固定條件機率路徑的向量場」做回歸來訓練 CNF，不需模擬；可用的高斯路徑族涵蓋既有擴散路徑；摘要並指出以 FM 搭配擴散路徑是訓練擴散模型的「更穩健、更穩定的替代方案」。本論文引言說明，flow matching 興起後 v-prediction 更為普及。

## Scaling Rectified Flow Transformers（Esser 等人 2024，arXiv:2403.03206）
摘要：rectified flow 以直線連接資料與噪聲；該文改進訓練時的噪聲取樣，使其偏向感知相關的尺度，並在高解析度文字生成影像上與既有擴散公式比較。本論文把它列為 [18]。

## 與本論文的關係
本論文的定理 3.2 在 flow matching 設定（α = t、σ = 1 − t）下推導，因此此設定是理論的前提。語言模型側的對應討論見[連續擴散語言模型的訓練目標與參數化](cdlm-training-objective.html)。廣義式見[廣義目標](generalized-target-k-parameterization.html)。
