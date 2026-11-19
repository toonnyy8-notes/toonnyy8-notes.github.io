---
slug: ttt-linear-mlp
title: 原版 TTT：TTT-Linear / TTT-MLP（隱藏狀態是一個在測試時被訓練的模型）
cluster: Architectures & Efficiency
kind: topic
tags: [test-time-training, ttt-linear, fast-weights, linear-attention, deltanet, rnn]
sources: [ttt-sequence-modeling.md]
arxiv: ["2407.04620", "2501.00663"]
related: ["ttt-sequence-modeling-overview", "ttt-theory-and-applications", "lact-large-chunk-ttt"]
---
# 原版 TTT：TTT-Linear / TTT-MLP

arXiv:2407.04620（已更新到 v4，2025-08-31）。屬於 [TTT 序列建模總覽](ttt-sequence-modeling-overview.html)。

## 直觀與數學原理
作者（Yu Sun, Xinhao Li, Karan Dalal 等 12 人，Stanford/UCSD/Berkeley）的統一觀點是：所有序列層都是「一個隨更新規則轉移的隱藏狀態」。Attention 的隱藏狀態是不斷成長的 KV cache（故 O(n²)）；RNN 的隱藏狀態是固定大小向量（故表達力受限）。TTT 的關鍵一步是讓隱藏狀態本身成為一個帶權重 `W` 的模型 `f`，而更新規則是對一個自監督損失做一步梯度下降：

- 更新規則：`W_t = W_{t-1} − η ∇ℓ(W_{t-1}; x_t)`
- 自監督損失（多視角重建）：`ℓ(W; x_t) = ‖f(θ_K x_t; W) − θ_V x_t‖²`
- 輸出規則：`z_t = f(θ_Q x_t; W_t)`

其中 θ_K、θ_V、θ_Q 是可學的低秩投影（類比 attention 的 K/V/Q）。直覺：fast weight `W` 是一個「聯想記憶」，要從被破壞的輸入 `θ_K x_t` 重建出 `θ_V x_t`。兩個實例化：**TTT-Linear**（f 是線性模型）與 **TTT-MLP**（f 是兩層 MLP）。

## 與周邊家族的關係（重要）
從 test-time regression 框架看：若 f 是線性、損失為負點積，更新規則就退化成 **linear attention**；若 f 是線性、損失為 MSE，就得到 **DeltaNet** 的 delta rule（MSE 強迫重建值在方向與大小上都接近目標，因而產生「遺忘」機制）。**Titans**（2501.00663）在此之上加了 momentum 與 weight decay（遺忘）；**Atlas / MesaNet / Miras** 則進一步改 optimizer 或改成對所有過去 token 全域最優。這說明 TTT 是這一整個「現代 RNN / 線性注意力」家族的母框架之一。反方向的理論還原（TTT≈learned linear attention）見 [TTT 理論與應用](ttt-theory-and-applications.html)。

## 平行化技巧
因更新是序列性的，原版用 **mini-batch TTT**：把序列切成 chunk，chunk 內所有梯度都對 chunk 開頭的 `W` 計算，於是可平行；再用 dual form 加速。但其官方 PyTorch 實作在 GPU 上 FLOPs 利用率 <5%，這正是 [LaCT](lact-large-chunk-ttt.html) 要解決的痛點。

## 自習資源
原論文 arXiv:2407.04620（JAX 與 PyTorch 兩版 code：github.com/test-time-training）；Yingfa Chen 的部落格〈The Rise of Test-Time Training〉對 attention→linear attention→TTT→LaCT 的推導極清晰；Erik Steiger〈Non-Linear Attention and Test-Time-Training〉系列把 optimizer/objective 的設計空間講透。
