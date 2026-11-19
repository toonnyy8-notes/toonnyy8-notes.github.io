---
slug: diffusion-dynamics-theory-background
title: 擴散模型學習與泛化動態的理論背景
cluster: Diffusion & Generative Models
kind: topic
tags: [learning-dynamics, generalization, memorization, deep-linear-networks]
arxiv: ["2412.20292", "2402.18491", "2505.17638", "1312.6120", "2601.21419"]
sources: [2601.21419, abstract]
related: ["prediction-targets-overview", "generalized-target-k-parameterization"]
---
# 擴散模型學習與泛化動態的理論背景

回到 [總覽](prediction-targets-overview.html)。本論文 related work 分兩條線。

## 擴散模型的泛化與記憶
- Kamb & Ganguli 2024（arXiv:2412.20292）：摘要指出局部性與等變性兩種歸納偏置使模型無法達到最佳 score matching，進而產生組合式創造力。
- Biroli、Bonnaire、De Bortoli、Mézard（*Dynamical Regimes of Diffusion Models*，arXiv:2402.18491；本論文引用 Nature Communications 2024 版）：以統計物理分析，反向生成過程有三個動力學區段，含 speciation 與 collapse 兩個轉變。
- Bonnaire 等人 2025（arXiv:2505.17638）：摘要指出兩個時間尺度 τ_gen 與 τ_mem，τ_mem 隨訓練集大小 n 線性增加而 τ_gen 不變。

本論文評語：這些工作談的是訓練與取樣的整體性質，預測目標的選擇對其影響「大多未被探討」。

## 神經網路學習動態
本論文引用 Saxe 等人 2013（arXiv:1312.6120）：以深層線性網路的精確解分析學習動態。本論文的單層線性擴散模型承接這種「簡化模型加解析推導」的做法。其餘引用（權重初始化、Advani 等人）此處未逐一查證，故不收。

接續：[廣義目標與 k-Diff](generalized-target-k-parameterization.html)。
