---
slug: exposure-bias-seq2seq-origin
title: 曝光偏差在序列生成中的來源：Ranzato 等人與 Schmidt
cluster: Diffusion & Generative Models
kind: topic
tags: [exposure-bias, teacher-forcing, sequence-generation]
arxiv: ["1511.06732", "1910.00292", "2308.15321"]
sources: [1511.06732, 1910.00292, 2308.15321]
related: ["exposure-bias-elucidation-overview", "diffusion-exposure-bias", "exposure-bias-prior-mitigations"]
---
# 曝光偏差在序列生成中的來源

回到 [總覽](exposure-bias-elucidation-overview.html)。

## 文獻
- Ranzato、Chopra、Auli、Zaremba，*Sequence Level Training with Recurrent Neural Networks*（2015，arXiv:1511.06732）：本論文（2308.15321）在引言以此作為「exposure bias」一詞的引用（Ranzato et al., 2016）。
- Schmidt，*Generalization in Generation: A closer look at Exposure Bias*（2019，arXiv:1910.00292）：本論文在同一處一併引用。

## 本論文如何使用
本論文引言寫道：Ning 等人（2023）主張迭代取樣鏈會造成曝光偏差，並引用上述兩篇。對擴散模型的敘述為：訓練時網路 ε_θ 的輸入是真實樣本 x_t，推論時的輸入是先前生成的 x̂_t，兩者的差異使 ε_θ(x_t) 與 ε_θ(x̂_t) 不同，導致誤差累積與取樣漂移（sampling drift，Li et al., 2023a）。

擴散模型版本的一般定義見 [擴散模型的曝光偏差](diffusion-exposure-bias.html)。本論文指出該問題缺乏深入探討：沒有量化曝光偏差的度量，也沒有明確的誤差分析（見 [預測誤差與取樣變異數分析](prediction-error-variance-analysis.html)）。
