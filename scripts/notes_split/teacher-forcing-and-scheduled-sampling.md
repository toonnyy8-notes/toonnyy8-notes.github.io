---
slug: teacher-forcing-and-scheduled-sampling
title: Teacher forcing、scheduled sampling 與序列生成的 exposure bias
cluster: Diffusion & Generative Models
kind: topic
tags: [exposure-bias, teacher-forcing, scheduled-sampling, sequence-generation]
arxiv: ["1506.03099", "1511.06732", "1612.00563", "1910.00292", "2301.11706", "2308.15321"]
sources: ["paper.txt §1, §4", "arXiv API metadata", "1511.06732", "1910.00292", "2308.15321"]
related: ["diffusion-exposure-bias", "exposure-bias-elucidation-overview", "exposure-bias-prior-mitigations"]
---
# Teacher forcing、scheduled sampling 與序列生成的 exposure bias

屬於[總覽](exposure-bias-elucidation-overview.html)。

## 定義

自回歸模型逐步生成序列。訓練時，每一步的條件是資料中的真實前文（teacher forcing）；測試時，條件是模型先前生成的結果。兩者的輸入分佈不同，稱為 exposure bias。

## 文獻（皆經 arXiv 查證，說法依本論文的描述）

| 論文 | arXiv | 本論文引用的說法 |
| --- | --- | --- |
| Bengio 等人 2015，*Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks* | 1506.03099 | 作為 Teacher-Forcing 的出處 |
| Ranzato、Chopra、Auli、Zaremba 2015，*Sequence Level Training with Recurrent Neural Networks* | 1511.06732 | 提出 exposure bias 的問題（與 Schmidt 並列引用）；2308.15321 在引言以此作為「exposure bias」一詞的引用（Ranzato et al., 2016） |
| Rennie 等人 2017，*Self-critical Sequence Training for Image Captioning* | 1612.00563 | 論點：以真實前一詞最大化下一詞似然，會在推論時累積誤差，因為模型從未接觸自己的預測 |
| Schmidt 2019，*Generalization in Generation: A closer look at Exposure Bias* | 1910.00292 | 與 Ranzato 並列引用；2308.15321 在同一處一併引用 |

這些條目的標題、作者與年份來自 arXiv API；各文的實際內容未在此重述，只記錄 Ning 等人（2301.11706）與 2308.15321 如何引用它們。

## 與 2301.11706 的關係

Ning 等人（2301.11706）指出 DDPM 的訓練輸入 ε_θ(x_t, t) 與取樣輸入 ε_θ(x̂_t, t) 之間的差異「類似」上述問題。對應方式與量測見 [擴散模型的 exposure bias](diffusion-exposure-bias.html)。

## 2308.15321 如何使用

2308.15321 引言寫道：Ning 等人（2023）主張迭代取樣鏈會造成曝光偏差，並引用 Ranzato 與 Schmidt 兩篇。對擴散模型的敘述為：訓練時網路 ε_θ 的輸入是真實樣本 x_t，推論時的輸入是先前生成的 x̂_t，兩者的差異使 ε_θ(x_t) 與 ε_θ(x̂_t) 不同，導致誤差累積與取樣漂移（sampling drift，Li et al., 2023a）。

2308.15321 指出該問題缺乏深入探討：沒有量化曝光偏差的度量，也沒有明確的誤差分析（見 [預測誤差與取樣變異數分析](prediction-error-variance-analysis.html)）。既有緩解方法見 [曝光偏差的既有緩解方法](exposure-bias-prior-mitigations.html)。
