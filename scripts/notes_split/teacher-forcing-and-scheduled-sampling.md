---
slug: teacher-forcing-and-scheduled-sampling
title: Teacher forcing、scheduled sampling 與序列生成的 exposure bias
cluster: Diffusion & Generative Models
kind: topic
tags: [exposure-bias, teacher-forcing, scheduled-sampling, sequence-generation]
arxiv: ["1506.03099", "1511.06732", "1612.00563", "1910.00292", "2301.11706"]
sources: ["paper.txt §1, §4", "arXiv API metadata"]
related: ["diffusion-exposure-bias"]
---
# Teacher forcing、scheduled sampling 與序列生成的 exposure bias

## 定義

自回歸模型逐步生成序列。訓練時，每一步的條件是資料中的真實前文（teacher forcing）；測試時，條件是模型先前生成的結果。兩者的輸入分佈不同，稱為 exposure bias。

## 文獻（皆經 arXiv 查證，說法依本論文的描述）

| 論文 | arXiv | 本論文引用的說法 |
| --- | --- | --- |
| Bengio 等人 2015，*Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks* | 1506.03099 | 作為 Teacher-Forcing 的出處 |
| Ranzato 等人 2016，*Sequence Level Training with Recurrent Neural Networks* | 1511.06732 | 提出 exposure bias 的問題（與 Schmidt 並列引用） |
| Rennie 等人 2017，*Self-critical Sequence Training for Image Captioning* | 1612.00563 | 論點：以真實前一詞最大化下一詞似然，會在推論時累積誤差，因為模型從未接觸自己的預測 |
| Schmidt 2019，*Generalization in Generation: A closer look at Exposure Bias* | 1910.00292 | 與 Ranzato 並列引用 |

這些條目的標題、作者與年份來自 arXiv API；各文的實際內容未在此重述，只記錄 Ning 等人如何引用它們。

## 與本論文的關係

Ning 等人（2301.11706）指出 DDPM 的訓練輸入 ε_θ(x_t, t) 與取樣輸入 ε_θ(x̂_t, t) 之間的差異「類似」上述問題。對應方式與量測見 [擴散模型的 exposure bias](diffusion-exposure-bias.html)。
