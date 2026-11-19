---
slug: fmlm-discrete-diffusion-factorization
title: 離散擴散語言模型與少步時的因子化誤差
cluster: Diffusion Language Models
kind: topic
tags: [discrete-diffusion, masked-diffusion, factorization-error]
arxiv: ["2602.16813", "2107.03006", "2310.16834", "2406.07524", "2407.15595", "2506.10892", "2502.09992", "2410.08709"]
sources: [paper-2602.16813-intro-background, arxiv-abstracts]
related: [fmlm-lineage-overview, lafa-parallel-decoding-background, fmlm-onehot-denoiser-two-time, hybrid-continuous-discrete-lm]
---
# 離散擴散語言模型與少步時的因子化誤差

屬於 [Flow Map Language Models 的研究脈絡總覽](fmlm-lineage-overview.html)。

## 發展順序

| 年 | 工作 | 內容（依本文描述或 arXiv 標題） |
|---|---|---|
| 2021 | D3PM（Austin 等，2107.03006） | *Structured Denoising Diffusion Models in Discrete State-Spaces*，在離散狀態空間定義擴散 |
| 2023 | SEDD（Lou 等，2310.16834） | *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* |
| 2024 | MDLM（Sahoo 等，2406.07524） | *Simple and Effective Masked Diffusion Language Models*，遮罩式噪聲 |
| 2024 | Discrete Flow Matching（Gat 等，2407.15595） | 把 flow matching 用在離散資料 |
| 2025 | Duo（Sahoo 等，2506.10892） | *The Diffusion Duality*，本文將其視為連續方法不如離散方法的證據之一 |
| 2025 | LLaDA（Nie 等，2502.09992） | *Large Language Diffusion Models*，本文引用為近期離散方法受到重視的例子 |

## 因子化近似

本文第 2 節說明：離散擴散要學轉移密度 p_{t|s}(y_t | y_s)，但它定義在整個序列空間 V^L 上，輸出維度不可行。實務上改用逐 token 的因子化近似（本文式 (2)），假設各 token 在給定較早狀態時條件獨立。本文引用 Campbell 等人，指出這個假設只在 t → s 的極限成立，因此推論需要很多步。

本文圖 3 用只含 new-york 與 san-diego 兩種模式的玩具資料說明：少步取樣時，因子化轉移會產生 new-diego、san-york 這類無效組合。本文也證明（命題 3.5）離散設定下不存在一般性的樣本層級決定性傳輸映射，所以這個困難不能靠改進模型品質消除。

## 少步蒸餾的既有嘗試

Deschenaux 與 Gulcehre（2410.21035）的 SDTT、Hayakawa 等人的 Di4C（2410.08709，*Distillation of Discrete Diffusion through Dimensional Correlations*）是本文比較少步表現時使用的蒸餾基線（Figure 1 的圖例含 MDLM+SDTT、MDLM+Di4C、Duo+Di4C、Duo+DCD、CANDI）。

混合連續與離散的做法見 [混合連續—離散擴散語言模型](hybrid-continuous-discrete-lm.html)。
