---
slug: in-sample-implicit-value-regularization-eql-ivr
title: EQL 與 IVR：指數型 critic 與隱含價值正則化
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [extreme-q-learning, implicit-value-regularization, in-sample-learning, offline-rl]
arxiv: ["2304.10573", "2301.02328", "2303.15810", "2110.06169", "1801.01290"]
sources: ["arXiv:2304.10573 §2, §4.1, §6.3", "arXiv 摘要 2301.02328", "arXiv 摘要 2303.15810"]
related: ["idql-implicit-policy-and-critic-choice", "expectile-regression-value-learning", "awr-advantage-weighted-policy-extraction", "idql-lineage-hub"]
---
# EQL 與 IVR：對 IQL 的同期推廣

## IDQL 對兩者的描述

IDQL §2 指出，IQL 的 AWR 抽取本身帶有 KL 散度限制，與 critic 所捕捉的策略不一致。其後有兩項工作調整 critic 目標：

- **Extreme Q-learning（EQL）**，Garg, Hejna, Geist, Ermon（arXiv 2301.02328，2023）。IDQL 描述為修改 IQL 的 Q 函數目標，估計與熵正則化 AWR 策略一致的 soft value。
- **Xu et al.**，"Offline RL with No OOD Actions: In-Sample Learning via Implicit Value Regularization"（arXiv 2303.15810，2023）。IDQL 描述為在 behavior-regularized MDP 中對 IQL 的另一種推廣，得到兩個新的隱含 Q-learning 目標；其推廣形式與 IDQL 不同。

## 與 IDQL 的關係

- IDQL 指出指數型損失 f(u) = α·exp(u) − αu 的 critic 目標與 Garg et al. 和 Xu et al. 所用的相同，並與 Haarnoja et al. 的 soft policy（SAC，arXiv 1801.01290）相似。
- 若搭配 AWR 抽取，指數型 critic 是對應的 critic；但 IDQL §6.3 的實驗中它不穩定，需要較多調整，這兩項工作也報告過該目標的不穩定。
- IDQL 與兩者的差別在實際貢獻：兩項工作聚焦於 Q-learning 的目標，IDQL 的主要貢獻在策略抽取。

推導與其他損失（expectile、quantile）見 [廣義 IQL 的隱含 actor 與 critic 損失選擇](idql-implicit-policy-and-critic-choice.html)，IQL 的 expectile 基礎見 [Expectile regression 與價值學習](expectile-regression-value-learning.html)，AWR 見 [Advantage-weighted regression 與策略抽取](awr-advantage-weighted-policy-extraction.html)。

整體脈絡見 [IDQL 的研究脈絡總覽](idql-lineage-hub.html)。
