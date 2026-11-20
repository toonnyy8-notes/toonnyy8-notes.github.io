---
slug: diffusion-behavior-model-policy-extraction
title: 以擴散行為模型做策略抽取：SfBC、ARQ、EMaQ 與 IDQL
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [diffusion-policy, behavior-cloning, importance-sampling, policy-extraction, offline-rl]
arxiv: ["2304.10573", "2209.14548", "2206.00695", "2007.11091", "2208.06193", "2103.15209"]
sources: ["arXiv:2304.10573 §2, §4.2, §5, 附錄 F", "arXiv 摘要 2209.14548、2206.00695、2007.11091、2208.06193"]
related: [idql-implicit-policy-and-critic-choice, diffusion-policy-class, advantage-weighted-bc-methods, diffusion-bc-ddpm-and-network-design, idql-evaluation-and-online-finetuning]
---
# 以擴散行為模型做策略抽取

## IDQL 的抽取方式（演算法 2）

1. 預訓練 Q_θ̂(s,a)、V_ψ(s) 與行為模型 μ_φ(a|s)。
2. 每個狀態從 μ_φ 取 N 個動作 a_i。
3. 用權重 w(s,a_i)（式 9、10、12）算出 p_i = w_i / Σ_j w_j，以類別分布選出動作。
4. 實務上取 argmax_{a_i} Q(s,a_i) 的表現較好，等於 w 取 one-hot。論文對照 SAC 等方法：訓練 critic 用隨機 actor，評估用確定性 actor。

為何不直接把 AWR 權重放進擴散模型的目標：論文引用文獻指出，表達力高的模型搭配重要性加權目標，可能提高所有訓練點的 likelihood，不論權重大小（引用 Byrd & Lipton 與 Xu et al. arXiv 2103.15209）；IDQL 在附錄 F 報告 AWR 加 DDPM 目標未改善表現。因此改為先擬合行為策略，再對樣本重新加權。

## 相關的擴散式離線 RL 方法（依 IDQL §2 的描述）

| 方法 | 做法 | 與 IDQL 的差異 |
| --- | --- | --- |
| EMaQ（2007.11091） | 自迴歸行為複製模型；critic 學習時取其 argmax 動作 | 非擴散模型 |
| SfBC（2209.14548） | 擴散行為模型的重要性重新加權定義策略，再以該策略的樣本做 value iteration 訓練 critic | critic 訓練與策略模型耦合 |
| ARQ（2206.00695） | 用擴散行為模型做 critic 更新，再用 AWR 抽取策略 | 抽取仍用 AWR |
| DQL（2208.06193） | 擴散模型作為 actor，TD3+BC 風格 | 見 [擴散策略作為表達力強的策略類別](diffusion-policy-class.html) |

IDQL 將擴散行為模型與 critic 學習分開，只在評估時互動，因此超參數調整較有效率；論文稱就其所知是第一個把擴散用於線上微調的方法。critic 側的推導見 [廣義 IQL 的隱含 actor 與 critic 損失選擇](idql-implicit-policy-and-critic-choice.html)，原抽取方式見 [Advantage-weighted regression 與策略抽取](advantage-weighted-bc-methods.html)。擴散模型本身的設計見 [擴散行為複製、DDPM 與 LN_Resnet](diffusion-bc-ddpm-and-network-design.html)。

實驗結果見 [IDQL 的實驗與線上微調](idql-evaluation-and-online-finetuning.html)。
