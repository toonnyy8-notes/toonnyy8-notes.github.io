---
slug: advantage-weighted-bc-methods
title: 優勢加權的行為複製：AWR、CRR、AWAC、BAIL
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-rl, advantage-weighted, behavior-cloning, awr, crr, awac, policy-extraction]
arxiv: ["1910.00177", "2006.15134", "2006.09359", "1910.12179", "2106.06860", "2110.06169"]
sources: [1910.00177 abstract, 2006.15134 abstract, 2006.09359 abstract, 1910.12179 abstract, 2106.06860 section 2, "arXiv:2110.06169 §1, §2, §4.3", "各論文 arXiv 摘要"]
related: ["td3bc-algorithm-and-d4rl", "offline-rl-distribution-shift", "iql-lineage-overview", "expectile-regression-value-learning", "iql-algorithm-and-d4rl-evaluation"]
---
# 優勢加權的行為複製

屬於[總覽](td3bc-lineage-overview.html)與[總覽](iql-lineage-overview.html)。

在 TD3+BC 論文的相關工作中寫道：另一類方法「使用加權版本的 BC，偏好高優勢（advantage）的動作」（Wang 等人 2018、Peng 等人 2019、Siegel 等人 2020、Wang 等人 2020、Nair 等人 2020），或「只對資料集的一個明確子集做 BC」（Chen 等人 2020）。

## 代表工作

- **AWR**（Advantage-Weighted Regression，Peng、Kumar、Zhang、Levine，*Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning*，arXiv 1910.00177，2019）：摘要描述為兩個標準監督式學習步驟，一是回歸價值函數目標，二是以加權的目標動作回歸策略。以 advantage 加權的監督回歸做 off-policy RL。
- **CRR**（Critic Regularized Regression，Wang 等，arXiv 2006.15134，2020）：摘要描述為離線 RL 演算法，以 critic 正則化的回歸學習策略。IQL 論文歸類為隱含散度約束的方法之一。
- **AWAC**（Nair、Gupta、Dalal、Levine，*AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*，arXiv 2006.09359，2020）：摘要的重點是利用先前資料加速線上學習，離線資料預訓練後以線上互動微調。TD3+BC 論文表 2 以其報告數字為基準，並說明最相似的前例之一是 Nair 等人（2020）的 SAC+BC 基準。IQL 論文 §5.3 沿用其 dexterous manipulation 設定。
- **BAIL**（Best-Action Imitation Learning，arXiv 1910.12179，2019）：摘要描述為學習 V 函數，選出被認為高表現的動作，再以模仿學習訓練策略。對應 TD3+BC 論文所說「對資料子集做 BC」。

## 在 IQL 中作為策略抽取

IQL 訓練完 Q 與 V 之後，沒有明確的策略，需要另外抽取。IQL 論文 §4.3 採用 advantage-weighted regression（引用 Peters 與 Schaal 2007、Wang 等 2018、Peng 等 2019、Nair 等 2020）：

`L_π(φ) = E_{(s,a)~D}[exp(β(Q(s,a) − V(s))) log π_φ(a|s)]`

β 是 inverse temperature。論文說明 β 小時目標接近 behavioral cloning，β 大時嘗試取回 Q 的最大值。目標只使用資料集中的 (s, a)，不查詢未見動作。論文引述的性質是：此目標學到的策略在分布約束下最大化 Q 值。論文也說明策略不影響價值函數訓練，所以可以與 TD 學習並行（供線上微調）或在其後執行。價值部分的設計見 [expectile regression 與 value learning](expectile-regression-value-learning.html)，整體演算法見 [IQL 演算法與 D4RL 評估](iql-algorithm-and-d4rl-evaluation.html)。

## 與 TD3+BC 的差別

這類方法的 BC 損失有權重，權重由優勢或價值估計決定；TD3+BC 的 BC 項無權重，價值資訊以另一項 λQ 進入同一個目標，並以 Q 的平均絕對值正規化 λ。兩者都不需要明確建模行為策略的機率密度。細節見 [TD3+BC 演算法與 D4RL 評估](td3bc-algorithm-and-d4rl.html)。

前置概念：[離線強化學習與分布偏移](offline-rl-distribution-shift.html)。
