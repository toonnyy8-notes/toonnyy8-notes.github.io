---
slug: awr-advantage-weighted-policy-extraction
title: Advantage-weighted regression 與策略抽取：AWR、AWAC、CRR
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-rl, awr, awac, crr, policy-extraction]
arxiv: ["2110.06169", "1910.00177", "2006.09359", "2006.15134"]
sources: ["arXiv:2110.06169 §1, §2, §4.3", "各論文 arXiv 摘要"]
related: ["iql-lineage-overview", "expectile-regression-value-learning", "iql-algorithm-and-d4rl-evaluation", "offline-rl-distribution-shift"]
---
# Advantage-weighted regression 與策略抽取

IQL 訓練完 Q 與 V 之後，沒有明確的策略，需要另外抽取。論文 §4.3 採用 advantage-weighted regression（引用 Peters 與 Schaal 2007、Wang 等 2018、Peng 等 2019、Nair 等 2020）：

`L_π(φ) = E_{(s,a)~D}[exp(β(Q(s,a) − V(s))) log π_φ(a|s)]`

β 是 inverse temperature。論文說明 β 小時目標接近 behavioral cloning，β 大時嘗試取回 Q 的最大值。目標只使用資料集中的 (s, a)，不查詢未見動作。

## 前人工作

- AWR（Peng、Kumar、Zhang、Levine，*Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning*，arXiv:1910.00177，2019）：以 advantage 加權的監督回歸做 off-policy RL。
- AWAC（Nair、Gupta、Dalal、Levine，*AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*，arXiv:2006.09359，2020）：離線資料預訓練後以線上互動微調。IQL 論文 §5.3 沿用其 dexterous manipulation 設定。
- CRR（Wang 等，*Critic Regularized Regression*，arXiv:2006.15134，2020）：IQL 論文歸類為隱含散度約束的方法之一。

論文引述的性質是：此目標學到的策略在分布約束下最大化 Q 值。論文也說明策略不影響價值函數訓練，所以可以與 TD 學習並行（供線上微調）或在其後執行。價值部分的設計見 [expectile regression 與 value learning](expectile-regression-value-learning.html)，整體演算法見 [IQL 演算法與 D4RL 評估](iql-algorithm-and-d4rl-evaluation.html)。

相關：[總覽](iql-lineage-overview.html)、[offline RL 的 distribution shift](offline-rl-distribution-shift.html)。
