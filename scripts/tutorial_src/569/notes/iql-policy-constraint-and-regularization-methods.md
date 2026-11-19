---
slug: iql-policy-constraint-and-regularization-methods
title: Offline RL 的 policy constraint 與 value regularization：BCQ、BEAR、BRAC、CQL、TD3+BC
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-rl, policy-constraint, value-regularization, bcq, cql, td3-bc]
arxiv: ["2110.06169", "1812.02900", "1906.00949", "1911.11361", "2006.04779", "2106.06860", "1802.09477"]
sources: ["arXiv:2110.06169 §1, §2, §3 對這些方法的描述", "各論文 arXiv 摘要"]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "iql-lineage-overview", "expectile-regression-value-learning"]
---
# Policy constraint 與 value regularization

這一類方法的共同點是在近似動態規劃（Q-learning 或 actor-critic）上加限制，使學到的策略或價值不偏離資料。背景見 [offline RL 的 distribution shift](offline-rl-distribution-shift.html)，Q-learning 的定義見 [MDP 與 Q-learning](rl-mdp-q-learning-basics.html)。

IQL 論文 §3 指出，TD 損失 `r + γ max Q(s', a') − Q(s, a)` 中的 out-of-distribution 動作 a' 會使 Q(s', a') 出錯，並因策略取 arg max 而造成高估。論文 §2 把對策分成下列幾種實作方式。

## 約束策略

- 用明確的密度模型約束：BCQ（Fujimoto、Meger、Precup，*Off-Policy Deep Reinforcement Learning without Exploration*，arXiv:1812.02900，2018）、BEAR（Kumar 等，*Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction*，arXiv:1906.00949，2019）、BRAC（Wu、Tucker、Nachum，*Behavior Regularized Offline Reinforcement Learning*，arXiv:1911.11361，2019）。IQL 論文將這三者列在 explicit density model 的引用中（Wu et al., 2019; Fujimoto et al., 2019; Kumar et al., 2019）。
- 在策略改進目標加上監督學習項：TD3+BC（Fujimoto 與 Gu，*A Minimalist Approach to Offline Reinforcement Learning*，arXiv:2106.06860，2021）。它建立在 TD3（arXiv:1802.09477，2018）之上。

## 正則化價值函數

- CQL（Kumar 等，*Conservative Q-Learning for Offline Reinforcement Learning*，arXiv:2006.04779，2020）：直接讓 Q 函數對 out-of-distribution 動作給出較低的值。IQL 論文將 CQL 歸為直接正則化 Q 函數的方法。

## 與 IQL 的差別

論文指出，這些方法在策略改進幅度與對價值誤估的脆弱度之間有取捨。論文的觀察是：資料集內的動作約束未必足以避免 value function extrapolation。IQL 的做法見 [expectile regression 與 value learning](expectile-regression-value-learning.html)，訓練價值時不查詢未見動作，因此不需要這類約束；後續的策略抽取步驟仍含隱含約束，但不影響價值函數訓練。

相關：[總覽](iql-lineage-overview.html)。
