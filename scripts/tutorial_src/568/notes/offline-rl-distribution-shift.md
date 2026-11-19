---
slug: offline-rl-distribution-shift
title: Offline RL 的分布偏移與 OOD 動作造成的 Q 值高估
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-RL, distribution-shift, OOD-actions, policy-constraint, overestimation]
arxiv: ["2006.04779", "1812.02900", "1906.00949", "1911.11361", "2005.01643"]
sources: ["arXiv:2006.04779 §1, §2, §6", "arXiv abstract 1812.02900", "arXiv abstract 1906.00949", "arXiv abstract 1911.11361", "arXiv abstract 2005.01643"]
related: ["rl-mdp-q-learning-basics"]
---
# Offline RL 的分布偏移

**設定。** Offline RL（亦稱 batch RL）只從事先收集的資料集學習，訓練期間不與環境互動。CQL（arXiv:2006.04779）指出，這在原則上能利用大量資料，但實務上的主要困難來自產生資料的策略與學到的策略之間的分布偏移。Q-learning 與 actor-critic 的記號見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)。

**動作分布偏移。** 依 CQL §2，以基本的 policy evaluation／improvement 流程訓練時，Bellman backup 的目標值使用學到的策略 `π^k` 取樣的動作，但 Q 函數只在行為策略 `π_β` 產生的動作上訓練。`π` 又被訓練去最大化 Q 值，因此可能偏向 Q 值被錯誤高估的 out-of-distribution（OOD）動作。Online RL 可以實際執行該動作並觀察真實價值來修正這種誤差；offline 設定無法互動，誤差就保留下來，表現為過度樂觀的價值估計。論文另指出這個問題與 overfitting 並列，是直接把 off-policy 演算法用於 offline 時效果不佳的原因。

**狀態分布偏移。** CQL 指出 offline 的 Q 函數訓練沒有狀態分布偏移，因為 Bellman backup 不會在 OOD 狀態上查詢 Q 函數；但策略在測試時仍可能遇到狀態分布偏移。

**既有對策：限制策略。** CQL 描述的典型做法是把學到的策略限制在遠離 OOD 動作的範圍：

- BCQ（Fujimoto、Meger、Precup，*Off-Policy Deep Reinforcement Learning without Exploration*，arXiv:1812.02900，2018）。依其 abstract，指出 off-policy 演算法在固定資料集上表現差，並提出只選擇與資料集中行為相近之動作的方法。
- BEAR（Kumar 等，*Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction*，arXiv:1906.00949，2019）。依其 abstract，把問題歸因於 bootstrapping error（以 OOD 動作做 bootstrap 造成的誤差累積），並以限制策略支撐集的方式降低。
- BRAC（Wu、Tucker、Nachum，*Behavior Regularized Offline Reinforcement Learning*，arXiv:1911.11361，2019）。依其 abstract，以行為正則化的 actor-critic 為框架，論文比較的是這類方法的變體。

綜述見 Levine、Kumar、Tucker、Fu 的 *Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems*（arXiv:2005.01643，2020），CQL 以它作為 offline RL 與策略限制的參考。

**CQL 的切入點。** CQL 不限制策略，而是學一個保守的 Q 函數：在標準 Bellman error 上加一項懲罰，使策略在學到的 Q 函數下的期望值成為其真實價值的下界，以此抑制 OOD 動作的過度樂觀。CQL 的實驗以 BEAR 與 BRAC 作為策略限制類的對照。
