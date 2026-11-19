---
slug: distributional-rl-return-distributions
title: Distributional RL：回報分布與分布式 Bellman 方程
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [distributional-rl, return-distribution, C51, quantile-regression, IQN, Wasserstein]
arxiv: ["1707.06887", "1710.10044", "1806.06923", "2611.02160"]
sources: ["arXiv 摘要 1707.06887、1710.10044、1806.06923", "arXiv:2611.02160 §1、§2、§3 對 distributional RL 的敘述", "Bellemare 等 2023（MIT Press，非 arXiv，僅依 2611.02160 的引用）"]
related: ["rl-mdp-q-learning-basics", "expectile-regression-value-learning", "flow-based-return-critics", "flow-bellman-operator-all-noise-levels"]
---
# Distributional RL：回報分布與分布式 Bellman 方程

**動機。** 標準 RL 的價值函數只學習回報的期望值 $Q^\pi(s,a)$（見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)）。Distributional RL 改為學習回報隨機變數 $Z^\pi(s,a)=\sum_t \gamma^t r(S_t,A_t)$ 的整個分布，$Q^\pi$ 是它的期望值。2611.02160 的例子：一個動作以相等機率得到 -1 與 +1，另一個動作固定得到 0，兩者期望值相同，但分布不同；以整個分布作為 TD 目標可提供較豐富的監督（該文引用 Wang 等 2023、Rowland 等 2023）。

**分布式 Bellman 方程。** 依 2611.02160 §3 的描述，$Z^\pi(s,a) \overset{d}{=} r(s,a)+\gamma Z^\pi(S',A')$（等號表示分布相等）。對應的分布式 Bellman 算子在最大 Wasserstein 距離下是 $\gamma$-contraction，唯一不動點為 $Z^\pi$；兩邊取期望則回到一般的 Bellman 方程。

**表示回報分布的方式**（2611.02160 §2 的分類）：
- 離散類別分布：C51，Bellemare 等 2017（arXiv 1707.06887）。摘要：主張價值分布的重要性，並設計把 Bellman 方程用於學習近似價值分布的演算法，在 Arcade Learning Environment 上評估。
- 分位數：QR-DQN，Dabney 等 2018（arXiv 1710.10044，分位數回歸）；IQN，Dabney 等 2018（arXiv 1806.06923，對隨機抽樣的分位數位置學習分位數函數）。IQN 使用 Huber 損失。
- 具有限制性建模假設的連續分布（該文列舉 DSAC 等）。
- 近期以 diffusion 或 flow matching 表示複雜回報分布的方法，見 [以 flow matching 表示回報分布的 critic](flow-based-return-critics.html)。

**與 expectile 的關係。** 分位數回歸與 expectile 回歸同屬非對稱損失，後者用於 IQL，見 [Expectile regression 與 value learning](expectile-regression-value-learning.html)。

**與本文的連結。** Value Forcing 在 $\tau=1$（去噪後）的端點恢復上述分布式 Bellman 方程，並把它擴展到所有 noise level，見 [Flow Bellman operator](flow-bellman-operator-all-noise-levels.html)。
