---
slug: td3bc-lineage-overview
title: TD3+BC 研究脈絡總覽：從 actor-critic 到最小化的離線 RL
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [offline-rl, td3, behavior-cloning, minimalism]
arxiv: ["2106.06860", "1509.02971", "1802.09477", "1812.02900", "2004.07219"]
sources: [2106.06860 abstract and section 1-2, 2106.06860 related work]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "ddpg-td3-actor-critic", "batch-constrained-policy-constraint-methods", "advantage-weighted-bc-methods", "rl-plus-behavior-cloning-prior-work", "rl-implementation-details-reproducibility", "td3bc-algorithm-and-d4rl"]
---
# TD3+BC 研究脈絡總覽

「A Minimalist Approach to Offline Reinforcement Learning」（arXiv 2106.06860，NeurIPS 2021）的主張是：只對 TD3 的策略更新加上一項行為複製（behavior cloning, BC）正則項，並對狀態做標準化，就能在 D4RL 上達到與當時最佳離線 RL 演算法相當的表現，且計算成本不到一半。

閱讀順序如下。

1. 先備知識：[強化學習與 MDP、Q-learning 基礎](rl-mdp-q-learning-basics.html)、[離線強化學習與分布偏移](offline-rl-distribution-shift.html)。前者說明 MDP、價值函數與 Q-learning；後者說明為什麼固定資料集會讓 off-policy 演算法失效。
2. 底層演算法：[DDPG 到 TD3：連續動作的 actor-critic](ddpg-td3-actor-critic.html)。TD3+BC 修改的是這一類演算法的策略更新。
3. 離線 RL 的第一批解法：[批次約束與策略約束方法（BCQ、BEAR、BRAC 等）](batch-constrained-policy-constraint-methods.html)，以及[優勢加權的行為複製（AWR、CRR、AWAC 等）](advantage-weighted-bc-methods.html)。
4. 與本論文最接近的前例：[RL 與行為複製的結合](rl-plus-behavior-cloning-prior-work.html)。這些工作的目的是加速線上學習，並非處理離線設定。
5. 本論文的動機之一：[深度 RL 的實作細節與可重現性](rl-implementation-details-reproducibility.html)。
6. 本論文的方法與實驗：[TD3+BC 演算法與 D4RL 評估](td3bc-algorithm-and-d4rl.html)。

年代順序見同資料夾的 timeline.json。論文自己的分類把相關工作分為 RL+BC、RL+Imitation、Offline RL 與 Meta Analyses 四類，本組筆記照這個分類拆分。
