---
slug: iql-lineage-overview
title: IQL 的研究脈絡總覽：從 policy constraint 與 value regularization 到 in-sample 的 expectile 學習
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [offline-rl, iql, expectile-regression, d4rl]
arxiv: ["2110.06169", "1812.02900", "1906.00949", "1911.11361", "2006.04779", "2106.06860", "1910.00177", "2006.09359", "2006.15134", "2106.08909", "2106.01345", "1710.10044", "1806.06923", "2005.04269", "2004.07219", "1802.09477"]
sources: ["arXiv:2110.06169 §1, §2, §3, §5", "各被引論文的 arXiv 摘要"]
related: [rl-mdp-q-learning-basics, offline-rl-distribution-shift, batch-constrained-policy-constraint-methods, advantage-weighted-bc-methods, one-step-offline-rl-and-sequence-modeling, expectile-regression-value-learning, iql-algorithm-and-d4rl-evaluation]
---
# IQL 的研究脈絡總覽

Implicit Q-Learning（IQL，arXiv:2110.06169，Kostrikov、Nair、Levine，2021-10-12 提交）處理的問題是：offline RL 要改進行為策略，就必須估計資料集中沒出現過的動作的價值，而這些估計容易因 distributional shift 出錯。論文的做法是訓練時完全不查詢資料集以外的動作。

## 閱讀順序

1. 基礎：[MDP、Q-learning 與 Bellman backup](rl-mdp-q-learning-basics.html)，以及 [offline RL 的 distribution shift 與 extrapolation error](offline-rl-distribution-shift.html)。這兩篇說明問題從何而來。
2. 既有解法一：[policy constraint 與 value regularization](batch-constrained-policy-constraint-methods.html)（BCQ、BEAR、BRAC、CQL、TD3+BC）。論文把這些歸為「multi-step dynamic programming」方法，並指出它們在 trade-off 中取捨。
3. 既有解法二：[單步方法與序列建模](one-step-offline-rl-and-sequence-modeling.html)（One-step RL、Decision Transformer）。這些方法不查詢未見動作，但在需要 stitching 的資料集上表現差。
4. 策略抽取：[advantage-weighted regression](advantage-weighted-bc-methods.html)（AWR、AWAC、CRR）。IQL 用它從 Q 與 V 取出策略。
5. 核心工具：[expectile regression 與 value learning](expectile-regression-value-learning.html)。IQL 用上 expectile 近似資料支撐範圍內的最大值。
6. 整合：[IQL 演算法與 D4RL 評估](iql-algorithm-and-d4rl-evaluation.html)。

## 論文的定位

論文把先前方法分成兩類：每輪都做動態規劃但需約束或正則化的方法，以及只做單步 policy improvement 或完全不用價值函數的方法。IQL 屬於前者，但訓練價值函數時不需要約束。時間序列與引用見 timeline.json。
