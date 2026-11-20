---
slug: policy-gradient-baselines-score-function
title: Policy gradient、baseline 與 score function
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [policy-gradient, REINFORCE, baseline, variance-reduction, score-function]
arxiv: ["2609.20807", "2503.20783"]
sources: ["arXiv:2609.20807 §1, §3 (footnote 4), §4", "arXiv abstract 2503.20783", "Williams 1992 與 Greensmith et al. 2004 為本文引用，無 arXiv ID，未另行查證"]
related: ["ppo-grpo-dapo-llm-rl-objectives", "score-centering-drift-decomposition", "rl-mdp-q-learning-basics"]
---
# Policy gradient、baseline 與 score function

**Policy gradient。** 本文把 PPO、GRPO、DAPO、IcePop 都視為建立在 policy gradient（Williams, 1992）之上。作法是對 rollout 取樣、為每條 rollout 指定獎勵 R，再以 R 加權計算梯度：$\nabla_\theta E_p[R] = E_p[R\,\nabla_\theta \log p_\theta(y)]$。其中 $\nabla_\theta \log p_\theta$ 稱為 score。MDP 與價值函數的記號見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)。

**期望 score 為 0。** 對任何分佈，$E_p[s_y] = \sum_v p_v \nabla_\theta \log p_v = \nabla_\theta \sum_v p_v = \nabla_\theta 1 = 0$。本文第 4 節以此說明：獎勵恆為 +1 時，on-policy 的期望梯度為 0，沒有學習訊號。

**Baseline 與 control variate。** 本文（§4）指出，獎勵 baseline（Williams, 1992）與 score function control variate（Ranganath et al., 2014）都依賴同一個恆等式，用來降低梯度變異。on-policy 時，baseline 改變變異而不改變期望。本文腳註 4 說明：獎勵平移一個常數不改變 on-policy 期望梯度，group centering 只把它縮放 $1-1/G$，中心化的理由是降低變異（Greensmith et al., 2004；Liu et al., 2025c）。Liu et al.（arXiv 2503.20783）在摘要中分析 R1-Zero 類訓練，並指出 GRPO 存在最佳化偏差。

**與 score centering 的差別。** 本文主張 score centering 是偏差修正而非變異降低：減去的向量在給定 prefix 下是確定的，目的是移動更新的平均值。off-policy 時，等於 prefix 期望獎勵的 baseline（精確的價值函數）在期望上也能消除 drift，score centering 不需要 critic 即可得到相同的期望更新。細節見 [Score centering：drift 與 signal 的分解](score-centering-drift-decomposition.html)。
