---
slug: trpo-monotonic-improvement-tv-bound
title: TRPO：trust region 與單調改進下界（GPPO 的理論起點）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [TRPO, trust-region, monotonic-improvement, TV-divergence, GAE, Pinsker]
arxiv: ["2610.13015", "1502.05477", "1506.02438"]
sources: ["arXiv:2610.13015 §2.1, §2.2", "arXiv abstract 1502.05477", "arXiv abstract 1506.02438"]
related: ["ppo-clip-tv-single-sample-proxy", "f-divergence-geometry-proximal-regions", "rl-mdp-q-learning-basics", "gppo-lineage-hub"]
---
# TRPO：trust region 與單調改進下界

屬於 [GPPO 的研究脈絡總覽](gppo-lineage-hub.html)。

**TRPO。** Schulman et al.（2015，Trust Region Policy Optimization，arXiv:1502.05477）；GPPO 論文 §2.1 引用它建立的 trust-region 下界作為 Theorem 1。TRPO 的摘要描述它是一個迭代程序，可保證策略改進為單調，並且在理論推導後做近似得到實用演算法。

**Theorem 1（GPPO 論文 §2.1 的敘述）。** 對行為策略 µ 與目標策略 π：

- 改進量 η(π) − η(µ) 大於等於 1/(1−γ) 乘以代理目標 L_µ(π) = E[r(s,a) A^µ(s,a)]，再減去 4ξγ/(1−γ)² 乘以 D_TV^max(π‖µ)²。
- r(s,a) = π(a|s)/µ(a|s) 是重要性比值，A^µ 是 µ 下的優勢函數，ξ = max |A^µ|，D_TV^max 是各狀態 TV 散度的最大值。
- 因此最大化 L_µ(π) 並控制 D_TV^max，即得單調改進保證。

**兩種約束寫法。** 論文式 (2) 以 TV 為約束：max L_µ(π)，約束 D_TV^max ≤ δ。另一種寫法用 KL，其依據是 Pinsker 不等式 D_TV² ≤ ½ D_KL。

**實作成本。** 論文 §2.2 指出 TRPO 的約束式二階最佳化有可觀的計算負擔，限制其在現代 LLM RL 的可擴展性；PPO 以一階 clipped 目標取代它（見 [PPO-Clip 作為 TV 散度的單樣本代理](ppo-clip-tv-single-sample-proxy.html)）。

**優勢估計。** 論文 §2.2 指出優勢 Â 在 PPO 中通常以 GAE 取得（Schulman et al., 2016，High-Dimensional Continuous Control Using Generalized Advantage Estimation，arXiv:1506.02438），在現代 LLM RL 的 GRPO 及其變體中則用 group 相對優勢。

**與 GPPO 的關係。** GPPO 的 Theorem 3 把式中的 D_TV² 以 f-散度取代（見 [f-散度幾何與 proximal region](f-divergence-geometry-proximal-regions.html)），保留同一個單調改進結構。MDP 與優勢函數的記號見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)。
