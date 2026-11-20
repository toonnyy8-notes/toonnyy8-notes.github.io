---
slug: gppo-lineage-hub
title: GPPO 的研究脈絡總覽：從 TRPO 到 Divergence × Estimator 設計空間
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [GPPO, PPO, TRPO, f-divergence, trust-region, LLM-RL]
arxiv: ["2610.13015", "1502.05477", "1707.06347", "2402.03300", "2503.14476", "2602.04879", "2607.10169"]
sources: ["arXiv:2610.13015 §1, §2, §3.4", "arXiv abstract 1502.05477", "arXiv abstract 1707.06347"]
related: ["trpo-monotonic-improvement-tv-bound", "ppo-clip-tv-single-sample-proxy", "llm-rl-clipping-variants-heuristics", "sequence-level-ratio-gspo-gmpo", "f-divergence-geometry-proximal-regions", "divergence-estimator-topologies", "gppo-design-space-unification", "ppo-grpo-dapo-llm-rl-objectives", "rlvr-grpo-credit-assignment", "rl-mdp-q-learning-basics"]
---
# GPPO 的研究脈絡總覽

Generalized Proximal Policy Optimization（GPPO，arXiv:2610.13015）把 LLM RL 中的 PPO 及其變體放進同一個框架。該論文的立場是：PPO 的 clipping 隱含地限制「全變差（TV）散度的單樣本 Monte Carlo 估計」，後續變體各自修改這個限制，缺少統一說明。GPPO 以「散度 f × 估計器 T」兩個維度描述所有變體。

建議閱讀順序如下。

1. [TRPO：trust region 與單調改進下界](trpo-monotonic-improvement-tv-bound.html)。GPPO 的理論起點，Theorem 1 來自此處。
2. [PPO-Clip 作為 TV 散度的單樣本代理](ppo-clip-tv-single-sample-proxy.html)。說明 PPO 為何對應 TV 與 Monte Carlo 估計，以及論文指出的兩項限制。
3. [LLM RL 中的 clipping 變體](llm-rl-clipping-variants-heuristics.html)。DAPO、DCPO、Clip-Cov、SAPO、CISPO 等針對實務問題的修改。
4. [序列層級的比值：GSPO 與 GMPO](sequence-level-ratio-gspo-gmpo.html)。改變取樣單位而保留 TV-MC 的方法。
5. [f-散度幾何與 proximal region](f-divergence-geometry-proximal-regions.html)。Vajda 不等式、Theorem 3，以及 TV、χ²、KL、Hellinger 的曲率差異。
6. [散度估計器拓撲：MC、Atomic、Binary](divergence-estimator-topologies.html)。DPPO 的 Binary 與 RIPO 的 Atomic 由此定位。
7. [GPPO 的設計空間與統一觀點](gppo-design-space-unification.html)。把 PPO、DAPO、DPPO、RIPO 對應到 (f, T) 格子並整理實驗結果。

背景筆記：MDP 與 Bellman 記號見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)；GRPO 與 RLVR 見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)；PPO、GRPO、DAPO 的目標函數比較見 [PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)。
