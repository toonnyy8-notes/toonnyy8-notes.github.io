---
slug: bpo-lineage-hub
title: Bellman Policy Optimization 的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [BPO, RLVR, GRPO, policy-mirror-descent, critic-free]
arxiv: ["2609.15987", "1707.06347", "2402.03300", "2005.09814", "2102.00135", "2305.18290", "2606.20008"]
sources: ["2609.15987 §1, §2", "arXiv abstract 1707.06347", "arXiv abstract 2402.03300", "arXiv abstract 2606.20008"]
related: [bpo-ppo-grpo-ratio-clipping, rlvr-grpo-credit-assignment, bpo-trust-region-variants-gspo-cispo-dppo, bpo-policy-mirror-descent, bpo-terminal-reward-bellman-telescoping, rl-mdp-q-learning-basics, bpo-likelihood-ratio-reparameterization, bpo-derivation-and-results]
---
# Bellman Policy Optimization 的研究脈絡總覽

Bellman Policy Optimization（BPO，arXiv:2609.15987）是用於 RLVR 的 critic-free 方法，由 Policy Mirror Descent（PMD）推導而來。本篇依閱讀順序列出理解 BPO 所需的脈絡。

## 閱讀順序
1. [PPO、GRPO 與 ratio clipping](bpo-ppo-grpo-ratio-clipping.html)：BPO 的比較對象與程式形式的來源。信用分配的背景見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。
2. [GSPO、CISPO、DPPO：GRPO 之後的 trust region 變體](bpo-trust-region-variants-gspo-cispo-dppo.html)：BPO 實驗中的四個基線。
3. [Policy Mirror Descent](bpo-policy-mirror-descent.html)：BPO 的理論起點。
4. [終端獎勵下的 Bellman 方程與 telescoping](bpo-terminal-reward-bellman-telescoping.html)：BPO 消去中間狀態價值的依據。MDP 與 value function 的基本記號見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)。
5. [以 policy likelihood ratio 重新參數化獎勵與價值](bpo-likelihood-ratio-reparameterization.html)：DPO 系列與 VIMPO。
6. [BPO 的推導與實驗結果](bpo-derivation-and-results.html)：本論文本身。

## 全貌
RLVR 以驗證器給出的結果獎勵訓練 LLM。主流做法 GRPO 不訓練價值模型，但所有 token 共用同一個 advantage。PMD 需要中間狀態的 action value，通常要另訓價值模型。BPO 利用 Bellman 方程把 token advantage 寫成相鄰狀態價值之差，沿回應求和後只剩終端獎勵與初始價值，因此不需估計中間狀態價值。
