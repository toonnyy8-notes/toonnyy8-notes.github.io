---
slug: bpo-policy-mirror-descent
title: Policy Mirror Descent：BPO 的理論起點
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [PMD, mirror-descent, KL-regularization, policy-optimization]
arxiv: ["2102.00135", "2201.07443", "1901.11275", "2005.09814", "2609.15987"]
sources: ["2609.15987 §1, §2.5, §3", "arXiv title check 2102.00135, 2201.07443, 1901.11275, 2005.09814", "2609.15987 reference list [20], [21], [32], [33]"]
related: [bpo-lineage-hub, bpo-terminal-reward-bellman-telescoping, bpo-derivation-and-results]
---
# Policy Mirror Descent：BPO 的理論起點

屬於 [BPO 的研究脈絡總覽](bpo-lineage-hub.html)。

## 脈絡
BPO 論文（§2.5）引用四項工作作為 PMD 的來源：
- Lan（2021，「Policy Mirror Descent for Reinforcement Learning: Linear Convergence, New Sampling Complexity, and Generalized Problem Classes」，arXiv:2102.00135）。
- Xiao（2022，「On the Convergence Rates of Policy Gradient Methods」，arXiv:2201.07443）。
- Geist, Scherrer, Pietquin（2019，「A Theory of Regularized Markov Decision Processes」，arXiv:1901.11275）。
- Tomar et al.（2020，「Mirror Descent Policy Optimization」，arXiv:2005.09814）。

以上四篇僅核對標題與作者；內容以 BPO 論文的描述為準。

## BPO 論文對 PMD 的描述
對每個狀態 s，PMD 求解 max E_{y∼π}[f(s,y)] − (1/η) D_KL(π ‖ µ)，其中 µ 為行為（rollout）策略、η 為步長。唯一解為 π⁺(y|s) = µ(y|s) exp(ηf(s,y)) / Z(s)。經典 PMD 使用 action value Q；BPO 改用 advantage A^µ，結果相同，因為 Q = A + V 且 V 與動作無關。

## 與 BPO 的關係
直接實作此更新需要每個中間狀態的 A^µ(s,y)，通常要學習的 critic 或昂貴的條件 rollout。BPO 要做的是在不估計這些量的前提下得到同一個解（見 [終端獎勵下的 Bellman 方程與 telescoping](bpo-terminal-reward-bellman-telescoping.html)）。
