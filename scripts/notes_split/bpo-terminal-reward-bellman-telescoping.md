---
slug: bpo-terminal-reward-bellman-telescoping
title: 終端獎勵下的 Bellman 方程與 telescoping
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [Bellman-equation, advantage, terminal-reward, VinePPO, value-model]
arxiv: ["2410.01679", "2609.15987"]
sources: ["2609.15987 §2.2, §2.4, §3.1", "arXiv abstract 2410.01679", "2609.15987 reference list [22], [31]"]
related: [bpo-lineage-hub, rl-mdp-q-learning-basics, bpo-policy-mirror-descent, bpo-derivation-and-results]
---
# 終端獎勵下的 Bellman 方程與 telescoping

屬於 [BPO 的研究脈絡總覽](bpo-lineage-hub.html)。MDP、value function 與 Bellman backup 的通用記號見 [MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)；本篇只寫 BPO 所用的終端獎勵特例。

## 設定（BPO 論文 §2.2、§2.4）
- 狀態 s_t = (x, y<t)，動作為第 t 個 token，轉移為確定性（附加 token）。
- 驗證器只在回應結束給終端獎勵 R(x, y)，中間獎勵為零。
- V^π(s_t) 為從 s_t 起依 π 完成回應的期望獎勵；Q^π(s_t, y_t) = V^π(s_{t+1})；A^π = Q^π − V^π；終端條件 V^π(s_{|y|+1}) = R(x, y)。

## Telescoping
A^µ(s_t, y_t) = V^µ(s_{t+1}) − V^µ(s_t)，沿整個回應求和得 Σ_t A^µ(s_t, y_t) = R(x, y) − V^µ(x)。只剩終端獎勵與初始價值；V^µ(x) 是 prompt 的期望獎勵，可由同一 prompt 的多個取樣回應估計。

## 為何避免中間價值
BPO 論文指出獨立價值模型增加記憶體與計算成本，且學得的價值在推理任務上可能不準，引用 VinePPO（Kazemnejad et al., 2024，arXiv:2410.01679）。該文摘要表示價值網路在推理密集任務中常給出不佳的期望回報估計，在比較替代步驟時幾乎不優於隨機基線。

## 與 BPO 的關係
此恆等式是 BPO 把 PMD 改寫成軌跡層級目標的第二步，見 [BPO 的推導與實驗結果](bpo-derivation-and-results.html)；PMD 本身見 [Policy Mirror Descent](bpo-policy-mirror-descent.html)。
