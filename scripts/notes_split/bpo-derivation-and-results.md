---
slug: bpo-derivation-and-results
title: BPO 的推導與實驗結果
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [BPO, critic-free, binary-KL, mismatch-correction, AIME]
arxiv: ["2609.15987", "2602.04879"]
sources: ["2609.15987 §3, §4, App. A-C"]
related: [bpo-lineage-hub, bpo-policy-mirror-descent, bpo-terminal-reward-bellman-telescoping, bpo-ppo-grpo-ratio-clipping, bpo-trust-region-variants-gspo-cispo-dppo]
---
# BPO 的推導與實驗結果

屬於 [BPO 的研究脈絡總覽](bpo-lineage-hub.html)。以下數字取自 Bellman Policy Optimization（arXiv:2609.15987）。

## 推導
1. 起點：以 advantage 的 PMD，[Policy Mirror Descent](bpo-policy-mirror-descent.html) 的解 π⁺ = µ exp(ηA^µ)/Z。
2. 消去 Z：對 µ 取期望得 log Z = D_KL(µ ‖ π⁺)；由此 log π⁺ − log µ − ηA^µ + D_KL(µ ‖ π⁺) = 0。
3. 沿軌跡求和並套用 [telescoping](bpo-terminal-reward-bellman-telescoping.html)，得軌跡殘差 δ = η(R − V^µ(x) − Σ_t[log(π/µ) + D_KL(µ ‖ π)])，目標 L(π) = E[ϕ(x)·δ²/(2η)]。
4. Theorem 1：此目標與 PMD 在 µ 可達狀態上有相同唯一最優解。必要性：對任何策略，每個狀態上 token 殘差在 µ 下的條件期望為 0，所以各步殘差兩兩不相關，E[δ²] 等於各步殘差平方期望之和；最佳解使 E[δ²] = 0，故每個可達 token 的殘差為 0（論文 App. A.1 以 martingale 表述同一論證）。
5. 實用化：線性化殘差；V^µ(x) 以組內平均估計，ϕ(x) 取 1/std，得 GRPO 的 advantage；以 binary KL（見 [DPPO](bpo-trust-region-variants-gspo-cispo-dppo.html)）近似完整 KL，得梯度因子 (1 − µ)/(1 − π)。

## 損失
BPO 損失保留 [GRPO 形式](bpo-ppo-grpo-ratio-clipping.html)，ratio 換成 ω = (1 + ε − µ)/(1 + ε − π)，上限 C。預設 ε = 0.1、C = 3.0。

## 結果
Qwen3-30B-A3B-Base，DAPO-Math-17k，400 步：AIME24/25/26 平均 50.5%（GRPO-ClipHigher 39.5、GSPO 43.5、CISPO 47.4、DPPO 46.4）。Qwen3-4B-Base 消融：ε ∈ {0.05, 0.1, 0.2, 0.3} 平均 25.8、25.4、25.5、24.1，C ∈ {2, 3, 4} 平均 25.3、25.4、25.8，GRPO-ClipHigher 為 20.5。
