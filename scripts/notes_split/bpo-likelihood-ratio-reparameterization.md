---
slug: bpo-likelihood-ratio-reparameterization
title: 以 policy likelihood ratio 重新參數化獎勵與價值
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [DPO, token-level-DPO, VIMPO, implicit-reward, Q-function]
arxiv: ["2305.18290", "2404.11999", "2404.12358", "2606.20008", "2609.15987"]
sources: ["arXiv title check 2305.18290, 2404.11999, 2404.12358", "arXiv abstract 2606.20008", "2609.15987 §1, §3.1", "2609.15987 reference list [11], [23], [24], [25]"]
related: [bpo-lineage-hub, bpo-derivation-and-results]
---
# 以 policy likelihood ratio 重新參數化獎勵與價值

屬於 [BPO 的研究脈絡總覽](bpo-lineage-hub.html)。

## 脈絡
BPO 論文（§1、§3.1）寫明其推導「建立在以 policy likelihood ratio 重新參數化獎勵與價值的先前工作上」，並引用四篇：
- DPO（Rafailov et al., 2023，「Direct Preference Optimization: Your Language Model is Secretly a Reward Model」，arXiv:2305.18290）。
- Token-level DPO（Zeng et al., 2024，「Token-level Direct Preference Optimization」，arXiv:2404.11999）。
- From r to Q*（Rafailov, Hejna, Park, Finn, 2024，「From r to Q*: Your Language Model is Secretly a Q-Function」，arXiv:2404.12358）。
- VIMPO（Kang et al., 2026，「VIMPO: Value-Implicit Policy Optimization for LLMs」，arXiv:2606.20008）。其摘要：critic-free，從 KL 正則化 RL 的最優條件導出由策略隱含的價值函數；自回歸生成下，價值遞迴可用 policy–reference log-ratio 表示，並以「軌跡結束後無剩餘獎勵」的終端條件固定。

前三篇僅核對標題與作者，內容以 BPO 論文對它們的歸類為準。

## 與 BPO 的關係
BPO 用同一類手法：把 PMD 最優條件寫成 log π − log µ 的式子，再結合 Bellman 方程，得到只含終端獎勵與初始價值的殘差（見 [BPO 的推導與實驗結果](bpo-derivation-and-results.html)）。
