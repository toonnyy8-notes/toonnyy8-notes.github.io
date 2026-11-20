---
slug: bpo-trust-region-variants-gspo-cispo-dppo
title: GSPO、CISPO、DPPO：GRPO 之後的 trust region 變體
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [GSPO, CISPO, DPPO, trust-region, baselines]
arxiv: ["2507.18071", "2506.13585", "2602.04879", "2609.15987"]
sources: ["arXiv abstract 2602.04879", "2609.15987 §1, §4, App. B", "2609.15987 reference list [7], [8], [13]"]
related: [bpo-lineage-hub, bpo-ppo-grpo-ratio-clipping, bpo-derivation-and-results]
---
# GSPO、CISPO、DPPO：GRPO 之後的 trust region 變體

屬於 [BPO 的研究脈絡總覽](bpo-lineage-hub.html)。BPO 論文把這三者與 GRPO-ClipHigher 作為實驗基線（[PPO、GRPO 與 ratio clipping](bpo-ppo-grpo-ratio-clipping.html)）。

## 脈絡
- GSPO（Zheng et al., 2025，「Group Sequence Policy Optimization」，arXiv:2507.18071）：BPO 論文將其列為基線，clipping 區間為 [1 − 3×10⁻³, 1 + 5×10⁻³]（依該文消融結果設定）。本筆記未讀取 GSPO 摘要，其方法內容不另述。
- CISPO 出自 MiniMax-M1（Chen et al., 2025，「MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention」，arXiv:2506.13585）。BPO 論文引用該文作為 CISPO 的來源；BPO 實驗中的 importance-weight 上限為 3.0。
- DPPO（Qi et al., 2026，「Rethinking the Trust Region in LLM Reinforcement Learning」，arXiv:2602.04879）：摘要指出 PPO 的 ratio clipping 以單一取樣 token 的機率比估計策略差異，低機率 token 被過度懲罰，高機率 token 的大幅變動則約束不足；DPPO 以對策略散度的約束取代 clipping。BPO 實驗中 DPPO 使用 binary total variation，門檻 δ = 0.1。
- DPPO 提出的 binary KL divergence 被 BPO 用來近似完整 KL。

## 與 BPO 的關係
BPO 在 Qwen3-30B-A3B-Base 上，以 AIME24–26 平均準確率 50.5% 對上 GRPO-ClipHigher 39.5%、GSPO 43.5%、CISPO 47.4%、DPPO 46.4%（論文 Table 1）。
