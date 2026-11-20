---
slug: sequence-level-ratio-gspo-gmpo
title: 序列層級的比值：GSPO 與 GMPO（保留 TV-MC、改變取樣單位）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [GSPO, GMPO, sequence-level, importance-ratio, GRPO]
arxiv: ["2610.13015", "2507.18071", "2507.20673"]
sources: ["arXiv abstract 2507.18071", "arXiv abstract 2507.20673", "arXiv:2610.13015 §3.4, §4.1"]
related: ["llm-rl-clipping-variants-heuristics", "gppo-design-space-unification", "ppo-clip-tv-single-sample-proxy", "gppo-lineage-hub"]
---
# 序列層級的比值：GSPO 與 GMPO

屬於 [GPPO 的研究脈絡總覽](gppo-lineage-hub.html)。

- **GSPO**（Zheng et al., 2025，Group Sequence Policy Optimization，arXiv:2507.18071）：摘要稱它不採用 token 級重要性比值，而以序列似然定義比值，並進行序列級 clipping、獎勵與最佳化；摘要並稱它穩定了 MoE 的 RL 訓練，用於 Qwen3 模型。
- **GMPO**（Zhao et al., 2026，Geometric-Mean Policy Optimization，arXiv:2507.20673）：摘要指出 GRPO 優化 token 級獎勵的算術平均，遇到離群的重要性加權獎勵時更新不穩；GMPO 改為最大化幾何平均，對離群值較不敏感。

**GPPO 論文的定位（§3.4）。** 兩者與 PPO 一樣保留 TV 散度與單樣本 MC 估計，只把取樣單位從個別 token 改為完整 rollout，所以在 (f, T) 格子中與 PPO 同一格（見 [GPPO 的設計空間](gppo-design-space-unification.html)）。

**實驗中的數字（論文 Table 1）。** 數學平均 GSPO 與 GMPO 皆為 35.1（相對 GRPO 的 27.7 為 +26.71%）；程式平均 GSPO 53.7、GMPO 54.4（GRPO 為 52.4）。

不改變取樣單位、只改 clipping 形狀的方法見 [LLM RL 中的 clipping 變體](llm-rl-clipping-variants-heuristics.html)。
