---
slug: ripo-isometric-clip-ric
title: Riemannian Isometric Clip 與 RIPO 目標
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [RIPO, RIC, dynamic-clipping, GRPO, isometric]
arxiv: ["2607.10169", "2402.03300", "2503.14476"]
sources: ["arXiv:2607.10169 §4.1, §4.3, Eq. 9–11, 15", "arXiv abstract 2402.03300", "arXiv abstract 2503.14476"]
related: ["ripo-lineage-hub", "ppo-grpo-dapo-llm-rl-objectives", "ripo-experiments-results"]
---
# Riemannian Isometric Clip 與 RIPO 目標

屬於 [RIPO 的研究脈絡總覽](ripo-lineage-hub.html)。

**要求。** 本文要求每個狀態-動作樣本的更新消耗相同的幾何距離，而不是相同的比值偏離：π_old(a|s)·(r−1)² ≤ δ（Eq. 9）。

**推導。** 解出 r 得 |r−1| ≤ √(2δ/π_old)（Eq. 10）。實作時把係數 2 併入超參數 δ，得到

ε(a|s) = √(δ / π_old(a|s))，1 − ε ≤ r ≤ 1 + ε　（Eq. 11）

這就是 Riemannian Isometric Clip（RIC）。π_old 越小，clip 範圍越寬。

**數值例（δ = 0.02，本文 §4.1）。**

| π_old | RIC 上限 | PPO-Clip 上限 | 消耗的幾何距離 |
|---|---|---|---|
| 0.8 | 0.92 | 0.96 | 0.01 |
| 0.01 | 0.024 | 0.012 | 0.01 |

兩個動作消耗的距離相同（0.01），PPO-Clip 則分別為 0.016 與 0.0002。

**RIPO 目標（Eq. 15）。** 將 GRPO 目標（群組標準化優勢、token 層級比值）中的固定 clip 範圍 (1−ε, 1+ε) 換成 (1−ε_{i,t}, 1+ε_{i,t})，ε_{i,t} 依 Eq. 11 由 π_old(o_{i,t}|q, o_{i,<t}) 計算。本文採用 DAPO 的 token 層級 policy gradient loss，使長短軌跡的梯度影響平衡。GRPO 與 DAPO 的定義見 [PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)。

**預設值。** 實驗預設 δ = 0.05（§5.1）；實驗另對 PPO 目標套用 RIC（見 [RIPO 的實驗設定與結果](ripo-experiments-results.html)）。
