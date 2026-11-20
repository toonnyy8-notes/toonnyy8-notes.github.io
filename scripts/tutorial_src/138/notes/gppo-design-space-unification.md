---
slug: gppo-design-space-unification
title: GPPO 的設計空間：把 PPO、DAPO、DPPO、RIPO 對應到 (f, T)
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [GPPO, design-space, unification, DAPO, DPPO, RIPO, benchmark]
arxiv: ["2610.13015", "1707.06347", "2503.14476", "2602.04879", "2607.10169", "2507.18071", "2507.20673"]
sources: ["arXiv:2610.13015 §3.4, §4, Table 1"]
related: ["f-divergence-geometry-proximal-regions", "divergence-estimator-topologies", "llm-rl-clipping-variants-heuristics", "sequence-level-ratio-gspo-gmpo", "ppo-clip-tv-single-sample-proxy", "gppo-lineage-hub"]
---
# GPPO 的設計空間

屬於 [GPPO 的研究脈絡總覽](gppo-lineage-hub.html)。GPPO（arXiv:2610.13015）的設計空間是 4 個散度（TV、Pearson χ²、Hellinger、KL）乘 3 個估計器（MC、Atomic、Binary）。

**既有方法的對應（§3.4、§4.1）。**

| 方法 | f | T | 論文的推導 |
|---|---|---|---|
| PPO / GRPO | TV | MC | \|r−1\| 是 TV 的單樣本估計 |
| GSPO、GMPO | TV | MC | 取樣單位改為完整 rollout（見 [序列層級比值](sequence-level-ratio-gspo-gmpo.html)） |
| DAPO | KL | MC | D̂ = −log r，−δ ≤ log r ≤ δ 得 e^{−δ} ≤ r ≤ e^{δ}；δ = 0.23 時 0.8 ≤ r ≤ 1.26，對應 DAPO 的不對稱 clip |
| DPPO | TV | Binary | 約束 \|π − µ\| ≤ δ，等價於 r 於 1 ± δ/µ(a\|s) |
| RIPO | χ² | Atomic | µ(r−1)² ≤ δ，等價於 r 於 1 ± √(δ/µ(a\|s)) |

論文的說法是：f 指定期望的阻力形狀，T 決定如何由取樣資料估計並執行約束。幾何與估計器的細節見 [f-散度幾何](f-divergence-geometry-proximal-regions.html)與 [散度估計器拓撲](divergence-estimator-topologies.html)。

**實驗設定（§4.1）。** Qwen3-8B-Base；數學訓練集 DAPO-Math-17k，程式訓練集 Eurus-Code-25k；group 大小 8、最大回應長度 20k；評估以 Avg@8，涵蓋 4 個數學與 4 個程式基準；所有方法採 GRPO 式 group 相對優勢、雙重 clipping（0.5 與 10）並移除 KL 懲罰。

**結果（Table 1 平均）。**

| 設定 | 數學平均 | 程式平均 |
|---|---|---|
| TV-MC（GRPO） | 27.7 | 52.4 |
| TV-Binary（DPPO） | 39.0 | 53.8 |
| χ²-Atomic（RIPO） | 40.5 | 57.2 |
| KL-MC（DAPO） | 34.6 | 54.3 |
| KL-Atomic | 41.5（+49.82%） | 59.7（+13.93%） |

論文結論：非零曲率的散度整體優於線性對稱的 TV，Atomic 估計在所有被評估的散度上一致優於單樣本 MC，KL 加 Atomic 最佳。對照基線 SAPO、CISPO、Clip-Cov 見 [LLM RL 中的 clipping 變體](llm-rl-clipping-variants-heuristics.html)。
