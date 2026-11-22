---
slug: ripo-euclidean-clip-mismatch
title: PPO-Clip 的歐氏假設與探索崩潰
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [PPO-Clip, exploration-collapse, Euclidean-metric, Clip-Higher, entropy]
arxiv: ["2607.10169", "1707.06347", "2503.14476"]
sources: ["arXiv:2607.10169 §1, §2.2, §3.1, §3.2, Proposition 3.1", "arXiv abstract 1707.06347", "arXiv abstract 2503.14476"]
related: ["ripo-lineage-hub", "ppo-grpo-dapo-llm-rl-objectives", "ripo-kl-fisher-ratio-geometry", "ppo-clip-tv-single-sample-proxy"]
---
# PPO-Clip 的歐氏假設與探索崩潰

屬於 [RIPO 的研究脈絡總覽](ripo-lineage-hub.html)。PPO-Clip 與 GRPO 的目標函數見 [PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)，此處只整理本文（arXiv 2607.10169）對其缺陷的論證。

**探索崩潰。** 本文引用 DAPO（Yu et al., 2025，arXiv 2503.14476）的觀察：PPO-Clip 使策略集中在少數高機率動作，罕見但有用的動作難以被提升。

**歐氏假設。** PPO-Clip 以 |r−1| < ε 限制比值，等於採用距離 d_clip = (r−1)²（Eq. 3），只要比值偏離相同，就視為相同的策略變化。本文（Proposition 3.1）主張此假設與 KL 誘導的 Riemannian 幾何不一致，見 [KL、Fisher 與比值的二次式](ripo-kl-fisher-ratio-geometry.html)。

**數值例（ε = 0.2）。**

| π_old | 上限 π_θ | 增量 | 幾何距離 0.5·π_old·(r−1)² |
|---|---|---|---|
| 0.8 | 0.96 | 0.16 | 0.016 |
| 0.01 | 0.012 | 0.002 | 0.0002 |

低機率動作只用掉信任區域預算的很小一部分，高機率動作的更新則相對激進。

**DAPO 的 Clip-Higher。** DAPO 將 ε_high 由 0.2 提高到 0.28。對 π_old = 0.01 的動作，上限由 0.012 變為 0.0128（增量 0.0008）；對 π_old = 0.8 的動作，上限由 0.96 變為 1。本文因此認為 Clip-Higher 沒有處理幾何不一致，並使高機率動作的更新更激進。

PPO-Clip 以 TV 散度為基礎的另一種解讀見 [PPO-Clip 作為 TV 散度的單樣本 Monte Carlo 代理](ppo-clip-tv-single-sample-proxy.html)，與本文的 KL 論證是不同的切入點。
