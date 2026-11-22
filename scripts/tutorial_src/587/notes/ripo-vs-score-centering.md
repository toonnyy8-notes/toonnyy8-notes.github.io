---
slug: ripo-vs-score-centering
title: RIPO 與 Score Centering：兩種處理 off-policy 比值的方式
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [RIPO, score-centering, importance-ratio, training-inference-mismatch, comparison]
arxiv: ["2607.10169", "2609.20807"]
sources: ["arXiv:2607.10169 §2, §3, §4", "arXiv:2609.20807 §4（見 Score Centering 教學頁）"]
related: ["ripo-lineage-hub", "training-inference-mismatch-sources", "importance-sampling-corrections-llm-rl", "ripo-isometric-clip-ric", "ppo-grpo-dapo-llm-rl-objectives"]
---
# 與 Score Centering 的異同

屬於 [RIPO 的研究脈絡總覽](ripo-lineage-hub.html)。Score Centering（arXiv 2609.20807）的完整說明見教學頁 [Score Centering](score-centering.html)。兩篇都涉及 LLM RL 中重要性比值的處理，但處理的問題不同。

| 項目 | RIPO（2607.10169） | Score Centering（2609.20807） |
|---|---|---|
| 關注的問題 | clip 的形狀（幾何）導致探索崩潰 | 訓練引擎與推論引擎機率不一致造成的 drift |
| 對 π_old 的假設 | π_old 是實際採樣的策略 | sampler q 與 trainer p 不同 |
| 做法 | 把固定 ε 換成與 π_old 相關的 ε = √(δ/π_old) | 對每個 score 減去 sampler 下的期望 score（加法修正） |
| 修改的位置 | clip 的上下界 | policy gradient 的 score 項 |

**可以疊加的理由與限制。** 兩者作用在不同位置：RIPO 決定哪些比值被截斷，Score Centering 修正 on-policy 假設不成立時 score 的期望不為 0 所造成的偏移。本文（RIPO）沒有討論訓練與推論不一致，也沒有與 Score Centering 組合的實驗，所以兩者組合的效果沒有資料可引用。

背景見 [訓練與推論不一致的來源](training-inference-mismatch-sources.html)、[LLM RL 中的 importance sampling 修正](importance-sampling-corrections-llm-rl.html)，以及 RIC 的定義 [Riemannian Isometric Clip 與 RIPO 目標](ripo-isometric-clip-ric.html)。PPO、GRPO 與 DAPO 的共同背景見 [PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)。
