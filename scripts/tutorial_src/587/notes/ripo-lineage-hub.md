---
slug: ripo-lineage-hub
title: RIPO 的研究脈絡總覽：從 TRPO 到等距 clipping
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [RIPO, PPO-Clip, TRPO, exploration-collapse, hub]
arxiv: ["2607.10169", "1502.05477", "1506.02438", "1707.06347", "2402.03300", "2501.12948", "2503.14476", "2507.18071", "2507.20673", "2509.02333", "2506.13585", "2508.07629", "2511.20347", "2505.22617", "2409.19256"]
sources: ["arXiv:2607.10169 §1–§5", "各引用論文的 arXiv 摘要"]
related: ["policy-gradient-baselines-score-function", "trpo-monotonic-improvement-tv-bound", "ppo-grpo-dapo-llm-rl-objectives", "ppo-clip-tv-single-sample-proxy", "llm-rl-clipping-variants-heuristics", "sequence-level-ratio-gspo-gmpo", "ripo-kl-fisher-ratio-geometry", "ripo-euclidean-clip-mismatch", "ripo-isometric-clip-ric", "ripo-bias-variance-homoscedasticity", "ripo-experiments-results", "ripo-vs-score-centering", "importance-sampling-corrections-llm-rl"]
---
# RIPO 的研究脈絡總覽

RIPO（Riemannian Isometric Policy Optimization，arXiv 2607.10169，Cai et al.，ICML 2026）主張 PPO-Clip 以歐氏距離衡量策略差異，與 KL 散度誘導的 Riemannian 幾何不一致，使低機率動作的更新過度保守、高機率動作過度激進，造成探索崩潰。解法是讓 clip 範圍隨 π_old 變化：ε = √(δ/π_old)。

建議閱讀順序：

1. **Policy gradient 與優勢估計**：見 [Policy gradient、baseline 與 score function](policy-gradient-baselines-score-function.html)。本文使用 GAE（Schulman et al., 2015，arXiv 1506.02438）估計 PPO 的優勢，GRPO 則以群組標準化。
2. **TRPO 與 PPO**：KL 約束下的單調改進，以及 PPO 以 clip 近似它。見 [TRPO：trust region 與單調改進下界](trpo-monotonic-improvement-tv-bound.html)、[PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)；另一個以 TV 散度解讀 clip 的觀點見 [PPO-Clip 作為 TV 散度的單樣本 Monte Carlo 代理](ppo-clip-tv-single-sample-proxy.html)。
3. **既有 clipping 修正**：DAPO、DCPO、Clip-Cov、SAPO、CISPO 見 [LLM RL 中的 clipping 變體](llm-rl-clipping-variants-heuristics.html)，GSPO、GMPO 見 [序列層級的比值](sequence-level-ratio-gspo-gmpo.html)。本文稱這些方法皆為經驗性修正。
4. **本文的理論**：[KL 與 Fisher 矩陣](ripo-kl-fisher-ratio-geometry.html) → [歐氏假設與探索崩潰](ripo-euclidean-clip-mismatch.html) → [Riemannian Isometric Clip](ripo-isometric-clip-ric.html) → [偏差—變異論證](ripo-bias-variance-homoscedasticity.html)。
5. **實驗**：[RIPO 的實驗設定與結果](ripo-experiments-results.html)。
6. **與同批教學的比較**：[RIPO 與 Score Centering](ripo-vs-score-centering.html)；importance sampling 的其他修正見 [LLM RL 中的 importance sampling 修正](importance-sampling-corrections-llm-rl.html)。
