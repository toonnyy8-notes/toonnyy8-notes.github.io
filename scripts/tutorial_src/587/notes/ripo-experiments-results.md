---
slug: ripo-experiments-results
title: RIPO 的實驗設定與結果
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [RIPO, AIME, GRPO, DAPO, DCPO, GSPO, GMPO, GPPO, Clip-Cov, ablation]
arxiv: ["2607.10169", "2402.03300", "2503.14476", "2507.18071", "2507.20673", "2509.02333", "2508.07629", "2505.22617", "2409.19256"]
sources: ["arXiv:2607.10169 §5, Table 1–6", "arXiv abstract 2508.07629", "arXiv abstract 2409.19256", "arXiv abstract 2505.22617"]
related: ["ripo-lineage-hub", "llm-rl-clipping-variants-heuristics", "sequence-level-ratio-gspo-gmpo"]
---
# RIPO 的實驗設定與結果

屬於 [RIPO 的研究脈絡總覽](ripo-lineage-hub.html)。以下數字取自本文（arXiv 2607.10169）。

**設定（§5.1）。** 四個模型：Qwen3-1.7B/4B/8B-Base 與 Llama3.2-3B-Instruct。訓練集 DAPO-Math-17k（17,917 題），每題 8 條 rollout，最長 16,384 token，AdamW、學習率 1e-6、300 步，框架為 VeRL（Sheng et al.，HybridFlow，arXiv 2409.19256），8×A100。獎勵為可驗證的 0/1。對照方法：GRPO、DAPO、GSPO、GMPO、DCPO。GRPO、DAPO、DCPO、RIPO 使用 dual clipping（下界 0.5、上界 10），所有方法移除 KL 項，δ 預設 0.05。評測為七個競賽題庫，報告 Avg@8。

**Table 1（平均 Avg@8，括號為相對 GRPO）。**

| 模型 | GRPO | DAPO | GSPO | GMPO | DCPO | RIPO |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | 11.2 | 12.4 | 13.3 | 13.9 | 14.8 | 15.4 (+37.2%) |
| Llama3.2-3B-Instruct | 6.4 | 7.4 | 7.2 | 7.4 | 6.9 | 8.6 (+34.4%) |
| Qwen3-4B-Base | 25.7 | 27.1 | 26.8 | 28.4 | 27.8 | 30.1 (+17.1%) |
| Qwen3-8B-Base | 28.5 | 30.6 | 32.1 | 35.3 | 34.5 | 38.5 (+35.1%) |

**訓練動態（§5.3，Qwen3-8B-Base）。** 本文報告 GRPO 熵快速降到接近 0，DAPO 熵持續上升，RIPO 熵先降後穩定；RIPO 的梯度範數幾乎無波動；clipped token 比例介於 DCPO/GMPO（很少 clip）與 GSPO/DAPO（clip 數量高出數個數量級）之間。

**消融（§5.4，AIME24）。** δ 在 0.02 至 0.08 的對稱設定下 Avg@8 為 40.8、43.8（δ=0.05）、42.1；δ_high 明顯大於 δ_low 時降到 27.5 至 28.8，並伴隨熵爆炸。

**其他比較。** Table 3（Qwen3-8B-Base，AIME24）：GRPO 31.7、GPPO 31.7、Clip-Cov 36.3、RIPO 43.8。GPPO 即 Klear-Reasoner 提出的 gradient-preserving clipping（Su et al.，arXiv 2508.07629），Clip-Cov 出自 Cui et al.（arXiv 2505.22617），兩者背景見 [LLM RL 中的 clipping 變體](llm-rl-clipping-variants-heuristics.html)。Table 4 在 GSM8K 上把 RIC 套到 PPO，0.5B/1.5B/7B/14B 的 RIPO-Clip 皆高於 PPO-Clip（+3.1、+2.4、+1.8、+1.2）。Table 5 的 Pass@128 在 AIME-25 為 RIPO 60.0、GRPO 53.3。Table 6 顯示程式（平均 44.9 對 GRPO 39.7）與搜尋（43.4 對 37.7）任務也較高。

GSPO 與 GMPO 的取樣單位見 [序列層級的比值：GSPO 與 GMPO](sequence-level-ratio-gspo-gmpo.html)。
