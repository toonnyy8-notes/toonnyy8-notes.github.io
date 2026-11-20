---
slug: score-centering-lineage-hub
title: Score Centering 的研究脈絡總覽：從 policy gradient 到訓練與推論不一致下的 drift
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [score-centering, training-inference-mismatch, policy-gradient, importance-sampling, hub]
arxiv: ["2609.20807", "1707.06347", "2402.03300", "2501.12948", "2503.14476", "2503.20783", "2503.14286", "2507.18071", "2510.18855", "2602.04879", "2603.19470", "2510.26788", "2505.24298", "2509.19128", "1503.02531", "2506.01347", "2505.18830"]
sources: ["arXiv:2609.20807 §1, §2, §3, §4", "各引用論文的 arXiv 摘要"]
related: ["policy-gradient-baselines-score-function", "ppo-grpo-dapo-llm-rl-objectives", "training-inference-mismatch-sources", "importance-sampling-corrections-llm-rl", "sft-vs-rl-offline-negative-rewards", "score-centering-drift-decomposition", "rlvr-grpo-credit-assignment", "knowledge-distillation-foundations", "on-policy-distillation-imitation"]
---
# Score Centering 的研究脈絡總覽

Score Centering（arXiv 2609.20807，Marek 與 Ryabinin）處理 LLM 強化學習中的訓練與推論不一致（training-inference mismatch，TIM）：rollout 由推論引擎的 sampler q 產生，梯度卻由訓練引擎的 trainer p 計算，兩者的 next-token 機率不完全相同。本文把 policy gradient 更新分解為 drift 與 signal 兩項，並以「從每個 score 減去 sampler 下的期望 score」消除 drift。建議依下列順序閱讀。

1. **Policy gradient 與 baseline**：期望 score 為 0 的恆等式是整個推導的起點。見 [Policy gradient、baseline 與 score function](policy-gradient-baselines-score-function.html)。
2. **LLM 的 RL 目標函數**：PPO、GRPO、DAPO 以及 group centering。見 [PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)；GRPO 與 RLVR 的背景另見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。
3. **TIM 的來源**：精度、kernel 非結合性、非同步更新造成的 staleness。見 [訓練與推論不一致的來源](training-inference-mismatch-sources.html)。
4. **以 importance sampling 修正**：TIS、MIS、TOPR、GSPO、DPPO 等。見 [LLM RL 中的 importance sampling 修正](importance-sampling-corrections-llm-rl.html)。
5. **SFT 與 RL 的差異**：離線時負獎勵不穩定，線上時全正獎勵不穩定。見 [SFT 與 RL：離線負獎勵與線上 drift](sft-vs-rl-offline-negative-rewards.html)；蒸餾背景另見 [知識蒸餾基礎](knowledge-distillation-foundations.html) 與 [On-policy distillation 與模仿學習](on-policy-distillation-imitation.html)。
6. **本文方法**：drift 與 signal 分解、score centering、top-k 近似。見 [Score centering：drift 與 signal 的分解](score-centering-drift-decomposition.html)。
