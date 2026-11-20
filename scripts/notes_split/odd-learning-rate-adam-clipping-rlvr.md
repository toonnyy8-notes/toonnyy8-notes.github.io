---
slug: odd-learning-rate-adam-clipping-rlvr
title: 學習率、Adam、梯度裁剪與蒸餾後的 RLVR
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [learning-rate, Adam, gradient-clipping, RLVR, DAPO]
arxiv: ["2609.35259", "2604.13627", "1412.6980", "1904.09237", "2503.14476", "2402.03300", "2501.12948"]
sources: ["2609.35259 §4.2, §6, 附錄 A.1, D.6", "arXiv abstract 2604.13627", "arXiv listing 1412.6980, 1904.09237, 2503.14476, 2402.03300, 2501.12948 (title only)"]
related: ["odd-lineage-hub", "odd-token-kl-direction-gradients", "odd-on-policy-forgetting-sparsity-claims", "rlvr-grpo-credit-assignment", "ppo-grpo-dapo-llm-rl-objectives", "r1-grpo-outcome-rewards"]
---
# 學習率、Adam、梯度裁剪與蒸餾後的 RLVR

屬於 [蒸餾動態研究的研究脈絡總覽](odd-lineage-hub.html)。

## 學習率
論文（arXiv:2609.35259）指出既有 SFT 與 RL 的比較還會混入學習率差異，影響學習與遺忘的取捨，並引 Rofin et al.（arXiv:2604.13627）。該文摘要指出學習率透過隱式正則化決定微調收斂到的模型，且學習率衰減提高預訓練模型的銳度，加劇 SFT 的遺忘。論文主要發現：遺忘與參數更新稀疏由學習率（1e-5 或 5e-5）主導，rollout policy 幾乎沒有影響。

## 梯度裁剪與 Adam（附錄 A.1）
- 主要實驗裁剪全域梯度範數於 1.0，且每個記錄步驟的裁剪前範數都超過此閾值；反向 KL 的範數更大且更不穩定。
- 移除裁剪後，正向 KL 在較大學習率仍有效，反向 KL 表現明顯惡化，與反向 KL 梯度含無界對數比的分析一致（見 [Token 層級 KL 方向與梯度](odd-token-kl-direction-gradients.html)）。
- 論文以 Adam（Kingma and Ba，arXiv:1412.6980）的正規化說明為何梯度範數大小不直接決定參數更新量，並引 Reddi et al.（arXiv:1904.09237）說明孤立的梯度尖峰會影響後續更新。這是論文的「可能解釋」，並非驗證結果。

## 蒸餾後的 RLVR（附錄 D.6）
蒸餾 Countdown-3 後接 Countdown-4 的 RLVR，使用 DAPO 損失與群組正規化（Yu et al., 2025，arXiv:2503.14476；GRPO 見 Shao et al., 2024，arXiv:2402.03300；DeepSeek-R1，arXiv:2501.12948），不加參考模型 KL 懲罰，答對獎勵為 2、否則 0。論文結果是 on-policy 蒸餾在較難變體上的優勢在 RLVR 後不穩定地持續。RL 目標細節見 [PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)、[GRPO 與規則式結果獎勵](r1-grpo-outcome-rewards.html) 與 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)；遺忘與稀疏的既有說法見 [On-policy 學習的遺忘與更新稀疏說法](odd-on-policy-forgetting-sparsity-claims.html)。
