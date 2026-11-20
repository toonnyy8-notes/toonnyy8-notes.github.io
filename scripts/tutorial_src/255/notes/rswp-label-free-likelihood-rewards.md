---
slug: rswp-label-free-likelihood-rewards
title: 以模型內部訊號作獎勵：自信心、自我熵與隨機獎勵
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [intrinsic-reward, confidence, self-entropy, RLVR]
arxiv: ["2510.14901", "2505.22660", "2505.19590", "2506.10947"]
sources: ["arXiv:2510.14901 §1, §2, §5.3", "arXiv abstract 2505.22660", "arXiv abstract 2505.19590", "arXiv abstract 2506.10947"]
related: ["rswp-distribution-sharpening-debate", "rswp-power-sampling-algorithm-results", "r1-grpo-outcome-rewards"]
---
# 以模型內部訊號作獎勵：自信心、自我熵與隨機獎勵

2510.14901 §2 指出，繼 RLVR 之後有一批工作改用不依賴外部驗證器的獎勵訊號：

- Zhao 等人（2505.19590，Learning to Reason without External Rewards）：論文歸類為使用 self-entropy 的訊號。
- Prabhudesai 等人（2505.22660，Maximizing Confidence Alone Improves Reasoning）：使用模型信心。論文在 §5.3 採用其信心定義：生成序列上各步 next-token 分布的平均負熵，Conf(x) = (1/(T+1)) Σ_t Σ_x p(x|x<t) log p(x|x<t)。
- Shao 等人（2506.10947）：論文列為使用隨機獎勵的工作。

論文表示本文與這些工作相似之處在於利用 base model 的概似度提升推理，差別在於本文不訓練。§5.3 的 Figure 4 以此信心指標顯示：GRPO 與冪次取樣的回答都落在 base model 的高信心區域，與較高的正確率相關。

這條脈絡說明為何「只用模型自身機率」就可能改善推理：與前述工作一致，改善可能來自把機率集中到模型原本就偏好的區域（見 [分布銳化之爭](rswp-distribution-sharpening-debate.html)）。數字與實驗見 [冪次取樣演算法與結果](rswp-power-sampling-algorithm-results.html)。相對的規則式結果獎勵見 [GRPO 與規則式結果獎勵](r1-grpo-outcome-rewards.html)。
