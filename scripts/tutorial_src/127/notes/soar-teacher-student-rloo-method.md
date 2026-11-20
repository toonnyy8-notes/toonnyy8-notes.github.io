---
slug: soar-teacher-student-rloo-method
title: SOAR：以 RLOO 訓練的 teacher–student 迴圈
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [SOAR, RLOO, meta-RL, teacher-student]
arxiv: ["2601.18778", "2402.14740"]
sources: ["arXiv:2601.18778 §3.1–§3.3", "arXiv:2402.14740 題名與作者（API 核對）；RLOO 的用法依本文 §3.2"]
related: ["soar-bilevel-meta-learning-dataset-distillation", "soar-rlvr-sparse-reward-plateau", "policy-gradient-baselines-score-function"]
---
# SOAR：以 RLOO 訓練的 teacher–student 迴圈

本文 §3 的方法（SOAR，Self-Optimization via Asymmetric RL）：兩份由同一基底模型初始化的副本，teacher π_φ^T 與 student π_θ^S。

**外層（teacher）。** 以 RLOO（Ahmadian et al.，arXiv 2402.14740，2024，Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs）訓練。每次迭代取樣 g·n 條 rollout，分成 g 個各含 n 題的資料集 X_k；teacher 同時產生問題與答案（不自動驗證）。資料集獎勵為 R(X_k) = Acc(π_θk^S′(Q_R)) − Acc(π_θ^S(Q_R))，即 student 在 D_train 子集 Q_R 上訓練後的貪婪成功率，減去訓練前的成功率。為降低變異，對每個資料集平均 r 次平行 student 訓練，此平均獎勵指派給 X_k 內每條 rollout。

**內層（student）。** 以 RLOO 在 X_k 上訓練 10 步（batch size 8），結束後 student 回復到基準策略。

**晉升機制。** 追蹤 teacher 獎勵的滾動平均；超過門檻 τ 時，將基準 student 更新為以最佳 X_k 訓練後的 student，後續獎勵相對於新基準計算。

背景：RLOO 屬於 policy gradient 加 baseline 的一族，見 [Policy gradient、baseline 與 score function](policy-gradient-baselines-score-function.html)。此設計要解決的停滯問題見 [RLVR 在低成功率資料上的停滯](soar-rlvr-sparse-reward-plateau.html)，雙層最佳化的先例與 BPTT 問題見 [雙層最佳化](soar-bilevel-meta-learning-dataset-distillation.html)。
