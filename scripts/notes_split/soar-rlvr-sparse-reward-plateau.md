---
slug: soar-rlvr-sparse-reward-plateau
title: RLVR 在低成功率資料上的停滯
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [RLVR, sparse-reward, DeepSeek-R1, Kimi-k1.5]
arxiv: ["2601.18778", "2501.12948", "2501.12599", "2402.14740"]
sources: ["arXiv:2601.18778 §1, §3", "arXiv:2501.12948 摘要（經本文 §1 引用）", "arXiv:2501.12599 題名（經本文 §1 引用）", "arXiv API（題名、作者、年份、ID 核對）"]
related: ["r1-grpo-outcome-rewards", "ppo-grpo-dapo-llm-rl-objectives", "rlvr-grpo-credit-assignment", "policy-gradient-baselines-score-function"]
---
# RLVR 在低成功率資料上的停滯

本文（SOAR，arXiv 2601.18778）§1 指出，可驗證獎勵的強化學習（RLVR）使 LLM 的推理能力明顯提升，代表工作為 DeepSeek-R1（Guo et al.，arXiv 2501.12948）與 Kimi k1.5（Kimi Team，arXiv 2501.12599，題名為 Scaling Reinforcement Learning with LLMs）。本文同時指出此範式的限制：模型無法從自己完全解不出的題目學習。題目過難時，獎勵稀疏或不存在，訓練信號趨近於零。

本文 §3 將此情況設為實驗場景：資料集 D 中的題目，模型 128 次取樣皆未成功（0/128），直接以 REINFORCE、GRPO 或 RLOO 在 D_train 上訓練時，正向獎勵過於稀疏而難以改善表現。本文的問題是模型能否自行產生中間難度的題目，讓這些困難題變得可學。

背景可先讀以下筆記：
- GRPO 與規則式結果獎勵：[GRPO 與規則式結果獎勵（DeepSeek-R1-Zero）](r1-grpo-outcome-rewards.html)
- PPO、GRPO、DAPO 的目標函數：[PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)
- RLVR 與信用分配：[RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)
- Policy gradient 與 baseline（RLOO 的 leave-one-out baseline 屬於此類）：[Policy gradient、baseline 與 score function](policy-gradient-baselines-score-function.html)

本文後續的所有設計（teacher 產題、以 student 進步量作獎勵）都是針對這個「0/128 仍無訊號」的情況。
