---
slug: rswp-rlvr-grpo-background
title: RLVR 與 GRPO：本文要對照的 RL 基準
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [RLVR, GRPO, RLHF, DeepSeek-R1]
arxiv: ["2510.14901", "2203.02155", "2402.03300", "2501.12948", "2503.24290"]
sources: ["arXiv:2510.14901 §1, §2", "arXiv abstract 2203.02155", "arXiv abstract 2402.03300", "arXiv abstract 2501.12948", "arXiv abstract 2503.24290"]
related: ["r1-rlhf-ppo-background", "r1-grpo-outcome-rewards", "rlvr-grpo-credit-assignment", "ppo-grpo-dapo-llm-rl-objectives"]
---
# RLVR 與 GRPO：本文要對照的 RL 基準

2510.14901 在 §2 把 LLM 的 RL 分成兩段。第一段是 RLHF：Ouyang 等人（2203.02155，2022）以人類回饋訓練的獎勵模型對齊 LLM 與人類偏好。第二段是 RLVR（可驗證獎勵的 RL）：以自動驗證器給出的生成結尾獎勵，提升數學與程式等困難推理任務的表現。論文引用 DeepSeek-R1（2501.12948）與 Open-Reasoner-Zero（2503.24290，在 base model 上擴大 RL）作為 RLVR 的代表。

GRPO 由 DeepSeekMath（2402.03300）提出，論文稱它是這波進展的核心，也是本文唯一對照的 RL 演算法。實驗中的 GRPO 基準使用 MATH 訓練集後訓練 Qwen2.5-Math-7B、Qwen2.5-7B 與 Phi-3.5-mini-instruct。

背景細節已有筆記，不在此重寫：[LLM 的 RL 前史：RLHF、PPO 與 GAE](r1-rlhf-ppo-background.html)、[GRPO 與規則式結果獎勵](r1-grpo-outcome-rewards.html)、[RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)、[PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)。

本文對 RL 的批評是三點：需要大量超參數搜尋以避免訓練不穩、需要建立多樣且大量的後訓練資料集、不保證能取得真實驗證器（ground truth verifier）。冪次取樣不需要這三項。
