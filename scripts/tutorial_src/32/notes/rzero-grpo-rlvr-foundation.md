---
slug: rzero-grpo-rlvr-foundation
title: R-Zero 使用的訓練基礎：GRPO 與 RLVR
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [GRPO, RLVR, DeepSeekMath, DeepSeek-R1, Qwen3]
arxiv: ["2508.05004", "2402.03300", "2501.12948", "2110.14168", "2505.09388"]
sources: ["arXiv 2508.05004 §1、§2.2、§2.4、§3.1", "arXiv abstract 2402.03300", "arXiv abstract 2501.12948", "arXiv abstract 2110.14168", "arXiv abstract 2505.09388"]
related: ["r1-grpo-outcome-rewards", "rlvr-grpo-credit-assignment", "ppo-grpo-dapo-llm-rl-objectives", "policy-gradient-baselines-score-function", "rzero-challenger-solver-pipeline", "rzero-majority-vote-pseudo-labels"]
---
# R-Zero 使用的訓練基礎：GRPO 與 RLVR

RLVR（reinforcement learning with verifiable rewards）以可自動檢查的結果當獎勵，例如最終答案是否等於標準答案。R-Zero 引言把 DeepSeekMath（Shao et al., 2024，arXiv 2402.03300）與 DeepSeek-R1（arXiv 2501.12948）列為這條路線的代表，並指出這類方法需要大量人工整理的題目與標籤。

GRPO 在 DeepSeekMath 提出，同一題取樣一組回答，以組內獎勵的相對差異計算優勢，不需要另外訓練 value 網路。R-Zero 的兩個角色都用 GRPO 更新（§2.2、§2.4）。GRPO 的目標函數、與 PPO 的差異見 [GRPO 與規則式結果獎勵（DeepSeek-R1-Zero）](r1-grpo-outcome-rewards.html)、[PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)；組內平均作為 baseline 的背景見 [Policy gradient、baseline 與 score function](policy-gradient-baselines-score-function.html)；RLVR 的限制見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。

R-Zero 與標準 RLVR 的差別只在獎勵與題目的來源：

- Solver 的獎勵是二元值，回答與偽標籤一致為 1，否則為 0（§2.4）。偽標籤來自 Solver 自己的多數決，見 [多數決偽標籤](rzero-majority-vote-pseudo-labels.html)。
- Challenger 的獎勵由 Solver 的不確定性、重複懲罰與格式檢查組成（§2.2），見 [R-Zero 的 Challenger–Solver 流程](rzero-challenger-solver-pipeline.html)。

實驗使用 Qwen3-4B/8B-Base（arXiv 2505.09388）與 OctoThinker-3B/8B，訓練程式基於 EasyR1（§3.1）。數學資料集 GSM8K（Cobbe et al., 2021，arXiv 2110.14168）是論文評測使用的基準之一，題目與答案皆可客觀比對，這是 R-Zero 選擇數學領域的原因（§2.2）。
