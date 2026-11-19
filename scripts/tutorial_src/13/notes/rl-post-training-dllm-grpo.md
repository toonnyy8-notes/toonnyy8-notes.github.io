---
slug: rl-post-training-dllm-grpo
title: dLLM 的強化學習後訓練與 GRPO
cluster: Diffusion Language Models
kind: topic
tags: [grpo, ppo, d1, reinforcement-learning, post-training]
arxiv: ["1707.06347", "2402.03300", "2504.12216", "2512.09106"]
sources: ["paper.txt §3.3, §5", "arXiv API metadata"]
related: [unmasking-policies-dllm-overview, learned-unmasking-concurrent-work, unmasking-mdp-policy-design]
---
# dLLM 的強化學習後訓練與 GRPO

屬於[總覽](unmasking-policies-dllm-overview.html)。

## 演算法來源

PPO（Schulman 等人，2017，「Proximal Policy Optimization Algorithms」，1707.06347）是 policy gradient 方法。GRPO 由 Shao 等人在「DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models」（2402.03300）提出。論文稱 GRPO 是比早期 policy gradient 更簡單、更可擴展的替代方案。

## dLLM 上的 RL

論文第 5 節說明，dLLM 的 RL 後訓練早期工作是 d1（Zhao 等人，2025a，「d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning」，2504.12216），提出適用於 dLLM 的 GRPO 變體。其後的工作多在改進 policy gradient 估計。這些工作的共同點：取樣策略固定（例如 high-confidence），更新的是 dLLM 本身（或其 LoRA 版本），目的是提升推理能力。

## 本論文的差異

論文不更新 dLLM，把 dLLM 當環境，只訓練獨立的輕量 policy，目標是自動發現取樣策略。訓練細節（依 §3.3）：

- 每個 prompt 取樣 G 條軌跡，advantage 為該軌跡 reward 減去組內平均。
- 訓練時 dLLM 溫度設為 0，使軌跡間的差異只來自 unmasking 動作。
- 結束步 T̂ 的 advantage 傳給所有先前時間步。
