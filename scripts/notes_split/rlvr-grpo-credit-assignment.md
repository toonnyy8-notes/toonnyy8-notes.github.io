---
slug: rlvr-grpo-credit-assignment
title: RLVR、GRPO 與信用分配瓶頸
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [RLVR, GRPO, PPO, PRM, credit-assignment]
arxiv: ["2601.20802", "1707.06347", "2402.03300", "2501.12948", "2305.20050", "2502.01456"]
sources: ["2601.20802 §6.1", "arXiv abstract 1707.06347", "arXiv abstract 2402.03300", "arXiv abstract 2501.12948", "arXiv abstract 2305.20050", "arXiv abstract 2502.01456"]
related: [sdpo-lineage-overview]
---
# RLVR、GRPO 與信用分配瓶頸

屬於 [SDPO 的研究脈絡總覽](sdpo-lineage-overview.html)。

## 脈絡
- PPO（Schulman et al., 2017, arXiv:1707.06347）是 SDPO 論文所指「依賴另外學習的價值網路」的傳統做法；SDPO 論文指出這帶來額外記憶體成本，且仍受純量獎勵的資訊瓶頸限制。
- PPO 的全名為「Proximal Policy Optimization Algorithms」，是 policy gradient 方法；Learning Unmasking Policies…（arXiv 2512.09106）稱 GRPO 是比早期 policy gradient 更簡單、更可擴展的替代方案。
- GRPO 由 DeepSeekMath（Shao et al., 2024，「DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models」，arXiv:2402.03300）提出，其摘要介紹了這個 PPO 的變體。SDPO 論文將 GRPO 與 STaR 列為使用 Monte Carlo 獎勵估計的 RLVR 方法。
- DeepSeek-R1（DeepSeek-AI, 2025, arXiv:2501.12948）的摘要表明以強化學習激發 LLM 的推理能力；SDPO 論文將它列為近年大規模 RL 提升推理表現的代表。
- 信用分配：結果獎勵通常只在序列結尾給出。Lightman et al.（2023, arXiv:2305.20050）訓練逐步驗證的 process reward model；Cui et al.（2025, arXiv:2502.01456）以隱含獎勵做 process reinforcement。SDPO 論文指出 PRM 是與學生分開的另一個模型，記憶體負擔大。

## 與 SDPO 的關係
SDPO 論文第 2.1 節的比較：GRPO 的優勢值 `A_GRPO = r_i − mean(r)` 在 t 上為常數；SDPO 的梯度是 logit 層級的 policy gradient，優勢值由自我老師估計，因此可沿用現有 RLVR 實作，只替換優勢值。
