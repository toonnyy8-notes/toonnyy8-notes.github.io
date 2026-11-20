---
slug: ppo-grpo-dapo-llm-rl-objectives
title: PPO、GRPO、DAPO 與 LLM 的 RL 目標
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [PPO, GRPO, DAPO, group-centering, DeepSeek-R1]
arxiv: ["2609.20807", "1707.06347", "2402.03300", "2501.12948", "2503.14476", "2503.20783", "2505.07291", "2505.09388", "2404.03683"]
sources: ["arXiv:2609.20807 §1, §3, §5.1, Table 1", "arXiv abstract 1707.06347", "arXiv abstract 2402.03300", "arXiv abstract 2501.12948", "arXiv abstract 2503.14476", "arXiv abstract 2503.20783"]
related: ["rlvr-grpo-credit-assignment", "policy-gradient-baselines-score-function", "importance-sampling-corrections-llm-rl", "score-centering-drift-decomposition"]
---
# PPO、GRPO、DAPO 與 LLM 的 RL 目標

本文（§1）指出 LLM 的 RL 訓練佔訓練算力的比例逐漸增加，常用演算法為 PPO、GRPO、DAPO 與 IcePop，皆以 policy gradient 為基礎。

- **PPO**（Schulman et al., 2017，arXiv 1707.06347）：以 clipped 的機率比限制每次更新的幅度。本文 Table 1 將其視為對 token 機率比（區間 0.8 至 1.2，超出則 drop）的修正，並使用 verl 預設的 dual clipping（係數 3）。
- **GRPO**（Shao et al., 2024，DeepSeekMath，arXiv 2402.03300）與 **DeepSeek-R1**（arXiv 2501.12948）：以同一 prompt 的多條 rollout 做 group centering，使優勢（advantage）在 group 內總和為 0。本文以 group-centered 獎勵作為共同目標（REINFORCE with group-centered rewards）來比較各修正方法。
- **DAPO**（Yu et al., 2025，arXiv 2503.14476）：本文 Table 1 列為不對稱 clipping（0.8 至 1.28），並指出原方法還包含 dynamic sampling 與 overlong reward penalty，本文未納入。
- **Dr. GRPO**（Liu et al., 2025c，arXiv 2503.20783）：本文引用其對 R1-Zero 類訓練的分析，用於說明中心化獎勵的目的。

本文的實驗使用 Qwen3（arXiv 2505.09388）模型，資料為 Countdown（取自 Stream of Search，arXiv 2404.03683）與 INTELLECT-2 的數學子集（arXiv 2505.07291）。GRPO 在 RLVR 與信用分配上的討論見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。各目標中 clipping 作為 IS 修正的角色見 [LLM RL 中的 importance sampling 修正](importance-sampling-corrections-llm-rl.html)。本文指出 group centering 在 prefix 層級並不能消除 drift，見 [Score centering：drift 與 signal 的分解](score-centering-drift-decomposition.html)。
