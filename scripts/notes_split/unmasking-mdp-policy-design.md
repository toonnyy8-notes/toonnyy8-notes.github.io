---
slug: unmasking-mdp-policy-design
title: Unmasking 作為 MDP 與輕量 policy 設計
cluster: Diffusion Language Models
kind: topic
tags: [mdp, policy, grpo, bernoulli, accuracy-efficiency]
arxiv: ["2512.09106", "2505.22618", "2502.09992", "2508.15487"]
sources: ["paper.txt §3, §4"]
related: [unmasking-policies-dllm-overview, confidence-thresholding-heuristics, rl-post-training-dllm-grpo, semi-ar-block-generation, dllm-llada-dream-models]
---
# Unmasking 作為 MDP 與輕量 policy 設計

屬於[總覽](unmasking-policies-dllm-overview.html)。本篇整理論文 2512.09106 的方法與結果。

## MDP

- 環境：凍結的 dLLM。狀態：目前部分遮罩的序列 y_t。動作：u_t ∈ {0,1}^L，標示哪些位置要解開。
- 轉移：以式 (2.2) 的標準取樣填入被選位置的 token。
- 終止：序列不再有遮罩，最後步為 T̂。

## Policy

- 輸入：各位置 confidence c_t、仍遮罩的二元向量 m_t、時間步 t。輸出每個位置的 logit b_t^k。
- 架構：單層 transformer，使用 adaptive layer normalization 條件化，約 30 萬參數。
- 動作：u_t^k ~ Bernoulli(σ(b_t^k / τ_π))，likelihood 有封閉形式。若沒有任何位置被選中，改解開分數最高的位置（argmax fallback）。論文也試過 Plackett-Luce 形式（附錄 E），表現相近。

## Reward 與訓練

reward 為最終答案的正確性，乘上與步數相關的項，超參數 α ≥ 0 控制準確度與效率的取捨；論文提到加法形式的懲罰容易發生 reward hacking。以 GRPO 訓練（見 [dLLM 的強化學習後訓練與 GRPO](rl-post-training-dllm-grpo.html)）。α 越大，policy 越快但越不準；α = 10.0 訓練較不穩定。

## 結果（論文敘述）

- semi-AR（BL = 32）：與 Fast-dLLM 相當，優於隨機與 high-confidence；中高 NFE 區間未超越 Fast-dLLM。
- full-diffusion（BL = 256）：優於 heuristic。
- 單一 policy 對應單一 α，每個 α 需另外訓練。
