---
slug: bpo-ppo-grpo-ratio-clipping
title: PPO、GRPO 與 ratio clipping
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [PPO, GRPO, DAPO, clip-higher, RLVR]
arxiv: ["1707.06347", "2402.03300", "2501.12948", "2503.14476", "2503.20783", "2609.15987"]
sources: ["arXiv abstract 1707.06347", "arXiv abstract 2402.03300", "arXiv abstract 2501.12948", "2609.15987 §1, §2.3, App. B", "2609.15987 reference list [4], [12]"]
related: [bpo-lineage-hub, rlvr-grpo-credit-assignment, bpo-derivation-and-results]
---
# PPO、GRPO 與 ratio clipping

屬於 [BPO 的研究脈絡總覽](bpo-lineage-hub.html)。

## 脈絡
- PPO（Schulman et al., 2017，「Proximal Policy Optimization Algorithms」，arXiv:1707.06347）：BPO 論文稱 GRPO 為 PPO-style 方法，並以 PPO 的 clipped surrogate objective 與 token-level importance-sampling ratio 描述其損失。
- GRPO 由 DeepSeekMath（Shao et al., 2024，arXiv:2402.03300）提出。BPO 論文的描述：每個 prompt 取樣 G 個回應，以組內獎勵的平均與標準差正規化得到 advantage；結果監督下，同一回應的所有 token 共用同一個 advantage；不訓練價值模型。
- DeepSeek-R1（DeepSeek-AI, 2025，arXiv:2501.12948）與 Qwen3 等被 BPO 論文列為以 RLVR 提升推理能力的代表。
- DAPO（Yu et al., 2025，「DAPO: An Open-Source LLM Reinforcement Learning System at Scale」，arXiv:2503.14476）：BPO 的基線 GRPO-ClipHigher 採用 DAPO 的非對稱 clipping 區間 [0.8, 1.28]；訓練資料為 DAPO-Math-17k 的英文子集。
- Liu et al.（2025，「Understanding R1-Zero-Like Training: A Critical Perspective」，arXiv:2503.20783）被 BPO 論文列為 GRPO 變體之一。

## 與 BPO 的關係
BPO 保留 GRPO 的組內 advantage 與 clipping mask 形式，只把 importance-sampling ratio 換成 mismatch-correction weight。細節見 [BPO 的推導與實驗結果](bpo-derivation-and-results.html)；信用分配問題見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。
