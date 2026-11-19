---
slug: learned-unmasking-concurrent-work
title: 學習式 unmasking 與 dLLM 加速的相關工作
cluster: Diffusion Language Models
kind: topic
tags: [learned-sampler, adaptive-computation, distillation, dllm]
arxiv: ["2510.05725", "2509.25188", "2509.22738", "2512.09106"]
sources: ["paper.txt §5", "arXiv API metadata"]
related: [unmasking-policies-dllm-overview, confidence-thresholding-heuristics, rl-post-training-dllm-grpo, unmasking-mdp-policy-design]
---
# 學習式 unmasking 與 dLLM 加速的相關工作

屬於[總覽](unmasking-policies-dllm-overview.html)。以下描述皆取自本論文第 5 節對這些工作的說明。

## 以 RL 學習 unmasking policy

Hong 等人（2025b，「Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies」，2510.05725）被論文視為最接近的同期工作之一：以 GRPO 訓練 dLLM 的 unmasking policy，但 policy 每步解開固定數量的 token。本論文透過 Bernoulli 取樣讓每步解開的 token 數量可變。

論文另提到 DiFFPO（Zhao 等人，2025b）：同時用 RL 學習自適應的 threshold 與更新基礎模型。該論文的 arXiv 編號本次未核對，不列入時間軸。

## 以蒸餾訓練輔助模組

Bansal 與 Sanghavi（「Enabling Approximate Joint Sampling in Diffusion LMs」，2509.22738）與 Bao 等人（「Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding」，2509.25188）在預訓練 dLLM 上訓練小型獨立模組。論文指出它們不是用 RL，而是蒸餾 heuristic 或基礎模型，因此預期仍承襲 heuristic 在非 semi-AR 設定的困難。

## 自適應計算

論文也把可變 NFE 連到自適應計算文獻：early exit（Schuster 等人，2022）、以 RL 決定 early exit 的長推理模型工作等，並稱本研究是第一個用 RL 學習 dLLM 自適應取樣 policy 的工作。
