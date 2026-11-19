---
slug: dllm-llada-dream-models
title: LLaDA 與 Dream：可比擬 AR 的 dLLM
cluster: Diffusion Language Models
kind: topic
tags: [llada, dream, dllm, pretrained-model]
arxiv: ["2502.09992", "2508.15487", "2407.21783", "2512.09106"]
sources: ["paper.txt §1, §4", "arXiv API metadata"]
related: [unmasking-policies-dllm-overview, mdm-discrete-diffusion-lineage, confidence-thresholding-heuristics]
---
# LLaDA 與 Dream：可比擬 AR 的 dLLM

屬於[總覽](unmasking-policies-dllm-overview.html)。

論文第 1 節指出，LLaDA（Nie 等人，2025，「Large Language Diffusion Models」，2502.09992）與 Dream（Ye 等人，2025，「Dream 7B: Diffusion Large Language Models」，2508.15487）在表現上可比擬同規模的自迴歸（AR）LLM。

論文引用 Llama 3（Dubey 等人，2024，「The Llama 3 Herd of Models」，2407.21783）作為 AR 比較對象，並說明開源 dLLM 在推論效率上長期落後 AR 模型，直到 Fast-dLLM 才在 LLaDA 上達到比同規模 LLaMA 更高的 token 吞吐量（見 [Confidence 門檻式 heuristic](confidence-thresholding-heuristics.html)）。

在本論文中，這兩個模型是 MDP 的環境，參數凍結：實驗主要使用 LLaDA-8B-Instruct，Dream-7B-Instruct 用於附錄與遷移實驗。policy 約 30 萬參數，小於預訓練 dLLM 的 0.01%。

資料集：GSM8K（Cobbe 等人，2110.14168，「Training Verifiers to Solve Math Word Problems」）與 MATH 的訓練集合計約 15,000 筆用於訓練 policy，評估用 GSM8K 與 MATH-500。
