---
slug: semi-ar-block-generation
title: Semi-AR 區塊生成
cluster: Diffusion Language Models
kind: topic
tags: [block-diffusion, semi-autoregressive, block-length, dllm]
arxiv: ["2503.09573", "2502.09992", "2512.09106"]
sources: ["paper.txt §1, §2.2", "arXiv API metadata"]
related: [unmasking-policies-dllm-overview, confidence-thresholding-heuristics, unmasking-mdp-policy-design]
---
# Semi-AR 區塊生成

屬於[總覽](unmasking-policies-dllm-overview.html)。

Semi-AR（半自迴歸）生成把答案切成連續的區塊（block），由左至右一次只解開一個區塊內的 token。論文引用 Arriola 等人「Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models」（2503.09573）與 LLaDA（Nie 等人，2502.09992）作為此設定的來源。

block length（BL）是超參數。BL 等於整個生成長度 L 時稱為 full-diffusion，論文實驗中 BL = 32 對應 semi-AR，BL = L = 256 對應 full-diffusion。

論文的觀察（圖 1，LLaDA-8B-Instruct，GSM8K）：

- 在 semi-AR 設定下，confidence 式 heuristic 優於隨機 unmasking。
- 在 full-diffusion 設定下，heuristic 的表現可能低於隨機 unmasking，也不再隨 NFE 增加而進步。

這個現象是論文提出學習式 policy 的動機之一：學到的 policy 在 semi-AR 與 Fast-dLLM 相當，在 full-diffusion 則勝過 heuristic。
