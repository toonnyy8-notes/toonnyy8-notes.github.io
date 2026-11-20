---
slug: r1-distillation-to-small-models
title: 以 DeepSeek-R1 輸出蒸餾小模型
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [distillation, SFT, Qwen, LLaMA]
arxiv: ["1503.02531", "2501.12948"]
sources: ["arXiv 摘要 1503.02531", "2501.12948 摘要、附錄 F"]
related: ["r1-lineage-hub", "knowledge-distillation-foundations"]
---
# 以 DeepSeek-R1 輸出蒸餾小模型

**背景。** Hinton 等（2015，arXiv 1503.02531）摘要：把模型集成的知識壓縮到單一模型以利部署。通用說明見 [知識蒸餾基礎](knowledge-distillation-foundations.html)。

**本文做法（附錄 F）。** 本文引用 Busbridge 等 2025 與 Hinton 等 2015，把蒸餾視為成熟的知識轉移技術。具體做法：以 DeepSeek-R1 產生 800,000 筆樣本，對 Qwen 與 LLaMA 等開源基礎模型做 SFT。蒸餾模型不含 RL 階段，本文說明目的是展示蒸餾本身的效果，RL 階段留給社群探索。本文稱由高品質教師輸出蒸餾的模型，表現持續優於直接以人類產生資料訓練者。

總覽：[總覽](r1-lineage-hub.html)。
