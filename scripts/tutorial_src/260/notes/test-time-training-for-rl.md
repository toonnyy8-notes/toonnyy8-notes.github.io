---
slug: test-time-training-for-rl
title: 測試時訓練與 test-time RL 與 SDPO 的 test-time self-distillation
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [test-time-training, test-time-RL, TTRL]
arxiv: ["2601.20802", "2411.07279", "2504.16084", "2510.04786"]
sources: ["2601.20802 §1", "2601.20802 reference list", "arXiv abstract 2411.07279", "arXiv abstract 2504.16084", "arXiv abstract 2510.04786"]
related: ["sdpo-lineage-overview", "test-time-training-naming", "ttt-theory-and-applications"]
---
# 測試時訓練與 test-time RL

屬於 [SDPO 的研究脈絡總覽](sdpo-lineage-overview.html)。

## 脈絡
- Akyürek et al.（2024, arXiv:2411.07279）：測試時訓練用於 few-shot 學習。
- TTRL（Zuo et al., 2025, arXiv:2504.16084）：測試時 RL；SDPO 論文將其多數決作為「學生加測試時搜尋」型 expert 的例子。
- Learning on the Job（Hübotter et al., 2025, arXiv:2510.04786）：test-time curricula for targeted RL，與 SDPO 同一研究群。

## 與 SDPO 的關係
SDPO 論文在 LiveCodeBench 上 pass@64 低於 0.03 的難題上，對單一題目做 Test-Time Self-Distillation，並報告發現解的速度為 RLVR 基線的 3 倍。此處的「測試時訓練」是對單一題目專門化，與序列建模層的 TTT 不同，命名差異見 [命名混淆](test-time-training-naming.html)，TTT 的應用擴散見 [TTT 的理論還原與應用擴散](ttt-theory-and-applications.html)。
