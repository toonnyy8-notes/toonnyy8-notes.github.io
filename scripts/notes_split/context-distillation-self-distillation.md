---
slug: context-distillation-self-distillation
title: Context distillation 與自蒸餾
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [self-distillation, context-distillation, in-context-learning]
arxiv: ["2601.20802", "2209.15189", "2212.08073", "2006.07733", "1705.08439"]
sources: ["2601.20802 §6.1", "2601.20802 §6.4", "arXiv abstract 2209.15189", "arXiv abstract 2212.08073", "arXiv abstract 2006.07733", "arXiv abstract 1705.08439"]
related: ["sdpo-lineage-overview"]
---
# Context distillation 與自蒸餾

屬於 [SDPO 的研究脈絡總覽](sdpo-lineage-overview.html)。

## 脈絡
- Snell, Klein et al.（2022, arXiv:2209.15189）Learning by Distilling Context。SDPO 論文第 6.4 節將其視為自蒸餾概念的最早提出：用額外 context 取樣，再訓練同一模型在沒有該 context 時重現這些預測。
- Constitutional AI（Bai et al., 2022, arXiv:2212.08073）：SDPO 論文將其列為把行為壓縮進模型權重的例子之一。
- BYOL（Grill et al., 2020, arXiv:2006.07733）與 Expert Iteration（Anthony et al., 2017, arXiv:1705.08439）：SDPO 論文第 6.1 節說明學生反覆模仿「自己的改進版」。expert 通常是學生加測試時搜尋（樹搜尋或多數決）；SDPO 的對應物是帶回饋 context 的自己，類似 BYOL 的 augmented views。
- 利用環境回饋的離線自蒸餾（如 Scheurer et al., 2023）：SDPO 論文指出這類方法是 off-policy，在實驗中明顯弱於 on-policy 的 SDPO。

## 與 SDPO 的關係
SDPO 把 context distillation 中的「額外 context」換成環境回饋 f，並改成在學生自己的 rollout 上計算。
