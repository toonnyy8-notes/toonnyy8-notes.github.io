---
slug: knowledge-distillation-foundations
title: 知識蒸餾基礎：logit 蒸餾與序列層級蒸餾
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [distillation, KL-divergence, logit-distillation]
arxiv: ["2601.20802", "1503.02531", "1606.07947"]
sources: ["2601.20802 §6.3", "arXiv abstract 1503.02531", "arXiv abstract 1606.07947"]
related: ["sdpo-lineage-overview"]
---
# 知識蒸餾基礎

屬於 [SDPO 的研究脈絡總覽](sdpo-lineage-overview.html)。

## 脈絡
- Hinton, Vinyals, Dean（2015, arXiv:1503.02531）：把大模型（或模型集合）的知識壓縮到小模型。SDPO 論文第 6.3 節將蒸餾描述為讓學生模仿老師的輸出分布或中間表示。
- Kim & Rush（2016, arXiv:1606.07947）：Sequence-Level Knowledge Distillation，把蒸餾從詞層級擴展到序列層級（用於神經機器翻譯）。
- SDPO 論文提到另有 FitNets、DistilBERT 等工作，本筆記未逐一查證。

## 與 SDPO 的關係
SDPO 的損失是標準的 logit 蒸餾：以 KL 散度比較學生 `π(·|x, y<t)` 與老師 `π(·|x, f, y<t)` 的下一個 token 分布，老師端施加 stopgrad。差別在於老師不是外部大模型，而是同一個模型加上回饋 f。
