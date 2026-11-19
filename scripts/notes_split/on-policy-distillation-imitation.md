---
slug: on-policy-distillation-imitation
title: On-policy 蒸餾與線上模仿學習
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [on-policy-distillation, imitation-learning, DAgger, distribution-shift]
arxiv: ["2601.20802", "1011.0686", "2306.13649", "2306.08543"]
sources: ["2601.20802 §6.3", "arXiv abstract 1011.0686", "arXiv abstract 2306.13649", "arXiv abstract 2306.08543"]
related: ["sdpo-lineage-overview", "knowledge-distillation-foundations"]
---
# On-policy 蒸餾與線上模仿學習

屬於 [SDPO 的研究脈絡總覽](sdpo-lineage-overview.html)。基礎的蒸餾損失見 [知識蒸餾基礎](knowledge-distillation-foundations.html)。

## 脈絡
- Ross, Gordon, Bagnell（arXiv:1011.0686，發表於 AISTATS 2011）：把模仿學習與結構化預測化約為 no-regret 線上學習。SDPO 論文稱 on-policy 蒸餾與這類線上模仿學習相關，用來處理訓練與推論的分布偏移。
- Agarwal et al.（2023, arXiv:2306.13649）：On-Policy Distillation of Language Models，學生從自己產生的輸出上獲得老師回饋。
- Gu et al.（2023, arXiv:2306.08543）：MiniLLM，大型語言模型的 on-policy 蒸餾。
- SDPO 論文另引 Lu & Thinking Machines Lab（2025）的部落格文章，非 arXiv 論文，未查證。

## 與 SDPO 的關係
這些方法的老師是外部模型。SDPO 保留「在學生自己的生成上評分」這一點，但老師換成帶回饋的自己；論文並指出「訓練學生避免自己生成中的錯誤」比離線的自蒸餾更有效。
