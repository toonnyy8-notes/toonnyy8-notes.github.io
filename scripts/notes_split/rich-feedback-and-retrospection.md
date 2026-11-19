---
slug: rich-feedback-and-retrospection
title: 豐富回饋、事後回顧與 feedback-conditioned policy
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [rich-feedback, retrospection, hindsight, self-refine, goal-relabeling]
arxiv: ["2601.20802", "2410.02089", "2303.17651", "2303.11366", "2302.02676", "1707.01495", "2106.01345"]
sources: ["2601.20802 §6.2", "arXiv abstract 2410.02089", "arXiv abstract 2303.17651", "arXiv abstract 2303.11366", "arXiv abstract 2302.02676", "arXiv abstract 1707.01495", "arXiv abstract 2106.01345"]
related: ["sdpo-lineage-overview"]
---
# 豐富回饋、事後回顧與 feedback-conditioned policy

屬於 [SDPO 的研究脈絡總覽](sdpo-lineage-overview.html)。

## 脈絡
- RLEF（Gehring et al., 2024, arXiv:2410.02089）：以程式執行回饋做 RL；SDPO 論文將其列為利用執行回饋引導生成的工作。
- 不放進 RL 迴圈的 in-context 改進：Self-Refine（arXiv:2303.17651）、Reflexion（arXiv:2303.11366）、Decision Transformer（arXiv:2106.01345）。SDPO 論文將這些歸為「不把回饋納入 RL 最佳化」的做法。
- Feedback-conditioned policy：Chain of Hindsight（Liu et al., 2023, arXiv:2302.02676）等以回饋為條件、用監督目標訓練。SDPO 論文將其與 goal-conditioned RL 的 goal relabeling（Hindsight Experience Replay, arXiv:1707.01495）類比，並說明這些方法把回饋當資料轉換，不對負面軌跡做信用分配。

## 與 SDPO 的關係
SDPO 論文的區分：feedback-conditioned policy 把回饋視為目標，RLRF（rich feedback）把回饋視為狀態，用來判斷目標 x 是否達成；SDPO 用這個狀態讓自我老師對負面軌跡逐 token 重新評分。
