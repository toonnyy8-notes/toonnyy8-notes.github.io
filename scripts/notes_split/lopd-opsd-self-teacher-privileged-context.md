---
slug: lopd-opsd-self-teacher-privileged-context
title: OPSD：同一模型帶特權脈絡當老師
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [OPSD, self-distillation, privileged-information, on-policy]
arxiv: ["2608.13040", "2601.18734", "2602.04942", "2601.19897", "2602.12275", "2601.20802"]
sources: ["arXiv abstract 2601.18734", "arXiv abstract 2602.04942", "arXiv abstract 2601.19897", "arXiv abstract 2602.12275", "2608.13040 §1", "2608.13040 §3"]
related: ["on-policy-distillation-imitation", "context-distillation-self-distillation", "sdpo-lineage-overview", "lopd-privileged-context-substrates", "lopd-method-learnable-context-and-margin"]
---
# OPSD：同一模型帶特權脈絡當老師

先備：[On-policy 蒸餾與線上模仿學習](on-policy-distillation-imitation.html)（學生自己取樣，老師在學生走過的狀態上給 token 層級監督）與 [Context distillation 與自蒸餾](context-distillation-self-distillation.html)（同一模型帶額外 context 當老師）。

## 脈絡
- Zhao et al.（2026, arXiv:2601.18734）Self-Distilled Reasoner：摘要指出一般 on-policy 蒸餾需要另一個較大的老師，也沒有利用推理資料集中的標準解答；OPSD 讓單一 LLM 以不同 context 同時當老師與學生，老師條件於特權的推理軌跡。
- Penaloza et al.（2026, arXiv:2602.04942）Privileged Information Distillation：摘要提出 π-Distill，同一模型同時訓練帶特權資訊（PI）的老師與不帶 PI 的學生，並提出 OPSD 作為另一種做法；場景是多輪 agent 環境。
- Shenfeld et al.（2026, arXiv:2601.19897）Self-Distillation Enables Continual Learning：SDFT 用「帶示範的模型」當自己的老師，產生 on-policy 訓練訊號，摘要宣稱比 SFT 更能保留既有能力。
- Ye et al.（2026, arXiv:2602.12275）On-Policy Context Distillation：摘要描述它把 on-policy 蒸餾與 context distillation 結合，對 context 條件化的老師最小化 reverse KL。
- Hübotter et al.（2026, arXiv:2601.20802）SDPO：老師帶的是環境回饋，見 [SDPO 研究脈絡總覽](sdpo-lineage-overview.html)。

## 與 LOPD 的關係
LOPD 論文把這些方法的共同結構寫成式 (2)(3)：老師為 π(·|[s; c_fix])，學生為 π(·|s)，c_fix 由固定轉換 Φ_fix 從經驗產生。論文指出監督品質受 Φ_fix 的設計限制，LOPD 改為把 Φ 參數化。各種 c_fix 的實例見 [特權脈絡的各種實體](lopd-privileged-context-substrates.html)。
LOPD 的設計見 [LOPD 方法](lopd-method-learnable-context-and-margin.html)。
