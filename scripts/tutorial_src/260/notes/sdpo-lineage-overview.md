---
slug: sdpo-lineage-overview
title: SDPO（Reinforcement Learning via Self-Distillation）的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [SDPO, RLVR, self-distillation, credit-assignment]
arxiv: ["2601.20802", "2402.03300", "2501.12948", "1503.02531", "2209.15189", "2306.13649", "2305.20050"]
sources: ["2601.20802 §6 Related Work", "2601.20802 reference list"]
related: ["rlvr-grpo-credit-assignment", "knowledge-distillation-foundations", "on-policy-distillation-imitation", "context-distillation-self-distillation", "rich-feedback-and-retrospection", "test-time-training-for-rl", "test-time-training-naming"]
---
# SDPO 的研究脈絡總覽

SDPO（Self-Distillation Policy Optimization，arXiv:2601.20802）把「現行 RLVR 每次嘗試只有一個純量結果獎勵」的問題，改成用同一個模型在看過回饋後當老師，對自己原本的輸出逐 token 做蒸餾。讀懂它需要依序理解下列五條脈絡，本文照閱讀順序排列。

1. [RLVR 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)：GRPO 等方法的優勢值在整段序列上為常數，PRM 以另一個模型提供逐步獎勵。這是 SDPO 要解決的問題。
2. [知識蒸餾基礎](knowledge-distillation-foundations.html)：用學生模仿老師輸出分布的 logit 蒸餾，是 SDPO 損失函數的形式。
3. [On-policy 蒸餾與模仿學習](on-policy-distillation-imitation.html)：在學生自己產生的軌跡上由老師給回饋，解決訓練與推論的分布不一致。
4. [Context distillation 與自蒸餾](context-distillation-self-distillation.html)：同一個模型帶額外 context 當老師，再學出不帶 context 的行為。SDPO 把額外 context 換成環境回饋。
5. [豐富回饋與事後回顧](rich-feedback-and-retrospection.html)：執行錯誤、評審文字等回饋如何被前人使用，以及 SDPO 與 feedback-conditioned policy、goal relabeling 的差別。
6. [測試時訓練與 test-time RL](test-time-training-for-rl.html)：SDPO 的 test-time self-distillation 對單一難題做專門化。與序列模型層的 TTT 是不同用法，見 [「Test-Time Training」的命名混淆](test-time-training-naming.html)。

時間順序的文獻清單見 timeline.json。本文所述的前人工作，內容來自該篇的 arXiv 摘要或 SDPO 論文第 6 節的描述，各筆記的 `sources` 欄位逐一標示。
