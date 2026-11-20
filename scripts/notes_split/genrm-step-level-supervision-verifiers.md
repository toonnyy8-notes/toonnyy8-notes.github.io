---
slug: genrm-step-level-supervision-verifiers
title: 步驟級與結果級監督的驗證器（Uesato、Lightman、Math-Shepherd、自動過程監督）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [process-supervision, outcome-supervision, PRM, verifier, MATH500]
arxiv: ["2408.15240", "2211.14275", "2305.20050", "2312.08935", "2406.06592"]
sources: ["arXiv 摘要 2211.14275", "arXiv:2408.15240 §1、§5、§6 對 2305.20050、2312.08935、2406.06592 的描述"]
related: ["genrm-lineage-hub", "genrm-best-of-n-discriminative-verifiers", "genrm-yes-token-method", "r1-process-supervision-prm"]
---
# 步驟級與結果級監督的驗證器

這一組工作決定判別式驗證器的監督粒度。完整的 PRM 脈絡見 [過程監督與 PRM](r1-process-supervision-prm.html)，這裡只記錄 GenRM 論文用到的部分。

**Uesato 等 2022（arXiv 2211.14275，Solving math word problems with process- and outcome-based feedback）。** 摘要：比較「監督最終結果」與「監督推理過程」兩種做法，並稱之為在自然語言任務上第一個全面的比較。GenRM §5 將其列為判別式驗證器。

**Lightman 等 2023（arXiv 2305.20050，Let's Verify Step by Step）。** GenRM §5 將其列為判別式驗證器；§1 的圖 1 以其 MATH500 作為易到難泛化的評測集；§6 把「利用過程級監督」列為未來方向，預期可得到更準確的生成式驗證器。

**Wang 等 2023（arXiv 2312.08935，Math-Shepherd）與 Luo 等 2024（arXiv 2406.06592，Improve Mathematical Reasoning in Language Models by Automated Process Supervision）。** GenRM §1、§5 把兩者與上述工作並列為判別式驗證器的代表。本筆記不引用它們摘要中的細節。

**與 GenRM 的關係。** 以上工作都輸出數值分數；GenRM 改以 Yes/No token 機率表示，且本文只做結果級驗證，過程級是未來方向（[GenRM 的 Yes token 方法](genrm-yes-token-method.html)）。回到 [總覽](genrm-lineage-hub.html)。
