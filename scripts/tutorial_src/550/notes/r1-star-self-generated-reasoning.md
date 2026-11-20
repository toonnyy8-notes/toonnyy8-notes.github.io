---
slug: r1-star-self-generated-reasoning
title: STaR：以模型自產的推理軌跡微調
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [STaR, rationale, self-improvement, outcome-based]
arxiv: ["2203.14465", "2501.12948"]
sources: ["arXiv 摘要 2203.14465", "2501.12948 §1、附錄 H.3"]
related: ["r1-lineage-hub", "r1-grpo-outcome-rewards", "rlvr-grpo-credit-assignment"]
---
# STaR：以模型自產的推理軌跡微調

**前作內容。** Zelikman 等（2022，arXiv 2203.14465）摘要：產生逐步思維鏈理由能提升複雜推理表現，但誘發理由需要大量理由資料集，或僅用 few-shot 而犧牲準確度。STaR 以少量理由範例加上無理由的大資料集，迭代地自我引導出越來越複雜的推理能力。DeepSeek-R1 H.3 描述為：以模型自己產生、且導向正確最終答案的思維鏈做微調（同類還有 Yuan 等 2023、Singh 等 2024）。

**與 DeepSeek-R1 的差異（依本文 H.3）。**
- STaR 類方法是對正確軌跡做微調；本文對 base language model 直接做結果導向 RL。
- 本文不做初始 SFT 階段，理由是避免限制模型只模仿人類範例，使其發展多樣的解法。
- 本文也提到後續研究（Face 2025、Liu 等 2025、Pan 等 2025）受本方法啟發。

**後續文獻中的歸類。** SDPO 論文把 STaR 與 GRPO 列為使用 Monte Carlo 獎勵估計的 RLVR 方法，見 [RLVR、GRPO 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)。

延伸：[總覽](r1-lineage-hub.html)；本文改採的獎勵設計見 [GRPO 與結果導向獎勵](r1-grpo-outcome-rewards.html)。
