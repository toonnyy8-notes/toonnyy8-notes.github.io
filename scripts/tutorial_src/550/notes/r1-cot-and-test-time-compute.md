---
slug: r1-cot-and-test-time-compute
title: 思維鏈、驗證器與推論時運算
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [chain-of-thought, self-consistency, verifier, test-time-compute, scaling]
arxiv: ["2201.11903", "2110.14168", "2203.11171", "2408.03314", "2407.21787", "2001.08361", "2501.12948"]
sources: ["arXiv 摘要 2201.11903、2110.14168、2203.11171、2408.03314、2407.21787、2001.08361", "2501.12948 §1、附錄 H.2"]
related: ["r1-lineage-hub", "r1-process-supervision-prm"]
---
# 思維鏈、驗證器與推論時運算

**思維鏈（CoT）。** Wei 等（2022，arXiv 2201.11903）的摘要：在 prompt 中提供幾個含中間推理步驟的範例，足夠大的語言模型就能在複雜推理任務上表現提升。DeepSeek-R1 摘要與 §1 把 CoT prompting 視為前提，並指出其成效仰賴人工標註的示範。

**規模與推論運算。** Kaplan 等（2020，arXiv 2001.08361）研究語言模型的 scaling laws。本文 H.2 提到預訓練規模可能受限於人類資料量，因此推論時擴增運算更重要（引用 Snell 等）。本文把「透過增加推論運算改善表現」的方法分為幾類：
- 多條推理鏈後挑選：以獨立的重排序器（Cobbe 等 2021，arXiv 2110.14168，提出 GSM8K 與訓練驗證器的做法；Brown 等 2024，arXiv 2407.21787）、過程式獎勵模型，或多數決（Wang 等 2022，arXiv 2203.11171，self-consistency：取樣多條推理路徑，選最一致的答案）。
- 搜尋（MCTS、beam search）、自我修正、使用工具、test-time training，與每個 token 配置更多運算。
- Snell 等（2024，arXiv 2408.03314）摘要：研究固定推論運算量下，LLM 在困難 prompt 上可以改善多少。

**本文的定位。** H.2 寫道：本文顯示 LLM 可藉由額外的 RL 運算與更多測試時運算（即更多 token）獲得可擴展的改善，並以 RL 激發更強的 in-context 搜尋能力。

相關：驗證器的逐步版本見 [過程監督與 PRM](r1-process-supervision-prm.html)；全貌見 [總覽](r1-lineage-hub.html)。
