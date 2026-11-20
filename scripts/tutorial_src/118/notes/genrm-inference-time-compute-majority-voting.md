---
slug: genrm-inference-time-compute-majority-voting
title: 推論時運算與多數決（CoT、Self-Consistency、重複取樣）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [chain-of-thought, self-consistency, majority-voting, inference-compute]
arxiv: ["2408.15240", "2201.11903", "2203.11171", "2407.21787"]
sources: ["arXiv 摘要 2407.21787", "arXiv:2408.15240 §1、§3.3"]
related: ["genrm-lineage-hub", "genrm-yes-token-method", "genrm-critiques-and-cot-reward-models", "r1-cot-and-test-time-compute"]
---
# 推論時運算與多數決

**CoT。** Wei 等 2022（arXiv 2201.11903，Chain-of-Thought Prompting Elicits Reasoning in Large Language Models）。GenRM §1、§3.3 引用它作為 CoT 推理的來源，並指出判別式 RM 無法利用 CoT。

**Self-Consistency。** Wang 等 2022（arXiv 2203.11171，Self-Consistency Improves Chain of Thought Reasoning in Language Models）。GenRM §3.3 引用它作為「邊際化多條推理路徑」的依據：取樣 K 條驗證 CoT，取各條 "Yes" 機率的平均，式 (7)：r_MajV@K = (1/K) Σ p_θ(Yes | x, y, I_CoT, v_CoT^(i), I)。論文說明個別驗證理由可能含錯，平均可降低其影響。

**重複取樣。** Brown 等 2024（arXiv 2407.21787，Large Language Monkeys）。摘要：把推論運算當作另一個縮放軸，重複取樣候選解；覆蓋率（至少一個樣本解出的題目比例）隨樣本數成長，跨四個數量級，且常呈對數線性。GenRM §1 引用它作為「利用額外推論時運算」的例子。

**GenRM 的連結。** 因驗證是生成，K 的增加直接提高驗證階段的運算量；論文結論稱 GenRM 在推論運算增加時優於 LLM-as-a-Judge。方法細節見 [GenRM 的 Yes token 方法](genrm-yes-token-method.html)；CoT 與測試時運算的更廣背景見 [思維鏈、驗證器與推論時運算](r1-cot-and-test-time-compute.html)。回到 [總覽](genrm-lineage-hub.html)。
