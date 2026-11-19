---
slug: test-time-training-naming
title: 「Test-Time Training」的命名混淆：序列建模層 vs 分布偏移適應 vs 測試時 RL（TEMPO）
cluster: Architectures & Efficiency
kind: topic
tags: [test-time-training, naming, tempo, test-time-rl, disambiguation]
sources: [ttt-sequence-modeling.md]
arxiv: ["2604.19295", "2411.07279"]
related: ["ttt-sequence-modeling-overview", "ttt-linear-mlp", "lact-large-chunk-ttt", "in-place-ttt", "ttt-e2e"]
---
# 「Test-Time Training」的命名混淆：序列建模層 vs 分布偏移適應 vs 測試時 RL（TEMPO）

屬於 [TTT 序列建模總覽](ttt-sequence-modeling-overview.html)。

「Test-Time Training」至少有三個語意：
- **(A) 序列建模層/RNN**：Sun et al. 2024 起的主線（[TTT-Linear / TTT-MLP](ttt-linear-mlp.html)、[LaCT](lact-large-chunk-ttt.html)、[In-Place TTT](in-place-ttt.html)、[TTT-E2E](ttt-e2e.html)）。
- **(B) 分布偏移下的測試時自監督適應**：Sun et al. 2020 原始定義，影像。
- **(C) 推理時用自生獎勵做 RL**：TEMPO、TTRL 等。

## TEMPO（arXiv:2604.19295）
TEMPO（Qingyang Zhang 等 10 人，2026-04-21）全名為「TEMPO: Scaling Test-time Training for Large Reasoning Models」，屬於 (C)「test-time training = 推理時對未標註測試題用自生獎勵做 RL 更新」這個語意分支：它用 EM 框架交替 E-step（在標註資料上重校準 critic）與 M-step（在未標註題目上精煉 policy），解決自獎勵 TTT 的飽和與多樣性崩潰（例如把 OLMO3-7B 在 AIME 2024 從 33.0% 提到 51.1%）。它與 TTT-Linear/TTT-MLP 的「序列建模層」主線無直接技術繼承關係。同類還有 TTRL、EMPO，以及把 TTT 用於 few-shot reasoning 的 Akyürek et al.（2411.07279）。

引用前務必確認屬哪一支。
