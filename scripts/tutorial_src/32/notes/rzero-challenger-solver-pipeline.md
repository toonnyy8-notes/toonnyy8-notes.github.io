---
slug: rzero-challenger-solver-pipeline
title: R-Zero 的 Challenger–Solver 迭代流程
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [R-Zero, Challenger, Solver, uncertainty-reward, BLEU-repetition-penalty]
arxiv: ["2508.05004", "2402.03300", "2505.03335"]
sources: ["arXiv 2508.05004 §2.1–2.4、§3.1、§4.1、附錄 F–G"]
related: ["rzero-lineage-hub", "rzero-grpo-rlvr-foundation", "rzero-majority-vote-pseudo-labels", "rzero-difficulty-curriculum", "rzero-self-play-task-generation", "rzero-self-training-collapse"]
---
# R-Zero 的 Challenger–Solver 迭代流程

每一輪迭代依序進行三步（論文 §2.1–2.4、附錄 G）：

1. **Challenger 訓練**：Qθ 產生問題，格式必須包在 question 標籤內，否則獎勵為 0。凍結的 Solver 回答 m 個答案，以多數決得偽標籤與經驗正確率 p̂。獎勵為 r = max(0, r_uncertainty − r_rep)，其中 r_uncertainty = 1 − 2|p̂ − 1/2|；r_rep = λ|C_k|/B，以 BLEU 距離 d = 1 − BLEU 小於 τ_BLEU 的題目成群，群越大懲罰越大（λ = 1、τ_BLEU = 0.5）。以 GRPO 更新。
2. **Solver 資料集建構**：Challenger 產生 N = 8,000 題，保留 p̂ 落在 |p̂ − 1/2| ≤ δ 的題目與其多數決答案。
3. **Solver 訓練**：以 GRPO 與二元獎勵訓練，之後 Solver 凍結，進入下一輪 Challenger 訓練。

各元件的來源脈絡：GRPO 見 [R-Zero 使用的訓練基礎：GRPO 與 RLVR](rzero-grpo-rlvr-foundation.html)；偽標籤見 [多數決偽標籤](rzero-majority-vote-pseudo-labels.html)；p̂ = 0.5 的理論依據與過濾見 [難度課程](rzero-difficulty-curriculum.html)；雙角色的設計與 Absolute Zero 的比較見 [自我對弈與模型自產任務](rzero-self-play-task-generation.html)。

消融結果（Qwen3-4B-Base，表 3）：完整方法的數學平均為 49.07、一般領域平均 31.15；移除重複懲罰為 45.76 與 28.73；移除過濾為 47.35 與 26.69。另一個基線 R-Zero（challenger）使用未訓練的 Challenger 出題，作為 Challenger RL 訓練的對照（表 1）。

論文承認的限制（§6）：方法依賴可客觀驗證答案的領域，不適用於創意寫作、對話等主觀任務；效能在多次迭代後下降，見 [自我訓練的崩壞風險](rzero-self-training-collapse.html)。
