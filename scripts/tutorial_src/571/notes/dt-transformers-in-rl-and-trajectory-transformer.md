---
slug: dt-transformers-in-rl-and-trajectory-transformer
title: RL 中的 Transformer 與同期的 Trajectory Transformer
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [transformer, trajectory-transformer, model-based-rl, relational-rl]
arxiv: ["2106.01345", "2106.02039", "1910.06764", "1806.01830"]
sources: ["2106.01345 §1, §6.2, §6.5", "arXiv abstract 2106.02039", "arXiv abstract 1910.06764", "arXiv abstract 1806.01830"]
related: ["dt-lineage-overview", "return-conditioned-sequence-modeling", "dt-method-and-results"]
---
# RL 中的 Transformer 與同期的 Trajectory Transformer

## 先前把 Transformer 當成元件的工作

Decision Transformer（arXiv 2106.01345）在 §1 與 §6.5 說明，先前的工作把 Transformer 當作傳統 RL 演算法內的架構選擇，仍以 actor-critic 優化。

- **Stabilizing Transformers for Reinforcement Learning**（1910.06764，Parisotto 等，2019）：abstract 指出標準 Transformer 在 RL 目標下難以優化，提出架構修改以改善穩定性。
- **Relational Deep Reinforcement Learning**（1806.01830，Zambaldi 等，2018；論文列出的標題為 *Deep reinforcement learning with relational inductive biases*）：以 self-attention 推理場景中實體的關係，指引 model-free 策略。DT 論文描述它在組合型環境改善表現。

DT 論文另引 Ritter 等 [69] 與模仿學習中的 Dasari 與 Gupta [70]、Abramson 等 [71]；這幾篇未在此逐一驗證。

## 同期工作：Trajectory Transformer

**Offline Reinforcement Learning as One Big Sequence Modeling Problem**（2106.02039，Janner、Li、Levine，2021）。abstract：用 Transformer 建模軌跡分布，並把 beam search 當成規劃演算法。DT 論文（§6.2）的描述：與 Decision Transformer 相似，但額外預測狀態與報酬並做離散化，帶有 model-based 成分。

兩者差異整理：

| 項目 | Decision Transformer | Trajectory Transformer |
| --- | --- | --- |
| 條件 | 目標 return-to-go | 以 beam search 規劃 |
| 預測目標 | 動作 | 狀態、動作、報酬（離散化） |
| 性質 | model-free | 含 model-based 成分 |

表中依 DT 論文 §3、§6.2 的描述與 2106.02039 的 abstract。DT 論文 §3 也說明，預測狀態或 returns-to-go 沒有改善其表現。與以報酬為條件的方法比較見 [以報酬為條件的監督式學習與序列建模](return-conditioned-sequence-modeling.html)。
