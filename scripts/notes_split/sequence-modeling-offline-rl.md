---
slug: sequence-modeling-offline-rl
title: 以序列模型處理離線 RL：Decision Transformer 與 Trajectory Transformer
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-rl, sequence-modeling, transformer, decision-transformer, trajectory-transformer]
arxiv: ["2106.01345", "2106.02039", "2205.09991", "1910.06764", "1806.01830"]
sources: ["paper.txt §5.3, §6, Appendix A.3", "arXiv API metadata (2106.01345, 2106.02039)", "2106.01345 §1, §6.2, §6.5", "arXiv abstract 2106.02039", "arXiv abstract 1910.06764", "arXiv abstract 1806.01830"]
related: ["dt-lineage-overview", "return-conditioned-sequence-modeling", "dt-method-and-results", "offline-rl-distribution-shift", "model-based-planning-abstraction-barrier", "trajectory-diffusion-planning"]
---
# 以序列模型處理離線 RL

屬於[總覽](dt-lineage-overview.html)。

## 兩篇前作

- Chen, Lu, Rajeswaran, Lee et al. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling*（2106.01345）。本論文（§5.3）把它歸為 return-conditioning 方法，簡稱 DT。
- Janner, Li, Levine (2021). *Offline Reinforcement Learning as One Big Sequence Modeling Problem*（2106.02039）。本論文歸為 model-based 方法，簡稱 TT，並在 §6 把 Transformer 列為被用於動態模型的一類生成模型。

Diffuser 的第一作者同時是 TT 的第一作者；論文把軌跡視為整體建模的出發點與 TT 相同，差別在 §1 所說：Diffuser 同時預測所有時間步，不依時間順序自迴歸。

## 先前把 Transformer 當成元件的工作

Decision Transformer 論文（2106.01345）在 §1 與 §6.5 說明，先前的工作把 Transformer 當作傳統 RL 演算法內的架構選擇，仍以 actor-critic 優化。

- **Stabilizing Transformers for Reinforcement Learning**（1910.06764，Parisotto 等，2019）：abstract 指出標準 Transformer 在 RL 目標下難以優化，提出架構修改以改善穩定性。
- **Relational Deep Reinforcement Learning**（1806.01830，Zambaldi 等，2018；論文列出的標題為 *Deep reinforcement learning with relational inductive biases*）：以 self-attention 推理場景中實體的關係，指引 model-free 策略。DT 論文描述它在組合型環境改善表現。

DT 論文另引 Ritter 等 [69] 與模仿學習中的 Dasari 與 Gupta [70]、Abramson 等 [71]；這幾篇未在此逐一驗證。

## DT 與 Trajectory Transformer 的差異

DT 論文（§6.2）的描述：Trajectory Transformer 與 Decision Transformer 相似，但額外預測狀態與報酬並做離散化，帶有 model-based 成分。2106.02039 的 abstract：用 Transformer 建模軌跡分布，並把 beam search 當成規劃演算法。

| 項目 | Decision Transformer | Trajectory Transformer |
| --- | --- | --- |
| 條件 | 目標 return-to-go | 以 beam search 規劃 |
| 預測目標 | 動作 | 狀態、動作、報酬（離散化） |
| 性質 | model-free | 含 model-based 成分 |

表中依 DT 論文 §3、§6.2 的描述與 2106.02039 的 abstract。DT 論文 §3 也說明，預測狀態或 returns-to-go 沒有改善其表現。與以報酬為條件的方法比較見 [以報酬為條件的監督式學習與序列建模](return-conditioned-sequence-modeling.html)，DT 的方法與結果見 [Decision Transformer 的方法與實驗結果](dt-method-and-results.html)。

## 在本論文中的角色

兩者都是 Table 2（D4RL locomotion）的比較對象。論文引用的分數來源：DT 取自 Chen et al. (2021b) Table 2，TT 取自 Janner et al. (2021) Table 1（Appendix A.3）。論文表中的九項平均：DT 74.7、TT 78.9、Diffuser 77.5。

## 與其他筆記的關係

離線設定下的分佈偏移問題見 [離線 RL 的分佈偏移](offline-rl-distribution-shift.html)。論文 §6 另提到 Lambert et al. (2020) 研究過非自迴歸的軌跡層級動態模型；該文 arXiv 編號（2012.09156）未能在本次以 API 逐一核對標題，因此不列入時間軸。Diffuser 的設計見 [軌跡層級擴散規劃（Diffuser）](trajectory-diffusion-planning.html)；規劃器與模型的分工背景見 [學習式動態模型與軌跡最佳化的分工](model-based-planning-abstraction-barrier.html)。
