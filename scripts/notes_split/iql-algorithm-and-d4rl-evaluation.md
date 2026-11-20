---
slug: iql-algorithm-and-d4rl-evaluation
title: IQL 演算法、超參數與 D4RL 評估
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [iql, d4rl, antmaze, online-finetuning, clipped-double-q]
arxiv: ["2110.06169", "2004.07219", "1802.09477", "2006.09359"]
sources: ["arXiv:2110.06169 §4.3, §5, Appendix B"]
related: [iql-lineage-overview, expectile-regression-value-learning, advantage-weighted-bc-methods, one-step-offline-rl-and-sequence-modeling, batch-constrained-policy-constraint-methods]
---
# IQL 演算法、超參數與 D4RL 評估

## 演算法（Algorithm 1）

兩階段：

1. TD learning：交替對 V 做 expectile 更新（[式 5](expectile-regression-value-learning.html)）與對 Q 做 MSE 更新（式 6），target network 以 Polyak 平均更新（α = 0.005）。
2. 策略抽取：以 [advantage-weighted regression](advantage-weighted-bc-methods.html) 擬合策略（式 7）。

V 與策略更新使用 clipped double Q-learning（Fujimoto 等，TD3，arXiv:1802.09477），取兩個 Q 函數的最小值。策略不影響價值函數，所以兩階段可並行，這使 IQL 能用於線上微調（§5.3）。

## 設定（Appendix B）

- Ant Maze：τ = 0.9，β = 10.0；MuJoCo locomotion：τ = 0.7，β = 3.0；Kitchen 與 Adroit：τ = 0.7，β = 0.5，另加 dropout 0.1。
- Adam，學習率 3·10⁻⁴；2 層 MLP、ReLU、256 單元；策略為狀態無關標準差的 Gaussian；actor 學習率用 cosine schedule；指數化 advantage 截斷至 (−∞, 100]。
- Ant Maze 的 reward 減 1；locomotion 的 reward 以資料集最佳與最差軌跡回報之差正規化。
- 以 JAX 與 Flax 實作。

## 評估（§5）

基準為 D4RL（Fu 等，arXiv:2004.07219，2020）。Table 1 的總分（locomotion-v2 加 antmaze-v0）：IQL 1070.4、CQL 1002.1、TD3+BC 841.2、One-step RL 809.9。antmaze-v0 總分：IQL 378.0、CQL 303.6、TD3+BC 163.8、One-step RL 125.3。locomotion-v2 總分：CQL 698.5、IQL 692.4。論文說明 IQL 在需要 stitching 的 Ant Maze 上超越先前最佳，在 locomotion 上與最佳方法相近。執行時間欄位為 IQL 約 20 分鐘，CQL 80 分鐘，Decision Transformer 960 分鐘。

§5.3 在 Ant Maze 與 Adroit（Rajeswaran 等 2018 的 dexterous manipulation 環境，AWAC 提議用於線上適應研究）上，先離線訓練再線上 RL 1M 步，與 AWAC 和 CQL 比較。單步與多步方法的差異見 [單步方法](one-step-offline-rl-and-sequence-modeling.html)，脈絡見 [總覽](iql-lineage-overview.html)；表中的 CQL、TD3+BC 見 [policy constraint 與 value regularization](batch-constrained-policy-constraint-methods.html)。
