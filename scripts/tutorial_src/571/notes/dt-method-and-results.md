---
slug: dt-method-and-results
title: Decision Transformer 的方法與實驗結果
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [decision-transformer, returns-to-go, GPT, offline-rl, D4RL, Atari]
arxiv: ["2106.01345", "2006.04779"]
sources: ["2106.01345 §3, §4, §5"]
related: ["dt-lineage-overview", "dt-transformer-lm-background", "return-conditioned-sequence-modeling", "dt-offline-rl-baselines-datasets", "dt-credit-assignment-lineage", "rl-mdp-q-learning-basics", "offline-rl-distribution-shift"]
---
# Decision Transformer 的方法與實驗結果

## 軌跡表示

Decision Transformer（arXiv 2106.01345，Chen 等，2021）不直接輸入報酬，而是輸入 returns-to-go（R̂_t = 從 t 起至結尾的報酬總和）。軌跡表示為 (R̂_1, s_1, a_1, R̂_2, s_2, a_2, ..., R̂_T, s_T, a_T)。

## 架構與訓練

- 輸入最近 K 個時間步，共 3K 個 token（return-to-go、狀態、動作各一）。
- 每種模態一個線性層（影像狀態改用卷積編碼器），之後 layer normalization；每個時間步一個學得的嵌入（一個時間步對應三個 token，與標準位置嵌入不同）。
- 以 GPT 預測動作 token；離散動作用 cross-entropy，連續動作用 MSE。
- 論文未發現預測狀態或 returns-to-go 能改善表現。

## 推論

指定目標報酬（例如成功為 1、失敗為 0）與起始狀態，執行生成的動作後，以實際獲得的報酬遞減目標報酬，重複至 episode 結束。

## 主要結果

- Atari（1% DQN replay，K = 30，Pong 為 50）：四個遊戲中與 CQL 競爭 3 個；Breakout DT 267.5、CQL 211.1。
- D4RL Gym（不含 Reacher 的平均）：DT 74.7、CQL 63.9、BEAR 48.2、BRAC-v 36.9、AWR 34.3、行為複製 46.4（Table 2）。
- 上下文長度（§5.3）：Atari 上 K = 1 明顯較差，例如 Pong 106.1 對 2.5。
- 報酬建模（§5.2）：目標報酬與實際報酬高度相關；Seaquest 等任務可用高於資料集最大值的目標，顯示有時能外推。

信用分配與延遲報酬的實驗見 [信用分配的相關研究](dt-credit-assignment-lineage.html)；比較的 TD 方法見 [離線 RL 基準方法與資料集](dt-offline-rl-baselines-datasets.html)；前身見 [以報酬為條件的監督式學習與序列建模](return-conditioned-sequence-modeling.html)。架構來源見 [Transformer 與語言模型背景](dt-transformer-lm-background.html)。共用背景見 [MDP 與 Q-learning 基礎](rl-mdp-q-learning-basics.html)、[離線 RL 的分布位移](offline-rl-distribution-shift.html)；整體脈絡見 [研究脈絡總覽](dt-lineage-overview.html)。
