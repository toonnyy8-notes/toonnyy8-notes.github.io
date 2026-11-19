---
slug: dt-credit-assignment-lineage
title: 信用分配的相關研究（Decision Transformer §6.3）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [credit-assignment, RUDDER, hindsight, key-to-door, delayed-reward]
arxiv: ["2106.01345", "1806.07857", "1810.06721", "1912.02503", "1907.08027", "2011.09464", "2102.12425", "1905.13420"]
sources: ["2106.01345 §1, §5.4, §5.5, §6.3", "arXiv abstract 1806.07857", "arXiv abstract 1810.06721", "arXiv abstract 1912.02503", "arXiv abstract 1907.08027", "arXiv abstract 2011.09464", "arXiv abstract 2102.12425", "arXiv abstract 1905.13420"]
related: ["dt-lineage-overview", "dt-method-and-results", "rl-mdp-q-learning-basics"]
---
# 信用分配的相關研究

信用分配指判斷長序列中哪些動作導致最終報酬。Decision Transformer（arXiv 2106.01345）主張 self-attention 能直接建立狀態與報酬的關聯，不必像 Bellman 備份那樣逐步傳遞報酬，後者容易受「干擾」訊號影響（§1，引 Hung 等）。

## 論文 §6.3 引用的方法

論文的描述：這些方法學習能把報酬函數分解的架構，使少數「重要」狀態分得大部分信用，並用學到的報酬改寫 actor-critic 的報酬。

- **RUDDER**（1806.07857，Arjona-Medina 等，2018）：針對有限 MDP 的延遲報酬；以報酬重分配（return-equivalent）與回報分解處理。
- **Optimizing Agent Behavior over Long Time Scales by Transporting Value**（1810.06721，Hung 等，2018）：abstract 指出既有短期信用分配無法處理動作與結果間有長延遲的任務；論文以此為「干擾訊號」的出處。
- **Hindsight Credit Assignment**（1912.02503，Harutyunyan 等，2019）：依「導致觀察到結果的可能性」事後分配信用。
- **Self-Attentional Credit Assignment**（1907.08027，Ferret 等，2019）：以自注意力做遷移用的信用分配。
- **Counterfactual Credit Assignment**（2011.09464，Mesnard 等，2020）：DT 的 Key-to-Door 環境即來自此論文。
- **Synthetic Returns**（2102.12425，Raposo 等，2021）：DT §5.5 引用其「狀態與報酬關聯」的觀點。
- **Sequence Modeling of Temporal Credit Assignment**（1905.13420，Liu 等，2019）：論文列為延遲報酬設定下的狀態關聯架構。

## Decision Transformer 的對應實驗

- §5.4 Key-to-Door：只在第三階段、且第一階段拿到鑰匙時才有二元報酬。以隨機行走資料訓練，1K 條軌跡時 DT 成功率 71.8%、CQL 13.1%；10K 條時 DT 94.6%、CQL 13.3%。
- §5.5：修改 DT 使其也輸出 return token，注意力集中在拿鑰匙與到達門等關鍵事件。
- §5.6：D4RL 延遲報酬版本中，CQL 明顯下降，DT 與 %BC 受影響很小。

論文的差異：這些性質在 Transformer 中自然出現，不需要另外學習報酬函數或 critic。見 [Decision Transformer 的方法與實驗結果](dt-method-and-results.html) 與 [研究脈絡總覽](dt-lineage-overview.html)。
