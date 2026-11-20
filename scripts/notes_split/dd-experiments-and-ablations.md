---
slug: dd-experiments-and-ablations
title: Decision Diffuser 的實驗結果與消融
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [d4rl, ablation, decision-diffuser, baselines, kitchen]
arxiv: ["2211.15657", "2004.07219", "2006.04779", "2110.06169", "2106.01345", "2106.02039", "2205.09991", "2112.10751"]
sources: ["paper.txt §4, §5, Table 1-3, Appendix L", "arXiv API metadata"]
related: ["decision-diffuser-lineage-hub", "offline-rl-benchmarks-and-baselines", "sequence-modeling-offline-rl", "return-conditioned-sequence-modeling", "trajectory-diffusion-planning", "inverse-dynamics-state-diffusion", "classifier-free-guidance-return-conditioning"]
---
# Decision Diffuser 的實驗結果與消融

屬於[總覽](decision-diffuser-lineage-hub.html)。基準與基線的一般說明見[離線 RL 基準與基線](offline-rl-benchmarks-and-baselines.html)。

## 比較對象
論文比較 BC、CQL（2006.04779）、IQL（2110.06169）、Trajectory Transformer（2106.02039）、MoReL、Decision Transformer（2106.01345）與 Diffuser（2205.09991），資料集為 D4RL（2004.07219）。

## 表 1（正規化回報）
- Locomotion 九個資料集平均：BC 51.9、CQL 77.6、IQL 77、DT 74.7、TT 78.9、MOReL 72.9、Diffuser 75.3、DD 81.8。
- Kitchen 平均：BC 44.8、CQL 51.2、IQL 48.7、DD 61（Mixed 65±2.8、Partial 57±2.5）。論文指出 DD 的差距在需要長期信用分配的 Kitchen 任務更明顯。
- 結果為 5 個隨機種子的平均與標準誤。

## 表 2（Hopper 消融）
Diffuser、CondDiffuser、CondMLPDiffuser、DD 的 Med-Expert 分別為 107.6、111.3、105.6、111.8；Medium 為 58.5、66.3、54.1、79.3；Med-Replay 為 96.8、76.5、66.5、100。無分類器引導使 CondDiffuser 在 3 個環境中的 2 個優於 Diffuser；逆動力學使 DD 在 3 個環境都最佳（見[逆動力學](inverse-dynamics-state-diffusion.html)與[無分類器引導](classifier-free-guidance-return-conditioning.html)）。

## 與報酬條件策略的關係（§5、附錄 L）
Decision Transformer 建模給定目標回報與過去軌跡的下一動作概似（見[序列建模離線 RL](sequence-modeling-offline-rl.html)與[報酬條件的監督式學習](return-conditioned-sequence-modeling.html)）；RvS（Emmons 等，arXiv 2112.10751）不用 Transformer，但依賴容量調校（依本論文描述）。論文指出 DD 建模未來狀態軌跡的分數函數，並可組合條件變數。

其餘實驗（限制、技能）見[組合限制與技能](composing-constraints-and-skills.html)。
