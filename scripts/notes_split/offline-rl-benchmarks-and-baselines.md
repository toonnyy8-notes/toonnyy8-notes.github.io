---
slug: offline-rl-benchmarks-and-baselines
title: 本論文使用的離線 RL 基準與比較方法
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [offline-rl, d4rl, cql, iql, mopo, morel, mbop, her]
arxiv: ["2004.07219", "2006.04779", "2110.06169", "2005.13239", "2005.05951", "2008.05556", "1707.01495", "2205.09991"]
sources: ["paper.txt §5, Appendix A", "arXiv API metadata (all listed IDs)"]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "sequence-modeling-offline-rl", "trajectory-diffusion-planning"]
---
# 本論文使用的離線 RL 基準與比較方法

離線 RL 的共通問題（分佈偏移、保守估計）見 [離線 RL 的分佈偏移](offline-rl-distribution-shift.html)，此處只記本論文怎麼用這些工作。

## 基準資料集

D4RL（Fu, Kumar, Nachum, Tucker, Levine, 2020；2004.07219）。論文用其中兩類：Maze2D（稀疏獎勵、長時程，§5.1）與 locomotion（HalfCheetah、Hopper、Walker2d；Medium-Expert、Medium、Medium-Replay，§5.3）。分數以標準化分數表示。

## 比較方法

| 方法 | 出處 | 在論文中的類別與用途 |
| --- | --- | --- |
| CQL | Kumar et al., 2020（2006.04779） | model-free；Maze2D、方塊堆疊、locomotion 都有比較 |
| IQL | Kostrikov, Nair, Levine, 2021/2022（2110.06169） | model-free；Maze2D 單任務最強基準 |
| MOPO | Yu et al., 2020（2005.13239） | model-based offline |
| MOReL | Kidambi et al., 2020（2005.05951） | model-based offline |
| MBOP | Argenson & Dulac-Arnold, 2020（2008.05556） | model-based offline planning |
| HER | Andrychowicz et al., 2017（1707.01495） | Multi2D 時改造 IQL 的目標重標記（hindsight relabeling） |

BCQ（Fujimoto et al., 2019）與 BC、AWAC 也出現在比較中，但本筆記未在 API 核對其 arXiv 編號，不列入 `arxiv`。DT、TT 見 [以序列模型處理離線 RL](sequence-modeling-offline-rl.html)。

## 論文的主要數字（照原文）

- Maze2D（Table 1）：單任務平均 MPPI 16.2、CQL 7.7、IQL 47.0、Diffuser 119.5；多任務平均 MPPI 21.5、IQL 16.9、Diffuser 129.4。
- 方塊堆疊（Table 3）：平均 BCQ 0.0、CQL 8.1、Diffuser 54.4。
- locomotion（Table 2）：九項平均 CQL 77.6、IQL 77.0、Diffuser 77.5。

結果與設計見 [軌跡層級擴散規劃（Diffuser）](trajectory-diffusion-planning.html)。MDP 與 Q 函數的記號見 [MDP 與 Q-learning 基礎](rl-mdp-q-learning-basics.html)。
