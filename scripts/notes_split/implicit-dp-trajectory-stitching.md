---
slug: implicit-dp-trajectory-stitching
title: 不用價值函數的軌跡拼接（隱式動態規劃）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [trajectory-stitching, dynamic-programming, td-learning, deadly-triad, offline-rl]
arxiv: ["2211.15657", "2004.07219", "2006.04779", "2110.06169", "2005.01643"]
sources: ["paper.txt §1, §2.1, §3.2, Appendix A.1, Appendix K", "arXiv API metadata"]
related: ["decision-diffuser-lineage-hub", "rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "offline-rl-benchmarks-and-baselines", "classifier-free-guidance-return-conditioning", "expectile-regression-value-learning"]
---
# 不用價值函數的軌跡拼接（隱式動態規劃）

屬於[總覽](decision-diffuser-lineage-hub.html)。

## 問題設定
離線決策的目標是把資料集中次佳的軌跡拼接成回報最大的軌跡。論文 §1 指出既有離線 RL 以動態規劃做拼接，需要學習價值函數（TD 損失見[MDP 與 Q-learning 基礎](rl-mdp-q-learning-basics.html)），而函數近似、off-policy 學習與 bootstrapping 並用會造成不穩定（論文稱為 deadly triad，引 Sutton & Barto 2018）。為穩定離線的價值估計，這些方法還需要讓策略貼近資料分布的啟發式（見[分布偏移](offline-rl-distribution-shift.html)，代表方法如 CQL 2006.04779、IQL 2110.06169，後者見[期望分位數回歸](expectile-regression-value-learning.html)）。

## DD 的主張
DD 不估計任何 Q 函數；論文 §2.1 的說法是生成模型以最大概似訓練，因此不會有狀態訪問分布偏移的風險。論文假設無分類器引導加低溫取樣會隱式執行動態規劃，從資料中取出最佳行為（這是假設，附錄 A 以例子支持）。

## 附錄 A.1 的例子
Maze2D-open（D4RL，Fu 等，arXiv 2004.07219）：目標為到達 C 點，獎勵為到 C 的負距離。資料集為 500 條 A→B 軌跡與 500 條 B→C 軌跡，最大長度 50。測試時從 A 出發，DD 生成由 A 到 C 近似直線的軌跡，論文據此稱它能拼接資料集中的軌跡。

## 說明
論文以「假設」與單一玩具環境的例子陳述隱式動態規劃；本筆記不補充論文以外的理論。
