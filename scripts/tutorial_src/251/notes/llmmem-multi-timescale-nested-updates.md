---
slug: llmmem-multi-timescale-nested-updates
title: 多時間尺度與巢狀更新：Nested Learning 與穩定性–可塑性取捨
cluster: Architectures & Efficiency
kind: topic
tags: [nested-learning, multi-timescale, stability-plasticity, continual-learning]
arxiv: ["2607.25380", "2512.24695", "2501.00663", "2512.23675", "1610.06258", "2509.06100", "2605.08949"]
sources: ["arXiv:2607.25380 §IV-D, §IV-E, §VI-B"]
related: ["llmmem-explicit-parameterized-test-time-memory", "llmmem-hybrid-management-evaluation", "llmmem-taxonomy-overview", "titans-lineage-overview"]
---
# 多時間尺度與巢狀更新

綜述（arXiv:2607.25380 §IV-D）的觀察是：顯式記憶的參數與主幹權重結構上分離，因此更新排程不必與訓練步或推論步一致，不同參數子集可以在不同時間頻率演化。

## 層次

1. 短時間尺度記憶：Titans（arXiv:2501.00663）逐 token 以 surprise 訊號調整記憶模組；TTT-E2E（arXiv:2512.23675）在推論時做批次式參數更新。兩者的更新頻率遠高於預訓練，論文稱為短時間尺度的顯式記憶：持久的參數在一個推論 episode 內動態演化。
2. 巢狀與階層式排程：Nested Learning（Behrouz、Razaviyayn、Zhong、Mirrokni，arXiv:2512.24695）讓不同參數子集採不同學習率與更新排程，形成快慢元件的分層，論文描述為時間可塑性的連續譜，不是主幹與記憶的二分。
3. 慢–快參數分解：與早期 fast–slow weights（如 arXiv:1610.06258）相連。把高頻更新限制在少數參數，可降低災難性干擾，同時仍累積情境資訊。

## 穩定性–可塑性

頻繁更新提高適應力但容易對暫態訊號過擬合，緩慢更新保持穩定但反應慢。論文指出顯式線上記憶必須調節更新的幅度、頻率與範圍；巢狀策略是一條可行路徑，但使收斂分析更複雜。另列出跨 episode 的一致性問題（記憶何時重置、長短期儲存如何分開），並指出與持續學習的關聯。

## 幾何約束

論文雖不把持續學習優化器視為主要記憶架構，仍引用 Lie 群上的正交低秩適應（arXiv:2509.06100）與 Muon-OGD（arXiv:2605.08949），說明當更多架構引入可寫入的參數子集，更新規則的幾何可能與儲存形式同樣重要。

此主題把 [參數化外顯記憶](llmmem-explicit-parameterized-test-time-memory.html) 的更新時序推廣成一條獨立的設計維度，並銜接 [混合架構、記憶管理與評估](llmmem-hybrid-management-evaluation.html) 中的記憶階層（token 層級的激活狀態、互動或批次層級的顯式記憶、緩慢或凍結的主幹）。Titans 的來源脈絡見 [Titans 的研究脈絡總覽](titans-lineage-overview.html)。

## 相關筆記

- [LLM 記憶機制的分類總覽：從 attention 與遞迴狀態到可定址的外顯記憶](llmmem-taxonomy-overview.html)
