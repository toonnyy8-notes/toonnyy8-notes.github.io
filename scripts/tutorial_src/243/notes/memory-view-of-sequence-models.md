---
slug: memory-view-of-sequence-models
title: 序列模型的記憶觀點：寫入、讀取與五個設計問題
cluster: Architectures & Efficiency
kind: topic
tags: [memory, rnn, transformer, associative-memory]
arxiv: ["2501.00663", "1706.03762", "2306.00802"]
sources: ["arXiv:2501.00663 §1 Memory Perspective, §2.1", "arXiv:2306.00802 abstract 僅用於確認題名；內容依 Titans 對它的引用"]
related: ["titans-architectures-mac-mag-mal", "titans-neural-memory-surprise-update"]
---
# 序列模型的記憶觀點

Titans 第 1 節把序列模型統一寫成兩個操作：寫入 `M_t = f(M_{t-1}, x_t)` 與讀取 `y_t = g(M_t, x_t)`（式 6、7）。

- **RNN**：記憶是向量值的隱藏狀態，每步壓縮寫入。
- **Transformer**：記憶是 key 與 value 矩陣，寫入就是附加，不壓縮；讀取是 query 與 key 的相似度加權 value。論文引用 Bietti 等人（Birth of a Transformer: A Memory Viewpoint，arXiv:2306.00802）說明 attention 可視為 key-value 聯想記憶。
- **Linear Transformer**：記憶是矩陣，寫入是壓縮的加法。

由此論文提出五個問題：(Q1) 好的記憶結構是什麼；(Q2) 適當的更新機制；(Q3) 適當的讀取方式；(Q4) 如何設計能整合多個互連記憶模組的高效架構；(Q5) 是否需要深層記憶模組來儲存長期歷史。

論文依神經心理學的區分，把 attention 視為短期記憶（視窗有限但依賴建模精確），把神經記憶視為長期記憶，並另加與資料無關的持久記憶。後續各筆記對應這些問題：Q1、Q5 見 [Titans 的神經記憶](titans-neural-memory-surprise-update.html)，Q4 見 [Titans 的三種架構](titans-architectures-mac-mag-mal.html)。
