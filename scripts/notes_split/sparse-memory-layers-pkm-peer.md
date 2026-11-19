---
slug: sparse-memory-layers-pkm-peer
title: 稀疏記憶層：PKM、PEER 與 UltraMem
cluster: Architectures & Efficiency
kind: topic
tags: [memory-layer, product-key, peer, ultramem]
sources: ["arXiv 2601.07372 全文（§1、§7 Related Work）"]
arxiv: ["2601.07372", "1907.05242", "2407.04153", "2411.12364", "2508.18756"]
related: ["engram-lineage-overview", "engram-module-design"]
---
# 稀疏記憶層：PKM、PEER 與 UltraMem

屬於 [Engram 條件式記憶的研究脈絡總覽](engram-lineage-overview.html)。論文 §7 將「參數式記憶」定義為把大規模稀疏 key-value 儲存直接整合進模型層，使容量增加而 FLOPs 幾乎不變。

- **PKM**（Lample, Sablayrolles, Ranzato, Denoyer, Jégou, 2019，arXiv:1907.05242，*Large Memory Layers with Product Keys*）：論文列為此類方法的起點。
- **PEER**（He, 2024，arXiv:2407.04153，*Mixture of A Million Experts*）：同被列為參數式記憶方法。
- **UltraMem**（Huang et al., 2025，arXiv:2411.12364，*Ultra-Sparse Memory Network*）與 **UltraMemV2**（arXiv:2508.18756，*Memory Networks Scaling to 120B Parameters with Superior Long-Context Learning*）：論文列為同類方法。
- Selfmem 與 Memory+ 也在同一列舉中，因未逐一驗證，本筆記不展開。

## 與 Engram 的差別

上述方法以學習到的 key 去比對候選項再取 value；Engram 的位址由局部 N-gram 經雜湊直接決定，不需要計算比對，因此位址在執行前已知，可以預先從主機記憶體取回。這點在論文的系統設計（§2.5）中用來重疊通訊與計算。此差異是依論文對 Engram 確定性定址的描述所作的整理，論文並未逐項比較上述方法的定址機制。
