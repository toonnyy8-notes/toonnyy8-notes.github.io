---
slug: engram-module-design
title: Engram 模組的設計
cluster: Architectures & Efficiency
kind: topic
tags: [engram, hashing, gating, systems]
sources: ["arXiv 2601.07372 全文（§1、§7 Related Work）"]
arxiv: ["2601.07372", "1607.04606", "1709.03933"]
related: ["engram-lineage-overview", "ngram-hash-embedding-lineage", "sparse-memory-layers-pkm-peer", "retrieval-augmented-external-memory"]
---
# Engram 模組的設計

屬於 [Engram 條件式記憶的研究脈絡總覽](engram-lineage-overview.html)。以下依論文 §2 整理，數字照原文。

1. **Tokenizer compression**：預先計算滿射函數 P: V→V'，依正規化文字等價（NFKC、小寫等）把 token ID 合併；對 128k tokenizer 有效詞表縮小 23%。N-gram 由壓縮後的 ID 組成後綴 g(t,n)。
2. **Multi-head hashing**：每個 N-gram 階數 n 使用 K 個雜湊頭，各自以輕量的 multiplicative-XOR 雜湊映射到質數大小的嵌入表；所有取回的向量串接成記憶向量 e(t)。雜湊做法承襲 [hash 嵌入研究](ngram-hash-embedding-lineage.html)。
3. **Context-aware gating**：以當前隱藏狀態 h(t) 為 query，e(t) 經投影為 key 與 value；RMSNorm 後計算純量閘門 α(t)∈(0,1)。若取回的記憶與上下文矛盾，α 趨近 0，抑制雜湊碰撞或多義造成的雜訊。
4. **Depthwise causal convolution**：核大小 4，dilation 為最大 N-gram 階數，加 SiLU 與殘差。
5. **整合**：以殘差連接加到隱藏狀態，之後接 Attention 與 MoE；並非每一層都放置，位置依效果與系統延遲取捨。與 multi-branch（mHC）骨幹的整合見論文 §2.4。
6. **系統面**：訓練時嵌入表分片到多張 GPU，以 All-to-All 取回；推論時表格放在主機記憶體，利用確定性位址非同步預取，與前面區塊的計算重疊。論文 Table 4 報告把 100B 參數嵌入表卸載到 CPU，在 8B 稠密骨幹上吞吐量損失最多 2.8%。

與其他記憶方法的比較見 [稀疏記憶層](sparse-memory-layers-pkm-peer.html) 與 [檢索增強與外部記憶](retrieval-augmented-external-memory.html)。
