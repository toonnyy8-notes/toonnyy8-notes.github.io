---
slug: ngram-hash-embedding-lineage
title: N-gram 與 hash 嵌入的發展
cluster: Architectures & Efficiency
kind: topic
tags: [ngram, hash-embedding, embedding-scaling, fasttext]
sources: ["arXiv 2601.07372 全文（§1、§7 Related Work）"]
arxiv: ["2601.07372", "1607.04606", "1709.03933", "2207.06366", "2501.16975", "2502.01637", "2412.09871", "2601.10639"]
related: ["engram-lineage-overview", "engram-module-design"]
---
# N-gram 與 hash 嵌入的發展

屬於 [Engram 條件式記憶的研究脈絡總覽](engram-lineage-overview.html)。Engram 的查表部分繼承自下列研究。

- **N-gram 語言模型**：論文 §7 追溯到 Shannon（1948），以局部歷史預測下一個 token，傳統上用平滑方法（Katz, 1987；Kneser and Ney, 1995）處理資料稀疏。
- **FastText**（Bojanowski et al., 2017，arXiv:1607.04606，*Enriching Word Vectors with Subword Information*）：論文 §1 以其作為 N-gram 嵌入的典型例子，並指出 N-gram 查找的計算效率在現代表示學習中被保留。
- **Hash Embeddings**（Tito Svenstrup, Hansen, Winther, 2017，arXiv:1709.03933）：Engram 的多頭雜湊（multi-head hashing）明確寫「Following Tito Svenstrup et al. (2017)」。
- **N-Grammer**（Roy et al., 2022，arXiv:2207.06366）：論文描述為直接用離散潛在文字表示建構的 N-gram 來擴充 Transformer。
- **OverEncoding / Over-Tokenized Transformer**（Huang et al., 2025，arXiv:2501.16975）：以 hash N-gram 嵌入在 token 層級捕捉局部相依；論文指出它在稀疏 MoE 骨幹上即使不限定同參數也沒有明顯改善，且嵌入只放在第 0 層。
- **SCONE**（*Scaling Embedding Layers in Language Models*，Yu et al., 2025，arXiv:2502.01637）：論文描述為以輔助編碼模型處理高頻模式，偏重推論，且有額外訓練 FLOPs。
- **Byte Latent Transformer**（Pagnoni et al., 2024，arXiv:2412.09871）：以 hash N-gram 嵌入處理位元組層級。
- **STEM**（Sadhukhan et al., 2026，arXiv:2601.10639）：論文列為以大型表格擴充容量的近期架構之一，與 Per-Layer Embeddings、L3、DeepEmbed 並列。

## 本論文的差異

論文 §7 指出兩點：一是把 N-gram 嵌入當成一等的建模原語，並在嚴格同參數、同 FLOPs 的 MoE 基線下評估；二是系統面上將記憶注入較深的層，使通訊與計算可以重疊，而前述方法多把嵌入放在輸入層。具體模組見 [Engram 模組的設計](engram-module-design.html)。

來源說明：描述取自本論文 §1、§7；標題、作者、年份與 arXiv ID 已對照 arxiv.org 驗證。
