---
slug: engram-lineage-overview
title: Engram 條件式記憶的研究脈絡總覽
cluster: Architectures & Efficiency
kind: hub
tags: [engram, conditional-memory, moe, overview]
sources: ["arXiv 2601.07372 全文（§1、§7 Related Work）"]
arxiv: ["2601.07372", "1701.06538", "1607.04606", "1907.05242"]
related: ["moe-conditional-computation-lineage", "ngram-hash-embedding-lineage", "sparse-memory-layers-pkm-peer", "retrieval-augmented-external-memory", "ffn-as-key-value-memory", "engram-module-design", "engram-sparsity-allocation-u-shape", "stages-of-inference-weight-sharing"]
---
# Engram 條件式記憶的研究脈絡總覽

本文整理理解《Conditional Memory via Scalable Lookup》（arXiv:2601.07372，DeepSeek-AI 與北京大學，提出 Engram 模組）所需的前置研究。論文的主張是：Mixture-of-Experts（MoE）以條件式計算擴充容量，但 Transformer 沒有原生的知識查表機制，只能用計算去模擬檢索；因此論文提出「條件式記憶」作為第二個稀疏軸，用 N-gram 嵌入表的 O(1) 查找實作。

## 建議閱讀順序

1. [MoE 與條件式計算的發展](moe-conditional-computation-lineage.html)：論文要補足的「第一個稀疏軸」，以及實驗用的 MoE 基線來源。
2. [FFN 作為 key-value 記憶體的研究](ffn-as-key-value-memory.html)：說明為何「模型用計算重建靜態知識」是論文的動機，以及早期層的作用。
3. [N-gram 與 hash 嵌入的發展](ngram-hash-embedding-lineage.html)：Engram 查表機制直接繼承的技術（FastText、hash embedding、N-Grammer、OverEncoding、SCONE、BLT）。
4. [稀疏記憶層：PKM、PEER 與 UltraMem](sparse-memory-layers-pkm-peer.html)：另一條「在層內加大型 key-value 儲存」的路線，與 Engram 的定址方式不同。
5. [檢索增強與外部記憶](retrieval-augmented-external-memory.html)：非參數記憶（REALM、RETRO），與 Engram 的參數式、確定性定址相對照。
6. [Engram 模組的設計](engram-module-design.html)：把上述元件組合起來的做法。
7. [稀疏度配置與 U 形縮放律](engram-sparsity-allocation-u-shape.html)：論文用來決定 MoE 與 Engram 參數比例的實驗。

## 與既有筆記的關係

論文的機制分析指出 Engram 讓骨幹的早期層不必重建靜態知識，增加可用於推理的有效深度。層級功能分工的相關討論見既有筆記 [推論四階段假說與權重共享](stages-of-inference-weight-sharing.html)。

## 時間軸

年代與作品見同資料夾的 timeline.json；從 2016 年的 FastText 與 2017 年的稀疏門控 MoE 開始，到 2026 年的本論文結束。
