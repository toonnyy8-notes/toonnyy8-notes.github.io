---
slug: recurrent-memory-segment-transformers
title: 分段 Transformer 與外掛記憶：Transformer-XL、RMT、Memory Layers
cluster: Architectures & Efficiency
kind: topic
tags: [segment-memory, rmt, transformer-xl, memory-layers]
arxiv: ["2501.00663", "1410.5401", "1901.02860", "2207.06881", "2304.11062", "2412.09764"]
sources: ["arXiv:2501.00663 Appendix A.2, A.3", "各論文題名與作者經 arXiv API 查證"]
related: ["titans-architectures-mac-mag-mal"]
---
# 分段 Transformer 與外掛記憶

Titans 附錄 A.2 描述另一條線：把序列切成片段（chunk），以記憶在片段間傳遞資訊。

- **Neural Turing Machines**（arXiv:1410.5401，Graves、Wayne、Danihelka，2014）：Titans 列為早期把記憶納入神經網路設計的例子。
- **Transformer-XL**（arXiv:1901.02860，Dai 等，2019）：片段式 Transformer 的代表，論文指出片段彼此分離，上下文受限於片段長度。
- **Recurrent Memory Transformer**（arXiv:2207.06881，Bulatov、Kuratov、Burtsev，2022）及 Scaling Transformer to 1M tokens and beyond with RMT（arXiv:2304.11062，2023）：以記憶向量跨片段傳遞。
- **Memory Layers at Scale**（arXiv:2412.09764，Berges 等，2024）：論文列為可替換 MLP 記憶的其他記憶架構。

Titans 對這類模型的比較：(1) 記憶是小的向量，表達力不足；(2) 沒有遺忘機制，記憶快速溢出；(3) 只依當下 surprise，忽略資訊流。並說明 Titans (MAC) 可視為 RMT 的推廣，差別是以神經記憶模組取代向量記憶。見 [Titans 的三種架構](titans-architectures-mac-mag-mal.html)。
