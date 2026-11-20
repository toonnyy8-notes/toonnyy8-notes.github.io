---
slug: asink-sink-applications-kv-cache-and-streaming
title: Attention sink 在 KV cache 與串流推論的應用
cluster: Architectures & Efficiency
kind: topic
tags: [attention-sink, kv-cache, streaming, long-context, head-pruning]
arxiv: ["2602.01203", "2309.17453", "2310.01801", "2410.10819"]
sources: ["arXiv:2309.17453 摘要", "arXiv:2310.01801 摘要", "arXiv:2410.10819 摘要", "arXiv:2602.01203 §1"]
related: [asink-lineage-hub, asink-softmax-normalization-and-first-token, asink-native-moe-and-head-collapse]
---
# Attention sink 在 KV cache 與串流推論的應用

屬於 [Attention Sink 與 Native MoE 的研究脈絡總覽](asink-lineage-hub.html)。本文 §1 指出 attention sink 已廣泛用於 KV cache 最佳化、長文本生成、attention head 剪枝與量化。本筆記整理其中與本文最相關的三項。

- **StreamingLLM**（Xiao et al., 2023，arXiv:2309.17453）：摘要中的設定是串流應用（多輪對話）需要處理超過訓練長度的文字，且 KV 快取佔用大量記憶體。保留初始 token 的 KV 加上近期視窗，可恢復 window attention 的表現。
- **FastGen**（Ge et al., 2023，*Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs*，arXiv:2310.01801）：本文 §1 把它列為 KV cache 最佳化的引用之一；此處不再引述其方法細節，因為本文沒有描述，且尚未逐一核對其摘要與 sink 的關係。
- **DuoAttention**（Xiao et al., 2024，arXiv:2410.10819）：摘要指出只有一部分 attention head（Retrieval Heads）對長文本處理是關鍵，需要對所有 token 做完整 attention；其餘 head（Streaming Heads）主要關注近期 token 與 attention sink，只需固定長度的 KV cache。本文 §1 引用它作為「只有固定一部分 head 有貢獻」的先前觀察。

這些工作把 sink 當成可利用的現象；本文則追問 sink 在 head 層級的功能，見 [Sink 即 native MoE 與 head collapse](asink-native-moe-and-head-collapse.html)。sink 為何出現見 [softmax 正規化與第一個 token](asink-softmax-normalization-and-first-token.html)。
