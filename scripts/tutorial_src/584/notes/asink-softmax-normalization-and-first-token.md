---
slug: asink-softmax-normalization-and-first-token
title: softmax 正規化與第一個 token 的高注意力
cluster: Architectures & Efficiency
kind: topic
tags: [softmax-attention, attention-sink, transformer]
arxiv: ["2602.01203", "1706.03762", "2309.17453"]
sources: ["arXiv:2602.01203 §2.1", "arXiv:1706.03762 摘要", "arXiv:2309.17453 摘要"]
related: [asink-lineage-hub, asink-sink-applications-kv-cache-and-streaming, asink-native-moe-and-head-collapse]
---
# softmax 正規化與第一個 token 的高注意力

屬於 [Attention Sink 與 Native MoE 的研究脈絡總覽](asink-lineage-hub.html)。

**Vaswani et al., 2017**，*Attention Is All You Need*（arXiv:1706.03762）：提出以 attention 為主體的 Transformer 架構。本文 §2.1 使用的是其中的 softmax attention：第 l 層第 h 個 head 在時間點 t 對 token j 的權重為

$A_{t,j} = \mathrm{Softmax}(q_t^\top k_j / \sqrt{d_h})$，輸出 $O_t = \sum_j A_{t,j}\, v_j$。

softmax 使同一個 query 對所有 token 的權重加總為 1。本文 §2.1 的說明是：當某個 head 只需要部分注意力或完全不需要注意力時，多餘的權重仍必須分配出去；模型把這些多餘的權重導向第一個 token，因為該 token 的 value 向量因語意內容極少而接近零。這些權重滿足正規化限制，但對輸出沒有實質貢獻。本文把這個現象稱為 attention sink，並引用 Xiao et al. (2023) 與 Miller (2023) 作為「由 softmax 正規化造成」的依據。

**Xiao et al., 2023**，*Efficient Streaming Language Models with Attention Sinks*（arXiv:2309.17453）：摘要指出，window attention 在文字長度超過 cache 大小時會失效；保留初始 token 的 KV 即可大幅恢復表現，作者將模型對初始 token 給予強注意力分數的現象稱為 attention sink，即使這些 token 在語意上不重要。本文 §1 以此為 attention sink 的起點。

Miller (2023) 為部落格文章（Attention is off by one），不是 arXiv 論文，未在此驗證其內容；此處僅記錄本文把它列為 softmax 正規化成因的引用。

接續：[Attention sink 在 KV cache 與串流推論的應用](asink-sink-applications-kv-cache-and-streaming.html)；本文如何把這個 sink 權重重新解讀為 gate，見 [Sink 即 native MoE 與 head collapse](asink-native-moe-and-head-collapse.html)。
