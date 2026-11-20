---
slug: asink-lineage-hub
title: Attention Sink 與 Native MoE 的研究脈絡總覽
cluster: Architectures & Efficiency
kind: hub
tags: [attention-sink, moe, gated-attention, head-collapse, hub]
arxiv: ["2602.01203", "1706.03762", "2309.17453", "2310.01801", "2410.10819", "2306.12929", "2410.13835", "2410.10781", "2504.02732", "2510.06477", "2504.20966", "2508.10925", "2505.06708", "1701.06538", "2101.03961", "2401.06066"]
sources: ["arXiv:2602.01203 摘要、§1、§2、§3.1、§4", "各引用論文的 arXiv 摘要（見各子筆記）"]
related: [asink-softmax-normalization-and-first-token, asink-sink-applications-kv-cache-and-streaming, asink-mechanism-studies-value-drain-over-mixing, asink-eliminating-sink-softmax-modification, asink-gated-attention-explicit-gating, asink-expert-collapse-and-load-balancing, asink-native-moe-and-head-collapse, moe-conditional-computation-lineage, llmmem-moe-as-conditional-memory, linear-attention-fast-weights, llmmem-implicit-attention-access-control]
---
# Attention Sink 與 Native MoE 的研究脈絡總覽

本文（arXiv 2602.01203）的主張有三層：Vanilla Attention 與 Sink Attention 中，sink 上的注意力權重等同隱式的 gating 因子，使每個 attention head 像 MoE 的專家；既有 LLM 只有固定的一部分 head 對生成有貢獻，稱為 head collapse，對應 MoE 的 expert collapse；因此提出針對 attention 層的 sink-aware 輔助 load balancing loss。

建議閱讀順序：

1. [softmax 正規化與第一個 token](asink-softmax-normalization-and-first-token.html)：sink 為何出現的基本前提。
2. [Attention sink 在 KV cache 與串流推論的應用](asink-sink-applications-kv-cache-and-streaming.html)：StreamingLLM 與其後續。
3. [Attention sink 的成因研究](asink-mechanism-studies-value-drain-over-mixing.html)：value drain、over-mixing、massive activation。
4. [以修改 softmax 消除 sink](asink-eliminating-sink-softmax-modification.html)：Softpick 與 GPT-OSS 的 Sink Attention。
5. [Gated Attention：顯式 gating](asink-gated-attention-explicit-gating.html)：head-specific sigmoid gate。
6. [Expert collapse 與 load balancing](asink-expert-collapse-and-load-balancing.html)：MoE 端的問題與解法。
7. [Sink 即 native MoE 與 head collapse](asink-native-moe-and-head-collapse.html)：本文新增的內容，彙整上述脈絡。

已有的相關筆記：MoE 與條件式計算見 [MoE 與條件式計算的發展](moe-conditional-computation-lineage.html)；MoE 作為條件式記憶見 [MoE 作為條件式記憶](llmmem-moe-as-conditional-memory.html)；線性注意力見 [線性注意力與 fast weights](linear-attention-fast-weights.html)；attention 作為隱式記憶存取見 [隱式記憶：attention 與存取控制](llmmem-implicit-attention-access-control.html)。時間線見 `timeline.json`。
