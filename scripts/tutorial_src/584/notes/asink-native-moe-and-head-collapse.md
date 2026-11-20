---
slug: asink-native-moe-and-head-collapse
title: Sink 即 native MoE、head collapse 與 sink-aware load balancing
cluster: Architectures & Efficiency
kind: topic
tags: [attention-sink, native-moe, head-collapse, load-balancing, gated-attention]
arxiv: ["2602.01203", "2309.17453", "2410.10819", "2505.06708", "1701.06538", "2101.03961"]
sources: ["arXiv:2602.01203 摘要、§1、§3.1、§4.1、§4.2、§6、§7"]
related: [asink-lineage-hub, asink-softmax-normalization-and-first-token, asink-sink-applications-kv-cache-and-streaming, asink-mechanism-studies-value-drain-over-mixing, asink-eliminating-sink-softmax-modification, asink-gated-attention-explicit-gating, asink-expert-collapse-and-load-balancing, moe-conditional-computation-lineage]
---
# Sink 即 native MoE、head collapse 與 sink-aware load balancing

屬於 [Attention Sink 與 Native MoE 的研究脈絡總覽](asink-lineage-hub.html)。本筆記只記錄 Fu, Zeng, Wang, Li（arXiv:2602.01203，ICML 2026）在前述脈絡上新增的部分，內容取自該論文。

**1. 三種 attention 的統一觀點（§3.1）。** Vanilla Attention 的第一個 token、Sink Attention 的 sink 參數，其上的注意力權重 $A_{sink}$ 扮演隱式 gating 因子，對應 Gated Attention 的顯式 gate。論文用同一個 gate 記號 $G^{l,h}_{t}$：Gated Attention 為 sigmoid gate（Eq. 2）；Vanilla 與 Sink Attention 為 $1-A_{sink}$。Sink Attention 的 gate 寫成 $\sigma(\mathrm{LSE}-\mathrm{sink})$。每個 head 因此像 MoE 的專家，gate 決定該 head 的貢獻。前置條件見 [softmax 正規化與第一個 token](asink-softmax-normalization-and-first-token.html) 與 [成因研究](asink-mechanism-studies-value-drain-over-mixing.html)（value 接近零）。

**2. 消除 sink 的好處（§3.2）。** 論文主張 Sink Attention 與 Gated Attention 在消除 sink 後，softmax 能更靈活、更精確地檢索相關 token，對長文本特別重要；Vanilla 的 sink 使 key 的分布受限，query-key 內積差異被壓縮。相關方法見 [修改 softmax](asink-eliminating-sink-softmax-modification.html) 與 [Gated Attention](asink-gated-attention-explicit-gating.html)。

**3. Head collapse（§4.1）。** 只有固定一部分 head 持續對生成有貢獻，其餘低度啟用，對應 expert collapse（見 [Expert collapse 與 load balancing](asink-expert-collapse-and-load-balancing.html) 與 [MoE 與條件式計算的發展](moe-conditional-computation-lineage.html)）。head 的重要性分數定義為 batch 內 gate 的平均，$\mathrm{Imp}^{l,h}=\frac{1}{|T|}\sum_{(i,t)\in T}G^{l,h}_{i,t}$。論文稱 head collapse 在三種 attention 皆出現，訓練初期不平衡度上升後在高位持平。類似觀察見 [DuoAttention](asink-sink-applications-kv-cache-and-streaming.html)。

**4. Sink-aware 輔助損失（§4.2）。** 從頭訓練時以重要性分數的 CV 作為輔助損失；微調時為避免 head pinning，只對 routed heads 平衡（細節見 [Expert collapse 與 load balancing](asink-expert-collapse-and-load-balancing.html)）。

**5. 限制（§6）。** 論文指出分析以第一個 token、sink 參數與顯式 gate 估計 head 啟用；其他吸收多餘注意力的 token 可能存在，Gated Attention 中也可能形成隱式 gate。
