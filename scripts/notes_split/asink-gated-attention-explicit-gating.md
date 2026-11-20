---
slug: asink-gated-attention-explicit-gating
title: Gated Attention：head-specific 的顯式 gating
cluster: Architectures & Efficiency
kind: topic
tags: [gated-attention, sigmoid-gate, qwen3-next, attention-sink]
arxiv: ["2602.01203", "2505.06708", "2306.12929", "2505.09388"]
sources: ["arXiv:2505.06708 摘要", "arXiv:2602.01203 §1、§2.2、Eq. (2)", "arXiv:2306.12929 摘要"]
related: [asink-lineage-hub, asink-eliminating-sink-softmax-modification, asink-mechanism-studies-value-drain-over-mixing, asink-native-moe-and-head-collapse, linear-attention-fast-weights]
---
# Gated Attention：head-specific 的顯式 gating

屬於 [Attention Sink 與 Native MoE 的研究脈絡總覽](asink-lineage-hub.html)。

**Qiu et al., 2025**，*Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*（arXiv:2505.06708）：摘要指出 gating 已用於 LSTM、Highway Network、狀態空間模型、線性注意力與 softmax attention，但鮮少有文獻檢驗其效果。作者比較 30 種 gating 變體（15B MoE 模型與 1.7B dense 模型，3.5T token），中心發現是在 SDPA 之後套用 head-specific sigmoid gate 能穩定提升表現，並改善訓練穩定性。本文列此為 Qwen3-Next 所採用的 Gated Attention。

本文 §1 把另一類方法（修改 softmax）整理於 [以修改 softmax 消除 sink](asink-eliminating-sink-softmax-modification.html)。

本文 §2.2 的描述包含兩項設計：每個 head 一個 gating 因子，以及 sigmoid 啟用。第 l 層第 h 個 head 的 gate 為

$G_t^{l,h} = \sigma(x_t^l W_\theta^{l,h})$，$W_\theta^{l,h}\in\mathbb{R}^{d_{model}\times 1}$，

層輸出為 $O_t^l = \sum_h G_t^{l,h}\, O_t^{l,h} W_O^{l,h}$。本文指出這等於在 attention 層引入 MoE 結構，使 head 稀疏地被使用，並消除 sink。

**Bondarenko et al., 2023**（arXiv:2306.12929）被本文列為 gating 類方法的先驅引用，見 [成因研究](asink-mechanism-studies-value-drain-over-mixing.html)。Qwen3 技術報告（Yang et al., 2025，arXiv:2505.09388）則被本文用於實驗模型家族（附錄的 Qwen2.5 與 Qwen3 head importance 分析）；Qwen3-Next 的架構細節以本文與 Qiu et al. 的描述為準，此處未核對 Qwen3 報告。

線性注意力與 fast weights 中的 gating 見 [線性注意力與 fast weights](linear-attention-fast-weights.html)。本文把 explicit gate 與 sink 權重對應起來，見 [Sink 即 native MoE 與 head collapse](asink-native-moe-and-head-collapse.html)。
