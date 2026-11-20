---
slug: asink-eliminating-sink-softmax-modification
title: 以修改 softmax 消除 sink：Softpick 與 Sink Attention
cluster: Architectures & Efficiency
kind: topic
tags: [sink-attention, softpick, gpt-oss, softmax-variants]
arxiv: ["2602.01203", "2504.20966", "2508.10925"]
sources: ["arXiv:2504.20966 摘要", "arXiv:2602.01203 §1、§2.2、Eq. (1)", "arXiv:2508.10925 為 GPT-OSS 模型卡，僅取其書目資料"]
related: [asink-lineage-hub, asink-gated-attention-explicit-gating, asink-mechanism-studies-value-drain-over-mixing, asink-native-moe-and-head-collapse]
---
# 以修改 softmax 消除 sink：Softpick 與 Sink Attention

屬於 [Attention Sink 與 Native MoE 的研究脈絡總覽](asink-lineage-hub.html)。本文 §1 把消除 sink 的方法分成兩類：修改 softmax 運算，以及加入 gating 機制。本筆記是第一類；第二類見 [Gated Attention](asink-gated-attention-explicit-gating.html)。

- **Miller, 2023**（Attention is off by one，部落格）、**Gu et al., 2024**（見 [成因研究](asink-mechanism-studies-value-drain-over-mixing.html)）與 **Zuhri et al., 2025** 被本文歸為第一類。
- **Softpick**（Zuhri et al., 2025，arXiv:2504.20966）：摘要稱它是 rectified、不要求加總為 1 的 softmax 替代品，在 340M 與 1.8B 模型上 sink rate 為 0%，並產生較稀疏的 attention map。
- **Sink Attention（GPT-OSS）**（OpenAI 等，2025，arXiv:2508.10925，*gpt-oss-120b & gpt-oss-20b Model Card*）：本文 Eq. (1) 的描述為在 softmax 分母加入可學習參數 sink：

$\mathrm{Softmax}_{sink}(x)_i = \exp(x_i) / (\sum_j \exp(x_j) + \exp(\mathrm{sink}))$

這放寬了正規化限制，並以一個 value 為零的合成 sink 取代 vanilla attention 中的第一個 token。模型卡本身的內容未在此核對，Eq. (1) 的描述取自本文。

本文把 Sink Attention 中的 sink 參數視為隱式 gate 的一部分，見 [Sink 即 native MoE 與 head collapse](asink-native-moe-and-head-collapse.html)。
