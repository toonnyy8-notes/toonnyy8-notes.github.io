---
slug: asink-mechanism-studies-value-drain-over-mixing
title: Attention sink 的成因研究：value drain、over-mixing、massive activation
cluster: Architectures & Efficiency
kind: topic
tags: [attention-sink, value-drain, massive-activation, interpretability]
arxiv: ["2602.01203", "2306.12929", "2410.13835", "2410.10781", "2504.02732", "2510.06477"]
sources: ["arXiv:2306.12929 摘要", "arXiv:2410.13835 摘要", "arXiv:2410.10781 摘要", "arXiv:2504.02732 摘要", "arXiv:2510.06477 摘要", "arXiv:2602.01203 §1、§3.1、附錄 A"]
related: [asink-lineage-hub, asink-native-moe-and-head-collapse, asink-gated-attention-explicit-gating]
---
# Attention sink 的成因研究：value drain、over-mixing、massive activation

屬於 [Attention Sink 與 Native MoE 的研究脈絡總覽](asink-lineage-hub.html)。本文 §1 說明「近期工作已研究 attention sink 並解釋其成因」，並列出下列四篇；Bondarenko et al. 則出現在量化與 gating 的脈絡。

- **Bondarenko et al., 2023**，*Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing*（arXiv:2306.12929）：摘要指出 Transformer 的激活有強離群值而難以量化，且離群值與 attention head 的特定行為有關，這些 head 試圖學會「不更新」殘差。本文把它列為 gating 機制的最早引用。其後的 gating 設計見 [Gated Attention](asink-gated-attention-explicit-gating.html)。
- **Guo et al., 2024**，*Active-Dormant Attention Heads*（arXiv:2410.13835）：摘要把 attention sink、value-state drain、residual-state peak 合稱 extreme-token phenomena，並在簡單模型上辨識出 active-dormant 機制：head 在不同輸入下於 active 與 dormant 之間切換。
- **Gu et al., 2024**，*When Attention Sink Emerges in Language Models: An Empirical View*（arXiv:2410.10781）：摘要指出 sink 普遍存在於各種語言模型，且在預訓練過程中隨有效最佳化與足夠資料而出現。本文附錄 A 採用其 head-wise sink attention ratio 定義。
- **Barbero et al., 2025**，*Why do LLMs attend to the first token?*（arXiv:2504.02732）：摘要的論點是 sink 讓 LLM 避免 over-mixing，並討論 context 長度、深度與資料 packing 的影響。
- **Queipo-de-Llano et al., 2025**，*Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin*（arXiv:2510.06477；本文列為 de Llano et al.）：摘要把 attention sink 與 compression valley 追溯到 residual stream 的 massive activation。

本文承接的核心是 value drain：第一個 token 的 value 向量範數很小，使它吸收多餘注意力而不改變輸出（本文引用 Guo、Gu、Barbero）。附錄 A 以把第一個 token 的 value 設為零的實驗檢驗：對高 sink 的 head 選擇性歸零，表現與原模型相當；對全部 head 歸零則部分任務明顯下降。此現象是 [Sink 即 native MoE 與 head collapse](asink-native-moe-and-head-collapse.html) 中隱式 gate 成立的前提。
