---
slug: asink-expert-collapse-and-load-balancing
title: Expert collapse 與 MoE 的 load balancing
cluster: Architectures & Efficiency
kind: topic
tags: [moe, expert-collapse, load-balancing, shared-experts, deepseekmoe]
arxiv: ["2602.01203", "1701.06538", "2101.03961", "2401.06066"]
sources: ["arXiv:2602.01203 §1、§4.2", "arXiv:1701.06538 摘要", "arXiv:2101.03961 摘要", "arXiv:2401.06066 摘要"]
related: [asink-lineage-hub, asink-native-moe-and-head-collapse, moe-conditional-computation-lineage, llmmem-moe-as-conditional-memory]
---
# Expert collapse 與 MoE 的 load balancing

屬於 [Attention Sink 與 Native MoE 的研究脈絡總覽](asink-lineage-hub.html)。MoE 本身（條件式計算、gating 的歷史）已在 [MoE 與條件式計算的發展](moe-conditional-computation-lineage.html) 說明，這裡只整理本文引用的部分。

- **Shazeer et al., 2017**，*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*（arXiv:1701.06538）與 **Fedus et al., 2021**，*Switch Transformers*（arXiv:2101.03961）：本文 §1 引用兩者，說明 expert collapse：模型反覆依賴少數主導專家。§4.2 說明本文的輔助 load balancing loss 由兩者的 MoE load balancing 技術啟發。
- **DeepSeekMoE**（Dai et al., 2024，arXiv:2401.06066）：本文 §4.2 說明對既有模型微調時，借用其 shared experts 的做法。

本文的對應：

- 從頭訓練時，以每層 head importance 的變異係數（CV）作為損失，$L_{aux}=\lambda\sum_{l}N_h\,\mathrm{CV}(\{\mathrm{Imp}^{l,h}\}_h)^2$，總目標為 $L=L_{base}+L_{aux}$。
- 微調既有模型時，直接套用上式會造成本文所稱的 head pinning effect：預訓練中已塌縮的 head 成為生成所必需，其 gate 難以改變，其他 head 反而提高自己的 gate 去追上，無法達成平衡。因此把最活躍的 head 視為 shared heads，只對其餘 routed heads 施加 load balancing。

這些設計的前提是 attention head 可被視為 MoE 專家，見 [Sink 即 native MoE 與 head collapse](asink-native-moe-and-head-collapse.html)。MoE 作為記憶結構的觀點見 [MoE 作為條件式記憶](llmmem-moe-as-conditional-memory.html)。
