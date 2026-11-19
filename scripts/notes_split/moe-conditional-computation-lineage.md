---
slug: moe-conditional-computation-lineage
title: MoE 與條件式計算的發展
cluster: Architectures & Efficiency
kind: topic
tags: [moe, conditional-computation, deepseekmoe]
sources: ["arXiv 2601.07372 全文（§1、§7 Related Work）"]
arxiv: ["2601.07372", "1701.06538", "2006.16668", "2101.03961", "2401.06066", "2412.19437"]
related: ["engram-lineage-overview", "engram-sparsity-allocation-u-shape"]
---
# MoE 與條件式計算的發展

屬於 [Engram 條件式記憶的研究脈絡總覽](engram-lineage-overview.html)。論文 §7 的描述是：MoE 以每個 token 只啟用一小部分專家的方式，把模型容量與計算成本分開。

- **Shazeer et al., 2017**，*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*（arXiv:1701.06538）：論文將其列為 MoE 範式的提出者，並用「條件式計算」描述這個稀疏軸。
- **GShard**（Lepikhin et al., 2020，arXiv:2006.16668）、**Switch Transformer**（Fedus, Zoph, Shazeer, 2021，arXiv:2101.03961）：論文列為後續創新，使參數量超線性成長而推論成本維持固定。
- **DeepSeekMoE**（Dai et al., 2024，arXiv:2401.06066）：論文描述其以細粒度專家切分與共享專家隔離，在相同啟用參數下明顯優於稠密模型。
- **DeepSeek-V3**（DeepSeek-AI, 2024，arXiv:2412.19437）：論文列為採用此架構、總參數達數百億規模的代表模型。

## 與本論文的關係

論文把 MoE 視為「條件式計算」，Engram 為互補的「條件式記憶」。實驗中的基線是嚴格同總參數、同每 token FLOPs 的 MoE 模型（Engram-27B 對 MoE-27B）。把一部分 MoE 專家的參數預算改配給 Engram 嵌入槽位的比例實驗見 [稀疏度配置與 U 形縮放律](engram-sparsity-allocation-u-shape.html)。

來源說明：以上對各作品的描述來自本論文 §1 與 §7 的敘述；未另行閱讀各作品摘要。
