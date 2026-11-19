---
slug: ffn-as-key-value-memory
title: FFN 作為 key-value 記憶體與深度使用效率
cluster: Architectures & Efficiency
kind: topic
tags: [knowledge-storage, ffn, interpretability, depth]
sources: ["arXiv 2601.07372 全文（§1、§7 Related Work）"]
arxiv: ["2601.07372", "2012.14913", "2104.08696", "2505.13898"]
related: ["engram-lineage-overview", "stages-of-inference-weight-sharing"]
---
# FFN 作為 key-value 記憶體與深度使用效率

屬於 [Engram 條件式記憶的研究脈絡總覽](engram-lineage-overview.html)。論文 §7「Mechanisms of Knowledge Storage」整理了 Transformer 內部如何儲存與檢索事實知識。

- **Geva et al., 2021**（arXiv:2012.14913，*Transformer Feed-Forward Layers Are Key-Value Memories*）：論文描述其框架為第一層作為模式偵測器（key），第二層把特定資訊投影回殘差流（value）。
- **Dai et al., 2022**（arXiv:2104.08696，*Knowledge Neurons in Pretrained Transformers*）：論文描述為找出負責儲存個別事實的「知識神經元」。
- 論文另引用 causal tracing（Meng et al., 2022）與模型編輯方法，未逐一驗證。
- **Csordás, Manning, Potts, 2025**（arXiv:2505.13898，*Do Language Models Use Their Depth Efficiently?*）：出現在論文參考文獻中；本筆記未閱讀其摘要，不陳述其結論。

## 與本論文動機的連結

論文 §1 主張：解析一個常見的多 token 實體需要消耗多個早期層的 attention 與 FFN，等於在執行期重建靜態查表，浪費序列深度。Engram 因此把這類局部靜態模式交給查表，論文的機制分析（LogitLens、CKA）據此主張骨幹的有效深度增加。層級分工的另一種觀點見 [推論四階段假說與權重共享](stages-of-inference-weight-sharing.html)。
