---
slug: depth-nonuniform-lora-rank
title: 深度非均勻 LoRA rank 分配 × 遞迴模型：研究空白與方向
cluster: Architectures & Efficiency
kind: topic
tags: [lora, rank-allocation, research-gap, recursive-transformer, interpretability]
sources: [weight-sharing-lora.md]
arxiv: ["2607.19391", "2607.09757", "2604.11791"]
related: ["weight-sharing-recursive-overview", "relaxed-recursive-transformer", "stages-of-inference-weight-sharing"]
---
# 深度非均勻 LoRA rank 分配 × 遞迴模型：研究空白與方向

屬於 [層間權重共享總覽](weight-sharing-recursive-overview.html)。

## 研究空白確認
- 在通用 LoRA 微調領域，已有多篇證明「均勻 rank 次佳、rank 應隨深度非均勻」：AdaLoRA、LAARA（Fisher 資訊導向，arXiv:2607.19391）、RSLoRA（表徵敏感度探測，arXiv:2607.09757，明確觀察到「depth-dependent sensitivity transition」）、PARA、La-LoRA、ARD-LoRA 等。
- **但在「遞迴/共享 transformer 的層級 LoRA 鬆綁」脈絡下，尚無論文以可解釋性（四階段/U 形）為依據做深度非均勻 rank 分配。** [RRT](relaxed-recursive-transformer.html) 本身用的是各 loop 均勻 rank；其 component-wise 消融（Table G.1）僅比較 Q/KV/Out/FFN 模組別（發現砍 FFN rank 傷害最大、砍 KV 最小），並未做「首尾大 rank、中間小 rank」的深度別實驗。**這是明確且高價值的研究空白。**

## 方向一：深度非均勻的 rank 分配（可解釋性導向）
- 依據：Lad et al. U 形敏感度（見 [四階段假說](stages-of-inference-weight-sharing.html)）+ 通用 LoRA 的 depth-dependent 敏感度（RSLoRA）。
- 已有證據：RRT 的 Stepwise（固定首尾）間接支持；component-wise 消融顯示模組別非均勻有效。
- 假設：在遞迴模型中，對應「首/尾 loop」或「detokenization/residual sharpening 階段」的層給更大 rank（甚至不共享），中間 feature-engineering 階段高度共享 + 低 rank，可在同參數預算下優於均勻 rank。
- 風險：Stepwise+SVD 已出現中 rank U 形退化，非均勻分配可能與 SVD 初始化交互產生不穩；需小心 rank–初始化耦合。

## 方向二：可解釋性導向的共享策略設計（元方向）
- 依據：mechanistic 分析已證明 looped 模型重現四階段。
- 假設：以「階段邊界」（attention→MLP 相變、prediction/suppression neuron 出現）為切點決定 block 邊界與共享粒度，而非機械地等分層。
- 風險：階段邊界跨模型不完全一致、需逐模型探測；目前無端到端驗證，屬推測性最高的方向。

## 給研究者的建議
- 直接做「深度非均勻 rank × 遞迴 LoRA」實驗，以 Lad et al. 四階段/U 形為先驗，對照均勻 rank baseline。這是目前文獻缺口且理論依據充分。
- 用 mechanistic 工具（如 arXiv:2604.11791 的不動點/軌跡分析）檢驗你的共享設計是否破壞或保留四階段結構。
