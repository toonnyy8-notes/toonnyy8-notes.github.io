---
slug: stages-of-inference-weight-sharing
title: 四階段推論假說（Lad et al.）與權重共享、looped 模型的可解釋性
cluster: Interpretability & Theory
kind: topic
tags: [stages-of-inference, layer-robustness, mechanistic-interpretability, looped-transformer]
sources: [weight-sharing-lora.md]
arxiv: ["2406.19384", "2410.20672", "2604.11791", "2606.31779"]
related: ["weight-sharing-recursive-overview", "depth-redundancy-adjacent-methods", "relaxed-recursive-transformer", "depth-nonuniform-lora-rank"]
---
# 四階段推論假說（Lad et al.）與權重共享、looped 模型的可解釋性

屬於 [層間權重共享總覽](weight-sharing-recursive-overview.html)。

## Lad et al. 四階段與 U 形敏感度的啟示
Lad et al.（The Remarkable Robustness of LLMs: Stages of Inference?, arXiv:2406.19384）：刪除/交換相鄰層後保留 72–95% top-1 準確率；層越多越穩健； [Hugging Face](https://huggingface.co/papers/2406.19384) **首尾層脆弱、中間層穩健**（ [Emergent Mind](https://www.emergentmind.com/papers/2406.19384) U 形）；四階段為 detokenization → feature engineering → prediction ensembling → residual sharpening。 [arxiv](https://arxiv.org/pdf/2406.19384)

對權重共享的直接設計啟示：
- 中間「feature engineering」階段本質上就是「對殘差流做迭代精修」，與 looped/遞迴的「重複套用同一 block」高度同構——這解釋了為何中間層最適合被 tie/loop，也解釋了 Gromov et al. 為何能安全移除中段層（見 [相鄰路線](depth-redundancy-adjacent-methods.html)）。
- 首尾層脆弱 → 不應共享、應保留獨立參數（[RRT](relaxed-recursive-transformer.html) 的 Stepwise 正是「固定首尾、只在中間取樣」， [arxiv](https://arxiv.org/html/2410.20672v3) 可視為對此假說的隱性呼應，雖然 Bae et al. 的動機引用的是 layer-skipping 文獻而非 Lad et al.）。
- 由此推出的「深度非均勻 rank 分配」空白見 [深度非均勻 LoRA rank](depth-nonuniform-lora-rank.html)。

## 對 looped/recurrent 模型的機械可解釋性分析
- A Mechanistic Analysis of Looped Reasoning Language Models（arXiv:2604.11791, 2026）：分析 Huginn-0125、retrofitted Llama、Ouro，發現每個 cycle 層收斂到不同不動點、recurrent block 沿 latent 空間走一致的循環軌跡、attention head 行為隨迭代穩定；**經驗上 recurrent block 學到的「stages of inference」與 feedforward 模型相似，並沿深度每次迭代重複這些階段**—— [arXiv](https://arxiv.org/html/2604.11791v1) 這直接把 Lad et al. 的四階段結構延伸到 looped 模型，是本題最相關的交叉工作。（註：亦觀察到並非所有模型都達到嚴格不動點，例如 Ouro 未如 Huginn 般快速收斂。）
- LOTUS（arXiv:2606.31779）：把 post-loop latent 經 base LM head 投影可還原 gold reasoning steps，顯示 latent 空間可解釋、與 CoT 對齊。 [arXiv](https://arxiv.org/html/2606.31779v2)
- Pappone et al.（2025）分析 recurrent-depth 的「two-scale」latent 動態（快 intra-loop、慢 inter-token）。
