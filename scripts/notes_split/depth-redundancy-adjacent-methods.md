---
slug: depth-redundancy-adjacent-methods
title: 深度冗餘的相鄰路線：Mixture-of-Depths、深度剪枝、MoEUT
cluster: Architectures & Efficiency
kind: topic
tags: [mixture-of-depths, layer-pruning, moeut, depth-redundancy]
sources: [weight-sharing-lora.md]
arxiv: ["2404.02258", "2403.17887", "2405.16039"]
related: ["weight-sharing-recursive-overview", "stages-of-inference-weight-sharing", "recursive-transformer-followups", "early-cross-layer-weight-sharing"]
---
# 深度冗餘的相鄰路線：Mixture-of-Depths、深度剪枝、MoEUT

屬於 [層間權重共享總覽](weight-sharing-recursive-overview.html)。

- **Mixture-of-Depths（MoD）**（Raposo et al., arXiv:2404.02258, 2024）：top-k routing 讓 token 動態決定是否參與某層運算，靜態計算圖、每次 forward FLOPs 較少，post-training 取樣最高快 50%。屬 per-token 運算分配，與遞迴的 per-token 深度互補。
- **深度剪枝**：The Unreasonable Ineffectiveness of the Deeper Layers（Gromov et al., arXiv:2403.17887, ICLR 2025）——移除相似度高的連續層區塊 + QLoRA heal（單卡 A100 即可），某些開源模型移除近半層仍幾乎不掉分，直接證明「深層冗餘」。其可安全移除中段層的解釋見 [四階段假說](stages-of-inference-weight-sharing.html)。
- **latent reasoning / recurrent depth**：Geiping et al. 與 Ouro，見 [後續工作](recursive-transformer-followups.html)。
- **深度方向 MoE**：MoEUT（Csordás et al., NeurIPS 2024, arXiv:2405.16039）以 MoE 補回共享層被砍掉的參數，首次讓共享層 Transformer 在 BLiMP/PIQA 上略勝標準 Transformer 且省運算記憶體。 [arXiv](https://arxiv.org/abs/2405.16039) 它也指出在其之前沒有共享層 Transformer 在語言建模上有競爭力（見 [早期權重共享](early-cross-layer-weight-sharing.html)）。

各方法的參數量、FLOPs、吞吐比較見總覽的優劣分析表。
