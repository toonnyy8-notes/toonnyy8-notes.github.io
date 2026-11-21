---
slug: tokc-learned-routing-dynamic-dit
title: 學得路由與動態 DiT：DyDiT、DiffCR、FlexiDiT、Sparse Guidance
cluster: Diffusion & Generative Models
kind: topic
tags: [DiffCR, DyDiT, FlexiDiT, Sparse-Guidance, router, mixture-of-depths]
arxiv: ["2404.02258", "2412.16822", "2410.03456", "2502.20126", "2601.01608"]
sources: ["2412.16822 全文（擷取）§4", "2601.01608 全文（摘要部分）", "2410.03456 arXiv 摘要", "2502.20126 arXiv 摘要", "2404.02258 arXiv 摘要", "research/lineage.md §3.10-§3.14"]
related: ["depth-redundancy-adjacent-methods", "moe-conditional-computation-lineage", "tokc-training-free-diffusion-merging", "tokc-lineage-hub"]
---
# 學得路由與動態 DiT

免訓練方法的壓縮比例需手調，且與層、時間步無關。這一類方法在既有 DiT 上微調出路由，或讓計算量隨時間步變化。

## Mixture-of-Depths

Mixture-of-Depths（arXiv 2404.02258）是語言模型中的前身：每層以 top-k router 決定哪 k 個 token 參與自注意力與 MLP，其餘沿殘差流跳過；k 事先固定。內容見既有筆記 [深度冗餘的相鄰路線](depth-redundancy-adjacent-methods.html)，上位概念見 [條件式計算脈絡](moe-conditional-computation-lineage.html)。

## DiffCR（arXiv 2412.16822）

- 每個 DiT 層加輕量 router，與模型權重聯合微調，預測 token 重要性並取 top-k 處理，其餘 token 繞過整層（輸出乘以 router 分數，同 MoD）。
- 層級壓縮比與時間步壓縮比皆可微，自零初始化學出。PixArt-Σ 20 步分為 4 區，每層 4 個可學比例。學到的模式：雜訊越大的時間步壓縮比例越高。
- 數字：PixArt-Σ 上品質與未壓縮模型相當，節省 20.68% 延遲與 8.33% 記憶體。

## DyDiT、FlexiDiT、Sparse Guidance（僅讀摘要）

- DyDiT（arXiv 2410.03456）：Timestep-wise Dynamic Width 加 Spatial-wise Dynamic Token。額外微調不到 3% 迭代，DiT-XL FLOPs 降低 51%、加速 1.73×，ImageNet FID 2.07。
- FlexiDiT（arXiv 2502.20126）：把預訓練 DiT 轉為可在不同計算預算下處理輸入的彈性模型。影像 FLOPs 降低 40% 以上；影片至多省 75%。具體機制未讀全文。
- Sparse Guidance（arXiv 2601.01608）：以 token 稀疏度作為引導訊號，處理 token 稀疏訓練模型對 CFG 反應不佳的問題。ImageNet-256 FID 1.58、FLOPs 少 25%。HiT 的 group guidance 明言受其啟發。

## 限制

路由類方法讓 token 跳過 block，屬於刪除而非總結。RTI 重實作的 Token Skip 只保留其分組三分之一的表徵。多數需要微調。
