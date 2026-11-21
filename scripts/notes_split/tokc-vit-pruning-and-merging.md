---
slug: tokc-vit-pruning-and-merging
title: ViT 的 token 剪枝與合併：DynamicViT、EViT、ToMe
cluster: Diffusion & Generative Models
kind: topic
tags: [token-merging, token-pruning, ToMe, DynamicViT, ViT]
arxiv: ["2106.02034", "2202.07800", "2210.09461", "2305.17997"]
sources: ["2210.09461 全文 §3, 摘要", "2106.02034 arXiv 摘要", "research/lineage.md §3.4-§3.5", "2202.07800 與 2305.17997 僅以 lineage.md 附帶提及為據，未核對"]
related: ["tokc-training-free-diffusion-merging", "tokc-learned-routing-dynamic-dit", "tokc-position-for-merged-tokens", "tokc-lineage-hub"]
---
# ViT 的 token 剪枝與合併

ViT 對每個 patch 配置相同計算，背景 patch 因而浪費計算。2021–2022 年的方法在分類任務中縮減 token 數。分類只需要 CLS token，被縮減的 token 不需要還原，這是之後搬到擴散時必須補上的部分（見 [免訓練的擴散 token 合併](tokc-training-free-diffusion-merging.html)）。

## 剪枝

DynamicViT（arXiv 2106.02034）以輕量預測模組依當前特徵估計 token 重要性，插入多層做階層式剪枝，並以注意力遮罩阻斷被剪 token 與其他 token 的互動，使剪枝可端對端微分。摘要報告：剪去 66% 輸入 token，FLOPs 降低 31%–37%，吞吐量提升 40% 以上，精度下降 0.5% 以內。EViT（arXiv 2202.07800）依 CLS 注意力保留重要 token 並融合其餘 token。DiffRate（arXiv 2305.17997）讓壓縮比可微，為 DiffCR 的前作（見 [學得路由與動態 DiT](tokc-learned-routing-dynamic-dit.html)）。

## ToMe

Token Merging（ToMe，arXiv 2210.09461）以合併取代丟棄，且不需訓練。每個 block 內執行二部軟配對（bipartite soft matching）：

1. 把 token 交替分為 A、B 兩集合。
2. 每個 A token 連一條邊到 B 中 key 餘弦相似度最高的 token。
3. 保留最相似的 r 條邊。
4. 把相連 token 合併，以 token 大小加權平均，大小相加。
5. 串回序列。每層減少 r 個 token。

合併後的 token 代表多個 patch，因此 ToMe 使用 proportional attention：softmax(QKᵀ/√d + log s)，s 為每個 token 代表的 patch 數，等同把該 key 複製 s 份（見 [合併後 token 的位置處理](tokc-position-for-merged-tokens.html)）。摘要報告：現成 ViT-L@512 與 ViT-H@518 吞吐量 2×、ViT-L 影片 2.2×，精度下降 0.2–0.3%。

限制：分組依外觀相似度決定，不考慮空間位置。RTI 的 Feat Sim 基線即此類分組；HiT 的 log-size bias 取自 proportional attention。
