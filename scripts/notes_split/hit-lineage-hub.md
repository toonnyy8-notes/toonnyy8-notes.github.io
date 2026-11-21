---
slug: hit-lineage-hub
title: HiT 研究脈絡總覽：從潛空間階層到可學習的影像分組
cluster: Diffusion & Generative Models
kind: hub
tags: ["hit", "hierarchy", "pixel-space", "token-grouping", "lineage"]
arxiv: ["2112.10752", "2212.09748", "2410.06940", "2510.11690", "2511.13720", "2412.09871", "2507.07955", "2202.11094", "2210.09461", "2303.17604", "2601.01608", "2611.22962"]
sources: ["2611.22962 Introduction、Related Work、Appendix", "各前驅工作 arXiv 摘要"]
related: ["hit-latent-hierarchy-and-pixel-space", "hit-hnet-blt-to-image-groups", "hit-merging-baselines-fixed-rates", "hit-affinity-grouping", "hit-compression-budget", "hit-group-guidance", "hit-compact-position-and-joint-grouping", "tokc-lineage-hub", "tokc-vit-pruning-and-merging", "tokc-training-free-diffusion-merging", "tokc-learned-routing-dynamic-dit", "tokc-hourglass-and-latent-interfaces", "tokc-dynamic-chunking-bytes-to-pixels", "tokc-content-adaptive-tokenization", "tokc-position-for-merged-tokens", "pixel-space-diffusion", "x-prediction-manifold-view"]
---
# HiT 研究脈絡總覽：從潛空間階層到可學習的影像分組

HiT（Hierarchical Diffusion Transformer，2611.22962）想回答的問題是：pixel-space 擴散在單一端到端階段訓練，品質與訓練效率仍落後於雙階段潛空間模型，原因是否在於骨幹缺少視覺階層。論文的假設是肯定的，並在 Transformer 內加入兩層可學習的分組。下列筆記依閱讀順序排列，每篇只涵蓋一條前驅脈絡與 HiT 新增的部分。

## 閱讀順序

1. [HiT 的出發點：潛空間的階層與 pixel-space 的固定 token 格點](hit-latent-hierarchy-and-pixel-space.html)：潛空間模型把階層放在外部編碼器，pixel-space 模型（JiT、MiniT2I 等）固定 token 數量與排列。
2. [H-Net 與 BLT：從動態切塊到影像分組](hit-hnet-blt-to-image-groups.html)：HiT 沿用 H-Net 的巢狀骨幹與 ratio loss，改為對雙向影像特徵做親和度分組。
3. [ToMe 與 ToMeSD 在 HiT 中的位置：預先設定的縮減率](hit-merging-baselines-fixed-rates.html)：視覺 token 合併使用預設的縮減率；HiT 讓數量可變。
4. [GroupViT 到 HiT：親和度、平均連結與軟分組](hit-affinity-grouping.html)：親和度、平均連結、軟權重與解組的具體算法。
5. [HiT 的壓縮預算：批次平均比例與可變群組數](hit-compression-budget.html)：以批次平均的 ratio loss 控制群組數。
6. [Group guidance：以分組容量差作為引導](hit-group-guidance.html)：由 Sparse Guidance 延伸的取樣方式。
7. [HiT 的 compact 階段位置編碼與文字影像聯合分組](hit-compact-position-and-joint-grouping.html)：群組的位置處理與文字到影像的聯合分組。

## 成果（論文 Abstract）

ImageNet 256×256 上 FID 1.48；與 JiT 達到相近生成品質時，累計訓練 FLOPs 約為其 40%。文字到影像上，HiT 在五項基準的平均分數超過 UniLiP，去噪器參數約少 6 倍、forward FLOPs 約少 3 倍。

## 共同背景（系列筆記，不在此重複）

- 總覽：[Token 合併、壓縮與階層：系列總覽](tokc-lineage-hub.html)
- [ViT 的 token 剪枝與合併](tokc-vit-pruning-and-merging.html)
- [免訓練的擴散 token 合併](tokc-training-free-diffusion-merging.html)
- [學得的路由與動態 DiT](tokc-learned-routing-dynamic-dit.html)
- [Hourglass 與 latent 介面](tokc-hourglass-and-latent-interfaces.html)
- [從 byte 到 pixel 的動態切塊](tokc-dynamic-chunking-bytes-to-pixels.html)
- [內容自適應的 tokenization](tokc-content-adaptive-tokenization.html)
- [合併 token 的位置處理](tokc-position-for-merged-tokens.html)
- 站內既有：[Pixel-space diffusion](pixel-space-diffusion.html)、[x-prediction 與流形觀點](x-prediction-manifold-view.html)
