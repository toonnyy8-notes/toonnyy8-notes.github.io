---
slug: tokc-lineage-hub
title: Transformer 的 token 縮減與階層：從序列縮短到 RTI 與 HiT 的研究脈絡總覽
cluster: Diffusion & Generative Models
kind: hub
tags: [token-merging, token-compression, hierarchy, pixel-space-diffusion, DiT]
arxiv: ["2006.03236", "2110.13711", "2210.09461", "2303.17604", "2401.11605", "2507.07955", "2603.12245", "2603.06351", "2608.29281", "2611.22962"]
sources: ["research/lineage.md §1-§2, §4, §7.3", "2110.13711 全文", "2210.09461 全文", "2303.17604 全文", "2401.11605 全文", "2507.07955 全文", "2603.12245 全文", "2603.06351 全文", "2006.03236 arXiv 摘要"]
related: ["tokc-vit-pruning-and-merging", "tokc-training-free-diffusion-merging", "tokc-learned-routing-dynamic-dit", "tokc-hourglass-and-latent-interfaces", "tokc-dynamic-chunking-bytes-to-pixels", "tokc-content-adaptive-tokenization", "tokc-position-for-merged-tokens", "rti-lineage-hub", "hit-lineage-hub", "pixel-space-diffusion", "x-prediction-manifold-view", "depth-redundancy-adjacent-methods", "moe-conditional-computation-lineage"]
---
# token 縮減與階層的研究脈絡總覽

Transformer 每個 block 的成本隨 token 數增加。在 DiT 常見的 256 至 1024 個 token，成本大致與 token 數呈線性關係，因此縮短序列可近似等比例降低成本。pixel-space 擴散模型沒有 VAE 預先壓縮空間冗餘，每個 block 都要處理完整的 patch 格點（512² 影像、patch 16 時為 1024 個 token），縮減 token 的需求因此更明顯。相關背景見 [Pixel-space diffusion](pixel-space-diffusion.html) 與 [x-prediction 的流形觀點](x-prediction-manifold-view.html)。

本系列的兩個終點是 RTI（arXiv 2608.29281，在凍結的 pixel-space 骨幹上以 Hilbert 序切出連續區域 token）與 HiT（arXiv 2611.22962，從頭訓練、以學得 affinity 做階層分組）。兩者的機制分別見 [RTI 研究脈絡總覽](rti-lineage-hub.html) 與 [HiT 研究脈絡總覽](hit-lineage-hub.html)。

## 分期與閱讀順序

1. 序列縮短的原型（2020–2021）：Funnel-Transformer、Hourglass、Perceiver IO。見 [Hourglass 與 latent 介面](tokc-hourglass-and-latent-interfaces.html)。
2. ViT 分類中的 token 縮減（2021–2022）：DynamicViT、ToMe。分類只用 CLS token，不需還原被縮減的 token。見 [ViT 的 token 剪枝與合併](tokc-vit-pruning-and-merging.html)。
3. 免訓練的擴散 token 合併（2023–2024）：ToMeSD、ToDo。擴散輸出為稠密預測，必須還原。見 [免訓練的擴散 token 合併](tokc-training-free-diffusion-merging.html)。
4. 學得的路由與架構層級瓶頸（2022–2025）：Mixture-of-Depths、DiffCR、DyDiT、HDiT。見 [學得路由與動態 DiT](tokc-learned-routing-dynamic-dit.html)；Mixture-of-Depths 另見 [深度冗餘的相鄰路線](depth-redundancy-adjacent-methods.html) 與 [條件式計算脈絡](moe-conditional-computation-lineage.html)。
5. 內容自適應的切分與動態切塊（2023–2025）：Quadformer、BLT、H-Net。見 [內容自適應的 tokenization](tokc-content-adaptive-tokenization.html) 與 [動態切塊：從 byte 到 pixel](tokc-dynamic-chunking-bytes-to-pixels.html)。
6. 擴散 Transformer 中的彈性與自適應 token（2026）：ELIT、DC-DiT、RTI、HiT。
7. 橫跨各期的位置處理問題：見 [合併後 token 的位置處理](tokc-position-for-merged-tokens.html)。

各方法的時間順序清單見 timeline.json。數字取自各論文原文；僅讀摘要者在各筆記的 `sources` 欄位標示。
