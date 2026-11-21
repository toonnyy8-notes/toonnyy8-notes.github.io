---
slug: hit-merging-baselines-fixed-rates
title: ToMe 與 ToMeSD 在 HiT 中的位置：預先設定的縮減率
cluster: Diffusion & Generative Models
kind: topic
tags: ["token-merging", "tome", "tomesd", "proportional-attention"]
arxiv: ["2210.09461", "2303.17604", "2611.22962"]
sources: ["2611.22962 Related Work、Appendix B.1", "2210.09461 abstract", "2303.17604 abstract"]
related: ["hit-compact-position-and-joint-grouping", "tokc-vit-pruning-and-merging", "tokc-training-free-diffusion-merging"]
---
# ToMe 與 ToMeSD 在 HiT 中的位置：預先設定的縮減率

ToMe（Bolya 等，2210.09461）與 ToMeSD（Bolya 與 Hoffman，2303.17604）是 HiT 引用的視覺 token 合併工作。論文對它們的描述是：合併相似 token，用來加速視覺 Transformer 與預訓練的擴散模型。論文把它們與 GroupViT 並列，認為這些方法確立了「分組」作為計算效率與視覺表示學習的有效機制，但使用**預先設定的 token 數或縮減率**。

## HiT 與它們的差別

- 目的：ToMe 系列在現成模型上加速；HiT 從頭訓練，並把分組視為提升生成品質與訓練效率的手段（Abstract）。
- 數量：ToMe 系列的縮減率事先給定；HiT 的群組數由壓縮預算與親和度門檻決定，隨影像與時間步變動。
- 還原：HiT 用與分組相同的軟權重 A 解組，再加上 skip 連接。

## HiT 借用的細節

Appendix B.1 測試了「log-size bias」：在注意力 logit 加上群組大小的對數，論文標明這是依 ToMe 的 proportional attention。結果是它與 learned positional embedding 對 hard-centroid 的結果略有幫助；預設設定中沒有使用 compact 階段的 RoPE（見 [HiT 的 compact 階段位置編碼與文字影像聯合分組](hit-compact-position-and-joint-grouping.html)）。

## 本站已有的系列筆記

- [ViT 的 token 剪枝與合併（含 ToMe）](tokc-vit-pruning-and-merging.html)
- [免訓練的擴散 token 合併（ToMeSD 等）](tokc-training-free-diffusion-merging.html)

這兩篇涵蓋 ToMe 的二部配對與 ToMeSD 的 unmerge，本筆記不重複。
