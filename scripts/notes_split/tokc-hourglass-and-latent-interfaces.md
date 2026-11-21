---
slug: tokc-hourglass-and-latent-interfaces
title: 沙漏結構與 latent 介面：Funnel、Hourglass、HDiT、Perceiver IO、RIN、ELIT
cluster: Diffusion & Generative Models
kind: topic
tags: [hourglass, HDiT, Perceiver-IO, RIN, ELIT, read-write]
arxiv: ["2006.03236", "2110.13711", "2401.11605", "2107.14795", "2212.11972", "2603.12245", "2209.12152", "2301.11093"]
sources: ["2110.13711 全文 §2, §3.4", "2401.11605 全文", "2603.12245 全文（擷取）", "2006.03236, 2107.14795, 2212.11972, 2209.12152, 2301.11093 僅 arXiv 摘要", "research/lineage.md §3.1-§3.3, §3.15-§3.19"]
related: ["rti-lineage-hub", "hit-lineage-hub", "tokc-dynamic-chunking-bytes-to-pixels", "tokc-position-for-merged-tokens", "tokc-lineage-hub", "pixel-space-diffusion"]
---
# 沙漏結構與 latent 介面

兩個原型：把序列縮短後在中段處理再還原；或讓固定數量的 latent 讀寫整個輸入。

## 沙漏（縮短、處理、上採樣）

- Funnel-Transformer（arXiv 2006.03236，僅讀摘要）：逐步壓縮隱藏序列，省下的 FLOPs 再投入加深或加寬；需要逐 token 預測時以解碼器還原。
- Hourglass（arXiv 2110.13711）：shorten（係數 k）、中段 Transformer、upsample 加殘差、全長 Transformer。比較的上採樣方式有重複、線性上採樣、注意力上採樣 x = U(x, x′) + Attention(Q = U(x, x′), K = V = x′)。ImageNet32 生成 3.741 bits/dim。
- HDiT（arXiv 2401.11605）：把 Hourglass 帶到 pixel-space 擴散。編碼端每層以 Pixel-UnShuffle 把 2×2 token 合併為一並投影，解碼端以 Pixel-Shuffle 分裂，skip 以可學係數線性插值，高解析度層用鄰域注意力、最內層（16×16 token）用全域注意力，整體 O(n)。FFHQ-1024² FID 5.23（85M 參數）。分組固定為 2×2，與內容無關。
- simple diffusion（arXiv 2301.11093，僅讀摘要）：下採樣是避免高解析度特徵圖的有效策略。U-ViT（arXiv 2209.12152，僅讀摘要）：長 skip 連接至關重要，下採樣與上採樣並非必要。

## latent 介面（讀入、處理、寫回）

- Perceiver IO（arXiv 2107.14795，僅讀摘要）：固定大小 latent 以 cross-attention 讀入輸入，在 latent 上做自注意力，再以輸出查詢讀出，成本對輸入、輸出大小呈線性。
- RIN（arXiv 2212.11972，僅讀摘要）：全域自注意力集中在 latent，以 cross-attention 在 latent 與資料 token 間讀寫。
- ELIT（arXiv 2603.12245）：block 分為短空間頭、latent 核心、空間尾。Read：l_CA = l + CA(Q = l; K, V = s)，Write 對稱。為降低成本採 grouped cross-attention：空間 token 分為 G 個不重疊網格群組，每群 J = K/G 個 latent。訓練時每次迭代抽 J̃ 並丟棄各群尾端 latent，使 latent 依重要性排序，推論時可任選 J̃。主實驗 16 個群組，512px 每群至多 64 個 latent，共 60 種推論預算。ImageNet-1K 512px FID 平均改善 35.3%。latent 由可學位置嵌入初始化，該嵌入在各群間重用並編碼群內位置（全文確認）。

## 與 RTI、HiT 的關係

RTI 的 Read/Write 與 prelude/core/coda 結構屬 latent 介面這一族，但把 latent 錨定在連續空間區域上（見 [RTI 研究脈絡總覽](rti-lineage-hub.html)）。HiT 屬沙漏巢狀結構，分組改為學得（見 [HiT 研究脈絡總覽](hit-lineage-hub.html)）。RTI 與 HiT 皆未引用 HDiT，本系列依主題補入。Pixel-space 背景見 [Pixel-space diffusion](pixel-space-diffusion.html)。
