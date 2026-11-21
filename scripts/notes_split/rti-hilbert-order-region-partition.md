---
slug: rti-hilbert-order-region-partition
title: Hilbert 順序與區域切割：把二維分組化為一維切割
cluster: Diffusion & Generative Models
kind: topic
tags: [rti, hilbert-curve, region-partition, quadtree, content-adaptive]
arxiv: ["2608.29281", "2312.10035"]
sources: ["2608.29281 §1, §2, §3.1, §3.3, §4.3, §5", "2304.00287 abstract", "2307.06304 abstract", "2503.12015 abstract", "2312.10035 abstract"]
related: ["rti-lineage-hub", "tokc-content-adaptive-tokenization", "tokc-dynamic-chunking-bytes-to-pixels"]
---
# Hilbert 順序與區域切割

屬於[RTI 研究脈絡總覽](rti-lineage-hub.html)。

## Hilbert 順序

Hilbert 曲線（Hilbert 1891）走訪 2^k × 2^k 網格的每個格子恰好一次，且曲線上相鄰位置在網格上必為相鄰格。因此序列中任何連續片段都占據影像中一個連通區域，長度任意、不必是正方形。RTI 把 32×32 的 patch 網格依此曲線排成長度 N = 1024 的序列，順序記為 π（§3.1）。每個二進位（quadtree）格子恰為序列中的一段連續區間，所以切割序列能表達任何 quadtree 切分，也能表達 quadtree 做不到的切分（lineage.md 對 32×32 格點驗證這兩個性質）。RTI 附錄 B.4 的對照：raster 在列尾斷開，Morton 在二進位區塊邊界跳躍；lineage.md 記載 Hilbert 對 raster 在 R = 64 的 PickScore 為 −0.268（符號方向以 RTI 附錄 B.4 為準，未核對）。

## 在特徵落差最大處切割（§3.3）

令 h⁰_j 為進入核心的特徵，相鄰兩 patch 的邊界分數為 d_j = ‖h⁰_{π(j+1)} − h⁰_{π(j)}‖₂（Eq. 1）。取最大的 R−1 個 d_j 為切點（Eq. 2），兩切點之間的連續片段即 R 個區域。

- 無參數，成本 O(N)：一次掃描算落差，一次線性選取。
- 平坦處成為長區域，細節處被切成單一 patch。
- 因骨幹預測乾淨影像，其特徵追蹤影像結構，大落差標示內容邊界。切割依據當前狀態 x_t，每個取樣步重切。
- 細區域與影像細節的對齊並非內建：切割只讀特徵，不讀細節量測（§4.4）。

## 與前作的關係

- Quadtree 方法（QDM、NaViT 等）讓 tokenizer 隨內容自適應但需從頭訓練；RTI §2 因此認為它們不適用於已釋出的 checkpoint，並指出 quadtree 把區域綁在二進位正方形，且每張圖每步都要建樹。這些方法本身見[內容自適應 tokenization](tokc-content-adaptive-tokenization.html)。
- Point Transformer V3（arXiv 2312.10035）用 Hilbert 曲線排列注意力所處理的序列；RTI 用它定義群組本身（§2）。
- 一維切塊的語言模型作法見[動態切塊：從 byte 到像素](tokc-dynamic-chunking-bytes-to-pixels.html)。

## 消融與限制

均勻切割（沿 Hilbert 序等距切）取代最大落差切割，R = 128 時損失 2.7 個 GenEval 點，預算越大差距越小；R = 256 時每區域約 4 個 patch，等距切割已接近自適應切割（§4.3）。論文承認並非每種分組都是 Hilbert 區間（§5）。
