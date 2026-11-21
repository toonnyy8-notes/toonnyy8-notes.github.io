---
slug: tokc-content-adaptive-tokenization
title: 內容自適應的 tokenization：Quadformer 與空間序列化
cluster: Diffusion & Generative Models
kind: topic
tags: [quadtree, Quadformer, Hilbert-curve, serialization, mixed-resolution]
arxiv: ["2304.00287", "2503.12015", "2307.06304", "2312.10035", "2608.29281"]
sources: ["2304.00287 摘要與擷取全文", "research/lineage.md §3.21, §5.1, W3", "2503.12015, 2307.06304, 2312.10035 僅以 lineage.md 附帶提及為據，未核對", "FlexTok、ElasticTok 僅在 lineage.md 附帶提及，未列 arXiv 編號"]
related: ["tokc-dynamic-chunking-bytes-to-pixels", "tokc-position-for-merged-tokens", "tokc-lineage-hub", "rti-lineage-hub", "aniso-pixel-patchified-setting"]
---
# 內容自適應的 tokenization

固定網格對平坦區與細節區分配相同 token 數。另一條路線在輸入端改變切分方式。patchified pixel 設定的背景見 [Patchified pixel 設定](aniso-pixel-patchified-setting.html)。

## Quadformer（arXiv 2304.00287）

以 quadtree 演算法加新的顯著性評分器，把影像切成大小不一的 patch 馬賽克，低顯著區以低解析度處理；架構同 vanilla ViT，加 2D 位置嵌入。摘要稱在控制計算預算下分類精度大幅提升（無具體數字）。需從頭訓練且限於分類。

相關工作（僅附帶提及，未核對）：QDM（arXiv 2503.12015）以 quadtree 做超解析擴散；NaViT（arXiv 2307.06304）以原生解析度打包；FlexTok、ElasticTok 為可變長度 tokenizer。tokenizer 類方法改變輸入表示，無法用於已釋出的 checkpoint（RTI §2 的批評）。

## 空間序列化：raster、Morton、Hilbert

把 2D 格點排成 1D 序列後，可在序列上切分。Hilbert 曲線有兩個性質（lineage.md 對 32×32 格點驗證）：相鄰序號必為格點鄰居；每個對齊的二進位（quadtree）格子恰為一段連續區間。因此切割 Hilbert 序列可表達任何 quadtree 切分，以及 quadtree 達不到的切分。RTI 附錄 B.4 的對照：raster 在列尾斷開，Morton 在二進位區塊邊界跳躍；lineage.md 記載 Hilbert 對 raster 在 R = 64 的 PickScore 為 −0.268（符號方向以 RTI 附錄 B.4 為準，未核對）。Point Transformer V3（arXiv 2312.10035）以曲線排序注意力序列；RTI 則用 Hilbert 序定義分組本身。RTI 的完整機制見 [RTI 研究脈絡總覽](rti-lineage-hub.html)。

沿序列切塊的方法見 [動態切塊](tokc-dynamic-chunking-bytes-to-pixels.html)。
