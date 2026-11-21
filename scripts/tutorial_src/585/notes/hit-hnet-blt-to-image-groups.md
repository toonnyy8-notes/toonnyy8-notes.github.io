---
slug: hit-hnet-blt-to-image-groups
title: H-Net 與 BLT：從動態切塊到影像分組
cluster: Diffusion & Generative Models
kind: topic
tags: ["h-net", "blt", "dynamic-chunking", "hierarchy", "ratio-loss"]
arxiv: ["2412.09871", "2507.07955", "2611.22962"]
sources: ["2611.22962 Related Work、Sec. 3.1、Sec. 3.3", "2412.09871 abstract", "2507.07955 abstract"]
related: ["hit-compression-budget", "hit-affinity-grouping", "tokc-dynamic-chunking-bytes-to-pixels"]
---
# H-Net 與 BLT：從動態切塊到影像分組

## 兩個前驅工作

BLT（2412.09871）依下一個 byte 的熵切成動態長度 patch；H-Net（2507.07955）以聯合學習的動態切塊取代 tokenization，階層可疊代為多階段。兩者的機制與數字見 [從 byte 到 pixel 的動態切塊](tokc-dynamic-chunking-bytes-to-pixels.html)。

## HiT 沿用的部分

- **巢狀骨幹**：HiT 的 Transformer 排成 encoder、compact network、decoder 的巢狀階段，引用 H-Net 的作法（Sec. 3.1）。J 層分組時，token 數為 N → K1 → … → KJ → … → K1 → N（Eq. 1）。每一層有自己的 router 與 skip projection W_ℓ。
- **Ratio loss**：壓縮預算沿用 H-Net 的 ratio loss（Eq. 4），細節見 [HiT 的壓縮預算：批次平均比例與可變群組數](hit-compression-budget.html)。
- **Router 的投影**：每層以投影後的單位長度向量計算親和度，論文標註這兩個投影沿用 H-Net（Sec. 3.2）。

## HiT 不同的部分

論文自己列出的差異（Related Work）：

1. BLT 與 H-Net 沿輸入序列形成切塊，較深層處理較少的單位；H-Net 的切塊沿因果序列。HiT 對雙向的影像特徵做**親和度分組**，空間上不相連的區域可以共用一個計算單位。
2. 先前的視覺分組方法使用預先設定的 token 數或縮減率；HiT 讓群組成員與群組數隨影像與擴散時間步改變。

這兩點的具體機制見 [GroupViT 到 HiT：親和度、平均連結與軟分組](hit-affinity-grouping.html)。

## 系列筆記

動態切塊從位元組到像素的整體脈絡，見 [從 byte 到 pixel 的動態切塊](tokc-dynamic-chunking-bytes-to-pixels.html)；本筆記只記錄 HiT 對 H-Net 的直接使用與差異。
