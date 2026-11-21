---
slug: rti-redundancy-probe-four-requirements
title: 冗餘探測：pixel-space 模型中冗餘的位置與四項需求
cluster: Diffusion & Generative Models
kind: topic
tags: [rti, redundancy, probing, pixel-space, jit]
arxiv: ["2608.29281", "2511.13720"]
sources: ["2608.29281 §1, §3.1, §3.2, Fig. 2", "2511.13720 abstract"]
related: ["rti-lineage-hub", "rti-prior-reductions-compared", "pixel-space-diffusion", "x-prediction-manifold-view", "tokc-lineage-hub"]
---
# 冗餘探測與四項需求

屬於[RTI 研究脈絡總覽](rti-lineage-hub.html)。

## 起點：均勻 token 的成本

DiT 在規則的 patch 網格上運算，每個 patch 一個 token，每層、每個時間步成本相同。RTI §1 指出影像細節集中在畫面的一小部分，天空、桌面、水面等區域大面積重複。Latent 模型在 VAE 中「一次付清」這項冗餘，pixel-space 模型（如 JiT，arXiv 2511.13720；背景見 [Pixel-space diffusion](pixel-space-diffusion.html)）則在每個 block 都付一次。

RTI 使用的 MiniT2I-B/16 是 JiT 系列的 pixel-space T2I 模型，包含 17 個雙流 block；512² 解析度、patch 16 時為 32×32 = 1024 個 patch token（§3.1）。模型訓練為 rectified flow，並預測乾淨影像而非雜訊殘差（x-prediction，見 [x-prediction 與流形觀點](x-prediction-manifold-view.html)）。

## 三個軸向的量測（§1、Fig. 2）

- 空間：細節集中於少數 patch。
- 時間：把 N 個 patch token 匯聚為數百個區域後，保留的表徵比例隨取樣進行而上升。
- 深度：保留比例在第一個 block 為 61%，中段約 75%，往最後一個 block 再下降（§3.2）。因此首尾 block 對每個 patch 都敏感，中段可壓縮。

## 由量測導出的四項需求

1. 連續：冗餘的 patch 在平坦區域彼此相鄰，分組必須是連通區域。
2. 大小隨內容：細節集中，故區域在牆面等平坦處大、在葉叢等細節處小。
3. 帶位置：每個群組覆蓋確定的位置，其 token 必須帶有位置。
4. 總結而非刪除：Fig. 2c 顯示丟棄損失的資訊多於匯聚保留的，冗餘內容應被「粗略表示」而非移除。

論文結論（§5）指出，這四項需求在訓練任何縮減方法之前就已固定，並事先預測了哪些方法會在哪個性質上失敗；各方法的比較見[既有縮減方法各自放棄什麼](rti-prior-reductions-compared.html)。

相關的縮減方法脈絡見 [token 縮減系列總覽](tokc-lineage-hub.html)。
