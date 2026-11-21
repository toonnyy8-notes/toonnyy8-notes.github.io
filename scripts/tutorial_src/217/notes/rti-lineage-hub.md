---
slug: rti-lineage-hub
title: RTI 研究脈絡總覽：從 token 縮減到 Hilbert 區域 token
cluster: Diffusion & Generative Models
kind: hub
tags: [rti, token-compression, pixel-space-diffusion, lineage]
arxiv: ["2608.29281", "2210.09461", "2303.17604", "2404.02258", "2107.14795", "2212.11972", "2603.12245", "2603.06351", "2304.00287", "2511.13720"]
sources: ["2608.29281 §1, §2", "各論文 arXiv 摘要"]
related: ["rti-redundancy-probe-four-requirements", "rti-prior-reductions-compared", "rti-hilbert-order-region-partition", "rti-read-write-interface", "rti-elastic-budget-lora-training", "rti-results-and-limits", "tokc-lineage-hub", "pixel-space-diffusion", "x-prediction-manifold-view"]
---
# RTI 研究脈絡總覽

*Elastic Token Compression for Pixel-Space Diffusion Transformers*（RTI，arXiv 2608.29281）處理的問題是：擴散 Transformer 在每一層、每個時間步，對每個 patch 都花一個 token 的計算；pixel-space 模型沒有 VAE 先吸收低階空間冗餘，因此在每個 block 都付這筆成本（論文 §1）。RTI 在已釋出、凍結的 pixel-space T2I 骨幹上，把中段 block 改為在「區域 token」上運算，並讓同一個 checkpoint 支援任意 token 預算。

共同背景（token 合併、階層結構、pixel-space DiT、動態切塊）已由系列筆記整理，請先參考 [token 縮減系列總覽](tokc-lineage-hub.html)、[Pixel-space diffusion](pixel-space-diffusion.html) 與 [x-prediction 與流形觀點](x-prediction-manifold-view.html)。本組筆記只說明 RTI 增加的部分。

## 閱讀順序

1. [冗餘探測與四項需求](rti-redundancy-probe-four-requirements.html)：在凍結模型上量測冗餘的位置與形狀，得出分組必須具備的四個性質。
2. [既有縮減方法各自放棄什麼](rti-prior-reductions-compared.html)：相似度合併、latent 瓶頸、token 跳過，分別失去四項需求中的哪幾項。
3. [Hilbert 順序與區域切割](rti-hilbert-order-region-partition.html)：以 Hilbert 曲線把二維分組化為一維切割，切在特徵落差最大處。
4. [Read / Write 介面](rti-read-write-interface.html)：把區域匯聚為 token、再把更新寫回 patch，以及區域 token 的位置。
5. [彈性預算與 LoRA 訓練](rti-elastic-budget-lora-training.html)：隨機抽取區域數、凍結骨幹，以及與 ELIT、DC-DiT 的差異。
6. [實驗結果、消融與限制](rti-results-and-limits.html)。

脈絡中的主要前作以時間序整理於 timeline.json，終點為 RTI。
