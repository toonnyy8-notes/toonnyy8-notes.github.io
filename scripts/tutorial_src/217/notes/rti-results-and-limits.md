---
slug: rti-results-and-limits
title: RTI 的實驗結果、消融與限制
cluster: Diffusion & Generative Models
kind: topic
tags: [rti, geneval, ablation, limitations]
arxiv: ["2608.29281", "2310.11513", "2511.13720"]
sources: ["2608.29281 §4.2, §4.3, §4.4, §5, Tab. 1, Tab. 4", "2310.11513 abstract"]
related: ["rti-lineage-hub", "rti-prior-reductions-compared", "rti-hilbert-order-region-partition", "rti-elastic-budget-lora-training"]
---
# RTI 的實驗結果、消融與限制

屬於[RTI 研究脈絡總覽](rti-lineage-hub.html)。數字取自論文 §4（512²，GenEval 取 553 個 prompt，GenEval 見 arXiv 2310.11513；速度為 RTX 4090 牆鐘）。

## 主結果（Tab. 1）

| 模型 | R | 速度 | GenEval |
|---|---|---|---|
| MiniT2I-B/16 稠密 | 1024 | 1.00× | 87.2 |
| RTI | 512 | 1.84× | 87.8 |
| RTI | 256 | 2.15× | 84.7 |
| Feat Sim | 256 | 2.15× | 80.5 |
| Token Skip | 576* | 1.19× | 82.3 |
| MiniT2I-L/16 稠密 | 1024 | 1.00× | 88.1 |
| RTI | 512 | 2.11× | 87.5 |
| RTI | 256 | 2.59× | 86.3 |

*不是預算，為 Token Skip 稠密與稀疏核心 block 的平均。R ≥ 384（2.00×）時與稠密的差距落在雜訊內，但稠密模型在各項偏好分數上仍領先（§4.2）。基線的比較見[既有縮減方法各自放棄什麼](rti-prior-reductions-compared.html)。

## 消融（§4.3）

- 均勻切割取代最大落差切割：R = 128 損失 2.7（見 [Hilbert 切割](rti-hilbert-order-region-partition.html)）。
- 以 mean 加 broadcast 取代 Read/Write：損失 7.2。
- 以 latent array 取代錨定的區域 token：降到 Latent Array 基線的水準；此替換同時移除分組與位置，其代價是錨定貢獻的上限。

## 切割的行為（§4.4）

單一 patch 的區域聚集在物體與其邊界，平坦區由少數大區域涵蓋。細區域與粗區域的細節比，在最吵的取樣步接近 1，影像成形後升至約 6 倍。

## 限制（§5）

- 加速有上限：prelude、coda 與文字流不縮減。
- 核心無法區分同一區域內的 patch。
- 並非每種分組都是 Hilbert 區間。
- 預算由使用者而非影像內容決定。

另有 JiT 類別條件實驗（arXiv 2511.13720 的模型）列於論文附錄，本筆記未涵蓋。預算相關設定見[彈性預算與 LoRA 訓練](rti-elastic-budget-lora-training.html)。
