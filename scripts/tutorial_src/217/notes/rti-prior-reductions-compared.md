---
slug: rti-prior-reductions-compared
title: 既有縮減方法各自放棄什麼：相似度合併、latent 瓶頸、token 跳過
cluster: Diffusion & Generative Models
kind: topic
tags: [rti, tome, perceiver-io, mixture-of-depths, baselines]
arxiv: ["2608.29281", "2210.09461", "2303.17604", "2107.14795", "2212.11972", "2404.02258", "2407.19985"]
sources: ["2608.29281 §1, §2, §4.2, Tab. 1, Tab. 4", "2210.09461 abstract", "2303.17604 abstract", "2107.14795 abstract", "2212.11972 abstract", "2404.02258 abstract", "2407.19985 abstract"]
related: ["rti-lineage-hub", "rti-redundancy-probe-four-requirements", "tokc-vit-pruning-and-merging", "tokc-training-free-diffusion-merging", "tokc-hourglass-and-latent-interfaces", "tokc-learned-routing-dynamic-dit", "depth-redundancy-adjacent-methods"]
---
# 既有縮減方法各自放棄什麼

屬於[RTI 研究脈絡總覽](rti-lineage-hub.html)。RTI 論文把既有方法分成三類，每類至少放棄[四項需求](rti-redundancy-probe-four-requirements.html)中的一項。以下前作的機制細節見系列筆記，這裡只記 RTI 如何歸類與量測它們。

| 類別 | 代表 | 失去的性質（RTI 的說法） |
|---|---|---|
| 相似度合併 | Token Merging（arXiv 2210.09461）、其擴散版本（arXiv 2303.17604） | 依外觀分組，群組可能收集畫面相反角落的 patch，同時失去連續性與位置 |
| latent 瓶頸 | Perceiver-IO（arXiv 2107.14795）、RIN（arXiv 2212.11972） | 放棄分組，同時失去連續性與位置 |
| 條件計算 | Mixture-of-Depths（arXiv 2404.02258）、Mixture of Nested Experts（arXiv 2407.19985） | 跳過 token 而非總結，刪除了應被總結的內容 |

背景：[ViT 的 token 剪枝與合併](tokc-vit-pruning-and-merging.html)、[免訓練的擴散 token 合併](tokc-training-free-diffusion-merging.html)、[Hourglass 與 latent 介面](tokc-hourglass-and-latent-interfaces.html)、[學得路由與動態 DiT](tokc-learned-routing-dynamic-dit.html)、[深度冗餘的相鄰路線](depth-redundancy-adjacent-methods.html)。

## RTI 的對照實驗（§4.2）

三個基線都在 RTI 自己的介面內重新實作，並以同一份訓練配方、同一個凍結骨幹訓練，只有縮減方式不同。

- Token Skip（依 Mixture-of-Depths）：核心 block 交替稠密與稀疏，稀疏 block 只收前 12.5% 影像 token。GenEval 82.3，速度 1.19×；稠密 block 使加速上限低於約 1.3×。論文量得它只保留 RTI 分組所保留表徵的三分之一。
- Feat Sim（依 Token Merging）：群組 patch 距群組中心 5.2 格（RTI 為 0.5 格）；資訊量保留與 RTI 區域 token 相當。R = 256 時 GenEval 80.5，RTI 為 84.7。
- Latent Array（依 Perceiver-IO、RIN）：R = 256 個 latent，GenEval 45.7；讀取注意力的散布為 11.4 格（整張圖為 12.2 格）。論文說明這些方法為從頭訓練而設計，此處檢驗的是 latent 瓶頸能否在凍結骨幹上運作。

在相同預算下，刪除而非總結的代價為 2.4 個 GenEval 點，打散群組為 4.2，移除位置為 39.0。
