---
slug: tokc-position-for-merged-tokens
title: 合併後 token 的位置處理
cluster: Diffusion & Generative Models
kind: topic
tags: [RoPE, proportional-attention, positional-embedding, merged-token]
arxiv: ["2210.09461", "2603.06351", "2401.11605", "2603.12245", "2107.14795", "2608.29281", "2611.22962"]
sources: ["2210.09461 全文 §3", "2603.06351 全文（擷取）", "2401.11605 全文", "2603.12245 全文", "RTI §3.4、§4.2、Tab. 4 與 HiT 附錄 B.1、Tab. 9 之描述取自 lineage.md §3.25-§3.26, §5.2, W4，未重讀兩篇全文", "Dirichlet 核公式為 lineage.md W4 數值驗證"]
related: ["rti-read-write-interface", "hit-compact-position-and-joint-grouping", "tokc-vit-pruning-and-merging", "tokc-hourglass-and-latent-interfaces", "tokc-dynamic-chunking-bytes-to-pixels", "tokc-lineage-hub", "rti-lineage-hub", "hit-lineage-hub"]
---
# 合併後 token 的位置處理

一個合併 token 代表多個 patch，其位置與大小需要新的表示。各方法的選擇不同。

## 大小：proportional attention

ToMe（arXiv 2210.09461）在 logit 加 log s：softmax(QKᵀ/√d + log s)，s 為該 token 代表的 patch 數，等同把 key 複製 s 份。HiT 附錄 B.1 的 log-size bias 取自此（見 [ViT 的 token 剪枝與合併](tokc-vit-pruning-and-merging.html)）。RTI 以 e_size[⌊log₂|c|⌋] 嵌入區域大小。

## 位置

- 保留原位置：DC-DiT（arXiv 2603.06351）對選取的邊界 token 加上依原始 2D 座標的正弦位置嵌入。
- 軸向 RoPE：HDiT（arXiv 2401.11605）使用 RoPE 的改作（axial），分組固定為 2×2。
- latent 位置：Perceiver IO 類的 latent 無位置；ELIT（arXiv 2603.12245）以可學位置嵌入初始化 latent，並在群組間重用以編碼群內位置。
- 平均 RoPE 相量（RTI §3.4）：位置取區域內各 patch RoPE 相量的平均，等於依區域大小做低通；公式見 [Read / Write 介面](rti-read-write-interface.html)。
- HiT（附錄 B.1）：預設 compact 層不加 RoPE、全解析度層保留 RoPE，另有質心與學得 RoPE 變體；細節與 FID 見 [HiT 的 compact 階段位置編碼](hit-compact-position-and-joint-grouping.html)。

## 位置是否必要

RTI 在凍結骨幹上以 R = 256 latent 的 Latent Array 為基線，GenEval 僅 45.7（RTI 為 84.7）；其推估移除位置的代價為 39.0 GenEval 點（§4.2）。相似度合併的群組散布 5.16 格，RTI 區域為 0.49 格（Tab. 4）。HiT 從頭訓練，預設不加位置仍有效。lineage.md 對差異的可能解釋（推論，未經實驗驗證）：凍結骨幹無法學習新的位置機制，必須提供它看得懂的位置；從頭訓練的模型可經由 skip 與全解析度層取得位置資訊。

合併 token 最佳位置表示仍無定論，HiT 各變體差距小。兩篇終點論文見 [RTI 研究脈絡總覽](rti-lineage-hub.html)、[HiT 研究脈絡總覽](hit-lineage-hub.html)；切塊方法見 [動態切塊](tokc-dynamic-chunking-bytes-to-pixels.html)；latent 介面見 [沙漏結構與 latent 介面](tokc-hourglass-and-latent-interfaces.html)。
