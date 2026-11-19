---
slug: llmmem-moe-as-conditional-memory
title: MoE 作為條件參數記憶：離線、長期、以路由定址
cluster: Architectures & Efficiency
kind: topic
tags: [mixture-of-experts, conditional-memory, routing, explicit-memory]
arxiv: ["2607.25380", "2101.03961", "2112.06905", "2401.04088", "2401.06066", "2601.07372"]
sources: ["arXiv:2607.25380 §IV-C, Table I"]
related: [moe-conditional-computation-lineage, llmmem-explicit-lookup-memory-engram, llmmem-explicit-parameterized-test-time-memory, llmmem-taxonomy-overview, llmmem-three-axes-taxonomy]
---
# MoE 作為條件參數記憶

綜述（arXiv:2607.25380 §IV-C）把 Mixture-of-Experts 歸為嵌在參數空間的離線顯式記憶。稠密 Transformer 層的所有參數都參與每次前向；MoE 維護一組專家子網路，由學到的路由函數依輸入只啟用稀疏子集合。給定隱藏狀態 h，路由器對 E 個專家產生稀疏閘控分布，輸出為被選專家輸出以閘控權重加總。

發展脈絡（Switch Transformer、GLaM、Mixtral、DeepSeek-MoE 及其論文描述）見 [MoE 與條件式計算的發展](moe-conditional-computation-lineage.html)。

## 記憶詮釋

每個專家是一個持久的參數化記憶區塊，路由網路對這組儲存做依情境的定址。與檢索式機制不同，MoE 從參數空間取出的是轉換，不是外部 key–value 條目；但兩者共有「對模組化知識儲存做明確的條件式存取」這個結構原則。論文提到的分析顯示專家傾向聚集在語言、推理或領域子功能上，因此支持模組化持久知識記憶的看法。論文把 MoE 放在參數化記憶與查表記憶之間。

## 與 Engram 的區別

Engram（arXiv:2601.07372）區分記憶稀疏性與專家稀疏性：MoE 稀疏啟用參數化轉換，Engram 稀疏定址儲存條目，因此查表記憶不能簡化為條件計算。詳見 [查表式外顯記憶](llmmem-explicit-lookup-memory-engram.html)。MoE 是離線更新，推論期的寫入由 [參數化外顯記憶與測試時更新](llmmem-explicit-parameterized-test-time-memory.html) 處理。在 Table II 中，MoE 與 Engram 同屬 objective-induced or structural updates，風險是僵固與測試時適應受限。

## 相關筆記

- [LLM 記憶機制的分類總覽：從 attention 與遞迴狀態到可定址的外顯記憶](llmmem-taxonomy-overview.html)
- [LLM 記憶的三條分類軸：representation、update dynamics、persistence](llmmem-three-axes-taxonomy.html)
