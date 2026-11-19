---
slug: llmmem-explicit-lookup-memory-engram
title: 查表式外顯記憶：從 kNN-LM 到 Engram 的可定址儲存
cluster: Architectures & Efficiency
kind: topic
tags: [lookup-memory, engram, knn-lm, retrieval, explicit-memory]
arxiv: ["2607.25380", "1911.00172", "2306.07174", "2511.01581", "2602.00398", "2601.07372", "2402.04624"]
sources: ["arXiv:2607.25380 §IV-B, §IV-E, Table I, Table II"]
related: [engram-module-design, llmmem-explicit-parameterized-test-time-memory, llmmem-hybrid-management-evaluation, llmmem-moe-as-conditional-memory, llmmem-taxonomy-overview, llmmem-three-axes-taxonomy]
---
# 查表式外顯記憶

綜述（arXiv:2607.25380 §IV-B）把這一類定義為長期、可定址的儲存結構，以內容式檢索存取。條目獨立儲存，內容不必每次從頭重算，可跨推論實例保留。許多儲存體是離線建構，也有架構支援線上納入新表示而不更新主幹權重。

## 發展順序

1. kNN-LM（Khandelwal 等，arXiv:1911.00172）：在預訓練模型外加上由 context–target 配對組成的持久 datastore，以最近鄰檢索，降低 perplexity 並改善稀有 token 預測。論文認為它比一般 KV cache 更清楚地代表顯式記憶，因為知識存在獨立維護、以另一套機制查詢的非參數資源。
2. 邊界類別：Memorizing Transformers 把內部 key–value 表示存入不可微的記憶並以近似最近鄰檢索；LongMem（arXiv:2306.07174）把凍結主幹抽出的 key–value 存入快取記憶庫，再以 SideNet 檢索並融合。論文把它們視為顯式 datastore 與隱式 activation 記憶之間的邊界類別，不放進主分類表。
3. 可編輯的記憶庫：PlugLM 以可微的外掛 key–value 記憶取代部分 FFN 層，起點是 FFN 本來就像隱式 key–value 儲存；ExplicitLM（arXiv:2511.01581）用逐層的顯式記憶庫，條目是人可讀的 token 序列，並以兩階段可微檢索存取。論文強調它們不是 RAG，因為記憶庫作為架構元件被訓練與呼叫，且提供在糾纏的參數記憶中難以做到的定向更新。
4. FFN 作為查表：MemoryLLM（Plug-n-play 可解釋 feed-forward 記憶，arXiv:2602.00398）把 FFN 重新詮釋為不依賴 context 的逐 token 檢索模組，從 token embedding 直接訓練解耦的 FFN 記憶，輸出可預先計算成 token 層級查表。論文指出它與可自我更新的 MEMORYLLM（arXiv:2402.04624）概念不同，後者是推論時重新整理的潛在記憶池。
5. Engram（Cheng 等，arXiv:2601.07372，論文標題為 Conditional memory via scalable lookup: A new axis of sparsity for large language models）：資訊存在稀疏啟用的槽位，以結構化雜湊檢索；模型依當前隱藏狀態做選擇性查表，不經完整 attention 重算歷史。論文強調它把記憶稀疏性與專家稀疏性分開：MoE 稀疏啟用參數化轉換，Engram 稀疏定址儲存的條目。核心貢獻是容量可獨立於稠密計算成長的定址基底，受限於索引方案的準確度與碰撞行為。Table I 把它標為離線、長期、雜湊查表槽。模組本身的設計見 [Engram 模組的設計](engram-module-design.html)。

## 與 RAG 的界線

論文明確區分：RAG 依賴外部文件語料、向量資料庫與編排流程，位於模型架構之外；這裡討論的是把持久儲存嵌入模型結構的機制，儲存庫（矩陣、datastore 或槽位）是架構元件。

## 結構性質與風險

論文列出三項性質：條目可個別定址、不被壓進單一演化的隱藏狀態；容量隨槽位數或 datastore 大小成長，不受固定隱藏維度限制；檢索通常是內容式，帶來稀疏與條件計算。更新規則以結構性或 admission 為主，表可離線建構，推論時動態決定哪些條目可被存取。挑戰是記憶成長、索引效率與干擾管理，以及過時或重複條目造成的檢索偏差。

與條件參數路由的對照見 [MoE 作為條件參數記憶](llmmem-moe-as-conditional-memory.html)；與測試時寫入的對照見 [參數化外顯記憶](llmmem-explicit-parameterized-test-time-memory.html)；兩者在架構中的搭配見 [混合架構、記憶管理與評估](llmmem-hybrid-management-evaluation.html)。

## 相關筆記

- [LLM 記憶機制的分類總覽：從 attention 與遞迴狀態到可定址的外顯記憶](llmmem-taxonomy-overview.html)
- [LLM 記憶的三條分類軸：representation、update dynamics、persistence](llmmem-three-axes-taxonomy.html)
