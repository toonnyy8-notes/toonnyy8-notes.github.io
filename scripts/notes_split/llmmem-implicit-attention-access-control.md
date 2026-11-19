---
slug: llmmem-implicit-attention-access-control
title: Attention 作為隱式記憶：KV cache、串流視窗與稀疏存取控制
cluster: Architectures & Efficiency
kind: topic
tags: [attention, kv-cache, sparse-attention, streaming, implicit-memory]
arxiv: ["2607.25380", "1706.03762", "1904.10509", "2007.14062", "2309.17453", "2405.04434"]
sources: ["arXiv:2607.25380 §III-A, §III-B, §III-D, §V-B"]
related: ["llmmem-explicit-parameterized-test-time-memory", "llmmem-hybrid-management-evaluation", "llmmem-implicit-recurrent-state", "llmmem-taxonomy-overview", "llmmem-three-axes-taxonomy", "memory-view-of-sequence-models"]
---
# Attention 作為隱式記憶

綜述（arXiv:2607.25380 §III）把 Transformer（arXiv:1706.03762）的 attention 視為內容可定址的工作記憶：query 與過去所有 key 比對，輸出是 value 的加權和。它是隱式的，因為存取語義由架構固定，內容隨前向計算產生並在推論結束時丟棄；分類上屬線上、短期。

## 容量與 context 長度

attention 記憶的容量與 context window 直接綁定，KV 儲存隨序列長度線性成長，計算成本隨長度平方成長。論文提到兩類緩解方式。一類縮小每個 token 的儲存量，例如 DeepSeek-V2 的 Multi-head Latent Attention（arXiv:2405.04434）把 key 與 value 聯合壓縮成潛在向量，改變的是粒度與效率，不改變隱式語義。另一類限制可存取的子集合。論文也引用研究指出，延長 context 不等於有效利用，名目容量與實際使用之間有差距。

## 串流與視窗

StreamingLLM（arXiv:2309.17453）結合滑動視窗與少量保留的 attention sink token，不需重新訓練即可穩定處理無界輸入流。論文把它解讀為隱式的逐出策略：多數過去狀態持續被覆寫，少數被保留作為 attention 的錨點。

## 稀疏、選擇與局部–全域

Sparse Transformer（arXiv:1904.10509）、BigBird（arXiv:2007.14062）與 Longformer 以局部視窗、跨步模式或少量全域 token 限制每步可存取的位置。論文的觀點是這些方法沒有增加記憶容量，而是在固定隱式記憶預算內重塑存取路徑；所有 token 仍被編碼，只是同一時間只有子集合可被讀取。

後續工作把固定模式換成學習式路由。論文描述 MoBA 把 KV context 分塊並讓每個 query 路由到少數相關區塊，NSA 結合壓縮區塊摘要、選擇性保留的細粒度區塊與局部視窗；兩者仍是 attention 式隱式記憶，貢獻在於自適應的存取控制。RATTENTION 分析滑動視窗大小與保留的全域 token 數的取捨，結論是小而經過挑選的全域記憶能明顯改善效能。Selective Attention 以資料相依的閘控調節 attention 分數，在論文的分類中扮演隱式的 admission 控制。

## 限制

論文列出隱式記憶的共同限制：容量有界且難以控制、缺少可調整的讀寫語義、跨 context 的持久性有限。這些限制促成下一類設計，即把儲存從計算中分離，見 [遞迴序列記憶](llmmem-implicit-recurrent-state.html) 與 [參數化外顯記憶](llmmem-explicit-parameterized-test-time-memory.html)。序列模型的寫入、讀取設計問題另見 [序列模型的記憶觀點](memory-view-of-sequence-models.html)。系統層級的 KV 管理（量化、分頁、整併）在 [混合架構、記憶管理與評估](llmmem-hybrid-management-evaluation.html)。

## 相關筆記

- [LLM 記憶機制的分類總覽：從 attention 與遞迴狀態到可定址的外顯記憶](llmmem-taxonomy-overview.html)
- [LLM 記憶的三條分類軸：representation、update dynamics、persistence](llmmem-three-axes-taxonomy.html)
