---
slug: llmmem-three-axes-taxonomy
title: LLM 記憶的三條分類軸：representation、update dynamics、persistence
cluster: Architectures & Efficiency
kind: topic
tags: [memory, taxonomy, update-rules]
arxiv: ["2607.25380"]
sources: ["arXiv:2607.25380 §II-A to §II-C, Fig. 2, Table II"]
related: ["llmmem-explicit-lookup-memory-engram", "llmmem-explicit-parameterized-test-time-memory", "llmmem-implicit-attention-access-control", "llmmem-taxonomy-overview", "memory-view-of-sequence-models", "titans-neural-memory-surprise-update", "ttt-e2e"]
---
# 三條分類軸

綜述（arXiv:2607.25380 §II）把「記憶」限定為在模型架構或推論動態內部實例化的機制，再用三條軸描述，並說明這是概念框架而不是嚴格分類。

| 軸 | 取值 | 判準 |
| --- | --- | --- |
| Representation | 隱式 / 顯式 | 記憶是否有獨立的定址、讀取、更新語義 |
| Update dynamics | 離線 / 線上 | 更新是否只發生在訓練期，或也發生在推論期 |
| Persistence | 短期 / 長期 | 保留範圍限於 context window，或跨 context、跨 session |

## Representation

隱式記憶是計算的副產物，包含 KV cache 與隱藏狀態。KV cache 雖然實體化成張量，其存取語義由架構固定，沒有獨立的定址與更新控制，因此仍算暫態的計算狀態。顯式記憶是獨立的儲存元件，可以是外部表、關聯緩衝區或專用記憶模組，也可以是以特殊規則更新的額外參數。論文強調判準不是參數化與否，而是該元件是否被設計並當作記憶來操作，有明確的讀寫語義與可控的更新策略。

## Update dynamics

離線記憶只在訓練期更新，推論時固定。線上記憶可在推論時更新，頻率可以是每個 token、每段或每批，更新規則可以不同於標準反向傳播。

## Persistence

短期記憶受限於 context window。長期記憶的特徵是有效範圍較長，來源可以是高壓縮、選擇性保留或持久儲存，論文明確說不是只靠儲存量大小。三條軸大致正交，例如顯式記憶可以離線也可以線上，線上記憶可以短期也可以長期。

## 更新規則（Table II）

離線 / 線上只說明何時可以更新，不說明內容如何改變，因此論文在 update dynamics 軸下加入第二層視角，列出五類規則，並註明這不是第四條主軸，規則之間也不互斥。

| 更新規則 | 機制 | 代表例（論文所列） | 論文所列主要風險 |
| --- | --- | --- | --- |
| Optimization-based writing | 以最小化明確目標的方式更新記憶參數或 fast weights | TTT-E2E、Titans、In-Place TTT、Nested Learning | 漂移、不穩定、目標不一致 |
| State-transition updates | 以學到的轉移、delta rule、投影或濾波式更新狀態 | Mamba、Gated Delta Networks、Gated DeltaNet-2 | 壓縮損失、狀態干擾 |
| Signal-gated writing or routing | 由 surprise、不確定性、預測誤差決定是否寫入或啟用 | Titans、GDWM、AMOR、HAM | 校準、門檻敏感度、相關性延遲 |
| Admission, eviction, consolidation | 決定哪些條目被納入、保留、壓縮、逐出或改寫 | StreamingLLM、RATTENTION、HAM、Bottlenecked Transformers | 不可逆損失、檢索偏差 |
| Objective-induced or structural updates | 由輔助目標、架構排程或離線建構決定記憶行為 | Engram、MoE、Priming、Next-Latent Prediction | 僵固、目標不一致、測試時適應受限 |

後續筆記依這個框架逐類說明各機制。Titans 與 TTT-E2E 的細節見 [Titans 的神經長期記憶](titans-neural-memory-surprise-update.html) 與 [TTT-E2E](ttt-e2e.html)。

## 相關筆記

- [查表式外顯記憶：從 kNN-LM 到 Engram 的可定址儲存](llmmem-explicit-lookup-memory-engram.html)
- [參數化外顯記憶：fast weights、Titans、TTT-E2E 與測試時更新](llmmem-explicit-parameterized-test-time-memory.html)
- [Attention 作為隱式記憶：KV cache、串流視窗與稀疏存取控制](llmmem-implicit-attention-access-control.html)
- [LLM 記憶機制的分類總覽：從 attention 與遞迴狀態到可定址的外顯記憶](llmmem-taxonomy-overview.html)
- [序列模型的記憶觀點：寫入、讀取與五個設計問題](memory-view-of-sequence-models.html)
