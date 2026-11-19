---
slug: llmmem-taxonomy-overview
title: LLM 記憶機制的分類總覽：從 attention 與遞迴狀態到可定址的外顯記憶
cluster: Architectures & Efficiency
kind: hub
tags: [memory, taxonomy, survey, implicit-memory, explicit-memory]
arxiv: ["2607.25380", "1706.03762", "2312.00752", "2501.00663", "2512.23675", "2601.07372", "2512.24695"]
sources: ["arXiv:2607.25380 Abstract, §I, §II, Fig. 2, Table I"]
related: ["llmmem-explicit-lookup-memory-engram", "llmmem-explicit-parameterized-test-time-memory", "llmmem-hybrid-management-evaluation", "llmmem-implicit-attention-access-control", "llmmem-implicit-recurrent-state", "llmmem-moe-as-conditional-memory", "llmmem-multi-timescale-nested-updates", "llmmem-three-axes-taxonomy", "memory-view-of-sequence-models", "titans-lineage-overview"]
---
# LLM 記憶機制的分類總覽

《Memory for Large Language Models》（arXiv:2607.25380，Zhoubian、Zhang、Kharlamov、Tang，2026-07-28 提交）是一篇綜述。它處理的問題是：attention 快取、遞迴狀態、測試時適應、查表式儲存與條件參數路由，常被各自獨立研究，但回答的是同一組問題，即儲存什麼、何時以何種方式更新、與前向計算如何互動、保留多久。論文提出以三條正交軸分類，並限定範圍為模型層級的記憶，不含 agent 層級或 prompt 層級的記憶系統，也不含預訓練後固定的一般權重。

閱讀順序如下。

1. [三條分類軸與更新規則](llmmem-three-axes-taxonomy.html)：先建立詞彙，包含 representation、update dynamics、persistence，以及 Table II 的五種更新規則。
2. [Attention 作為隱式記憶](llmmem-implicit-attention-access-control.html)：KV cache、稀疏與串流存取，記憶容量由 context 長度決定。
3. [遞迴序列記憶](llmmem-implicit-recurrent-state.html)：把歷史壓進固定大小狀態；線性 attention、SSM、delta rule 這一支在 Titans 的研究脈絡中已有專門筆記，見 [Titans 的研究脈絡總覽](titans-lineage-overview.html)。
4. [參數化外顯記憶與測試時更新](llmmem-explicit-parameterized-test-time-memory.html)：Titans、TTT-E2E、In-Place TTT；序列模型的寫入與讀取設計見 [序列模型的記憶觀點](memory-view-of-sequence-models.html)。
5. [查表式外顯記憶](llmmem-explicit-lookup-memory-engram.html)：kNN-LM 到 Engram，容量可獨立於稠密計算成長。
6. [MoE 作為條件參數記憶](llmmem-moe-as-conditional-memory.html)：離線、長期、以路由定址的專家。
7. [多時間尺度與巢狀更新](llmmem-multi-timescale-nested-updates.html)：Nested Learning 與穩定性–可塑性取捨。
8. [混合架構、記憶管理與評估](llmmem-hybrid-management-evaluation.html)：把前述機制組合起來的設計、系統層級成本與評估，以及論文列出的開放問題。

## 對應表

Table I 以 representation 分成隱式與顯式兩大類。隱式記憶（步驟 2、3）的內容是前向計算的副產物，推論結束即丟棄。顯式記憶（步驟 4 到 7）是有獨立定址與更新語義的儲存元件。論文的結論是沒有單一範式同時滿足所有需求：token 層級 attention 召回精確但難以擴展，遞迴記憶壓縮後可涵蓋長距離但有干擾，顯式記憶提高持久性但增加更新穩定性與定址的複雜度，因此設計趨向混合架構。

本綜述引用的代表性工作包含 Transformer（arXiv:1706.03762）、Mamba（arXiv:2312.00752）、Titans（arXiv:2501.00663）、TTT-E2E（arXiv:2512.23675）、Engram（arXiv:2601.07372）與 Nested Learning（arXiv:2512.24695），分別落在後續各筆記。
