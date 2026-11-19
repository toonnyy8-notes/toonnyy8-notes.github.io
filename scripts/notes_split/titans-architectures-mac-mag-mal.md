---
slug: titans-architectures-mac-mag-mal
title: Titans 的三種架構：記憶作為上下文、閘控分支與一層
cluster: Architectures & Efficiency
kind: topic
tags: [titans, mac, mag, mal, persistent-memory, sliding-window-attention]
arxiv: ["2501.00663"]
sources: ["arXiv:2501.00663 §3.3, §4, §5"]
related: ["titans-neural-memory-surprise-update"]
---
# Titans 的三種架構

Titans 由三部分組成：**核心**（視窗有限的 attention，短期記憶）、**長期記憶**（見 [神經記憶](titans-neural-memory-surprise-update.html)）與**持久記憶**。持久記憶是 `N_p` 個可學習、與資料無關的向量 `p_1…p_Np`，接在序列最前面（式 19），推論時固定，用來儲存任務知識；論文並指出它可緩解 causal attention 偏重開頭 token 的現象。

**MAC（Memory as a Context）**：序列切成片段；以片段查詢長期記憶得到 `h_t`，與持久記憶、當前片段串接後做 attention（式 21–23）；attention 輸出用來更新記憶（式 24）。論文提出兩個優點：attention 能決定是否需要長期記憶，且能幫記憶只存有用資訊。

**MAG（Memory as a Gate）**：不分段；一支以 sliding window attention 處理（含持久記憶前綴），另一支由神經記憶處理整個輸入，兩者以非線性閘合併（式 26–28）。

**MAL（Memory as a Layer）**：先過記憶層再接 sliding window attention（式 29–31）。論文指出缺點是能力受限於單一層，無法利用兩者的互補。另有不含 attention 的 LMM 變體。

實驗涵蓋語言建模、常識推理、needle-in-haystack（RULER 的 S-NIAH）、BABILong、時間序列與 DNA。論文表示 Titans 可擴展到超過 2M 的上下文長度；此為論文自述，未在此重現。論文 Theorem 4.1 主張 Titans 能解 TC⁰ 以外的問題。
