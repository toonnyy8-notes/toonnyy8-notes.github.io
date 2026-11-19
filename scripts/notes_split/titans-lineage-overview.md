---
slug: titans-lineage-overview
title: Titans 的研究脈絡總覽：從 attention 與 linear recurrence 到測試時學習的長期記憶
cluster: Architectures & Efficiency
kind: hub
tags: [titans, long-term-memory, linear-attention, test-time-training]
arxiv: ["2501.00663", "1706.03762", "2006.16236", "2102.11174", "2312.00752", "2406.06484", "2407.04620", "2412.06464"]
sources: ["arXiv:2501.00663 §1, §2, Appendix A"]
related: ["delta-rule-deltanet-family", "gated-linear-recurrence-forgetting", "linear-attention-fast-weights", "memory-view-of-sequence-models", "recurrent-memory-segment-transformers", "titans-architectures-mac-mag-mal", "titans-neural-memory-surprise-update", "ttt-linear-mlp", "ttt-sequence-modeling-overview"]
---
# Titans 的研究脈絡總覽

Titans（arXiv:2501.00663，Behrouz、Zhong、Mirrokni，Google Research，2024-12-31 提交）要解決的問題是：attention 能精確建模上下文內的依賴，但成本隨長度平方成長；linear recurrent 模型成本低，但把整段歷史壓進固定大小的狀態。論文提出一個在測試時以梯度更新參數的神經長期記憶，再與 attention 組合。

閱讀順序如下，每一步對應論文引用的一條研究線。

1. [序列模型的記憶觀點](memory-view-of-sequence-models.html)：論文把 RNN 與 Transformer 都視為「寫入＋讀取」的記憶，並由此提出五個設計問題。
2. [Linear attention 與 fast weight](linear-attention-fast-weights.html)：softmax 換成核函數後得到遞迴形式 `M_t = M_{t-1} + K_t^T V_t`，其記憶是矩陣。
3. [閘控線性遞迴與遺忘](gated-linear-recurrence-forgetting.html)：Mamba、GLA、Griffin 等加入資料相依的遺忘，處理加法寫入造成的記憶溢出。
4. [Delta rule 與 DeltaNet 系列](delta-rule-deltanet-family.html)：寫入前先移除舊值，並可視為對重建損失做一步梯度下降。
5. [分段與外掛記憶](recurrent-memory-segment-transformers.html)：Transformer-XL、RMT 等以向量或快取在片段間傳遞資訊。
6. [Titans 的神經記憶與 surprise 更新](titans-neural-memory-surprise-update.html)：以動量與權重衰減的梯度下降更新深層 MLP 記憶。
7. [Titans 的三種架構](titans-architectures-mac-mag-mal.html)：MAC、MAG、MAL 三種把記憶接入模型的方式。

測試時訓練（TTT）這條線與 Titans 同期，已有現成筆記：[原版 TTT](ttt-linear-mlp.html) 與 [TTT 主系列總覽](ttt-sequence-modeling-overview.html)。Titans 在附錄 C 把 LMM 與 TTT 層、Gated DeltaNet 的差異逐項列出。
