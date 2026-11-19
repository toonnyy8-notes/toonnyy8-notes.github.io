---
slug: gated-linear-recurrence-forgetting
title: 閘控線性遞迴：以資料相依的遺忘處理記憶溢出
cluster: Architectures & Efficiency
kind: topic
tags: [ssm, mamba, gating, forgetting]
arxiv: ["2501.00663", "2312.00752", "2405.21060", "2312.06635", "2402.19427", "2405.04517"]
sources: ["arXiv:2501.00663 §2.1, §3.1, Appendix A.1", "各論文的題名與作者經 arXiv API 查證；內容依 Titans 的描述"]
related: ["delta-rule-deltanet-family", "titans-neural-memory-surprise-update"]
---
# 閘控線性遞迴與遺忘

Titans 在 §2.1 指出，加法寫入的記憶需要兩個改進方向，第一個是**加入遺忘機制**：以資料相依的閘讓模型在需要時清除記憶。論文列舉的代表：

- GLA：Gated Linear Attention Transformers with Hardware-Efficient Training（arXiv:2312.06635，Yang 等，2023）。
- Griffin：Mixing Gated Linear Recurrences with Local Attention（arXiv:2402.19427，De 等，2024）。
- xLSTM：Extended Long Short-Term Memory（arXiv:2405.04517，Beck 等，2024）。
- Mamba：Linear-Time Sequence Modeling with Selective State Spaces（arXiv:2312.00752，Gu 與 Dao，2023），與其後的 Mamba2（Transformers are SSMs，arXiv:2405.21060，Dao 與 Gu，2024）。論文說 Mamba2 與離散化的狀態空間模型有關。

Titans 附錄 A.1 把線性遞迴模型分成三代：第一代使用與資料無關的轉移或衰減（RetNet、LRU、RWKV、S4、S5）；第二代加入閘控（Griffin、Mamba 等）；第三代使用以元學習、線上學習或 delta rule 為基礎的更新（Longhorn、Gated DeltaNet、TTT、DeltaNet），見 [Delta rule 系列](delta-rule-deltanet-family.html)。

Titans 的式 13 以 `(1-α_t)` 衰減舊記憶，論文表示這是現代遞迴模型遺忘機制的推廣，並等同於權重衰減，見 [Titans 的神經記憶](titans-neural-memory-surprise-update.html)。
