---
slug: retrieval-augmented-external-memory
title: 檢索增強與外部記憶
cluster: Architectures & Efficiency
kind: topic
tags: [retrieval, non-parametric-memory, realm, retro]
sources: ["arXiv 2601.07372 全文（§1、§7 Related Work）"]
arxiv: ["2601.07372", "2002.08909", "2112.04426"]
related: ["engram-lineage-overview", "engram-module-design"]
---
# 檢索增強與外部記憶

屬於 [Engram 條件式記憶的研究脈絡總覽](engram-lineage-overview.html)。論文 §7 將非參數記憶方法描述為把知識儲存與模型處理分開，外部記憶是可編輯、可擴充的 key-value 儲存，模型不必重新訓練就能適應新資訊。

- **REALM**（Guu, Lee et al., 2020，arXiv:2002.08909，*REALM: Retrieval-Augmented Language Model Pre-Training*）。
- **RETRO**（Borgeaud, Mensch et al., 2022，arXiv:2112.04426，*Improving language models by retrieving from trillions of tokens*）。
- 論文同列 CoG 與 PlugLM，未逐一驗證。

## 與 Engram 的差別

Engram 的記憶表是模型參數，隨模型一起訓練，不是外部可編輯的語料索引；其查找鍵只由當前 token 的局部 N-gram 決定。因此它屬於論文所稱的參數式記憶一側，與本筆記的非參數方法相對照。
