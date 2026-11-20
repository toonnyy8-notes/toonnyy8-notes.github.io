---
slug: soar-curriculum-and-environment-design
title: 課程學習與非監督環境設計
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [curriculum-learning, UED, PAIRED, PLR, SEC]
arxiv: ["2601.18778", "2012.02096", "2010.03934", "2505.14970"]
sources: ["arXiv:2601.18778 §2, §A.1（本文對各工作的描述）", "arXiv API（題名、作者、年份、ID 核對）"]
related: ["soar-asymmetric-self-play-origins", "soar-llm-data-free-self-play"]
---
# 課程學習與非監督環境設計

本文 §2、§A.1 把課程相關工作分成兩類。

**重排或挑選既有資料。** 起點為 Bengio et al.（2009，Curriculum learning，ICML）與 Graves et al.（2017，Automated curriculum learning for neural networks）。Narvekar et al.（2020，JMLR，RL 領域課程學習的框架與綜述）與 Parashar et al.（2026）屬於 RL 中由易到難的課程。LLM 後訓練的課程以梯度範數或優勢（advantage）估計等代理訊號挑選題目，例：Self-Evolving Curriculum（Chen et al.，arXiv 2505.14970，2025）、Online Difficulty Filtering（Bae et al.，2026）、DAPO 的動態取樣。本文指出這些方法預設難度可事先分桶，且只間接獎勵 student 的進步。

**生成環境分佈。** Dennis et al.（arXiv 2012.02096，2020，Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design）、Racaniere et al.（2020）、Prioritized Level Replay（Jiang et al.，arXiv 2010.03934，2020）與 Jiang et al.（2021）。本文描述這些工作發現無約束的目標（如 minimax 對抗）會產生退化的課程，因此改為最佳化 regret 或學習潛力。

SOAR 與兩類的差異（本文自述）：目標不是重排資料，而是在固定且可驗證的困難資料集上生成任務；獎勵直接綁定 student 在固定目標集上的進步。非對稱 self-play 的來源見 [非對稱 self-play 的來源](soar-asymmetric-self-play-origins.html)，LLM 上的資料自由 self-play 見 [LLM 資料自由 self-play 與自我獎勵的崩潰](soar-llm-data-free-self-play.html)。
