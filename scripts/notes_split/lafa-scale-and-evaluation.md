---
slug: lafa-scale-and-evaluation
title: 規模、資料量與評估（Lafa 與其他大型擴散模型的比較）
cluster: Diffusion Language Models
kind: topic
tags: [scaling, benchmarks, token-efficiency, gen-ppl]
arxiv: ["2610.28559", "2508.15487", "2512.14067", "2506.17298", "2508.02193", "2605.18530", "2606.08417", "2604.07023"]
sources: ["paper.txt §1, §3, §4", "arXiv abstracts"]
related: ["lafa-lineage-overview", "lafa-self-speculative-decoding", "lafa-ar-to-block-adaptation", "cdlm-scaling", "cdlm-benchmarks-and-eval"]
---
# 規模、資料量與評估

屬於 [Lafa 研究脈絡總覽](lafa-lineage-overview.html)。

## 規模
- 論文指出既有 flow 語言模型只在小規模與簡單任務上展示。
- Lafa 訓練 0.5B、7B、8B 三種規模，使用少於 30B tokens；論文對照的擴散模型使用最多 500B tokens。
- 離散擴散在數十億參數規模的代表有 Dream 7B（2508.15487）、Mercury（2506.17298，商用規模）、Seed Diffusion（2508.02193）。連續擴散方面，RePlaid（2605.18530）摘要稱其重新檢視 Plaid 並顯示連續擴散可與離散擴散競爭。各模型規模比較見 [規模化策略與大模型結果](cdlm-scaling.html)。

## 評估
- 論文以指令遵循、數學與程式基準評估，比較對象包含 AR SFT（Qwen2.5 7B）、MARS、DeepSeek-MTP、Efficient-DLM、Fast-dLLM v2、Dream（Figure 1）。
- 以 generative perplexity 評估無條件文字生成有已知問題：Hacking Generative Perplexity（2606.08417）摘要稱 gen-PPL 不可靠。討論見 [基準結果對照與 Gen PPL 評估陷阱](cdlm-benchmarks-and-eval.html)。
- 吞吐量取決於硬體與批次設定，引用 2.02–2.54 倍時需回查論文實驗設定。

## 未驗證
- 二手來源提到的 macro-average 數字未在此引用，現有筆記 [規模化策略](cdlm-scaling.html) 也標示其無法從原文表格直接核對。
