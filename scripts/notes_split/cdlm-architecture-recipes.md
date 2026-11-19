---
slug: cdlm-architecture-recipes
title: 連續擴散語言模型的架構與訓練配方
cluster: Diffusion Language Models
kind: topic
tags: [dit, architecture, cfg, repa, hyperparameters, continuous-diffusion-lm]
sources: [flow-matching-lm-2026.md]
arxiv: ["2610.22330", "2605.11125", "2605.10938", "2609.29102", "2604.09784", "2610.02665"]
related: ["continuous-flow-lm-overview", "optimization-dilemma-semantic-alignment", "cdlm-open-problems"]
---
# 連續擴散語言模型的架構與訓練配方

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。

- **DiT 骨幹：** 小規模工作多沿用 SEDD / MDLM 的 DiT（約 170M），以 adaLN 注入時間。FMLM 對 s、t 雙時間做 adaLN 條件化。
- **LangDiT（2610.22330）：**
  - 系統比較 Transformer 設計後發現，因果 LLM 或影像 DiT 的有效設計不一定適用於 CDLM。 [arxiv](https://arxiv.org/html/2610.22330v1)
  - 採用 QK normalization 等設計，learned embedding 的 row norm 設為 0.1√d。 [arxiv](https://arxiv.org/html/2610.22330v1)
  - 作為 LangFlow 的骨幹時，OWT Gen PPL 在 1024 NFE 為 21.72，在 128 NFE 為 44.72。 [arxiv](https://arxiv.org/abs/2610.22330)
- **S-arch（S-FLM）：** 參考 nGPT，使激活值保持在球面上。在 T=1 下，GSM8K 由 11.1% 提升到 12.4%。作者也寫明 S-arch 訓練較慢。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
- **ELF：**
  - logit-normal 時間取樣、training-time CFG、in-context conditioning、128 維 bottleneck、Muon 優化器。 [arxiv](https://arxiv.org/pdf/2605.10938)
  - 約 45.2B 有效訓練 tokens（OWT 5 epochs）；同篇論文稱其他 DLM 通常使用超過 500B。 [arxiv](https://arxiv.org/pdf/2605.10938)
  - 模型大小：ELF-B 105M、ELF-M 342M、ELF-L 652M。 [arxiv](https://arxiv.org/pdf/2605.10938)
  - WMT14 De-En BLEU 26.4（AR 25.2）；XSum ROUGE-1/2/L 36.0 / 12.2 / 27.8。 [arxiv](https://arxiv.org/pdf/2605.10938)
- **ELF-REG（2609.29102）：**
  - 以 REPA + REG 對齊凍結的 AR teacher 表示（REPA 本身見 [語意對齊與 optimization dilemma](optimization-dilemma-semantic-alignment.html)）。
  - GSM8K 55.96%（64 NFE）、MATH-500 13.39%、HumanEval 22.56%。
- **CFG：**
  - DFM 的 block 條件生成中，ω 由 0 增至 2 時，Gen PPL 由 56.31 降到 30.98，entropy 由 5.20 降到 4.81。 [arxiv](https://arxiv.org/pdf/2604.09784)
  - Sigma 在 embedding 空間使用 CFG 與 score temperature，每次前向開銷少於 5%。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- **優化超參數：** Scaling CFM 在 400M 規模做了 702 + 324 組掃描，Gen PPL 與 MAUVE 主要受學習率與 weight decay 影響。β₂=0.95 優於較大值。Complete(d)P 超參數遷移反而使 Gen PPL 變差 35%。時間條件層不在 µP / Complete(d)P 的涵蓋範圍內（見 [未解決問題](cdlm-open-problems.html)）。
