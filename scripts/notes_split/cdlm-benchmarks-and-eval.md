---
slug: cdlm-benchmarks-and-eval
title: 連續擴散語言模型的基準結果對照與 Gen PPL 評估陷阱
cluster: Diffusion Language Models
kind: topic
tags: [benchmarks, generative-perplexity, evaluation, gsm8k, owt, lm1b]
sources: [flow-matching-lm-2026.md]
arxiv: ["2610.22330", "2605.10938", "2604.09784", "2605.11125", "2610.13834", "2606.08417", "2607.00588", "2610.00497"]
related: ["continuous-flow-lm-overview", "cdlm-scaling", "flow-map-few-step-distillation", "self-conditioning-flow-lm", "generative-eval-metrics"]
---
# 連續擴散語言模型的基準結果對照與 Gen PPL 評估陷阱

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。大模型（Sigma）結果見 [規模化策略](cdlm-scaling.html)；少步表格見 [少步蒸餾](flow-map-few-step-distillation.html)。

## LM1B / OWT 多步

| 模型 | 資料 | PPL（上界） | Gen PPL | 設定 |
|---|---|---|---|---|
| LangFlow | LM1B | 30.0 | — | ODE NLL 上界 |
| LangFlow | OWT | 24.6 | 36.5 | — |
| LangFlow + LangDiT | OWT | — | 21.72 | 1024 NFE |
| LangFlow + LangDiT | OWT | — | 44.72 | 128 NFE |
| RePlaid | OWT | 22.1 | — | ELBO |
| ELF | OWT | — | 24 | 32 步，SDE，CFG=3；去除重複後 ELF-B 由 19.5 → 27.7 |
| DFM 對角（未蒸餾） | OWT | — | 47.07 / entropy 5.12 | 1024 步 |
| DFM ESD | OWT | — | 36.48 / 4.95 | 1024 步 |

（LangDiT：[arxiv](https://arxiv.org/abs/2610.22330)；ELF：[arxiv](https://arxiv.org/pdf/2605.10938)；DFM：[arxiv](https://arxiv.org/pdf/2604.09784)）

## 推理任務（TinyGSM 訓練、GSM8K 零樣本、約 170M）

| 模型 | GSM8K |
|---|---|
| AR（sample / greedy） | 53.9 / 63.3 |
| MDLM / Duo（T=1） | 18.0 / 17.2 |
| MDLM / Duo（T=0.1） | 約 33 / 約 36 |
| FLM | 0.3 |
| CANDI | 0.2 |
| S-FLM（T=1，top-1 velocity） | 18.0 |
| S-FLM（T=0.1） | 約 18 |
| FMLM+ PR（32 NFE） | 19.0 |
| FMLM+ Init（64 NFE） | 33.6 |
| SemiCat（調過取樣器） | 49.94 |

（AR 至 S-FLM 列：[arxiv](https://arxiv.org/pdf/2605.11125v3)；SemiCat：[arxiv](https://arxiv.org/abs/2610.13834)）

- S-FLM 在 GSM8K 上的逐項消融：1.2 → 7.7（截斷）→ 8.1（不重新正規化）→ 11.1（自適應排程）→ 12.4（S-arch）→ 18.0（top-1）。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
- 在這個基準上，one-hot FLM 幾乎無法解題。 [arxiv](https://arxiv.org/pdf/2605.11125v3)

## 評估方法的限制
- **Gen PPL 可被操縱。** Franca & Tong 的 Hacking Generative Perplexity（2606.08417）構造了零參數的簡單取樣器。這些取樣器在 LM1B 與 OWT 上以非退化的 entropy 取得 SOTA Gen PPL，輸出卻在構造上就不連貫。作者建議改用分佈距離指標。
- **重複內容會壓低 Gen PPL。** 2607.00588 顯示，帶 self-conditioning 的連續模型重複遠多於人類文本，而 Gen PPL 獎勵重複（機制見 [self-conditioning](self-conditioning-flow-lm.html)）。 [arxiv](https://arxiv.org/abs/2607.00588) [arxiv](https://arxiv.org/html/2607.00588)
- **少步退化也會壓低 Gen PPL。** Scaling CFM 的未蒸餾模型在 1 步時的 Gen PPL 為 8.11，但輸出是「the the the」。DFM ESD 在 OWT 1 步時 Gen PPL 5.33、entropy 0.26。 [arxiv](https://arxiv.org/pdf/2604.09784)
- **替代指標。** S-FLM 改用有標準答案的任務（Sudoku、GSM8K），並畫出溫度掃描下的 Gen PPL–entropy 前緣。 [arxiv](https://arxiv.org/pdf/2605.11125v3) GSF 使用 Gradient Moment。 [arxiv](https://arxiv.org/html/2610.00497) Scaling CFM 使用 MAUVE⋆ 與 ΔPPL。
- **跨論文比較的限制。** Gen PPL 評分模型不一致：GPT-2 Large 與 Qwen2.5-7B 都有人用。tokenizer、序列長度與溫度設定也不同，跨論文的數字只能作為參考。

## 評估時的做法（建議）
引用任何 Gen PPL 數字時，應同時檢查 entropy、重複率（Rep-n）、評分模型與溫度，並優先參考 MAUVE、GSM8K、HumanEval 這類有標準答案或分佈距離的指標。

影像生成端類似的「指標被 Goodhart 化」問題見 [生成評估指標的演進](generative-eval-metrics.html)。
