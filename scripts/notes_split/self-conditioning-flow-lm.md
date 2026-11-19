---
slug: self-conditioning-flow-lm
title: self-conditioning 在連續 flow 語言模型中的作用與理論解釋
cluster: Diffusion Language Models
kind: topic
tags: [self-conditioning, fixed-point, repetition, continuous-diffusion-lm]
sources: [flow-matching-lm-2026.md]
arxiv: ["2610.13834", "2607.00714", "2608.22244", "2607.00588"]
related: ["continuous-flow-lm-overview", "cdlm-scaling", "cdlm-benchmarks-and-eval"]
---
# self-conditioning 在連續 flow 語言模型中的作用與理論解釋

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。

**舊做法：** Analog Bits（Chen et al. 2022）提出 self-conditioning，訓練時以 50% 機率把上一步的預測作為額外輸入。CDCD、Plaid 沿用此做法。

## 新的實證
- LangFlow：self-conditioning 使 LM1B Gen PPL 由 154 降到 81.5，PPL 由 49 降到 30。
- Design Space（2610.13834）的解釋：self-conditioning 修正 train/inference mismatch，也解決 token identifiability 問題。
- RePlaid 追上 AR 的算力倍數：有 SC 約 20×，無 SC 約 27×（見 [規模化策略](cdlm-scaling.html)）。

## 新的理論解釋
- **FMLM⋆（2607.00714）** 證明帶 self-conditioning 的 flow LM 在做定點迭代（fixed-point iteration），並由此定義 fixed-point flow。這類 flow 有對應的 flow map，可以同時壓縮定點迭代與 flow。作者報告 OWT 一步與少步結果達到 SOTA。 [arxiv](https://arxiv.org/html/2607.00714) [arxiv](https://arxiv.org/pdf/2607.00714)
- **Untied Self-Conditioning（2608.22244）** 指出另一種推論時的不匹配：步長大時，solver 會耦合 latent 與 self-conditioning 兩條通路，而訓練時沒有這種耦合。 [arxiv](https://arxiv.org/pdf/2608.22244)
  - 方法是不需重訓、每步一次前向的取樣器：以 overlap transform 衰減冗餘方向，並以 solver 修正把預測移向步內平均。 [arxiv](https://arxiv.org/html/2608.22244)
  - 8 NFE 時，LangFlow 的 OWT Gen PPL 由 531 降到約 62；ELF-B 由 71 降到約 43。 [awesomepapers](https://awesomepapers.io/llm-papers/papers/2608.22244)
  - 在 ELF-B 上，8 NFE 的 Gen PPL 為 45.63，HiGS 為 56.90，官方取樣器為 74.18。 [arxiv](https://arxiv.org/html/2608.22244)
- **Low Perplexity is Repetition（2607.00588）** 發現 self-conditioning 迴路沿單一方向形成收縮吸引子，使輸出偏向重複內容。 [arxiv](https://arxiv.org/abs/2607.00588)
  - 去除重複後，ELF-B 的 Gen PPL 由 19.5 升為 27.7。 [arxiv](https://arxiv.org/abs/2607.00588)
  - 在 Gen PPL 上，最小的 ELF 模型因重複最多而得到最佳分數。 [arxiv](https://arxiv.org/abs/2607.00588)
  - 作者提出 ACE：每步從回饋中減去該方向。 [arxiv](https://arxiv.org/abs/2607.00588)
  - 這對 Gen PPL 評估的影響見 [基準結果對照與評估方法的限制](cdlm-benchmarks-and-eval.html)。

## 結論
- self-conditioning 是性能提升的主要來源之一。
- 它同時是低 entropy / 重複問題的來源之一。
- 以 Gen PPL 報告的 self-conditioning 增益，有一部分來自重複內容。
