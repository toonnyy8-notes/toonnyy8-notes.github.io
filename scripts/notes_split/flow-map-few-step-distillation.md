---
slug: flow-map-few-step-distillation
title: 連續 flow 語言模型的少步生成：flow map / consistency / 分佈匹配蒸餾
cluster: Diffusion Language Models
kind: topic
tags: [flow-map, few-step, distillation, consistency, dmd]
sources: [flow-matching-lm-2026.md]
arxiv: ["2406.07507", "2505.18825", "2604.09784", "2602.16813", "2602.12233", "2610.00497", "2609.40235", "2605.10938", "2608.22244"]
related: ["continuous-flow-lm-overview", "self-conditioning-flow-lm", "cdlm-benchmarks-and-eval", "cdlm-open-problems"]
---
# 連續 flow 語言模型的少步生成：flow map / consistency / 分佈匹配蒸餾

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。flow map 理論來源（Boffi、Albergo、Vanden-Eijnden，2406.07507、2505.18825）見總覽。

離散擴散的蒸餾受因子化誤差限制。DFM 與 FMLM 都論證，離散更新在大步長下函數形式不正確，連續 flow 的 flow map 是唯一存在的。 [arxiv](https://arxiv.org/pdf/2604.09784) [arxiv](https://arxiv.org/html/2602.16813v2)

## 蒸餾方法

| 方法 | 論文 | 一致性條件 / 目標 | 備註 |
|---|---|---|---|
| PSD（semigroup + KL） | FMLM, DFM, Scaling CFM | X_{u,t}∘X_{s,u}=X_{s,t}，以 KL 監督 | 大規模下穩定 |
| ESD（Eulerian logit） | DFM | logit 空間的 Eulerian 恆等式 | 小規模最佳；OWT 1 步崩潰（entropy 0.26）；1.7B 規模下全部發散 |
| 端點一致性 | CFM（2602.12233） | endpoint consistency | — |
| Gumbel 耦合 + AR teacher | GSF（2610.00497） | AR 的 Gumbel-max 耦合給出不相交的直線路徑 | 見下文 |
| 分佈匹配（reverse KL） | DMD for CDLMs（2609.40235） | Simplex-DMD（pathwise）、Reinforce-DMD（REINFORCE + density ratio） | — |
| 漸進蒸餾 | ELF+PD | — | 見下表 |

（DMD for CDLMs：[arxiv](https://arxiv.org/abs/2609.40235)）

- **GSF** 證明在無限容量假設下，AR teacher 的 Gumbel-max 耦合給出的最佳路徑直線度 S=0。這同時讓所需的時間 warping 變得溫和許多。 [arxiv](https://arxiv.org/html/2610.00497)
- 帶 self-conditioning 的 fixed-point flow 也有對應 flow map（FMLM⋆），見 [self-conditioning](self-conditioning-flow-lm.html)。

## 少步結果（Gen PPL / entropy，GPT-2 Large 評分，OWT L=1024）

| 方法 | 1 步 | 2 步 | 4 步 | 8 步 |
|---|---|---|---|---|
| MDLM+SDTT | 1260.86 / 5.26 | 877.22 / 5.34 | 339.73 / 5.38 | — |
| Duo+DCD | 5743.29 / 6.02 | 891.16 / 5.41 | 250.86 / 5.37 | — |
| Duo+Di4C | 370.51 / 3.92 | 210.22 / 4.63 | 154.67 / 4.85 | — |
| FMLM | 168.30 / 5.17 | 133.29 / 5.25 | 111.31 / 5.26 | 86.50 |
| DFM（PSD） | 180.29 / 4.91 | 152.83 / 5.03 | 122.32 / 5.10 | 98.54 / 5.11 |
| DFM（ESD） | 5.33 / 0.26（崩潰） | 108.91 / 5.15 | 77.08 / 5.27 | 62.98 / 5.23 |
| GSF（AR teacher） | 114.9 / 5.07 | 105.6 / 5.30 | 99.7 / 5.32 | — |
| ELF+PD | 136.10 / 5.26 | — | 34.33 / 5.16 | 23.18 / 5.07 |

（MDLM/Duo/DFM 列：[arxiv](https://arxiv.org/pdf/2604.09784)；FMLM、ELF+PD 列：[arxiv](https://arxiv.org/pdf/2605.10938)；GSF 列：[arxiv](https://arxiv.org/html/2610.00497)）

## 讀表注意
- 各論文的 teacher、資料與 tokenizer 不同。表中 ELF+PD 的 entropy 接近該文自設的 5.0 退化線，部分 Gen PPL 改善來自較低的多樣性。 [arxiv](https://arxiv.org/pdf/2605.10938)
- **GSF：** 一步的 Gradient Moment（GM）為 0.94，FMLM 為 1.18；作者重新實作的 FMLM 一步為 140.0 / 4.93。 [arxiv](https://arxiv.org/html/2610.00497)
- **Untied SC：** 在 ELF-B 上，8 NFE 的 Gen PPL 為 45.63，HiGS 為 56.90，官方取樣器為 74.18。 [arxiv](https://arxiv.org/html/2608.22244)
- **規模化後的少步結果：** Scaling CFM 的 1.7B 模型（2.1T tokens）在 PSD 蒸餾後，4 步 MAUVE 提升最多，token entropy 接近資料水準。但作者也寫明「Gen PPL remains high」。未蒸餾模型在 1–2 步時輸出退化（「the the the」），反而讓 Gen PPL 偏低。
- **Sigma：** DDIM-32 平均 60.45，DDIM-16 平均 55.17，DDIM-4 平均僅 7.74（DDPM-256 為 64.34）。Sigma 摘要稱連續軌跡在低 NFE 時「graceful degradation」，但上述數字顯示少步品質在大模型上仍明顯下降。
- JSD 與 forward KL 蒸餾在 entropy 與品質之間有取捨（Scaling CFM）。
- Gen PPL 本身的可靠性問題見 [基準結果對照與評估方法的限制](cdlm-benchmarks-and-eval.html)；大模型少步品質是 [未解決問題](cdlm-open-problems.html) 之一。
