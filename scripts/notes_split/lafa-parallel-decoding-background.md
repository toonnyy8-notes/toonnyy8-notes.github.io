---
slug: lafa-parallel-decoding-background
title: 自回歸的序列瓶頸與平行解碼方法（非自回歸、Blockwise decoding、MTP、Medusa）
cluster: Diffusion Language Models
kind: topic
tags: [autoregressive, non-autoregressive, multi-token-prediction, speculative-decoding, parallel-decoding]
arxiv: ["2610.28559", "1811.03115", "2401.10774", "2404.19737", "2604.07023", "2509.04185", "2602.16813", "1711.02281", "2410.21035", "2505.22618"]
sources: ["arXiv abstracts", "paper.txt §1, §4", "paper-2602.16813-intro-background", "arxiv-abstracts"]
related: [lafa-lineage-overview, lafa-self-speculative-decoding, lafa-ar-to-block-adaptation, fmlm-discrete-diffusion-factorization]
---
# 自回歸的序列瓶頸與平行解碼方法

屬於 [Lafa 研究脈絡總覽](lafa-lineage-overview.html)與 [Flow Map Language Models 的研究脈絡總覽](fmlm-lineage-overview.html)。

## 問題
Lafa 論文指出，自回歸（AR）模型每次前向只產生一個 token，推論延遲是瓶頸。論文把平行生成方法分為 multi-token prediction（MTP）、離散擴散與連續 flow 語言模型三類。

## 自回歸分解（在 Flow Map Language Models（2602.16813）中）
該文背景一節把自回歸語言模型寫成 p(y) = p(y1) p(y2|y1) ... p(yL|y<L)，推論時每個 token 需要前面所有 token，生成過程因此是序列化的。訓練時可以用 teacher forcing 與 Transformer 平行處理，瓶頸只在取樣。該文引用的早期自回歸模型包括 Jordan（1986）、Elman（1990）、Bengio 等人（2003）。

## 非自回歸生成
Gu 等人的 *Non-Autoregressive Neural Machine Translation*（1711.02281，2017）是 FMLM 引用的非自回歸做法之一，目標是一次輸出整個序列以避免序列化。FMLM 把這類方法視為「一次建模整個序列」的替代方案，動機是降低生成延遲。（說明依該文描述。）

## 擴散語言模型的承諾與問題
FMLM 指出，離散擴散語言模型每步可平行輸出多個 token，因此有機會加速，但少步時品質快速下降。引用的證據是 Deschenaux 與 Gulcehre 的 *Beyond Autoregression: Fast LLMs via Self-Distillation Through Time*（2410.21035，2024）。Fast-dLLM（2505.22618，2025）與 ParallelBench 被 FMLM 引用為說明平行解碼取捨的工作。（以上依該文描述與標題；Fast-dLLM 標題為 Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding。）FMLM 要解決的就是這個少步品質問題，詳見 [離散擴散與因子化誤差](fmlm-discrete-diffusion-factorization.html)。

## 各方法（說明依各論文摘要）
| 工作 | arXiv | 做法 |
| --- | --- | --- |
| Blockwise Parallel Decoding（Stern et al., 2018） | 1811.03115 | 額外的 proposal 模型一次預測多個未來位置，再由原模型驗證。 |
| Medusa（Cai et al., 2024） | 2401.10774 | 在同一模型上加多個解碼 head 以加速解碼。 |
| Multi-token Prediction（Gloeckle et al., 2024） | 2404.19737 | 訓練時每個位置預測多個未來 token，摘要稱可提高 sample efficiency。 |
| MARS（Jin et al., 2026） | 2604.07023 | 以輕量微調讓指令微調後的 AR 模型一次預測多個 token。 |
| Set Block Decoding（Gat et al., 2025） | 2509.04185 | 針對 AR 模型的推論加速器，摘要稱為一種解碼方法。 |

## 與 Lafa 的關係
- 論文在 Related work 指出，一次更新內獨立預測多個 token 的做法，在一次提交很多 token 時品質可能下降；部分方法以 AR 驗證草稿來緩解。
- Lafa 同樣使用 AR 驗證，但草稿來自同一個模型的 flow（見 [自我推測解碼](lafa-self-speculative-decoding.html)）。
- 論文也指出 flow 訓練有類似 multi-token prediction 的效果，並與這些方法在速度與品質上比較（見 [規模、資料量與評估](lafa-scale-and-evaluation.html)）。
