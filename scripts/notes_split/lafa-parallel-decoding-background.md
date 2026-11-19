---
slug: lafa-parallel-decoding-background
title: AR 生成的延遲與平行解碼方法（Blockwise decoding、MTP、Medusa）
cluster: Diffusion Language Models
kind: topic
tags: [autoregressive, multi-token-prediction, speculative-decoding, parallel-decoding]
arxiv: ["2610.28559", "1811.03115", "2401.10774", "2404.19737", "2604.07023", "2509.04185"]
sources: ["arXiv abstracts", "paper.txt §1, §4"]
related: ["lafa-lineage-overview", "lafa-self-speculative-decoding", "lafa-ar-to-block-adaptation"]
---
# AR 生成的延遲與平行解碼方法

屬於 [Lafa 研究脈絡總覽](lafa-lineage-overview.html)。

## 問題
Lafa 論文指出，自回歸（AR）模型每次前向只產生一個 token，推論延遲是瓶頸。論文把平行生成方法分為 multi-token prediction（MTP）、離散擴散與連續 flow 語言模型三類。

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
