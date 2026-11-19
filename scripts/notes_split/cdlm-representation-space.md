---
slug: cdlm-representation-space
title: 連續擴散語言模型的表示空間與 rounding / decoder 問題
cluster: Diffusion Language Models
kind: topic
tags: [continuous-diffusion-lm, embedding, one-hot, hypersphere, embedding-collapse]
sources: [flow-matching-lm-2026.md]
arxiv: ["2605.11125", "2610.02665", "2610.28559", "2605.07748", "2610.12898", "2605.10938", "2605.07820", "2606.08810", "2608.23551"]
related: ["continuous-flow-lm-overview", "cdlm-training-objective", "cdlm-open-problems", "cdlm-noise-schedule"]
---
# 連續擴散語言模型的表示空間與 rounding / decoder 問題

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。訓練目標的部分見 [訓練目標與參數化](cdlm-training-objective.html)。

| 表示 | 代表論文 | 主要優點 | 主要成本／問題 |
|---|---|---|---|
| 可學低維 embedding（舊） | Diffusion-LM, Plaid | 維度低 | embedding collapse、rounding 誤差 |
| one-hot（R^{\|V\|}） | FLM, CFM, DFM, GSF | 去噪器輸出即 simplex，可直接用 CE / KL | L×\|V\| 張量，記憶體隨詞彙量線性增加 |
| 單位超球面 S^{d−1} | S-FLM（2605.11125） | 不需實體化 one-hot，訓練成本接近離散擴散 | 需截斷噪聲區間 |
| 凍結 contextual embedding（T5） | ELF | 語義結構好、訓練 token 少 | 依賴外部編碼器 |
| 16 維 unit-norm embedding | Sigma（2610.02665） | 參數占比 <1% | 未正則化時塌縮到約 4 個有效維度 |
| norm 固定、只學方向 | Lafa（2610.28559） | 保留預訓練 AR embedding 的 norm | — |
| VAE / latent | TextLDM（2605.07748）、PSF（2610.12898） | 擴散對象維度低 | 需額外解碼器 |

（S-FLM：[arxiv](https://arxiv.org/pdf/2605.11125v3)；ELF：[arxiv](https://arxiv.org/pdf/2605.10938)；Sigma：[arxiv](https://arxiv.org/pdf/2610.02665v1)）

## one-hot 的成本
- Scaling CFM（2605.07820）的限制章節寫明：L=2048、|V|≈100k 時「manageable given sufficient hardware (256 H100s)」，但成本隨序列長度與詞彙量線性增加。作者以 Cut Cross-Entropy 吸收這個成本，並未解決它。
- S-FLM 以此為動機改用超球面，並以 SLERP 插值。 [arxiv](https://arxiv.org/pdf/2605.11125v3)

## S-FLM 的噪聲截斷
- 在高維球面上，embedding 分得很開，後驗很快變成 one-hot。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
- 作者推導出截斷上界 α⋆(δ) ≈ (2/π)·arcsin(√(2 log(2(|V|−1)/δ)/d))，只在高噪聲段訓練。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
- 作者稱此截斷界假設 embedding 隨機分佈，只是啟發式（見 [未解決問題](cdlm-open-problems.html)）。截斷搭配的自適應排程見 [噪聲／時間排程](cdlm-noise-schedule.html)。

## decoder 問題的三種處理方式
- ELF 讓最後一步由同一網路做 CE 解碼。 [arxiv](https://arxiv.org/pdf/2605.10938)
- Du & Ma（2606.08810）以 decoder basin / margin 分析並提出 BGEE early exit。
- ConvergeFlow 證明收斂性，從而取消 decoder。 [arxiv](https://arxiv.org/html/2608.23551v1)
