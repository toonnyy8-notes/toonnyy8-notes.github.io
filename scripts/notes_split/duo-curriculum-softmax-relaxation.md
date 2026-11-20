---
slug: duo-curriculum-softmax-relaxation
title: 課程學習與 softmax 溫度鬆弛：Gumbel-Softmax、Concrete 到 Duo
cluster: Diffusion Language Models
kind: topic
tags: [curriculum-learning, gumbel-softmax, concrete-distribution, variance-reduction, duo]
arxiv: ["2506.10892", "1611.01144", "1611.00712", "2503.09573"]
sources: ["arXiv:2506.10892 §4.1、§5.1、§6", "1611.01144 與 1611.00712 的 arXiv 摘要"]
related: ["duo-lineage-hub", "duo-argmax-duality-theory"]
---
# 課程學習與 softmax 溫度鬆弛

屬於 [Duo 研究脈絡總覽](duo-lineage-hub.html)。對偶理論見 [arg max 對偶的理論](duo-argmax-duality-theory.html)。

## 前人
- Bengio 等人 2009，*Curriculum learning*（ICML；無 arXiv 版本）：從簡單樣本逐步進到困難樣本。Duo 以此命名其訓練策略。
- Jang、Gu、Poole 2017，*Categorical Reparameterization with Gumbel-Softmax*（arXiv:1611.01144）與 Maddison、Mnih、Teh 2017，*The Concrete Distribution*（arXiv:1611.00712）：以帶溫度的 softmax 連續鬆弛類別變數的取樣，使 arg max 可微。Duo §6 說明自己把這個技巧用在不同目的：不是為了反向傳播，而是為了降低訓練變異。

## Duo 的做法（§4.1）
1. 離散 NELBO 可改寫為對 Gaussian latent w_t 取期望、輸入為 arg max(w_t)（式 19）。
2. 訓練時把輸入的 arg max 換成 softmax(w_t/τ)（式 20–21）。τ 越大，保留的訊號越多、任務越容易，但估計有偏；τ → 0 時還原為離散 NELBO。
3. 時間 t 從子區間 U[β, γ] 取樣；沿用 Arriola 等人 2025（*Block Diffusion*，arXiv:2503.09573）將上界設為 b < 1 的做法。
4. 去噪模型須同時接受連續與離散輸入。

## 論文報告的設定與結果
τ = 0.001、β = 0.03、γ = 0.15，前 500K 步使用鬆弛，其後 τ = 0 至 1M 步。Fig. 2 報告梯度變異下降約一個數量級；LM1B 上 510K 步的 PPL 為 35.2，優於 UDLM 訓練 1M 步的結果，論文據此主張收斂至少快 2 倍。Table 3：移除課程學習為 35.0，再移除改進目標為 36.7（完整 Duo 為 33.7）。

## 限制（論文 §7.1）
τ 與課程長度為兩個超參數，其他領域可能需要調整；無法調整時，論文建議不用課程學習，仍可靠 Rao-Blackwellized 目標優於前人。
