---
slug: duo-curriculum-softmax-relaxation
title: 課程學習與 softmax 溫度鬆弛：Gumbel-Softmax、Concrete 到 Duo
cluster: Diffusion Language Models
kind: topic
tags: [curriculum-learning, gumbel-softmax, concrete-distribution, variance-reduction, duo, gaussian-relaxation, top-k]
arxiv: ["2506.10892", "1611.01144", "1611.00712", "2503.09573", "2602.21185", "2602.15014"]
sources: ["arXiv:2506.10892 §4.1、§5.1、§6", "1611.01144 與 1611.00712 的 arXiv 摘要", "arxiv.org/abs 摘要（2506.10892）", "paper:2602.21185 第 1、2.1.2、4 節", "arXiv API metadata（標題、作者、年份）"]
related: ["duo-lineage-hub", "duo-argmax-duality-theory", "psi-samplers-lineage-hub", "psi-superposition-posteriors-method"]
---
# 課程學習與 softmax 溫度鬆弛

屬於 [Duo 研究脈絡總覽](duo-lineage-hub.html)與 [Ψ-Samplers 脈絡總覽](psi-samplers-lineage-hub.html)。對偶理論見 [arg max 對偶的理論](duo-argmax-duality-theory.html)。

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

## 摘要層級的描述與課程學習的成本（Ψ-Samplers 論文第 1、4 節）
Duo（Sahoo 等人，2025；arXiv 2506.10892）摘要：uniform-state 擴散過程由底層 Gaussian 擴散自然產生；Duo 把 Gaussian 擴散的技巧移植過來，包含以 Gaussian 過程引導的課程學習（訓練速度加倍），以及 Discrete Consistency Distillation（少步生成，取樣加速約兩個數量級）。Duo 的完整介紹見 [The Diffusion Duality](diffusion-duality.html)。

*The Diffusion Duality, Chapter II: Ψ-Samplers*（arXiv 2602.21185）指出課程的成本：課程以 softmax 後的 Gaussian 潛變數取代離散 token 作為 transformer 輸入，溫度為 $\tau=10^{-3}$。直接實作需要為每個 token 在每個訓練步驟建立 $K$ 維權重向量，詞表 $K>100{,}000$ 時不可行。該論文的觀察是低溫下 softmax 幾乎只集中在少數項，因此只用 top-$k$ 個 embedding 近似，並以均勻亂數的 order statistics 直接取樣 top-$k$ 值，而不建立完整向量。該論文報告相對 Duo 訓練時間減少 25%、記憶體減少 33%（摘要）。

## 同一作者群的相關後續
Sahoo 等人（2026，arXiv 2602.15014），*Scaling Beyond Masked Diffusion Language Models*，為 Ψ-Samplers 論文引用的同系列工作；本筆記未讀其內容。
