---
slug: cdlm-noise-schedule
title: 連續擴散語言模型的噪聲／時間排程
cluster: Diffusion Language Models
kind: topic
tags: [noise-schedule, time-warping, decoding-error-schedule, continuous-diffusion-lm]
sources: [flow-matching-lm-2026.md]
arxiv: ["2610.00894", "2605.11125", "2605.10938", "2610.00497", "2610.28559"]
related: ["continuous-flow-lm-overview", "representation-autoencoder-rae", "hybrid-continuous-discrete-lm", "flow-map-few-step-distillation"]
---
# 連續擴散語言模型的噪聲／時間排程

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。影像 latent 擴散中依資料維度做的 noise schedule shift 見 [RAE](representation-autoencoder-rae.html)。

| 做法 | 論文 | 內容 | 證據 |
|---|---|---|---|
| time warping（舊） | CDCD（2022） | 依 loss 的 CDF 重新分配時間取樣 | — |
| 解碼錯誤率排程 | FLM、DFM、GSF | 選 β(t) 使 P(argmax x_t ≠ argmax x₁) 隨 t 近似線性變化 | 見下方說明 |
| 混合排程 | DFM、Scaling CFM | α_t=(1−λ)t+λγ_t | 見下方說明 |
| 可學 Gumbel / logNSR 排程 | LangFlow | information-uniform 原則 | Gen PPL 約 1000 → 154 |
| 截斷 + 自適應排程 | S-FLM | 依 \|dL̂/dt\| 的反 CDF 每 50 步更新，參考 InfoNoise | GSM8K 7.7% → 11.1%（加入自適應） |
| logit-normal 時間取樣 | ELF | 沿用影像 rectified flow | — |
| 常數 per-timestep loss | RePlaid | 最小化 ELBO 估計變異數 | — |
| 逐位置時間 | Clock Diffusion（2610.00894）、Lafa、DFM | 位置相依時鐘 u^ℓ(t)；Lafa 以 loss 反 CDF 學 per-token time | — |

（S-FLM：[arxiv](https://arxiv.org/pdf/2605.11125v3)；ELF：[arxiv](https://arxiv.org/pdf/2605.10938)）

- **解碼錯誤率排程。** FLM 稱此舉「greatly improves training stability」。GSF 圖 4 顯示，在大詞彙量下 FLM 的解碼錯誤集中在 t≈0.8 附近，是一個急遽的相變，必須做時間重參數化。 [arxiv](https://arxiv.org/html/2610.00497) 解碼錯誤率排程最早由 CANDI 提出（見 [混合連續—離散方法](hybrid-continuous-discrete-lm.html)）。
- **混合排程的消融。** Scaling CFM 在 400M 規模掃 λ，最佳值約 0.75，λ=1（純錯誤率排程）時崩潰。DFM 使用 λ=0.9。
- **逐位置時間。** Lafa 學到的 per-token time 自然呈現由左到右的順序，論文寫明「this left-to-right bias is not manually imposed」。 [arxiv](https://arxiv.org/html/2610.28559)
- GSF 的 Gumbel 耦合讓所需的時間 warping 溫和許多（見 [少步蒸餾](flow-map-few-step-distillation.html)）。
