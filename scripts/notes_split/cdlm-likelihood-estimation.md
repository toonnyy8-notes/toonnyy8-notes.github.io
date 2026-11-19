---
slug: cdlm-likelihood-estimation
title: 連續擴散語言模型的 SDE vs ODE 與 likelihood 估計
cluster: Diffusion Language Models
kind: topic
tags: [likelihood, elbo, ode, sde, continuous-diffusion-lm]
sources: [flow-matching-lm-2026.md]
arxiv: ["2610.02665", "2610.00894"]
related: ["continuous-flow-lm-overview", "cdlm-benchmarks-and-eval", "cdlm-open-problems"]
---
# 連續擴散語言模型的 SDE vs ODE 與 likelihood 估計

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。

- **舊做法：** Plaid 使用 VDM ELBO，可以直接報告 PPL 上界。
- **確定性 ODE 取樣：** FLM 系列以確定性 ODE 取樣，論點是 ODE 能表示序列的疊加並避免逐 token 祖先取樣。
- **ODE-based NLL 上界：** LangFlow 用它報告 PPL：LM1B 30.0、OWT 24.6。
- **Scaling CFM 的半離散 ELBO：**
  - 用 Pinsker 不等式把 L² 擴散損失上界為 CE，從而在 MCQA 上評分。
  - 作者表示 bound 的緊度未知。
  - 在 6 項 MCQA 中，1.7B CFM 低於同資料同算力的 MDLM / Duo（例如 ARC-E 47.1 vs 50.5 / 53.4），BoolQ 42.5 甚至低於機率水準 50.4。
  - 作者表示透過反解 ODE 計算精確 likelihood 在 L×|V| 維度下過於昂貴。
- **Sigma** 使用 VDM ELBO，並以 DDPM（SDE）與 DDIM（ODE）兩種取樣器報告結果。DDPM-256 品質最好。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- **Clock Diffusion** 的 ClockDLM 以 sliding window 變分界，在 OWT 上取得連續 DLM 中最佳的 likelihood bound，並勝過 block 半自迴歸離散擴散（BD3LM）。 [arxiv](https://arxiv.org/html/2610.00894)

各模型的 PPL 數字對照見 [基準結果對照](cdlm-benchmarks-and-eval.html)；likelihood 評估的未解問題見 [未解決問題](cdlm-open-problems.html)。
