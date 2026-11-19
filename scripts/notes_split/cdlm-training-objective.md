---
slug: cdlm-training-objective
title: 連續擴散語言模型的訓練目標與參數化：從 MSE 到 CE 去噪器
cluster: Diffusion Language Models
kind: topic
tags: [continuous-diffusion-lm, cross-entropy, x-prediction, parameterization]
sources: [flow-matching-lm-2026.md]
arxiv: ["2602.16813", "2604.09784", "2604.11748", "2605.10938", "2608.23551"]
related: ["continuous-flow-lm-overview", "cdlm-representation-space", "flow-map-few-step-distillation"]
---
# 連續擴散語言模型的訓練目標與參數化：從 MSE 到 CE 去噪器

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。表示空間的選擇見 [表示空間與 decoder 問題](cdlm-representation-space.html)。

## 舊做法（2022–2024）
- Diffusion-LM（Li et al. 2022）在可學 embedding 上以 MSE 預測 x₀，最後一步用 rounding 把向量對回 token。
- Plaid（Gulrajani & Hashimoto 2023）使用 VDM 式 ELBO。
- CDCD（Dieleman et al. 2022）已經用 CE 訓練 softmax 去噪器（score interpolation）。但 CDCD 之後的工作多數沒有延續這條路線。

## 新做法與證據
- **FLM（2602.16813）**
  - 在 one-hot 上做線性 interpolant。去噪器輸出逐 token 的 softmax，以 CE 訓練。 [arxiv](https://arxiv.org/html/2602.16813v2)
  - 去噪器的最佳解是 E[x₁|x_t]，即每個位置在 simplex 上的後驗。
  - 消融：CE 去噪器優於 velocity MSE，one-hot 優於可學 embedding，Euclidean 優於 Riemannian / simplex 擴散。
- **DFM（2604.09784）** 指出 flow map 的「mean denoiser」ψ_{s,t} 是時間加權的條件期望，因此恆落在 simplex 上。所以 semigroup、Lagrangian、Eulerian 三種一致性條件都能寫成 KL / CE 目標，取代 Euclidean L² 回歸（蒸餾用法見 [少步蒸餾](flow-map-few-step-distillation.html)）。 [arxiv](https://arxiv.org/pdf/2604.09784)
- **LangFlow（2604.11748）** 用 Bregman divergence 把 embedding 空間的擴散連到 flow matching，從而導出 CE 目標。 [arxiv](https://arxiv.org/html/2604.11748v3) 它並指出 Plaid 式 MSE 會使 embedding collapse（NND 0.058 vs 0.877）。
- **ELF（2605.10938）** 採混合做法：
  - 中間步以 x-prediction + MSE 去噪；最後一步由同一網路以 CE 解碼成 token。 [arxiv](https://arxiv.org/pdf/2605.10938)
  - 論文說明選 x-prediction 而非 v-prediction 的理由：x-prediction 能讓去噪與解碼共用權重。 [arxiv](https://arxiv.org/pdf/2605.10938)
- **ConvergeFlow（2608.23551）** 是反例，只用 MSE：
  - 把資料預測器限制在 token embedding 的凸包內（softmax 加權平均）。 [arxiv](https://arxiv.org/html/2608.23551v1)
  - 證明在正則條件下，即使預測器有誤差，ODE 仍會收斂到合法的 token embedding，因此不需要 CE 監督的解碼器。 [arxiv](https://arxiv.org/html/2608.23551v1)
  - OWT 結果為「與現有連續與離散擴散 LM 相當」。 [arxiv](https://arxiv.org/html/2608.23551v1)

## 結論
- 實驗支持的部分：CE 目標，或至少把預測器的輸出限制在 simplex / 凸包，是性能提升的主因之一（在總覽的證據排序中列第一）。
- ConvergeFlow 的結果顯示，必要條件是參數化把輸出限制在合法集合，而不是 CE 本身。
- RePlaid 報告：在 learned embedding 上加輔助 CE，PPL 反而由 22.1 升到 26.1。CE 的效益取決於表示空間，不是普遍成立。
