---
slug: dllm-three-corruption-paradigms
title: 三種 corruption 與 dLLM 的發展：masked、uniform、Gaussian
cluster: Diffusion Language Models
kind: topic
tags: [dllm, masked-diffusion, uniform-diffusion, gaussian-diffusion, d3pm, sedd, mdlm, llada]
arxiv: ["2609.20539", "2107.03006", "2205.14217", "2211.15089", "2310.16834", "2406.07524", "2406.04329", "2502.09992", "2506.10892", "2602.15014", "2602.16813", "2605.10938"]
sources: [2609.20539 introduction, 2609.20539 section 2, arXiv abstracts via titles]
related: ["dllm-parallelism-lineage-hub", "dllm-parallelism-separation-result", "eps-prediction-ddpm-score", "continuous-flow-lm-overview"]
---
# 三種 corruption 與 dLLM 的發展

屬於 [dLLM 平行生成的理論脈絡總覽](dllm-parallelism-lineage-hub.html)。

## 本論文的分類（arXiv:2609.20539）
本論文把 dLLM 依 corruption process 分成三類（Figure 1）：
- **Uniform**：每個 token 以機率 1 − e^{−t} 被重新取樣為均勻分布的 token。
- **Gaussian**：one-hot 表示 X0 加上 Gaussian 噪聲，Xt = e^{−t}X0 + σt G。
- **Masked**：每個 token 以機率 1 − e^{−t} 被遮蔽；未遮蔽的 token 一經 unmask 即固定（commit）。

autoregressive LLM 被視為特例：corruption 為由右至左的 erasure。

## 前人工作（本論文引用）
- Austin et al. 2021, *Structured Denoising Diffusion Models in Discrete State-Spaces*（arXiv:2107.03006）：離散狀態空間的 diffusion。
- Li et al. 2022, *Diffusion-LM Improves Controllable Text Generation*（arXiv:2205.14217）與 Dieleman et al. 2022, *Continuous diffusion for categorical data*（arXiv:2211.15089）：本論文歸為 Gaussian（連續）路線。
- Lou et al. 2023, *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*（arXiv:2310.16834）：uniform diffusion 的 score entropy loss。
- Sahoo et al. 2024（arXiv:2406.07524）與 Shi et al. 2024（arXiv:2406.04329）：masked diffusion 的簡化訓練目標。
- Nie et al. 2025, *Large Language Diffusion Models*（arXiv:2502.09992）：大規模 masked dLLM。
- Sahoo et al. 2025, *The Diffusion Duality*（arXiv:2506.10892）、Sahoo et al. 2026（arXiv:2602.15014）：本論文引用為 uniform diffusion 與「超越 masked」的實證主張。
- Lee et al. 2026（arXiv:2602.16813）、Hu et al. 2026（arXiv:2605.10938）：本論文引用為 Gaussian 路線的 few-step 實證，連續語言模型的整理見 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。

## 常見直覺
本論文指出，常見說法認為 masked diffusion 必須 commit token 值，錯誤無法修正，因而較難平行化；uniform 與 Gaussian 可逐步修正輸出。這些說法在本論文之前主要是啟發式（heuristic）。結果見 [本論文的分離結果](dllm-parallelism-separation-result.html)。
