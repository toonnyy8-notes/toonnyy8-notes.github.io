---
slug: confidence-thresholding-heuristics
title: Confidence 門檻式 unmasking heuristic
cluster: Diffusion Language Models
kind: topic
tags: [fast-dllm, confidence, sampling-heuristic, parallel-decoding, nfe]
arxiv: ["2505.22618", "2510.21961", "2502.06768", "2512.09106"]
sources: ["paper.txt §1, §2.2, §5", "arXiv API metadata"]
related: [unmasking-policies-dllm-overview, semi-ar-block-generation, learned-unmasking-concurrent-work, unmasking-mdp-policy-design]
---
# Confidence 門檻式 unmasking heuristic

屬於[總覽](unmasking-policies-dllm-overview.html)。

## 三種基準

論文比較三種規則，confidence c_t^k 定義為位置 k 預測分佈的最大機率。

- Random：每步隨機解開固定數量 K 個位置。
- High confidence：每步解開 confidence 最高的 K 個位置。
- Fast-dLLM（Wu 等人，2025，「Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding」，2505.22618）：解開所有 confidence 超過門檻 λ 的位置，每步解開的數量可變。

論文說明 Fast-dLLM 使 LLaDA 的 token 吞吐量超過同規模 LLaMA，並使 confidence 門檻成為後續 heuristic 的主流做法。論文第 5 節列出的後續工作多為此規則的重新詮釋或延伸，另有納入空間資訊、替代 confidence 度量，以及 Azangulov 等人（2025，「Parallel Sampling from Masked Diffusion Models via Conditional Independence Testing」，2510.21961）對 token 相依性的建模。Kim 等人的「Train for the Worst, Plan for the Best: Understanding Token Ordering in Masked Diffusions」（2502.06768）出現在論文的基準比較中（附錄圖 11）。

## 論文指出的缺點

- 需要手動調參（門檻 λ、confidence 度量）。
- 表現依賴 semi-AR 生成；BL 變大時退化（見 [Semi-AR 區塊生成](semi-ar-block-generation.html)）。

論文的立場是這類規則是序列決策問題的手工解，因此改用[學習式 policy](unmasking-mdp-policy-design.html)。
