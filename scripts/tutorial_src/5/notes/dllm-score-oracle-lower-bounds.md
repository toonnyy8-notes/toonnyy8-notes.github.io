---
slug: dllm-score-oracle-lower-bounds
title: score oracle 模型與 query 下界：Xun & Price 與 random empirical measure
cluster: Diffusion & Generative Models
kind: topic
tags: [score-oracle, query-complexity, lower-bound, random-empirical-measure]
arxiv: ["2609.20539", "2604.10857"]
sources: [2609.20539 introduction, 2609.20539 related work, 2609.20539 section 5]
related: ["dllm-parallelism-lineage-hub", "dllm-critical-windows-speciation", "dllm-parallelism-separation-result"]
---
# score oracle 模型與 query 下界

屬於 [dLLM 平行生成的理論脈絡總覽](dllm-parallelism-lineage-hub.html)。

## 比較單位：score oracle query
本論文（arXiv:2609.20539）把一次 forward pass 視為一次對 (近似) score oracle 的 query：給定 noise level t 與條件 z，模型同時輸出所有座標的 posterior marginal law((X0)_i | Xt = z)。這些 marginal 與 annealed score 之間只差一個仿射轉換。上界分析具體演算法，下界排除所有使用此 oracle 的演算法。

## Xun & Price 的下界
Xun & Price 2026, *Query Lower Bounds for Diffusion Sampling*（arXiv:2604.10857）：本論文描述其證明，對於從 Boolean hypercube 均勻隨機取 e^{κd} 個點構成的 random empirical measure（0 < κ < log 2），給定近似 Gaussian diffusion score oracle，至少需 Ω̃(√d) 次 query，連估計 κ 到足夠精度都做不到，更不用說取樣。

## 本論文的延伸
- 附錄 A 以相似想法證明 uniform diffusion 同樣有 Ω̃(√d) 下界。
- 第 6、7 節證明此界對 uniform 與 Gaussian 也是上界（Θ̃(√d)）。
- 第 8 節對 masked diffusion 構造一個近似 oracle，使 õ(d) 次 query 的演算法輸出與 q 的 TV 距離至少 0.99。

細節見 [本論文的分離結果](dllm-parallelism-separation-result.html)；下界依賴的 noise level 窗口見 [critical window 與 speciation](dllm-critical-windows-speciation.html)。
