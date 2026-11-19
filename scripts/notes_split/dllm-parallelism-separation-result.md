---
slug: dllm-parallelism-separation-result
title: 本論文結果：DTC 縮放與 masked / uniform / Gaussian 的平行度分離
cluster: Diffusion Language Models
kind: topic
tags: [dllm, separation, dual-total-correlation, critical-window, reverse-dpi]
arxiv: ["2609.20539", "2604.10857", "2511.04647", "2510.25544"]
sources: [2609.20539 abstract, 2609.20539 section 1]
related: ["dllm-parallelism-lineage-hub", "dllm-masked-schedule-dtc-theory", "dllm-critical-windows-speciation", "dllm-score-oracle-lower-bounds", "dllm-three-corruption-paradigms"]
---
# 本論文結果

屬於 [dLLM 平行生成的理論脈絡總覽](dllm-parallelism-lineage-hub.html)。

## 結果 1：uniform 與 Gaussian 也達到 DTC 縮放（Theorem 1.1、1.2）
對任意 Σ^d 上的 q，存在演算法以 Õ(DTC(q)/ε) 次 query 取樣，使 KL(q ∥ q̂) ≤ ε。此前只知道 masked diffusion 能做到（[Chen–Cong–Li、Lavenant–Zanella](dllm-masked-schedule-dtc-theory.html)）。

證明方式與 masked 不同：masked 有取樣誤差的精確刻劃，這裡改以 reverse data-processing inequality 控制單步誤差，使其不超過步長乘以 DTC 在該步的下降量，再對各步 telescoping，得到總誤差 ≤ a · DTC(q)，取 a ≍ ε/DTC(q)。

## 結果 2：分離（Theorem 1.3、1.4）
對由 e^{Θ(d)} 個隨機 hypercube 點構成的 random empirical measure（DTC = Θ(d)）：
- uniform 與 Gaussian：Θ̃(√d) 次 query 即足夠（對 score error 1/poly(d) 的 oracle），超出 DTC 縮放。
- masked：存在近似 oracle，使 õ(d) 次 query 無法取樣，即需 Ω̃(d)。

## 機制
下界的共同想法是：除非在正確的 noise level 附近 query，oracle 的近似誤差可被設計成抹除關於 q 的有用訊號。差別在可接受 noise level 的範圍：masked 為 O(log d / d) 比例，uniform 與 Gaussian 為 O(√(log d / d)) 比例（critical window 較寬），所以前者需 Ω̃(d) 次 query 才找得到，後者 Õ(√d) 次。論文強調這與「masked 必須 commit」的常見說法無關。背景見 [critical window 與 speciation](dllm-critical-windows-speciation.html) 與 [score oracle 與下界](dllm-score-oracle-lower-bounds.html)。
