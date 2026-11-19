---
slug: dllm-parallelism-lineage-hub
title: dLLM 平行生成的理論脈絡總覽：從三種 corruption 到 critical window 分離結果
cluster: Diffusion Language Models
kind: hub
tags: [dllm, parallelism, dual-total-correlation, critical-window, lineage]
arxiv: ["2609.20539"]
sources: [2609.20539 introduction, 2609.20539 related work]
related: ["dllm-three-corruption-paradigms", "dllm-discrete-sampling-theory", "dllm-masked-schedule-dtc-theory", "dllm-critical-windows-speciation", "dllm-score-oracle-lower-bounds", "dllm-parallelism-separation-result", "eps-prediction-ddpm-score"]
---
# dLLM 平行生成的理論脈絡總覽

本論文（Chen & Wang, arXiv:2609.20539）比較 masked、uniform、Gaussian 三種 diffusion language model 在「所需 forward pass 數」上的差異。閱讀順序如下。

1. [三種 corruption 與 dLLM 的發展](dllm-three-corruption-paradigms.html)：D3PM、SEDD、MDLM 到 LLaDA，以及 uniform 與 Gaussian 路線。基礎的連續擴散見 [ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)。
2. [離散 diffusion 的取樣理論](dllm-discrete-sampling-theory.html)：continuous-time Markov chain 框架與 uniformization。
3. [masked diffusion 的排程與 DTC 界](dllm-masked-schedule-dtc-theory.html)：query 數隨 dual total correlation 成長的前人結果。
4. [critical window 與 speciation](dllm-critical-windows-speciation.html)：生成結果在窄範圍的 noise level 內定形。
5. [score oracle 的 query 下界](dllm-score-oracle-lower-bounds.html)：Xun & Price 的 Gaussian 下界與 random empirical measure。
6. [本論文的分離結果](dllm-parallelism-separation-result.html)：uniform 與 Gaussian 達到 DTC 縮放，並在 random empirical measure 上與 masked diffusion 分離。

時間線見 `timeline.json`。
