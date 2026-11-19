---
slug: dllm-discrete-sampling-theory
title: 離散 diffusion 取樣的理論基礎：CTMC 框架與 uniformization
cluster: Diffusion Language Models
kind: topic
tags: [discrete-diffusion, ctmc, uniformization, convergence-theory]
arxiv: ["2609.20539", "2205.14987", "2402.08095", "2602.15008"]
sources: [2609.20539 related work, arXiv abstracts via titles]
related: ["dllm-parallelism-lineage-hub", "dllm-masked-schedule-dtc-theory"]
---
# 離散 diffusion 取樣的理論基礎

屬於 [dLLM 平行生成的理論脈絡總覽](dllm-parallelism-lineage-hub.html)。

- Campbell et al. 2022, *A Continuous Time Framework for Discrete Denoising Models*（arXiv:2205.14987）：本論文描述其以 continuous-time Markov chain（CTMC）形式化連續時間的離散去噪。
- Chen & Ying 2024, *Convergence Analysis of Discrete Diffusion Model: Exact Implementation through Uniformization*（arXiv:2402.08095）：本論文描述其以 uniformization 實作估計 score 所對應的 reverse chain，並在 hypercube 上得到 TV 與 KL 保證；uniform diffusion 以 O(d) 次 score oracle query 達到零離散化誤差。
- Dmitriev et al. 2026, *Efficient Sampling with Discrete Diffusion Models: Sharp and Adaptive Guarantees*（arXiv:2602.15008）：本論文描述其證明標準 uniform 與 masking τ-leaping 的 sharp 收斂保證。

本論文另引用以 stochastic integral 分析時間離散化（Ren et al.）及對字母表大小依賴性的一批收斂分析；這些條目在此未逐一驗證 arXiv ID，不列入。

與本論文的關係：Chen & Ying 的 O(d) 是 uniform diffusion 在 d 維下不利用資料結構時的 query 數；本論文的目標是把 query 數降到 o(d)，其縮放量見 [masked diffusion 的排程與 DTC 界](dllm-masked-schedule-dtc-theory.html)。
