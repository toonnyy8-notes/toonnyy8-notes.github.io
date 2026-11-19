---
slug: fmlm-flow-map-distillation-lineage
title: Flow map 與少步蒸餾的發展：consistency model 到 MeanFlow
cluster: Diffusion Language Models
kind: topic
tags: [flow-map, consistency-model, meanflow, shortcut-model, distillation]
arxiv: ["2602.16813", "2202.00512", "2303.01469", "2410.12557", "2406.07507", "2505.18825", "2505.13447", "2512.02012", "2511.19797", "2506.14603"]
sources: [paper-2602.16813-section-3.3, arxiv-abstracts]
related: ["fmlm-lineage-overview", "fmlm-flow-matching-interpolant", "fmlm-onehot-denoiser-two-time", "flow-map-few-step-distillation"]
---
# Flow map 與少步蒸餾的發展：consistency model 到 MeanFlow

屬於 [Flow Map Language Models 的研究脈絡總覽](fmlm-lineage-overview.html)。

## flow map 的定義

本文式 (14)：flow map X_{s,t} 是機率流 ODE 的解算子，滿足 X_{s,t}(x_s) = x_t。參數化為 X_{s,t}(x) = x + (t − s) v_{s,t}(x)，v 稱為平均速度（mean flow）。一步取樣為 x̂1 = X_{0,1}(x0)。

## 發展順序

| 年 | 工作 | 與本文的關係（依本文描述） |
|---|---|---|
| 2022 | Progressive Distillation（Salimans、Ho，2202.00512） | 逐次減半步數的蒸餾；本文說半群條件的蒸餾與它相關 |
| 2023 | Consistency Models（Song 等，2303.01469） | 少步／一步生成模型，本文列為少步方法 |
| 2024 | Shortcut Models（Frans 等，2410.12557） | 以步長為條件的一步擴散；與半群條件蒸餾相關 |
| 2024 | Flow map matching（Boffi 等，2406.07507） | 以 stochastic interpolant 統一 consistency model 的數學框架 |
| 2025 | How to build a consistency model（Boffi 等，2505.18825） | 以自蒸餾學 flow map，提出 Lagrangian 與 Eulerian 等特徵化 |
| 2025 | MeanFlow（Geng 等，2505.13447）、Improved MeanFlow（2512.02012） | 平均速度參數化，需 Jacobian-vector product |
| 2025 | Terminal Velocity Matching（2511.19797）、Align Your Flow（2506.14603） | 本文列為同類 flow map／少步方法 |

## 三種特徵化

本文式 (16) 指出 flow map 由對角條件、切線條件與半群條件 X_{u,t}(X_{s,u}(x)) = X_{s,t}(x) 完全刻畫。半群條件可換成 Lagrangian 或 Eulerian 兩種微分形式，後兩者導出 MeanFlow 與 Lagrangian 自蒸餾。本文實驗聚焦半群蒸餾，另外兩者留作未來工作；摘要提到比較三種蒸餾選擇，細節在附錄 C。

語言模型上的各種蒸餾方法比較見 [連續 flow 語言模型的少步生成](flow-map-few-step-distillation.html)。
