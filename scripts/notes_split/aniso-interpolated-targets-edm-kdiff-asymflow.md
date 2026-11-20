---
slug: aniso-interpolated-targets-edm-kdiff-asymflow
title: 內插目標的前驅：EDM、k-Diff 與 AsymFlow
cluster: Diffusion & Generative Models
kind: topic
tags: [edm, k-diff, asymflow, interpolated-target, preconditioning]
arxiv: ["2611.17767", "2206.00364", "2202.00512", "2601.21419", "2605.12964"]
sources: [2611.17767 related work, 2611.17767 appendix A.2, 2605.12964 abstract, 2601.21419 abstract]
related: ["aniso-lineage-overview", "aniso-target-normal-form-and-recovery", "v-prediction-and-preconditioning", "generalized-target-k-parameterization", "aniso-pixel-patchified-setting"]
---
# 內插目標的前驅：EDM、k-Diff 與 AsymFlow

回到[總覽](aniso-lineage-overview.html)。

本論文 Related Work 把「標準目標」與「內插目標」分開處理。標準目標指 ε、v、x；內插目標在其間取折衷。

## EDM（Karras 等，arXiv:2206.00364）
各向同性內插：以依 σ_t 而定的 skip 係數解析地決定資料與噪聲的混合比例。本論文 Appendix A.2 指出，把 EDM 的純量資料方差換成方向 i 的方差 λ_i，得到 c_skip,i = λ_i / (λ_i + s_t²)，對應的噪聲權重為 w_i = (α_t/σ_t) λ_i；這正是網路沒有預測能力（只知道時間步）時本論文分析所預測的結果。EDM 與 Progressive Distillation（Salimans & Ho，arXiv:2202.00512）的目標設計見 [v-prediction 與 preconditioning](v-prediction-and-preconditioning.html)。

## k-Diff（Jin & Wang，arXiv:2601.21419）
各向同性內插，以單一純量學習混合係數。理論與方法見 [廣義預測目標與 k-Diff](generalized-target-k-parameterization.html)。本論文把 k-Diff 視為目標族中「所有方向共用同一個權重 w_i ≡ w」的特例（差一個尺度與符號）。

## AsymFlow（Chen 等，arXiv:2605.12964）
摘要：只在低秩子空間內預測噪聲，資料預測保持全維度，再解析地還原完整速度。本論文將其描述為各向異性但二元：在低秩子空間做速度預測、在正交補空間做資料預測，對應 w_i ∈ {0, 1}，基底為 PCA。Table 1：AsymFlow（二元權重、凍結 PCA 基底）在 JiT-H/16 上 600 epoch FID 為 1.76。

## 小結
三者都落在 AnisoFlow 的目標族內：EDM 的各方向推廣由理論給定權重，k-Diff 是共用純量，AsymFlow 是二元權重。AnisoFlow 讓每個方向的權重與基底都由訓練學得。
