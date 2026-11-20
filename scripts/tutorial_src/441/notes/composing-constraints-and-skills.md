---
slug: composing-constraints-and-skills
title: 組合限制與技能（組合式擴散）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [composable-diffusion, constraints, skills, classifier-free-guidance, compositionality]
arxiv: ["2211.15657", "2206.01714", "2011.13456"]
sources: ["paper.txt §3.3, §4.2, §4.3, Appendix A.2, D, J, M", "2206.01714 依本論文描述與 arXiv API 的題名資訊"]
related: ["decision-diffuser-lineage-hub", "classifier-free-guidance-return-conditioning", "eps-prediction-ddpm-score", "dd-experiments-and-ablations", "dd-limitations-stochastic-dynamics"]
---
# 組合限制與技能（組合式擴散）

屬於[總覽](decision-diffuser-lineage-hub.html)。

## 方法
除了回報，DD 的條件 y 可以是限制（one-hot 的 1(τ∈C_i)）或技能（one-hot 的 1(τ∈B_i)）。訓練時資料中每條軌跡只滿足一個限制或展示一個技能；推論時可同時條件化多個變數，噪聲改為

ε̂ := ε_θ(x_k,Ø,k) + ω Σ_i (ε_θ(x_k,y^i,k) − ε_θ(x_k,Ø,k))

此式出自 Liu 等（2022），"Compositional Visual Generation with Composable Diffusion Models"，arXiv 2206.01714。

## 推導與假設（附錄 D）
利用 ∇log q(x_k|y^i) ∝ −ε_θ(x_k,y^i,k)（分數與噪聲的關係，Song 等 arXiv 2011.13456），並假設 {y^i} 在給定狀態軌跡 x_k(τ) 時條件獨立，由貝氏定理得 q(x_k|{y^i}) ∝ q(x_k)∏ q(x_k|y^i)/q(x_k)。論文表示此假設不必嚴格成立，只要條件的組合可行即可。附錄 J 另說明 NOT 組合。

## 實驗
- 線性系統導航（附錄 A.2）：單一限制成功率 100%，組合限制（r ≤ ‖s_T‖ ≤ R）為 91.3%(±2.6%)。
- Kuka 積木堆疊：單一限制平均 DD 60.4（Diffuser 52.3），多重限制平均 63.8；BCQ 與 CQL 成功率為 0.0。
- Unitree 步態組合：條件為 Bound + Pace 時，1000 條軌跡中 60.1% 時間被分類為 Bound、38.5% 為 Pace。

## 限制（附錄 M）
DD 不提供各條件變數的顯式密度估計，只支援 AND 與 NOT 組合，不支援 OR。
