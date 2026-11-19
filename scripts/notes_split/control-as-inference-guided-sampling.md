---
slug: control-as-inference-guided-sampling
title: 控制即推論：把獎勵最大化改寫為條件取樣
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [control-as-inference, optimality-variable, planning]
arxiv: ["1805.00909", "2205.09991"]
sources: ["paper.txt §3.2 and footnote 1", "arXiv API metadata (1805.00909)"]
related: ["rl-mdp-q-learning-basics", "diffusion-classifier-guidance-background", "trajectory-diffusion-planning"]
---
# 控制即推論：把獎勵最大化改寫為條件取樣

## 來源

Levine, S. (2018). *Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review*（arXiv 1805.00909）。本論文只引用它的圖模型，不重述其推導。

## 論文如何使用

在 §3.2，論文對每個時間步 t 定義二元隨機變數 O_t 表示該步是否最佳，並令 p(O_t = 1) = exp(r(s_t, a_t))。於是從最佳軌跡取樣可寫成

p̃(τ) = p(τ | O_{1:T} = 1) ∝ p(τ) p(O_{1:T} = 1 | τ)

即公式 (1) 中令 h(τ) = p(O_{1:T} | τ)。強化學習問題被換成條件取樣問題。

論文腳註 1 指出：在一般 RL 中，對未來的條件來自「未來最佳」這個假設，用來寫出動態規劃遞迴；具體形式是動作分佈 log p(a_t | s_t, O_{t:T})（Levine, 2018）。

## 後續銜接

條件取樣由擴散模型的引導機制完成，見 [擴散模型的分類器引導與餘弦排程](diffusion-classifier-guidance-background.html)；實際演算法見 [軌跡層級擴散規劃（Diffuser）](trajectory-diffusion-planning.html)。

r(s, a) 與軌跡回報的基本定義見 [MDP 與 Q-learning 基礎](rl-mdp-q-learning-basics.html)。
