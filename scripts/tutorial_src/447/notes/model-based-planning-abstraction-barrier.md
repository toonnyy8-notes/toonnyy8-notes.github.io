---
slug: model-based-planning-abstraction-barrier
title: 學習式動態模型與軌跡最佳化的分工
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [model-based-rl, trajectory-optimization, mppi, planning]
arxiv: ["1708.02596", "1805.12114", "1509.01149", "1907.02057", "2205.09991"]
sources: ["paper.txt §1, §2.1, §6", "arXiv API metadata (1708.02596, 1805.12114)", "paper.txt reference list (1509.01149, 1907.02057)"]
related: ["rl-mdp-q-learning-basics", "trajectory-diffusion-planning", "control-as-inference-guided-sampling"]
---
# 學習式動態模型與軌跡最佳化的分工

## 問題設定

本論文（2205.09991）的 §2.1 把決策問題寫成軌跡最佳化：系統動態為 s_{t+1} = f(s_t, a_t)，目標是找出動作序列 a_{0:T}，使各時間步獎勵總和 J 最大。軌跡 τ = (s_0, a_0, ..., s_T, a_T)。MDP 與 Q-learning 的基本定義見 [MDP 與 Q-learning 基礎](rl-mdp-q-learning-basics.html)。

## 慣用做法：模型與規劃器分開

軌跡最佳化需要已知動態 f。常見做法是先用監督式學習訓練近似動態模型，再把模型放進既有的規劃或最佳化程序。論文（§6）把這種做法稱為在模型與規劃器之間設了一道抽象屏障（abstraction barrier）：學習只負責逼近環境動態，規劃器的形式不太依賴模型的形式。

## 論文指出的問題

- 強力的軌跡最佳化器會利用學習模型的誤差，產生的計畫較像對抗樣本，而不是最佳軌跡（論文 §1，引用 Talvitie 2014 與 Ke et al. 2018）。
- 因此近期的 model-based RL 較常沿用 model-free 的價值函數與策略梯度（論文引用 Wang et al., 2019，arXiv 1907.02057，Benchmarking Model-Based Reinforcement Learning）。
- 使用線上規劃的方法多半只用不需梯度的簡單最佳化：random shooting（Nagabandi et al., 2018，arXiv 1708.02596）或 cross-entropy method（Chua et al., 2018，PETS，arXiv 1805.12114）。論文另以 MPPI（Williams et al., 2015，arXiv 1509.01149）作為規劃器與基準。

## 與本論文的關係

Diffuser 的目標是拆掉這道屏障：模型與規劃一起訓練，取樣與規劃幾乎相同（見 [軌跡層級擴散規劃（Diffuser）](trajectory-diffusion-planning.html)）。論文在 §5.1 讓 MPPI 使用真實動態，仍得到較低分數，說明長時程規劃本身就困難；§5.3 把 Diffuser 當動態模型接上 MPPI，表現不優於隨機，論文據此推論效果來自模型與規劃的耦合。

條件取樣如何由獎勵得到，見 [控制即推論](control-as-inference-guided-sampling.html)。
