---
slug: diffusion-policy-class
title: 擴散模型作為策略類別：以反向擴散鏈表示 π(a|s)
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [diffusion-policy, conditional-diffusion, policy-class, ddpm, offline-rl]
arxiv: ["2208.06193", "2006.11239", "2112.07804", "2011.13456", "2301.10677", "2211.15657"]
sources: ["paper.txt §2, §3.1, Appendix G", "arXiv API 驗證 2006.11239、2112.07804、2011.13456、2301.10677、2211.15657 的題名與作者；對這些工作的描述依 paper.txt §2 的敘述"]
related: ["ddpm-training-and-sampling-recap", "eps-prediction-ddpm-score", "score-geometry-score-matching-sde", "gaussian-policy-multimodality-limits", "diffusion-ql-loss-coupling", "diffusion-policy-inference-cost", "diffusion-in-rl-concurrent-work"]
---
# 擴散模型作為策略類別

Wang, Hunt, Zhou（ICLR 2023，arXiv 2208.06193）把策略 `π_θ(a|s)` 直接寫成條件擴散模型的反向過程。本筆記說明這個策略類別的定義；其他以擴散模型表示策略的筆記（IDQL、Value Forcing 等）可連到此處作為共同背景。擴散模型本身的訓練與取樣見 [DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html) 與 [ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)。

## 定義

動作 `a` 取代影像 `x`，狀態 `s` 作為條件。為區分兩種時間，擴散步數用上標 `i ∈ {1,…,N}`，軌跡時間用下標 `t`。策略為

`π_θ(a|s) = p_θ(a^{0:N}|s) = N(a^N; 0, I) ∏_{i=1}^{N} p_θ(a^{i-1}|a^i, s)`

反向鏈最後的 `a^0` 是實際執行的動作。依 Ho et al.（2020）以雜訊預測模型 `ε_θ(a^i, s, i)` 參數化，共變異數固定為 `β_i I`，取樣為

`a^{i-1} = a^i/√α_i − β_i/(√α_i(1−ᾱ_i)^{1/2}) · ε_θ(a^i,s,i) + √β_i · ε`，`ε ~ N(0,I)`；`i=1` 時 `ε=0`。

訓練使用簡化的 DDPM 目標（論文式 2）：`L_d(θ) = E_{i,ε,(s,a)~D}‖ε − ε_θ(√ᾱ_i a + √(1−ᾱ_i) ε, s, i)‖²`，其中 `i` 在 `{1,…,N}` 均勻取樣，`D` 為離線資料集。

## 此損失的性質

- `L_d` 是行為複製（behavior cloning）損失：目標是從與資料相同的分布取樣動作。
- 反向鏈的邊際分布是隱式分布，可表示偏態與多峰；成因與對照見 [Gaussian、CVAE 與 MMD 策略的多峰限制](gaussian-policy-multimodality-limits.html)。
- 正則化只需要來自 `D` 與當前策略的樣本，不需要行為策略的密度值，因此不需先另外訓練行為複製模型。
- 每個資料點只取樣單一擴散步 `i` 即可計算 `L_d`；取樣時則需把 `ε_θ` 迭代計算 `N` 次（見 [擴散策略的推論成本與加速方向](diffusion-policy-inference-cost.html)）。

## 雜訊排程

為了在小 `N` 下運作，論文依 Xiao et al.（2021，arXiv 2112.07804）採用 variance-preserving SDE（Song et al., 2021，arXiv 2011.13456）導出的排程：`β_min=0.1`、`β_max=10.0`，`β_i = 1 − α_i = 1 − exp(−β_min/N − 0.5(β_max−β_min)(2i−1)/N²)`。

## 與其他擴散式決策方法的差別

- Diffuser 對整段軌跡做擴散，屬 model-based 規劃；本論文只對單一動作做擴散，屬 model-free，見 [擴散模型在 RL 中的同期與前驅工作](diffusion-in-rl-concurrent-work.html)。
- Pearce et al.（2023，arXiv 2301.10677）以擴散模型模仿人類行為；Ajay et al.（2022，arXiv 2211.15657）論述條件生成模型用於決策。論文引用這兩者作為相關工作。
- 此策略類別如何加入 Q 值項見 [Diffusion-QL 的損失耦合](diffusion-ql-loss-coupling.html)。
