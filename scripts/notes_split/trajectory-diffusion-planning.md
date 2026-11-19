---
slug: trajectory-diffusion-planning
title: 軌跡層級擴散規劃（Diffuser）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [diffuser, trajectory-diffusion, planning, classifier-guidance, inpainting, offline-rl]
arxiv: ["2205.09991", "1503.03585", "2006.11239", "2105.05233", "1805.00909", "2004.07219"]
sources: ["paper.txt §1-§5, Appendix C"]
related: ["ddpm-training-and-sampling-recap", "diffusion-classifier-guidance-background", "control-as-inference-guided-sampling", "model-based-planning-abstraction-barrier", "offline-rl-benchmarks-and-baselines", "sequence-modeling-offline-rl"]
---
# 軌跡層級擴散規劃（Diffuser）

Janner, Du, Tenenbaum, Levine. *Planning with Diffusion for Flexible Behavior Synthesis*（ICML 2022，arXiv 2205.09991）。本筆記是其他 diffusion × RL 筆記（決策擴散、策略擴散等）可引用的共同基礎。

## 核心設計

1. **軌跡即資料**：以二維陣列 τ（每欄為一個時間步的 [s_t; a_t]）表示整段計畫，狀態與動作一起預測，動作視為額外的狀態維度。
2. **非自迴歸**：一次產生所有時間步。論文理由：決策可以是反因果的（現在的決策取決於未來），如 p(s_1 | s_0, s_T)。
3. **時間局部性**：模型為一維時間卷積的 U-Net（6 個殘差區塊，每塊兩個時間卷積、group norm、Mish）。單一去噪步只強制局部一致，許多步疊加後得到全域一致。
4. **訓練**：簡化的 ε 預測損失；超參數 N = 20（locomotion）、N = 100（方塊堆疊）；Adam、學習率 4e-5、batch 32、500k 步。
5. **規劃 = 取樣 + 擾動 h(τ)**：p̃_θ(τ) ∝ p_θ(τ)h(τ)。

## 兩種條件方式

- **獎勵引導**（Algorithm 1）：另訓練 J_φ 預測軌跡累積獎勵，每個反向步把平均值改為 μ + αΣ∇J(μ)，α = 0.1（hopper-medium-expert 為 0.0001）。理論依據見 [控制即推論](control-as-inference-guided-sampling.html) 與 [分類器引導與餘弦排程](diffusion-classifier-guidance-background.html)。
- **修補（inpainting）**：以 Dirac delta 型 h(τ) 固定起點、終點或其他約束，其餘由去噪補齊。執行時採 receding-horizon：只執行計畫的第一個動作並重新規劃。

## 論文列出的性質

長時程規劃（不受單步誤差累積）、時間組合性（拼接訓練資料中的子序列）、可變長度計畫（時程由輸入雜訊大小決定）、任務組合性（獎勵可加總梯度）；以前一步計畫暖啟動可降低擴散步數（§5.4，Walker2d Medium-Expert）。

## 與前作的位置

規劃器與模型分工的背景見 [學習式動態模型與軌跡最佳化的分工](model-based-planning-abstraction-barrier.html)；實驗基準與比較方法見 [離線 RL 基準與比較方法](offline-rl-benchmarks-and-baselines.html)、[以序列模型處理離線 RL](sequence-modeling-offline-rl.html)。DDPM 本身見 [DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html)。論文自述（§6）：據其所知，擴散模型此前未被用於 RL 或決策。
