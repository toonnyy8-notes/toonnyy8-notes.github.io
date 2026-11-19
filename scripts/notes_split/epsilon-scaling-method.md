---
slug: epsilon-scaling-method
title: Epsilon Scaling：不需重新訓練的曝光偏差緩解方法
cluster: Diffusion & Generative Models
kind: topic
tags: [epsilon-scaling, training-free, adm, fid, sampling]
arxiv: ["2308.15321", "2105.05233", "2206.00364", "2010.02502"]
sources: [2308.15321]
related: [exposure-bias-elucidation-overview, prediction-error-variance-analysis, exposure-bias-prior-mitigations, diffusion-sampling-variance-schedule, ddpm-training-and-sampling-recap]
---
# Epsilon Scaling

回到 [總覽](exposure-bias-elucidation-overview.html)。承接 [預測誤差與取樣變異數分析](prediction-error-variance-analysis.html)。

## 動機
本論文固定已訓練網路，比較訓練情境（輸入真實 x_t）與取樣情境（輸入 x̂_t）下 ε_θ 輸出的 L2 範數，在 CIFAR-10 上發現取樣時的範數在每個時間步都較大。本論文據此推論網路在 x̂_t 附近學到的向量場長度偏長。

## 方法
取樣時將網路輸出除以係數 λ(t)：μ_θ(x_t,t) = (1/√α_t)(x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t,t)/λ_t)。這是只改取樣公式的 plug-in 方法，不增加計算量。
- λ(t) 由範數比 ΔN(t) = ‖ε_θ^s‖₂/‖ε_θ^t‖₂ 推得；ΔN(t) 在 t ∈ (5, T) 可用二次函數擬合，因此 λ(t) = kt + b 為線性。
- 取樣步數增加時 k 趨近 0，故多數實驗採用均勻 λ(t) = b，搜尋約需 6 至 10 次試驗。
- 本論文指出此法只修正 ε_θ 的大小誤差，不修正方向誤差，所以無法使 δ_t 降為 0。

## 結果（皆取自本論文）
- 以預訓練 ADM 為基線（[ADM 見此](diffusion-sampling-variance-schedule.html)），CIFAR-10 無條件生成 100 步 FID 由 3.37 降至 2.17；FFHQ 128×128 100 步由 14.52 降至 6.77。
- 在表 3 中，ADM-ES 以 100 步達 2.17 FID，優於 EDM（2.27 與 2.23）；本論文指出取樣速度比 EDM 與 Improved SDE 快 5 至 20 倍。
- 在 DDIM/DDPM（η=0/1）、EDM、LDM、DiT、PFGM++ 上均驗證有效（見 [DDPM 與 DDIM 公式](ddpm-training-and-sampling-recap.html)）。
- 圖 4 顯示 ADM-ES 在取樣終點的 δ_t 較低。

與需要重訓的方法相比見 [既有緩解方法](exposure-bias-prior-mitigations.html)。
