---
slug: lact-large-chunk-ttt
title: 從 One-Minute Video 到 LaCT：大 chunk Test-Time Training
cluster: Architectures & Efficiency
kind: topic
tags: [lact, test-time-training, large-chunk, hardware-efficiency, muon, video-generation]
sources: [ttt-sequence-modeling.md]
arxiv: ["2504.05298", "2505.23884"]
related: ["ttt-sequence-modeling-overview", "ttt-linear-mlp", "ttt-theory-and-applications"]
---
# 從 One-Minute Video 到 LaCT：大 chunk Test-Time Training

屬於 [TTT 序列建模總覽](ttt-sequence-modeling-overview.html)。前身見 [原版 TTT](ttt-linear-mlp.html)。

## 影片轉折點：One-Minute Video Generation（arXiv:2504.05298，CVPR 2025）
Karan Dalal、Yu Sun 等把 TTT 層加進預訓練的 Diffusion Transformer（5B），fine-tune 後能從文字 storyboard 生成一分鐘、多場景、時間一致的 Tom and Jerry 卡通。abstract 載明：「Compared to baselines such as Mamba~2, Gated DeltaNet, and sliding-window attention layers, TTT layers generate much more coherent videos...leading by 34 Elo points in a human evaluation of 100 videos per method」。這是 TTT 主系列首次大規模進入生成式長序列，也暴露了「小 chunk 更新在 N 維資料（影片）上既慢又不自然」的問題，直接動機了 LaCT。限制：5B 基模限制、有 artifact。code：github.com/test-time-training/ttt-video-dit。

## LaCT / Test-Time Training Done Right（Zhang et al., arXiv:2505.23884，ICLR 2026）
**設計理念。** Zhang et al.（MIT + Adobe Research）觀察到：原版 TTT 的算術強度（FLOPs/記憶體存取）被 chunk 大小 b 上限約束，小 b 使 TTT 層 memory-bound，FLOPs 利用率 <5%、難以超過 10%。LaCT 反其道而行，用 **2,048 到 1,000,000 token** 的超大 chunk 作為更新單位。

**數學原理（與 mini-batch TTT 的差異）。** LaCT 的 chunk 更新對 chunk 內「所有」key-value pair 的損失和求梯度，再一次更新：
- `g = ∇_W Σ_{i=1}^{b} η_i ℓ(f_W(k_i), v_i)`（對整個 chunk 的加總損失）
- `W ← weight-update(W, g)`，接著對所有 query 套用同一份更新後權重 `o_i = f_W(q_i)`

關鍵差異：原版每 token（或每 16/64 token）更新一次、chunk 內維持細粒度因果；LaCT 一個大 chunk 才更新一次、chunk 內 token 被當成「無序集合」。為補回 chunk 內局部依賴，LaCT 在同一層搭配 **sliding-window attention**（SWA 與 TTT 共用 QKV）。compute-to-memory 比例 `r = 2h²b/(2h²+4hb) ≤ min(h/2, b)`，清楚顯示算術強度被 chunk 大小 b 限制。
- 內層損失用負點積：`ℓ = − f_W(k_i)^⊤ v_i`
- 非線性 fast weight 用無 bias 的 SwiGLU-MLP：`f_W(x) = W_2[SiLU(W_1 x) ∘ (W_3 x)]`，三個矩陣 W₁,W₂,W₃
- 更新可選 vanilla GD + L2 weight normalization，或 **Muon**（對矩陣梯度做 Newton–Schulz 正交化：若 g=USVᵀ 則 Muon(g)≈UVᵀ），再 L2 normalize；不使用顯式 weight decay

**實驗結果。** GPU 利用率達「up to 70% on NVIDIA A100s」，僅需數十行純 PyTorch、無需自訂 kernel；非線性 state size「up to 40% of model parameters」（前人僅 0.1%–5%）。
- Novel view synthesis：prefill 從 16.1s 降到 1.4s，最長 1M token context（128 張影像），在稀疏視角下勝過 3D Gaussian Splatting（GSO 256-res 上 full-attention baseline 達 37.9 dB @48 views，LaCT 自身約 37.6 dB）。
- 語言建模：大 chunk 線性遞迴在語言上初期略遜 GLA/DeltaNet（語言無天然 chunk 結構），但搭配大非線性 state + Muon 後反超。
- 影片：把 14B Wan 2.1 雙向影片 diffusion 改造成自回歸，每 8.8s 片段約 56,160 token。

**優缺點。** 優勢：硬體效率、可放大 state 容量、可用先進 optimizer、跨模態（集合/N 維格點）、無需客製 kernel。缺陷：chunk 內失去因果順序（靠 SWA 補）、語言任務不一定贏純線性注意力、單一超大 chunk 在很長序列上易 catastrophic forgetting/overfitting（後續 FSM/LaCET 即針對此，見 [TTT 理論與應用](ttt-theory-and-applications.html)）。

**注意。** LaCT 的「>70% on H100」未在正文逐字證實，正文明載的是「up to 70% on NVIDIA A100s」。

**自習。** arXiv:2505.23884、project page tianyuanzhang.com/projects/ttt-done-right、code github.com/a1600012888/LaCT。
