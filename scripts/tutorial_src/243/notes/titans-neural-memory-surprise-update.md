---
slug: titans-neural-memory-surprise-update
title: Titans 的神經長期記憶：surprise、動量、遺忘與平行訓練
cluster: Architectures & Efficiency
kind: topic
tags: [titans, neural-memory, surprise, momentum, weight-decay]
arxiv: ["2501.00663", "2407.04620"]
sources: ["arXiv:2501.00663 §3.1, §3.2, Appendix C"]
related: ["delta-rule-deltanet-family", "ttt-linear-mlp"]
---
# Titans 的神經長期記憶

記憶 `M` 是 `L_M ≥ 1` 層的 MLP，在測試時以線上學習更新其權重。論文把這個內迴圈稱為元模型：`W_K`、`W_V` 等其他參數在外迴圈訓練。

**損失**：`ℓ(M_{t-1}; x_t) = ‖M_{t-1}(k_t) − v_t‖²`，其中 `k_t = x_t W_K`、`v_t = x_t W_V`（式 11、12）。

**Surprise**：以損失對記憶的梯度度量。單純的 `M_t = M_{t-1} − θ_t ∇ℓ` 在連續幾次大 surprise 後梯度變小，會漏掉其後的資訊，因此改為過去 surprise 加當下 surprise：`S_t = η_t S_{t-1} − θ_t ∇ℓ(M_{t-1}; x_t)`（式 10），形式與帶動量的梯度下降相同。`η_t`、`θ_t` 為資料相依。

**遺忘**：`M_t = (1−α_t) M_{t-1} + S_t`（式 13、14），`α_t→1` 清除記憶，`α_t→0` 不影響過去。論文將其對應到權重衰減。

**讀取**：不更新權重的前向傳遞，`y_t = M*(q_t)`（式 15）。

**平行訓練（§3.2）**：沿用 TTT 的 mini-batch 梯度下降（[原版 TTT](ttt-linear-mlp.html)），把序列切成大小 `b` 的 chunk，chunk 內的梯度都對 chunk 起點的權重計算，轉為 matmul；動量項是線性遞迴，以平行 associative scan 計算。

**深層記憶的理由**：矩陣記憶等於線上線性回歸，假設歷史資料間的依賴為線性；論文引用 MLP 至少兩層嚴格比線性模型更具表達力（Hornik 等 1989），並在 §5.5 以實驗比較記憶深度。與 [Delta rule 系列](delta-rule-deltanet-family.html) 的對照見該筆記。
