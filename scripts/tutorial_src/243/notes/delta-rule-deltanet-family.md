---
slug: delta-rule-deltanet-family
title: Delta rule 與 DeltaNet、Gated DeltaNet：把寫入視為線上回歸
cluster: Architectures & Efficiency
kind: topic
tags: [delta-rule, deltanet, online-learning, linear-rnn]
arxiv: ["2501.00663", "2406.06484", "2412.06464", "2407.04620"]
sources: ["arXiv:2501.00663 §2.1, Appendix C", "arXiv:2406.06484 與 arXiv:2412.06464 的題名與作者經 arXiv API 查證"]
related: ["ttt-linear-mlp"]
---
# Delta rule 與 DeltaNet 系列

加法寫入的第二個改進方向是**改良寫入**。Titans 引用 Widrow 與 Hoff（1988）的 delta rule：加入一組 key-value 之前，先移除該 key 的舊值。

- **DeltaNet**：Parallelizing Linear Transformers with the Delta Rule over Sequence Length（arXiv:2406.06484，Yang、Wang、Zhang 等，2024），提出可平行訓練的演算法。
- **Gated DeltaNet**：Gated Delta Networks: Improving Mamba2 with Delta Rule（arXiv:2412.06464，Yang、Kautz、Hatamizadeh，2024），加上遺忘閘。

Titans 附錄 C 把 DeltaNet 寫成線上學習：對 `L = ½‖S_t k_t − v_t‖²` 做一步梯度下降，得到 `S_{t+1} = S_t(I − θ_t k_t k_t^T) + θ_t v_t k_t^T`（式 34）。Gated DeltaNet 即再加權重衰減。Titans 稱動量係數 `η_t = 0` 時兩者等價；由 Titans 的式 13–14 只能推出 `η_t = α_t = 0` 時即 DeltaNet，與 Gated DeltaNet 則是更新規則形式相近（衰減是否也作用在修正項需查原論文）。

Titans 因此主張 LMM 推廣了 Gated DeltaNet，三點差異：以動量規則納入過去的 surprise、可使用深層記憶而非僅矩陣、具跨 chunk 的非線性遞迴（chunk 內為線性遞迴）。附錄也提到 Longhorn 與 RWKV-7 使用相同的損失形式。

同一觀點下，[TTT-Linear](ttt-linear-mlp.html) 以 MSE 損失得到 delta rule；該筆記有更完整的 TTT 對照。
