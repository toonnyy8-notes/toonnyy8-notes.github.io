---
slug: dowser-functional-importance-theory-and-probing
title: 功能重要性分數、Theorem 3.1 與合成探測
cluster: Memory & Continual Learning
kind: topic
tags: [Model-Dowser, importance-score, Jacobian, Hutchinson, synthetic-probing, data-free]
arxiv: ["2602.04509"]
sources: ["2602.04509 §3.1-§3.3", "2602.04509 Appendix B, C, I, J"]
related: ["dowser-lineage-hub", "dowser-parameter-importance-pruning-scores"]
---
# 功能重要性分數、定理與合成探測

本篇記錄 Model-Dowser（arXiv:2602.04509）的方法本體，前人工作的位置見 [參數重要性與剪枝分數](dowser-parameter-importance-pruning-scores.html)。

**Theorem 3.1（單一權重擾動）。** 在一階 Taylor 近似下，對第 l 層權重 W_ij 做擾動 ΔW_ij 時，輸出位移為
‖Δf‖₂ ≈ ‖J_i‖₂ · |ΔW_ij| · |h_j^(l-1)|，
其中 J_i = ∂f/∂z_i 是網路輸出對第 i 個預活化值的 Jacobian 欄，h^(l-1) 是該層的輸入激活。Corollary 3.2 將它推廣到多權重擾動，得到總輸出位移的上界（各參數敏感度之和）。證明在 Appendix B 與 C。

**重要性分數。** 把潛在擾動以現有權重大小代入：
S_ij = ‖J_i‖₂ · |W_ij| · |h_j|。
三項分別是輸出敏感度、連接強度、輸入活動量。

**Hutchinson 估計。** 完整 Jacobian 計算成本高。論文用 Rademacher 隨機向量 ξ 投影輸出，以 E_ξ[(∂(ξᵀf)/∂z_i)²] = ‖J_i‖₂² 估計，只需少量反向傳播。

**合成探測（data-free）。** 預訓練資料常不可得，因此從 tokenizer 詞表隨機取 token 當種子，由預訓練模型自己生成 N 個提示，在這些提示上累積分數的 Monte Carlo 平均（式 5）。實驗設定 N = 64、R = 8（R 為每個樣本的 Rademacher 向量數）。總成本為 O(N·R) 次前向與反向傳播，遠低於顯式 Jacobian 所需的 d_final 次反向傳播。

**無資料估計的驗證（Table 5）。** 與真實資料算出的分數排序比較：N = 64、R = 8 時，Hamming 距離 0.050 ± 0.002、Spearman 相關 0.891 ± 0.003；隨機選取的對照組 Hamming 距離為 0.226。Appendix I、J 另有細節與純文字探測驗證。

**Sparse fine-tuning。** 每層內依分數排序，給定更新比例 ρ，分數落在最低 ρ 百分位的參數遮罩為 1（可更新），其餘為 0（凍結）。更新式為 θ* = θ − λ·(M ⊙ ∂L/∂θ)。論文實驗用 ρ = 0.1 作為設定之一。遮罩在微調前算一次，之後固定。

**論文的 justification。** 凍結高分參數即壓低輸出擾動的主要貢獻者；論文主張因為分數考慮了非齊次激活的非線性（透過 Jacobian），比單純 magnitude 凍結更穩定。註：Theorem 3.1 是一階近似，論文把 |W_ij| 當作 |ΔW_ij| 的代理，這個代入是定義上的選擇。

總覽：[Model-Dowser 的研究脈絡總覽](dowser-lineage-hub.html)
