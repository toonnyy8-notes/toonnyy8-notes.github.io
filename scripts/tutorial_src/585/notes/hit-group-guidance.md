---
slug: hit-group-guidance
title: Group guidance：以分組容量差作為引導
cluster: Diffusion & Generative Models
kind: topic
tags: ["group-guidance", "sparse-guidance", "cfg", "token-sparse"]
arxiv: ["2601.01608", "2611.22962"]
sources: ["2611.22962 Sec. 4.1、Appendix A.6", "2601.01608 abstract"]
related: ["hit-compression-budget"]
---
# Group guidance：以分組容量差作為引導

## 前驅：Sparse Guidance

Guiding Token-Sparse Diffusion Models（Krause 等，2601.01608）的摘要指出：以稀疏 token 訓練的擴散模型對 classifier-free guidance（CFG）反應不佳，推論表現不理想；Sparse Guidance（SG）改用 token 層級的稀疏性作為引導訊號，摘要報告在 ImageNet-256 取得 1.58 FID 並少用 25% FLOPs。

## HiT 的 group guidance（Appendix A.6）

動機：HiT 的中央骨幹是在壓縮表示上訓練的；推論時增加群組數雖然表示較細，但改變了骨幹訓練時的運作範圍，預測不一定變好。論文受 SG 啟發，用同一組網路權重，把條件預測配上較細的分組、無條件預測配上較粗的分組。

公式（Eq. 14）：v̂_t = v_∅^(r_u(t)) + s·(v_c^(r_c(t)) − v_∅^(r_u(t)))，其中 r_c(t) ≥ 1 ≥ r_u(t)。r = (r1, r2) 是各階段「目標群組數相對預設平均值的倍數」，以調整門檻實現；兩個倍數都為 1 時退回標準 CFG。

- 只改分組門檻，不改學到的權重、軟 pooling 與解組。
- 不需要比 CFG 更多的網路評估。
- 校準：對每個 checkpoint 以記錄的合併分數擬合門檻。文字到影像用 128 組固定的圖文對、六個雜訊等級、跨時間步共用門檻；ImageNet 收集 64 條取樣軌跡，分成十個等長時間步區間分別擬合。

論文 Sec. 4.1 說明：HiT 在 ImageNet 與文字到影像的主要結果（含收斂比較）使用 group guidance。

預算本身的設計見 [HiT 的壓縮預算：批次平均比例與可變群組數](hit-compression-budget.html)。
