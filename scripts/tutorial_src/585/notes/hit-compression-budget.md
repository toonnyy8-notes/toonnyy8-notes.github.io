---
slug: hit-compression-budget
title: HiT 的壓縮預算：批次平均比例與可變群組數
cluster: Diffusion & Generative Models
kind: topic
tags: ["compression-ratio", "ratio-loss", "h-net", "adaptive-token-count"]
arxiv: ["2507.07955", "2611.22962"]
sources: ["2611.22962 Sec. 3.3、Appendix A.2", "2507.07955 abstract"]
related: ["hit-merging-baselines-fixed-rates", "hit-group-guidance"]
---
# HiT 的壓縮預算：批次平均比例與可變群組數

## 設計

HiT 不為每張影像固定群組數，而是在整個批次上調節壓縮量，因此群組數可隨影像內容與擴散時間步變動（Sec. 3.3）。第 ℓ 個分組階段的目標壓縮比為 r > 1，目標保留比例 ρ = 1/r。

- 實際保留比例 f_h：整個批次與所有裝置的群組總數除以進入該階段的 token 總數。
- 可微估計 f_s：以軟群組數估計，分母相同。
- Ratio loss（沿用 H-Net，2507.07955）：L_comp^(ℓ) = 1 + (f_h − ρ)(f_s − ρ) / (ρ(1 − ρ))（Eq. 4）。f_h 視為常數，因此損失經 f_s 更新 router 特徵。f_h > ρ 時鼓勵更多合併；f_h < ρ 時鼓勵保留更多群組。
- 總損失 L = L_diff + λ·L_comp（Eq. 5），L_comp 為各階段損失之和。

附錄 A.2 指出 Eq. 4 與 H-Net 的原始寫法 r/(r−1)·[(r−1)f_h f_s + (1−f_h)(1−f_s)]（Eq. 10）在 ρ = 1/r 下等價，損失尺度相同。

## 可微群組數估計（Eq. 11）

一個樣本進入階段時有 N_b 個 token，完整合併樹有 N_b − 1 次合併。把每個門檻判斷換成 sigmoid，得 K̃_b = N_b − Σ σ((s_{b,m} − τ)/T_b)，其中 s_{b,m} 是第 m 次合併的平均親和度，T_b 控制平滑度。合併樹的拓樸在反向傳播中固定，但各次合併的親和度可微。第二階段的 N_b 取第一階段實際的群組數。

## 與其他預算設計的差別

批次平均比例讓預算只約束平均值；單張影像可以高於或低於目標。這與每層固定 r 的合併方法不同，見 [ToMe 與 ToMeSD 在 HiT 中的位置：預先設定的縮減率](hit-merging-baselines-fixed-rates.html)。群組數也可在推論時被調整，作為引導訊號使用：見 [Group guidance：以分組容量差作為引導](hit-group-guidance.html)。
