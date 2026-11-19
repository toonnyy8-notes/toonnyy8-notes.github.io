---
slug: latent-diffusion-open-problems
title: Latent diffusion（diffusability 與 RAE）的開放問題與研究方向
cluster: Diffusion & Generative Models
kind: topic
tags: [open-problems, latent-diffusion, diffusability, rae]
sources: [latent-diffusion-rae.md]
arxiv: []
related: ["latent-diffusion-rae-overview", "velocity-irreducible-variance", "rae-distillation-drifting", "generative-eval-metrics", "diffusability-systematic-study", "pixel-space-diffusion", "diffusability-latent-regularizers"]
---
# Latent diffusion（diffusability 與 RAE）的開放問題與研究方向

屬於 [Latent Diffusion 總覽](latent-diffusion-rae-overview.html)。

1. **VIV 假設的普適性**：Kronecker 可分離協方差與 anisotropic Gaussian/GMM 假設在極端非高斯、超高各向異性的 RAE latent 上是否成立？若不成立，需要非高斯版的 irreducible variance 估計。（見 [VIV](velocity-irreducible-variance.html)）
2. **訓練–蒸餾張力的統一解**：能否設計一個 latent space（或 reparameterization）同時滿足「訓練時高各向異性（低 VIV）」與「蒸餾時低曲率（近 isotropic）」？Drifting Models 是一種繞道，但是否有 latent 側的根本解？（見 [RAE 蒸餾](rae-distillation-drifting.html)）
3. **T2I 遷移性**：DiffusionBench 顯示 ImageNet↔T2I 排名負相關，代表整條 diffusability/RAE 研究線的結論在真實 T2I 上可能失效。亟需在 freeform T2I 上重做相關性分析。（見 [生成評估指標](generative-eval-metrics.html)）
4. **指標的因果驗證**：目前 VIV/LNC/SRSS 與 gFID 的高相關仍是相關而非因果。需以介入式實驗（直接操控單一 latent 屬性、其他不變）建立因果。（見 [系統性比較研究](diffusability-systematic-study.html)）
5. **pixel-space 的表徵預訓練上限**：若把 RAE 的語意先驗以自監督預訓練形式注入 pixel-space（EPG 方向），能否在統一框架下複現 latent 的加速？這將直接回答「latent 層是否必要」。（見 [pixel-space 第三路線](pixel-space-diffusion.html)）
6. **統一 tokenizer**：Prism Hypothesis/UAE 指出語意（低頻）與 pixel（高頻）可頻帶融合。能否設計一個同時最優 rFID、gFID、understanding linear-probe 的單一 tokenizer，讓生成/理解/編輯共用？（見 [其他 diffusability 觀點](diffusability-latent-regularizers.html)）
7. **評估的最終形態**：FDr⁶ 仍繼承 Fréchet 的 Gaussian 假設；當多個 SOTA 已「FID 低於真實 validation」，需要超越 moment-matching 的分布距離或與人評對齊的自動指標。
