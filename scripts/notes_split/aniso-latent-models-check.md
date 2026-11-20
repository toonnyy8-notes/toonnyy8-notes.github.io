---
slug: aniso-latent-models-check
title: Latent 模型上的對照：SiT-XL/2-REPA 與 RAEv2
cluster: Diffusion & Generative Models
kind: topic
tags: [latent-diffusion, sit, rae, repa, anisoflow]
arxiv: ["2611.17767", "2401.08740", "2112.10752", "2605.18324"]
sources: [2611.17767 section 6.3, 2401.08740 title, 2112.10752 title, 2605.18324 title]
related: ["aniso-lineage-overview", "aniso-pixel-patchified-setting", "latent-diffusion-rae-overview", "rae-followups"]
---
# Latent 模型上的對照

回到[總覽](aniso-lineage-overview.html)。

## 設定
本論文 Sec. 6.3 在 SiT-XL/2-REPA（SiT：Ma 等，arXiv:2401.08740；搭配 SD-VAE，Rombach 等，arXiv:2112.10752）與 RAEv2（Singh 等，arXiv:2605.18324）上，只更換預測目標，其餘架構與訓練配方不變。

## 結果（Table 4、Fig. 7）
- SiT-XL/2-REPA：速度目標 FID 1.44，AnisoFlow 1.44；以 JiT 的參考統計計算分別為 1.41 與 1.40。學到的權重接近速度預測。
- RAEv2：資料預測 FID 1.05，AnisoFlow 1.05；JiT 參考統計下為 0.98 與 0.96。學到的權重接近資料預測。
- 本論文的解讀：預設目標與這些表徵相配；latent 的權重曲線比 pixel 平坦許多，因此近線性縮放的假設不直接適用。

RAE 與 latent 路線的背景見 [Latent Diffusion 兩大主線總覽](latent-diffusion-rae-overview.html) 與 [RAE 後續改進](rae-followups.html)。
