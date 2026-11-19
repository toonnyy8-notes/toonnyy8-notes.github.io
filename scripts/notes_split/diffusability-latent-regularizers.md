---
slug: diffusability-latent-regularizers
title: 其他 diffusability 觀點：仿射等變性、頻譜平滑度、流形連續性、分布均勻性
cluster: Diffusion & Generative Models
kind: topic
tags: [diffusability, eq-vae, spectral-bias, ifid, latent-regularization]
sources: [latent-diffusion-rae.md]
arxiv: ["2502.09509", "2502.14831", "2512.19693", "2512.05394", "2503.03206", "2606.03578", "2603.05630", "2603.21085"]
related: ["latent-diffusion-rae-overview", "optimization-dilemma-semantic-alignment", "velocity-irreducible-variance", "diffusability-systematic-study"]
---
# 其他 diffusability 觀點：仿射等變性、頻譜平滑度、流形連續性、分布均勻性

屬於 [Latent Diffusion 總覽](latent-diffusion-rae-overview.html)。語意與空間觀點見 [optimization dilemma 與語意/空間對齊](optimization-dilemma-semantic-alignment.html)；速度歧義見 [VIV](velocity-irreducible-variance.html)。

## (c) 仿射等變性 (affine equivariance) — EQ-VAE（arXiv:2502.09509，ICML 2025）
- 診斷：既有 AE 對 scaling/rotation 缺乏等變性，導致 latent 複雜。
- 做法：fine-tune 使 decoder 對縮放/旋轉等變。
- 結果：DiT-XL/2 ×7、REPA ×4 加速，不損 reconstruction。
- **但 Zhong et al. 裁決此觀點與 gFID 相關性最低、兩個度量彼此不一致，故被剔除**——這是重要的反面證據（見 [系統性比較研究](diffusability-systematic-study.html)）。

## (d) 頻譜平滑度 (spectral smoothness)
- Skorokhodov et al.（Improving Diffusability of Autoencoders，arXiv:2502.14831，Snap Research，ICML 2025）：DCT 分析發現 bottleneck channel 越大，latent 高頻越過量；高頻干擾 diffusion 的 coarse-to-fine 過程。scale equivariance 對齊 latent 與 RGB 頻譜，僅需 ≤20K fine-tune steps，ImageNet-1K 256² FID −19%、Kinetics-700 影片 FVD ≥−44%。限制：大 channel bottleneck 才明顯。
- Prism Hypothesis（arXiv:2512.19693）：語意 encoder 偏低頻（抽象語意）、pixel encoder 保留高頻（細節）； [Semantic Scholar](https://www.semanticscholar.org/paper/The-Prism-Hypothesis:-Harmonizing-Semantic-and-via-Fan-Diao/e3e8cca6a793e617836ade10016d43218cce4ed8) 提出 UAE 用 frequency-band modulator 融合，DINOv2-L 版 PSNR 29.65 / SSIM 0.88 / rFID 0.19，reconstruction 勝過 RAE。 [HyperAI](https://hyper.ai/en/papers/2512.19693)
- Video VAE 頻譜偏置（Delving into Latent Spectral Biasing of Video VAEs，arXiv:2512.05394）為影片場景對應版本。

### 理論：頻譜偏置（Wang & Pehlevan）
Wang & Pehlevan（arXiv:2503.03206，NeurIPS 2025 Spotlight）以 Gaussian-equivalence 原理解出線性/卷積 denoiser 的 gradient-flow 動態，得到普適的 **inverse-variance spectral law：某模態收斂時間 τ∝λ⁻¹**——高變異（粗結構）比低變異（細節）早數個數量級被學會（線性網路沿主成分、卷積網路沿 Fourier modes 收斂）。這直接支撐了 diffusability 研究的「latent 高頻能量少 = 較易學」直覺（Zhong et al. 在 Latent Smoothness 一節明確引用）。 [arxiv](https://arxiv.org/pdf/2606.03578)

## (e) 流形連續性 (manifold continuity) — iFID（arXiv:2603.05630）
- 動機：rFID 與 gFID 甚至負相關，因為 reconstruction 偏好「分離/斷裂」的 latent，而 diffusion 偏好連續 latent。 [Hugging Face](https://huggingface.co/papers/2603.05630)
- 做法：對每個 latent 取最近鄰、插值、解碼、與真實分布算 FID。 [ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260305630X/abstract)
- 理論：diffusion 樣本集中在連接鄰近訓練樣本的 ridge set 附近，故插值品質預測生成品質。 [arXiv](https://arxiv.org/html/2603.05630v2) 論文並細分：rFID 對應 diffusion 的 refinement phase、iFID 對應 navigation phase。
- 結果：跨多種 VAE 的 Pearson/Spearman ≈0.85，是首個與 gFID 強相關的指標。 [ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260305630X/abstract) 限制：不可直接最小化。VE（Variance Expansion Loss，arXiv:2603.21085）為同一流形連續性思路的訓練損失版本。

## (f) 分布均勻性 (distribution uniformity) — VA-VAE 協定
- 用 t-SNE 投影 + density coefficient of variation / Gini / normalized entropy 評估 latent 是否被均勻利用。 [arxiv](https://arxiv.org/pdf/2606.03578) Zhong et al. 直接沿用，但發現其相關性偏弱且與頻譜/VIV 共線。
