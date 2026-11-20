---
slug: duo-gaussian-diffusion-foundations
title: Gaussian diffusion 的基礎：從 DDPM 到 PF-ODE
cluster: Diffusion & Generative Models
kind: topic
tags: [gaussian-diffusion, ddpm, pf-ode, nelbo, ddim]
arxiv: ["2506.10892", "1503.03585", "2006.11239", "2011.13456", "2010.02502", "2107.00630"]
sources: ["arXiv:2506.10892 §1、§2.2、§2.3", "各論文 arXiv 摘要"]
related: ["duo-lineage-hub", "duo-argmax-duality-theory", "duo-consistency-distillation-to-discrete", "eps-prediction-ddpm-score", "ddpm-quality-and-ddim-sampling"]
---
# Gaussian diffusion 的基礎：從 DDPM 到 PF-ODE

屬於 [Duo 研究脈絡總覽](duo-lineage-hub.html)。Duo 的出發點是 Gaussian diffusion 已有成熟的訓練與取樣技巧，離散擴散沒有。以下依 Duo 的引用整理這些前提。

## 擴散模型的提出與 DDPM
- Sohl-Dickstein 等人 2015，*Deep Unsupervised Learning using Nonequilibrium Thermodynamics*（arXiv:1503.03585）：Duo 在 §1 以此作為擴散模型的出處。
- Ho 等人 2020，*Denoising Diffusion Probabilistic Models*（arXiv:2006.11239）：Duo 引用其為影像擴散的代表，並引用其 ϵ-parameterization 作為相對於 mean-parameterization 的改進。
 
通用的 DDPM 推導見 [ε-prediction、DDPM 與 score 的關係](eps-prediction-ddpm-score.html)，本文不重複。

## Duo 使用的 Gaussian 設定（論文 §2.2）
前向邊際為 N(α̃_t x, (1 − α̃_t²) I_K)，α̃_t 隨 t 單調遞減。NELBO 依 Kingma 等人 2021（*Variational Diffusion Models*，arXiv:2107.00630）寫成以 signal-to-noise ratio ν(t) = α̃_t²/(1 − α̃_t²) 的時間導數加權的平方誤差：−E ν′(t)‖x − x_θ(w_t, t)‖²。論文把 Kingma 等人列為「faster training techniques」。

## 確定性取樣與 PF-ODE
Song 等人 2020，*Score-Based Generative Modeling through Stochastic Differential Equations*（arXiv:2011.13456）提出 probability-flow ODE（PF-ODE）。Song 等人 2021，*Denoising Diffusion Implicit Models*（arXiv:2010.02502）給出確定性取樣。Duo 稱 consistency model 建立在這些確定性取樣器之上（§2.3）。論文的論點是：MDM 因先驗為確定性的遮罩狀態而缺少從雜訊到資料的確定性映射（論文稱為 "implicit" property），所以無法直接套用依賴 PF-ODE 的蒸餾。USDM 經由 Gaussian 的對偶取得這個性質，見 [arg max 對偶的理論](duo-argmax-duality-theory.html)。DDIM 在站內的整理見 [生成品質改進與 DDIM 取樣](ddpm-quality-and-ddim-sampling.html)。
