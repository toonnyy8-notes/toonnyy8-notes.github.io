---
slug: psi-superposition-posteriors-method
title: Ψ-posterior 與 Ψ-sampler：以 κ_t 疊加前向與反向 posterior
cluster: Diffusion Language Models
kind: topic
tags: [psi-sampler, superposition-posterior, predictor-corrector, non-markovian]
arxiv: ["2602.21185", "2503.00307", "2205.14987", "2407.15595"]
sources: ["paper:2602.21185 第 3、6 節"]
related: [psi-samplers-lineage-hub, duo-discrete-diffusion-masked-and-uniform, duo-self-correction-and-correctors, psi-ctmc-discrete-diffusion-samplers, uniform-noise-gidd-clean-supervision]
---
# Ψ-posterior 與 Ψ-sampler

屬於 [Ψ-Samplers 脈絡總覽](psi-samplers-lineage-hub.html)。前置見 [前向過程與兩種 prior](duo-discrete-diffusion-masked-and-uniform.html)。

## 定義（論文第 3 節）
許多聯合分佈可產生與式 (1) 相同的邊際。論文定義非 Markovian 的 Ψ-posterior：

$$\Psi_{s|t}(\cdot|x,z_t)=\kappa_t\, q_{s|t}(\cdot|z_t,x)+(1-\kappa_t)\, q_s(\cdot|x),\quad \kappa_t\in[0,1].$$

這是標準反向 posterior 與前向邊際的線性疊加（superposition），適用於 $\pi=m$ 與 $\pi=1/K$。

## Ψ-sampler
以 denoiser $x_\theta$ 取代 $x$ 即得式 (12)。$\kappa_t=1$ 時還原為 MDM 的式 (2) 或 USDM 的式 (3) ancestral sampler；$\kappa_t<1$ 時是較噪的版本，$q_t$ 作 corrector，$q_{s|t}$ 作 predictor（類比 [Gaussian PC](duo-self-correction-and-correctors.html)）。

## 與既有方法的差異（論文第 6 節）
- 推廣 ReMDM（見 [ReMDM](duo-self-correction-and-correctors.html)）到任意噪聲。
- 與 [CTMC 系 PC](psi-ctmc-discrete-diffusion-samplers.html) 不同，不使用 rate matrix。
- 不訓練額外 corrector 模組。
- 支援任意步長排程。
- 論文的實驗結論（摘要）：搭配 uniform-state 擴散時，Gen. PPL 與 FID/IS 優於 ancestral sampling，且步數增加時持續改善。與 USDM 的關係見 [USDM 與 guidance](uniform-noise-gidd-clean-supervision.html)。
