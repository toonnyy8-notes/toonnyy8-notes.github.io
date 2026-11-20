---
slug: duo-discrete-diffusion-masked-and-uniform
title: 離散擴散：遮罩式與均勻狀態（Duo 的背景）
cluster: Diffusion Language Models
kind: topic
tags: [discrete-diffusion, usdm, mdm, d3pm, sedd, mdlm, udlm, masked-diffusion, uniform-state, noise-schedule, posterior]
arxiv: ["2506.10892", "2107.03006", "2310.16834", "2406.07524", "2406.04329", "2412.10193", "2205.14987", "2602.21185", "1503.03585", "2406.03736"]
sources: ["arXiv:2506.10892 §1、§2.1、§6", "各論文 arXiv 摘要", "paper:2602.21185 第 2.1 節", "arXiv API metadata（標題、作者、年份）"]
related: ["duo-lineage-hub", "psi-samplers-lineage-hub", "psi-superposition-posteriors-method", "duo-argmax-duality-theory", "duo-self-correction-and-correctors", "discrete-diffusion-foundations-d3pm", "mdlm-masked-diffusion-objective", "uniform-noise-gidd-clean-supervision", "dllm-three-corruption-paradigms"]
---
# 離散擴散：遮罩式與均勻狀態（Duo 的背景）

屬於 [Duo 研究脈絡總覽](duo-lineage-hub.html)與 [Ψ-Samplers 脈絡總覽](psi-samplers-lineage-hub.html)。

## 發展順序（依 Duo 的引用）
- Sohl-Dickstein 等人（2015，arXiv 1503.03585）提出以逐步加噪與學習反向過程生成資料的框架；Austin 等人（2021）把它推到離散狀態空間。Ψ-Samplers 論文第 2.1 節把兩者列為離散擴散的基礎。
- Austin 等人 2021，*Structured Denoising Diffusion Models in Discrete State-Spaces*（D3PM，arXiv:2107.03006）：在離散狀態空間定義擴散，包含 absorbing 與 uniform 兩種轉移；Duo 將其列為 USDM 與慢速 ancestral sampling 的出處。
- Campbell 等人 2022，*A Continuous Time Framework for Discrete Denoising Models*（arXiv:2205.14987）：Duo 將其列為 MDM 的 predictor-corrector 方法之一。
- Lou 等人 2023，*Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution*（SEDD，arXiv:2310.16834）：Duo 的 USDM 基線之一（SEDD Uniform）。
- Sahoo 等人 2024，*Simple and Effective Masked Diffusion Language Models*（MDLM，arXiv:2406.07524）與 Shi 等人 2024，*Simplified and Generalized Masked Diffusion for Discrete Data*（arXiv:2406.04329）：簡化遮罩式擴散的目標。Ψ-Samplers 論文另引 Ou 等人（2024，arXiv 2406.03736，*Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data*）為 MDM 的來源工作。Duo 以 MDLM 為最強 MDM 基線。
- Schiff 等人 2024，*Simple Guidance Mechanisms for Discrete Diffusion Models*（UDLM，arXiv:2412.10193）：把 interpolating 框架用於均勻先驗，Duo 以其為直接前身（Table 3 的 "w/o improved training loss" 退化為 UDLM）。

## Duo 採用的統一記號（§2.1）
前向邊際 q_t(·|x) = Cat(·; α_t x + (1 − α_t)π)。MDM 的 π 為遮罩 token m；USDM 的 π = 1/K。USDM 的轉移矩陣 Q_t = (α′_t/(Kα_t))[11ᵀ − K I_K]。取樣由先驗 z_1 ∼ 1/K 開始做 ancestral denoising。

## 差異
MDM 的 token 一旦解除遮罩即固定；USDM 允許每一步更新 token（§2.1）。論文報告：USDM 的困惑度長期落後 MDM（Austin 等人 2021；Lou 等人 2023），Duo 將與 MDLM 的差距縮小到 2 PPL 以內（§5.1）。站內對應整理見 [離散擴散的基礎：D3PM](discrete-diffusion-foundations-d3pm.html)、[MDLM](mdlm-masked-diffusion-objective.html)、[均勻噪聲與 GIDD](uniform-noise-gidd-clean-supervision.html)。

## Ψ-Samplers 論文使用的形式（第 2.1 節）
- 序列 $x$ 的每個位置獨立加噪，邊際為 $q_t(\cdot|x)=\mathrm{Cat}(\alpha_t x+(1-\alpha_t)\pi)$，$\alpha_t$ 隨 $t$ 單調遞減（noise schedule）。
- prior $\pi$ 的選擇決定模型類型：$\pi=m$（one-hot 的 [MASK]）為 Masked Diffusion Model（MDM），$\pi=1/K$ 為 Uniform-state Diffusion Model（USDM）。
- MDM 的 posterior 見該論文式 (2)：已解遮罩的 token 不再改變，因此無法修正錯誤。
- USDM 的 posterior 見式 (3)，token 在生成過程中可多次改變。
- USDM 的來源工作為 Schiff 等人（2024，arXiv 2412.10193）與 Duo；見 [USDM、guidance 與自我修正](uniform-noise-gidd-clean-supervision.html)。
- 該論文的 Ψ-posterior 對 $\pi$ 不設限，因此同時適用於兩者（見 [Ψ-posterior](psi-superposition-posteriors-method.html)）。
