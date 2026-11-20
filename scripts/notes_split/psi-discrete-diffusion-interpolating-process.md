---
slug: psi-discrete-diffusion-interpolating-process
title: 離散擴散的 interpolating 前向過程與 masked／uniform 兩種 prior
cluster: Diffusion Language Models
kind: topic
tags: [discrete-diffusion, masked-diffusion, uniform-state, noise-schedule, posterior]
arxiv: ["2602.21185", "1503.03585", "2107.03006", "2406.07524", "2406.04329", "2406.03736", "2412.10193"]
sources: ["paper:2602.21185 第 2.1 節", "arXiv API metadata（標題、作者、年份）"]
related: [psi-samplers-lineage-hub, psi-superposition-posteriors-method, mdlm-masked-diffusion-objective, discrete-diffusion-foundations-d3pm, uniform-noise-gidd-clean-supervision]
---
# 離散擴散的 interpolating 前向過程與兩種 prior

屬於 [Ψ-Samplers 脈絡總覽](psi-samplers-lineage-hub.html)。

## 來源脈絡
Sohl-Dickstein 等人（2015，arXiv 1503.03585）提出以逐步加噪與學習反向過程生成資料的框架；Austin 等人（2021，arXiv 2107.03006，D3PM）把它推到離散狀態空間。論文第 2.1 節把兩者列為離散擴散的基礎，詳細機制見既有筆記 [離散擴散的基礎](discrete-diffusion-foundations-d3pm.html)。

## 論文使用的形式（第 2.1 節）
- 序列 $x$ 的每個位置獨立加噪，邊際為 $q_t(\cdot|x)=\mathrm{Cat}(\alpha_t x+(1-\alpha_t)\pi)$，$\alpha_t$ 隨 $t$ 單調遞減（noise schedule）。
- prior $\pi$ 的選擇決定模型類型：$\pi=m$（one-hot 的 [MASK]）為 Masked Diffusion Model（MDM），$\pi=1/K$ 為 Uniform-state Diffusion Model（USDM）。
- MDM 的 posterior 見論文式 (2)：已解遮罩的 token 不再改變，因此無法修正錯誤。
- USDM 的 posterior 見式 (3)，token 在生成過程中可多次改變。

## MDM 的來源工作（論文引用）
- Sahoo 等人（2024，arXiv 2406.07524）：[MDLM](mdlm-masked-diffusion-objective.html)。
- Shi 等人（2024，arXiv 2406.04329）：*Simplified and Generalized Masked Diffusion for Discrete Data*。
- Ou 等人（2024，arXiv 2406.03736）：*Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data*。

## USDM 的來源工作
Schiff 等人（2024，arXiv 2412.10193）與 Duo。見 [USDM、guidance 與自我修正](psi-uniform-state-guidance-and-self-correction.html) 及 [均勻噪聲與 GIDD](uniform-noise-gidd-clean-supervision.html)。

論文的 Ψ-posterior 對 $\pi$ 不設限，因此同時適用於兩者（見 [Ψ-posterior](psi-superposition-posteriors-method.html)）。
