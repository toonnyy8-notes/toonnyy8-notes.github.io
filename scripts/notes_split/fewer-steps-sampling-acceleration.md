---
slug: fewer-steps-sampling-acceleration
title: 減少取樣步數與加速訓練的相關工作
cluster: Diffusion & Generative Models
kind: topic
tags: [sampling-acceleration, distillation, latent-diffusion, ddgan, edm]
arxiv: ["2202.00512", "2112.10752", "2112.07804", "2206.00364", "2106.00132", "2010.02502", "2301.11706"]
sources: ["paper.txt §2 (Related Work)", "arXiv API metadata"]
related: ["ddpm-quality-and-ddim-sampling", "diffusion-exposure-bias", "input-perturbation-ddpm-ip"]
---
# 減少取樣步數與加速訓練的相關工作

以下內容依本論文 Related Work 對各文的描述整理；本論文說這些方法與 DDPM-IP 正交，可能可以合併使用。

- **DDIM**（Song 等人，2010.02502）：見 [生成品質改進與 DDIM 取樣](ddpm-quality-and-ddim-sampling.html)。
- **Progressive Distillation**（Salimans & Ho，2202.00512）：把預測網路逐步蒸餾成取樣步數更少的新網路。論文指出缺點是需要訓練多個網路。
- **Latent Diffusion**（Rombach 等人，2112.10752）：先壓縮、再在潛在空間做 DDPM，分成壓縮與生成兩階段。
- **Denoising Diffusion GANs**（Xiao 等人，2112.07804）：加入對抗損失，使用較大的取樣步長與非高斯反向過程。
- **EDM**（Karras 等人，2206.00364）：使用 Heun 二階確定性取樣，品質高且取樣快。本論文在 Sec. 5.3 也引用其結論：推論軌跡的後段對效能影響最大。
- **Kong & Ping**（2106.00132，*On fast sampling of diffusion probabilistic models*）：引入連續的擴散步，形成快速取樣的統一框架。

另外，論文提到 Hoogeboom 等人（2022）的 order-agnostic DDPM、Chen 等人（2021，WaveGrad）以連續雜訊強度取代離散步 t 作為條件、Xu 等人（2022）的 Poisson flow。這三項在此未逐一查證 arXiv ID，僅作為論文敘述的轉述。

## 與本論文的關係

這些方法都針對「步數太多」造成的計算成本。本論文指出另一個與長取樣鏈有關的現象，即誤差累積（見 [擴散模型的 exposure bias](diffusion-exposure-bias.html)），並用訓練端的 [DDPM-IP](input-perturbation-ddpm-ip.html) 處理。
