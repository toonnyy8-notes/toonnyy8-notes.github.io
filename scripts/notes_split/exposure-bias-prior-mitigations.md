---
slug: exposure-bias-prior-mitigations
title: 曝光偏差的既有緩解方法：DDPM-IP、shifted time steps 與誤差傳播分析
cluster: Diffusion & Generative Models
kind: topic
tags: [exposure-bias, input-perturbation, time-shift, error-propagation, pfgm]
arxiv: ["2301.11706", "2305.15583", "2308.05021", "2302.04265", "2308.15321"]
sources: [2308.15321]
related: ["exposure-bias-elucidation-overview", "diffusion-exposure-bias", "input-perturbation-ddpm-ip", "exposure-bias-seq2seq-origin", "epsilon-scaling-method"]
---
# 曝光偏差的既有緩解方法

回到 [總覽](exposure-bias-elucidation-overview.html)。以下描述皆取自本論文（2308.15321）對這些工作的敘述。

## 文獻
- Ning、Sangineto、Porrello 等人 2023，*Input Perturbation Reduces Exposure Bias in Diffusion Models*（arXiv:2301.11706）：本論文稱其首次指出擴散模型的曝光偏差，並引入輸入擾動來模擬取樣時的預測誤差，其依據是 Lipschitz 連續性；在第 3.4 節歸類為資料增強：以高斯項擴增每個訓練樣本 x_t，FID 有大幅改善。該方法需要重新訓練。細節見 [DDPM-IP](input-perturbation-ddpm-ip.html)。
- Li、Qu、Yao 等人 2023a，*Alleviating Exposure Bias in Diffusion Models through Sampling with Shifted Time Steps*（arXiv:2305.15583）：本論文稱其無需重新訓練，方式是在反向生成過程中調整時間步的選擇。
- Li & van der Schaar 2023，*On Error Propagation of Diffusion Models*（arXiv:2308.05021）：本論文稱其估計累積誤差的上界並加以最佳化；本論文 4.2 節設計 scaling schedule 時，也引用其累積誤差分析。
- Xu 等人 2023b，*PFGM++*（arXiv:2302.04265）：本論文引述 Xu 等人（2022）的主張，認為 Poisson Flow 對預測誤差較不敏感，並在附錄 A.7 驗證 Epsilon Scaling 能進一步改善 PFGM++。

## 本論文的定位
本論文歸納的解法有三類：學得更準確的 ε 或 score 網路（如 EDM）、資料增強（如 Ning 等人 2023）、以及不需重新訓練的取樣階段方法。本論文提出的 [Epsilon Scaling](epsilon-scaling-method.html) 屬第三類，且不增加計算量。一般定義見 [擴散模型的曝光偏差](diffusion-exposure-bias.html)，術語來源見 [曝光偏差在序列生成中的來源](exposure-bias-seq2seq-origin.html)。
