---
slug: diffusion-exposure-bias
title: 擴散模型的 exposure bias
cluster: Diffusion & Generative Models
kind: topic
tags: [exposure-bias, ddpm, error-accumulation, train-inference-mismatch]
arxiv: ["2301.11706", "2102.09672", "2105.05233", "1511.06732", "1910.00292", "1612.00563", "1506.03099"]
sources: ["paper.txt §1, §4, §5.2, §5.3, Tab. 1", "arXiv API metadata"]
related: ["ddpm-training-and-sampling-recap", "teacher-forcing-and-scheduled-sampling", "input-perturbation-ddpm-ip", "vicinal-risk-and-smoothness-regularization"]
---
# 擴散模型的 exposure bias

本篇是跨論文共用的背景筆記：說明 exposure bias 在擴散模型中指什麼、如何被量測。由 Ning 等人（2301.11706）提出此對應。

## 定義

[DDPM 訓練與取樣](ddpm-training-and-sampling-recap.html)的 Algorithm 1 第 4 行輸入 ε_θ(x_t, t)，其中 x_t 由真實 x_0 與雜訊算出；Algorithm 2 第 4 行輸入 ε_θ(x̂_t, t)，其中 x̂_t 來自前一個取樣步的輸出。訓練與取樣的輸入因此不同。論文將此類比序列生成中的問題（見 [teacher forcing 與 scheduled sampling](teacher-forcing-and-scheduled-sampling.html)）：取樣時的誤差會被後續步驟當作輸入，沿取樣鏈累積。

## 量測方法（論文 Sec. 4）

從一張真實影像 x_0 出發，用 Eq. 4 算 x_t，再以 Algorithm 2 反向取樣，與原始路徑比較。論文在 ImageNet 32×32 上的估計值（Tab. 1，反向步數 100 / 300 / 500 / 700 / 1,000）：

- ADM：0.983、1.808、2.587、3.105、3.544
- ADM-IP：0.972、1.594、2.198、2.539、2.742

兩者都隨步數增加，但 ADM-IP 增加較少。

## 預測誤差的形狀（Sec. 5.3）

固定 t，將預測的 x̂_0 與真實 x_0 相比，得到誤差 e_t。論文實驗確認 e_t 近似 N(0, ν_t² I)；Fig. 2 的說明寫兩條曲線（藍、橘）的平均分別為 0.20 與 0.19；依圖例順序（ImageNet 32×32 在前、CIFAR10 在後）推斷，ImageNet 32×32 約 0.20、CIFAR10 約 0.19，論文正文沒有明說對應關係。

## 與「步數越少越好」的關係

標準 DDPM 以 T = 1,000 步訓練，往往在 100 至 300 步取樣時 FID 最佳（Tab. 3）。論文的解釋：長取樣鏈使反向過程較接近高斯，卻累積較多誤差。這個解釋是作者的推論，不是獨立驗證的結論。

## 處理方式

論文的作法見 [DDPM-IP](input-perturbation-ddpm-ip.html)，其動機與正則化的關係見 [Vicinal risk 與平滑性正則化](vicinal-risk-and-smoothness-regularization.html)。
