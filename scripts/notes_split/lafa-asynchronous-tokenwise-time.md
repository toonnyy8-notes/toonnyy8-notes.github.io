---
slug: lafa-asynchronous-tokenwise-time
title: block 內各位置的非同步時間（tokenwise time）
cluster: Diffusion Language Models
kind: topic
tags: [tokenwise-time, asynchronous-flow, time-warping, left-to-right-bias]
arxiv: ["2610.28559", "2211.15089", "2606.19662", "2407.01392"]
sources: ["arXiv abstracts", "paper.txt §2.1, §2.2"]
related: ["lafa-lineage-overview", "lafa-embedding-flow-denoiser", "lafa-self-speculative-decoding", "diffusion-forcing"]
---
# block 內各位置的非同步時間

屬於 [Lafa 研究脈絡總覽](lafa-lineage-overview.html)。

## Lafa 的做法
一般 flow 讓所有位置共用同一個時間 t。Lafa 對 block 內第 i 個位置使用各自的時間函數 τ^i(t)，滿足 τ^i(0)=0、τ^i(1)=1，且對 i<j 有 τ^i(t) > τ^j(t)，使左側 token 較早完成去噪。動機是讓 block 內的生成順序接近 AR 的由左到右方向（paper.txt §2.1）。

## 前人
- 論文將 tokenwise time 的先例引為 [26–30]，其中包含 Learning When to Denoise（Qian and Cheng，2606.19662）；依其摘要，該文處理的是影像的多表示擴散，學習各表示的去噪時機。Token Time Continuous Diffusion（Bansal and Sanghavi）與 AR-Diffusion（Wu et al., 2023）也在引用清單中，本筆記未驗證其 arXiv ID。
- Diffusion Forcing 對序列的每個 token 使用獨立噪聲，是連續序列生成中的相近想法，說明見 [Diffusion Forcing](diffusion-forcing.html)。
- CDCD 的 time warping（2211.15089）：Lafa 的 τ^i 以學到的分布的反向 CDF 建構，使離散化的時間步多配置在困難的噪聲水準（paper.txt §2.2，Figure 5）。

## 實作要點
- 時間 τ 由資料學得而非固定。
- 乾淨前綴的時間設為 1，block 內各位置各自接收時間條件（adaLN）。
