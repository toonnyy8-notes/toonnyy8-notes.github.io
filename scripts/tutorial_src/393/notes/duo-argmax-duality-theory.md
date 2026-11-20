---
slug: duo-argmax-duality-theory
title: Duo 的 arg max 對偶：Diffusion Transformation 與兩個定理
cluster: Diffusion Language Models
kind: topic
tags: [duo, duality, argmax, pushforward, elbo]
arxiv: ["2506.10892", "1904.09751"]
sources: ["arXiv:2506.10892 §3、附錄 A.1–A.6（以正文敘述為準）"]
related: ["duo-lineage-hub", "duo-gaussian-diffusion-foundations", "duo-discrete-diffusion-masked-and-uniform", "duo-curriculum-softmax-relaxation"]
---
# Duo 的 arg max 對偶

屬於 [Duo 研究脈絡總覽](duo-lineage-hub.html)。本篇整理論文本身的理論，前置知識見 [Gaussian diffusion 的基礎](duo-gaussian-diffusion-foundations.html) 與 [離散擴散背景](duo-discrete-diffusion-masked-and-uniform.html)。

## 設定
x ∈ V 為 one-hot 向量。Gaussian 前向過程 w_t ∼ N(α̃_t x, (1 − α̃_t²) I_K)。令 z_t = arg max(w_t)。

## 邊際對應（§3.1）
z_t 的分佈為 Cat(·; T(α̃_t) x + (1 − T(α̃_t))/K)，其中 Diffusion Transformation
T(α̃_t) = (K/(K−1)) [∫ φ(z − α̃_t/√(1 − α̃_t²)) Φ^{K−1}(z) dz − 1/K]。
論文證明 P_t 的時間演化滿足以 T(α̃_t) 為擴散參數的 USDM 線性 ODE（式 12），因此 [arg max]★ 把 Gaussian 擴散的邊際推送為 USDM 的邊際。論文同時註明：離散化後的 Gaussian 軌跡本身不一定是離散擴散過程，只有邊際對應成立；兩者是互不轉移的獨立 Markov 過程。

## 定理 3.1（§3.2）
給定 Gaussian 反向過程 p̄_θ，存在離散反向核 p_θ(s|t) 使 p_θ_t = [arg max]★ p̄_θ_t 對所有 t 成立，且兩者共用同一個以 Gaussian 空間參數 θ 定義的去噪 Transformer。

## 定理 3.2（§3.2）
E_{x∼q_data} log p_θ_data(x) ≥ E_{x∼q_data} log p̄_θ_data(x)：離散取樣器誘導的邊際似然不低於對應的 Gaussian 取樣器。論文據此選擇讓去噪模型作用在離散 latent 上，並以離散 NELBO 作為訓練與評估目標。

## 其他設計
- Rao-Blackwellized NELBO（式 17）避免實體化 one-hot 向量，降低記憶體與變異。論文 Table 3 的消融中，改進訓練目標約貢獻 1.7 PPL。
- Greedy-Tail 取樣器：最後一步以 arg max 取代取樣，作用類似 AR 的 nucleus sampling（Holtzman 等人 2020，arXiv:1904.09751）。此引用由論文正文給出。
