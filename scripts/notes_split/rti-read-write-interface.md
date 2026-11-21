---
slug: rti-read-write-interface
title: Read / Write 介面：核心區段、區域 token 與位置
cluster: Diffusion & Generative Models
kind: topic
tags: [rti, read-write, prelude-core-coda, rope, latent-interface]
arxiv: ["2608.29281", "2603.12245", "2107.14795", "2502.05171", "2110.13711", "1810.00825", "2308.00951"]
sources: ["2608.29281 §2, §3.2, §3.4", "2603.12245 abstract", "2502.05171 abstract", "2110.13711 abstract", "2107.14795 abstract", "1810.00825 abstract", "2308.00951 abstract"]
related: ["rti-lineage-hub", "rti-hilbert-order-region-partition", "tokc-hourglass-and-latent-interfaces", "tokc-position-for-merged-tokens"]
---
# Read / Write 介面

屬於[RTI 研究脈絡總覽](rti-lineage-hub.html)。切割方式見 [Hilbert 順序與區域切割](rti-hilbert-order-region-partition.html)。

## 只壓縮核心（§3.2）

沿用 recurrent-depth Transformer 的 prelude / core / coda 結構（Geiping 等，arXiv 2502.05171），RTI 指定連續的中段 block [s0, s1] 為核心。MiniT2I-B/16 為 17 個 block 中的 [3, 13]，L/16 為 23 個 block 中的 [3, 19]。Read 在 s0 前把 N 個 patch token 匯聚為 R 個區域 token，核心所有 block 在這段短序列上運算，Write 在 s1 後寫回 N 個 patch。文字流始終全長。

首尾保持全解析度的理由相反：prelude 仍在把原始 patch 轉為特徵，過早匯聚會在模型理解前丟失像素；coda 要輸出逐像素結果，共享的區域更新會使其模糊。

## Read（Eq. 3）

z_c = Σ_{j∈c} α_j h_j + e_size[⌊log₂|c|⌋]，α 為區域內的 softmax(wᵀh_j)，w 是單一可學向量。Read 因此是限於區域內的單 query 注意力，並加上區域大小的嵌入，使核心知道一個 token 代表一個還是十六個 patch。位置取區域內各 patch 的 RoPE 相量平均：單位相量的平均在波長長於區域範圍時接近單位圓、短於範圍時趨近零，等於依區域大小做低通。長度 L 的連續區間、頻率 ω 的平均相量長度為 |sin(Lω/2) / (L·sin(ω/2))|（Dirichlet 核，lineage.md W4 數值驗證）（一般的合併 token 位置處理見[合併 token 的位置](tokc-position-for-merged-tokens.html)）。

## Write（Eq. 4）

h^{(s1+1)}_j = h^{(s0)}_j + g([h^{(s0)}_j ∥ z′_{c(j)} − z_{c(j)}])。不把區域的變化量 z′ − z 原樣加到每個 patch，而讓每個 patch 以自己的前池化特徵與該變化量，經小型可學映射 g 產生專屬更新。這同時是繞過核心的 skip：每個 patch 保留進核心前的特徵，核心的工作以修正量加入。

Read 與 Write 只在區域內作用，成本 O(N)，與 R 無關。

## 與 latent 介面的關係

RTI 稱自己的 Read/Write 屬於交叉注意力瓶頸一族：Perceiver-IO（arXiv 2107.14795）、Set Transformer 的 inducing points（arXiv 1810.00825）、Soft MoE（arXiv 2308.00951），以及最接近的 ELIT（arXiv 2603.12245）。差別是 RTI 把 latent 錨定在空間區域上，讀取限於區域內（O(N)，而全域 query 為 O(NR)），且每個 token 帶有凍結的旋轉注意力可讀取的位置。沙漏形狀來自階層 Transformer（Nawrot 等，arXiv 2110.13711），RTI 以內容自適應的切割取代其固定步幅池化。見 [Hourglass 與 latent 介面](tokc-hourglass-and-latent-interfaces.html)。
