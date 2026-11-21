---
slug: tokc-training-free-diffusion-merging
title: 免訓練的擴散 token 合併：ToMeSD、ToDo、AT-EDM、Importance-Based Token Merging
cluster: Diffusion & Generative Models
kind: topic
tags: [ToMeSD, ToDo, AT-EDM, unmerge, CFG, training-free]
arxiv: ["2303.17604", "2402.13573", "2405.05252", "2411.16720", "2210.09461"]
sources: ["2303.17604 全文 §3.1, §4.1, Tab. 3-4", "2402.13573 全文 §2.2, 摘要", "2405.05252 arXiv 摘要", "2411.16720 arXiv 摘要", "research/lineage.md §3.6-§3.9"]
related: ["tokc-vit-pruning-and-merging", "tokc-learned-routing-dynamic-dit", "tokc-lineage-hub"]
---
# 免訓練的擴散 token 合併

擴散模型需要為每個 token 預測輸出，因此合併後必須還原；同一張圖在 classifier-free guidance（CFG）下會跑條件與無條件兩個分支，並在多個時間步重複運算。這一期的方法皆不需訓練。ToMe 的基礎見 [ViT 的 token 剪枝與合併](tokc-vit-pruning-and-merging.html)。

## ToMeSD（arXiv 2303.17604）

- 在 self-attention、cross-attention、MLP 之前合併，輸出後 unmerge：若合併 token 為 x* = (x₁ + x₂)/2，則 x′₁ = x′₂ = x*，即複製回所有來源位置。
- 以比例 r% 而非固定個數合併。
- src/dst 劃分：ToMe 的交替劃分使 dst 形成規則欄位，等同沿列減半解析度。ToMeSD 改為每個 2×2 區域隨機挑一個 dst，且在整個 batch 固定隨機性，使 CFG 的條件與無條件樣本以相同方式配置 dst。
- 設計實驗：只對 self-attention 合併最划算；早期步驟多合併、後期少合併有小幅改善。
- 數字：摘要稱最多減少 60% token、最高 2× 速度、記憶體最多降低 5.6×。SD v1.5、512²：基線 FID 33.12（3.09 s/im），r = 50% 時 FID 33.02。以 0 取代的剪枝使影像迅速崩壞。

## ToDo（arXiv 2402.13573）

ToMe 的相似度矩陣為 O(n²)，且 merge–unmerge 在高合併率下損失細節。ToDo 假設空間相鄰 token 相似，以最近鄰下採樣 D(·) 取代配對，且只對 key、value 下採樣，保留全部 query：softmax(Q·D(K)ᵀ/√d_k)·D(V)。輸出仍為全長，因此不需 unmerge。摘要報告一般尺寸加速至多 2×，2048×2048 等高解析度至多 4.5× 以上。只省注意力，MLP 仍為全長。

## AT-EDM 與 Importance-Based Token Merging

- AT-EDM（arXiv 2405.05252，僅讀摘要）：以注意力圖的 Generalized Weighted Page Rank（G-WPR）排序冗餘 token 後剪枝，以相似度為基礎的方法為卷積層補回 token；Denoising-Steps-Aware Pruning 依去噪步調整剪枝預算。
- Importance-Based Token Merging（arXiv 2411.16720，僅讀摘要）：合併時優先保留高資訊 token，重要性分數取自現成訊號，例如 CFG。

## 共同限制

分組由外觀相似度或重要性決定，不依位置；比例需手調；大幅壓縮時品質下降。學得比例與路由的後續做法見 [學得路由與動態 DiT](tokc-learned-routing-dynamic-dit.html)。
