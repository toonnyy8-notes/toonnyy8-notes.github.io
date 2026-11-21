---
slug: tokc-dynamic-chunking-bytes-to-pixels
title: 動態切塊：BLT、H-Net、DC-DiT 到影像的系譜
cluster: Diffusion & Generative Models
kind: topic
tags: [dynamic-chunking, BLT, H-Net, DC-DiT, ratio-loss]
arxiv: ["2412.09871", "2507.07955", "2603.06351", "2608.29281", "2611.22962"]
sources: ["2507.07955 全文（擷取）", "2603.06351 全文（擷取）", "2412.09871 arXiv 摘要", "research/lineage.md §3.22-§3.24, §5.1, §5.4", "RTI 與 HiT 的描述取自 lineage.md，未重讀兩篇全文"]
related: ["tokc-hourglass-and-latent-interfaces", "tokc-content-adaptive-tokenization", "tokc-lineage-hub", "rti-lineage-hub", "hit-lineage-hub"]
---
# 動態切塊：從 byte 到 pixel

固定步幅的縮短（見 [沙漏結構與 latent 介面](tokc-hourglass-and-latent-interfaces.html)）對平坦與複雜區段分配相同長度。動態切塊讓區段長度隨內容變化。

## BLT（arXiv 2412.09871，僅讀摘要）

依下一個 byte 的熵把 byte 切成動態長度 patch，資料可預測時用長 patch。摘要稱首次在 FLOP 控制下把 byte 層級模型擴展到 8B 參數、4T 訓練 byte。

## H-Net（arXiv 2507.07955）

結構為 encoder、chunking、main network、dechunking、decoder，可遞迴多階段。

- 路由：q_t = W_q x̂_t、k_t = W_k x̂_t，邊界機率 p_t = ½(1 − cos(q_t, k_{t−1}))，b_t = 1{p_t ≥ 0.5}（p₁ = 1）。downsampler 直接選取邊界位置的向量。
- 平滑：z̄_t = P_t ẑ_t + (1 − P_t) z̄_{t−1}（EMA），使低信心邊界被平滑；upsampler 重複每個壓縮向量直到下一邊界。
- Ratio loss：L_ratio = N/(N−1)·((N−1)FG + (1−F)(1−G))，F 為實際選取比例（不可微），G 為平均邊界機率（可微），F = G = 1/N 時達最小值 1。
- 摘要：一階段 byte 層級 H-Net 在計算與資料對齊下優於 BPE Transformer，兩階段可匹敵兩倍大的 token 模型，DNA 資料效率近 4×。

## DC-DiT（arXiv 2603.06351）

把動態切塊帶到 2D latent DiT。在 flattened latent 格點上，isotropic encoder 混合局部資訊；router 以 3×3 卷積預測器由鄰域估計 ẑ，殘差 r = z − ẑ 表示局部不可預測性，p_i = σ(f_score(r)_i)；以 p_i > 0.5 硬遮罩選取邊界 token（STE），加上依原始 2D 座標的正弦位置嵌入後進入 DiT 骨幹，de-chunking 還原格點並加上由邊界機率門控的 encoder 殘差。彈性推論時依 p_i 排序丟棄最低的 ρ|B| 個邊界（tail dropping）；Lite-CFG 讓無條件分支丟更多。摘要：ImageNet 類別條件上推論 FLOPs 至多降低 36.8%、FID 至多改善 37.8%。可由預訓練 DiT upcycle。

## 與 RTI、HiT 的關係

HiT 明言受 H-Net 啟發（q/k 投影、ratio loss、巢狀 encoder/main/decoder），HiT 附錄 A.2 證明其 compression loss 與 H-Net 的 ratio loss 代數等價。RTI 以 Hilbert 序把 2D 化為 1D，依相鄰特徵落差切塊，可視為把 H-Net 式切塊搬到 2D 的無參數作法，並以精確的 R 取代門檻 0.5。機制見 [HiT 研究脈絡總覽](hit-lineage-hub.html) 與 [RTI 研究脈絡總覽](rti-lineage-hub.html)。DC-DiT 在已吸收冗餘的 VAE latent 上運作，且只控制平均比例而非精確預算（RTI §2 的批評）。
