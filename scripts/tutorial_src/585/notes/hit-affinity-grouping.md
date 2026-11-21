---
slug: hit-affinity-grouping
title: GroupViT 到 HiT：親和度、平均連結與軟分組
cluster: Diffusion & Generative Models
kind: topic
tags: ["groupvit", "affinity", "average-linkage", "soft-pooling", "ungrouping"]
arxiv: ["2202.11094", "2611.22962"]
sources: ["2611.22962 Sec. 3.2、Appendix A.1、Sec. 4.4", "2202.11094 abstract"]
related: ["hit-compression-budget"]
---
# GroupViT 到 HiT：親和度、平均連結與軟分組

## GroupViT

GroupViT（Xu 等，2202.11094）的摘要指出：它把分組機制放回深度網路，學習把影像區域分成逐漸變大、形狀任意的 segment；與文字編碼器以對比損失聯合訓練，只用文字監督，不用像素級標註，就能零樣本轉移到語義分割。HiT 論文把它列為「以影像文字監督學習階層式語義分組」的先例。

## HiT 的分組步驟（Sec. 3.2）

1. **親和度**：token 特徵 h_i 經兩個投影並正規化為單位長度，得 q_i 與 k_i；對稱親和度 S_ij = (q_iᵀk_j + q_jᵀk_i) / 2（Eq. 2）。
2. **硬分組**：從個別 token 出發，每次合併平均親和度最高的兩個群組（平均連結法，average linkage），保留親和度至少為門檻 τ 的合併，得到 K 個群組。群組平均親和度為成員兩兩親和度的平均（Eq. 8）。同一群組的 token 在影像上可以相距很遠。
3. **軟權重**：每個 token 對每個硬群組計算平均親和度，經溫度 T_r 的 softmax 得 A ∈ R^{N×K}，每列和為 1（Eq. 9）。對 A 的每一欄在所有輸入 token 上正規化，得到 pooling 矩陣 P；群組特徵是所有輸入特徵的加權平均。
4. **解組與 skip**：Z = PᵀH；H̃ = A·M(Z) + H·W_ℓᵀ（Eq. 3），其中 M 是 compact 網路，W_ℓ 初始化為單位矩陣。

硬分組本身不可微，反向傳播時視為常數；梯度經軟 pooling 與解組的權重傳回親和度投影。附錄指出解組還原 token 位置，但不會反轉有損的 pooling。

## 學到的結構（Sec. 4.4）

論文在 ImageNet、t = 0.75 的雜訊輸入上觀察到：部分群組對應連貫的物件區域，部分群組收集重複的背景結構（如樹葉），且空間上分離的 patch（如車身標記）可屬同一群組。論文稱這種結構在沒有分割標註的生成訓練中自行出現。

預算如何決定群組數，見 [HiT 的壓縮預算：批次平均比例與可變群組數](hit-compression-budget.html)。
