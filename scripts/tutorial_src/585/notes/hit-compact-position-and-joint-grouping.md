---
slug: hit-compact-position-and-joint-grouping
title: HiT 的 compact 階段位置編碼與文字影像聯合分組
cluster: Diffusion & Generative Models
kind: topic
tags: ["rope", "positional-encoding", "text-to-image", "joint-grouping", "minit2i"]
arxiv: ["2210.09461", "2611.22962"]
sources: ["2611.22962 Appendix A.5、Appendix B.1", "2210.09461 abstract"]
related: ["hit-merging-baselines-fixed-rates", "tokc-position-for-merged-tokens"]
---
# HiT 的 compact 階段位置編碼與文字影像聯合分組

## 位置問題（Appendix B.1）

一個 compact token 可以代表多個、可能互不相連的影像區域，因此需要決定群組的位置資訊。所有變體都保留全解析度的影像 RoPE。比較的固定規則：

- hard centroid：以硬成員座標平均值評估一般二維 RoPE。
- soft centroid：以正規化 pooling 權重加權的座標。
- hard / soft table average：直接平均原始 cos、sin 表。

四者中 hard centroid 的 FID 最低，12 block 對照組由 12.27 降到 11.35。第二層分組時，硬支撐集是組成群組的聯集，軟權重跨 pooling 層級組合。

Learned RoPE 以各階段係數混合硬、軟質心與二階矩，並以 exp(−γω²v_g/2) 衰減頻率，另有控制 log 群組大小偏置的係數；加上 per-group MLP（66,700 參數）使 12 block 的 FID 由 11.48 變為 11.18，18 block 主模型由 6.61 變為 6.25。log-size bias 依 ToMe 的 proportional attention（2210.09461）。論文結論：各種位置變體的表現相近，為求簡單，預設設定省略 compact 階段的 RoPE。

與 ToMe 位置處理的比較見 [ToMe 與 ToMeSD 在 HiT 中的位置：預先設定的縮減率](hit-merging-baselines-fixed-rates.html)，與其他合併 token 位置方法的整理見 [合併 token 的位置處理](tokc-position-for-merged-tokens.html)。

## 文字影像聯合分組（Appendix A.5）

文字到影像沿用 MiniT2I 的設定：凍結的 FLAN-T5-Large 提供 caption 特徵，經投影與兩個純文字 preamble block。外層 encoder 與 decoder 保留各自的影像與文字投影，並做聯合注意力；一個 time token 在第一個聯合 block 前加入，貫穿整個階層，但繞過分組與解組。

分組對象是影像特徵與有效 caption 特徵的合集，樣本 b 的合格 token 數 N_b = N_img + Σ_j m_bj（Eq. 13），padding 不進入合併樹、軟 pooling 與壓縮預算。同一個 affinity router 與軟聚合形成 compact token，一個群組可含影像特徵、caption 特徵或兩者；compact 階層內使用共享的 Transformer block。解組把處理後的特徵分配回原本的影像與 caption 位置，再加 skip。Caption dropout 的樣本只有影像特徵進入分組。

論文 Sec. 4.4 報告 HiT 在沒有詞—區域標註下，把 caption 詞與對應的影像 patch 分在一起（視覺化在 Appendix D）。
