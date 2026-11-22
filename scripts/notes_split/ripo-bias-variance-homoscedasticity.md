---
slug: ripo-bias-variance-homoscedasticity
title: 幾何等距與同質變異：RIC 的偏差—變異論證
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [importance-sampling, variance, bias, homoscedasticity, clipping]
arxiv: ["2607.10169"]
sources: ["arXiv:2607.10169 §4.2, Eq. 12–14"]
related: ["ripo-lineage-hub", "importance-sampling-corrections-llm-rl"]
---
# 幾何等距與同質變異

屬於 [RIPO 的研究脈絡總覽](ripo-lineage-hub.html)。重要性取樣在 LLM RL 中的各種修正見 [LLM RL 中的 importance sampling 修正](importance-sampling-corrections-llm-rl.html)；本篇只整理本文（arXiv 2607.10169）§4.2 的論證。

**設定。** off-policy 目標以 E_{x∼π_old}[r(x)A(x)] 估計（Eq. 12），無偏但變異可能很大。本文引用 Tokdar & Kass（2010）與 Murphy（2012），主張變異主要來自二階矩 E[r²A²]，平方均值項通常可忽略。

**變異貢獻。** 樣本 x 的貢獻為 v(x) = π_old(x)·r(x)² = π_θ(x)²/π_old(x)。當 π_old(x) 很小時 v(x) 可任意大，變異爆炸來自 π_old 的長尾。

**PPO-Clip。** 截斷 r ≤ 1+ε 後，clip 邊界附近樣本的貢獻 π_old(x′)(1+ε)² 隨 π_old → 0 趨近 0，降低變異，但被截斷的樣本對目標沒有貢獻，本文稱此引入嚴重偏差。

**RIC。** 上界改為 r ≤ 1 + √(δ/π_old)，邊界附近樣本的貢獻為 π_old(1 + √(δ/π_old))² ≈ O(δ)（Eq. 14），不再依賴 π_old，即同質變異（homoscedasticity）。本文據此認為：變異嚴格小於標準 importance sampling，偏差小於 PPO-Clip，因為更多罕見樣本被納入。

**幾何與統計的對應。** 本文的說法是，等距更新使每個樣本對信任區域的二階貢獻相等，進而使 importance sampling 的變異同質；PPO-Clip 的歐氏假設則導致異質變異與低機率區域的保守更新。

這是本文的分析論證；O(δ) 為近似量級，文中未給出誤差界。
