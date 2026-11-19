---
slug: in-place-ttt
title: In-Place TTT：把現成 LLM 的 MLP 投影矩陣當 fast weight
cluster: Architectures & Efficiency
kind: topic
tags: [in-place-ttt, test-time-training, drop-in, long-context, continual-learning]
sources: [ttt-sequence-modeling.md]
arxiv: ["2604.06169"]
related: ["ttt-sequence-modeling-overview", "ttt-linear-mlp"]
---
# In-Place TTT：把現成 LLM 的 MLP 投影矩陣當 fast weight

arXiv:2604.06169，ICLR 2026 Oral。屬於 [TTT 序列建模總覽](ttt-sequence-modeling-overview.html)。

**作者與狀態。** Guhao Feng、Shengjie Luo、Kai Hua、Ge Zhang、Di He、Wenhao Huang、Tianle Cai（ByteDance Seed + 北京大學），2026-04-07，**已被 ICLR 2026 接收為 Oral**（官方 repo github.com/ByteDance-Seed/In-Place-TTT 公告），非僅未評審 preprint。

## 設計理念（解決三個障礙）
（i）架構不相容：既有 TTT 多需專用層、得從頭預訓練 → In-Place 直接「就地」把每個 MLP block 的最後投影矩陣當 fast weight，不改架構、保留預訓練權重，可 drop-in；（ii）計算低效：用 chunk-wise 更新取代逐 token，且因 in-place 設計與 attention 互補，不需要獨立 TTT 層所需的小 chunk；（iii）目標錯配：用一個理論上對齊 Next-Token-Prediction 的目標取代通用重建損失。

## 數學直覺
沿用 TTT 的 update/apply 兩步：`W_i ← W_{i-1} − η∇_W ℓ(f_{W_{i-1}}(k_i), v_i)`、`o_i = f_{W_i}(q_i)`；創新在「ℓ 改成與 NTP 對齊的目標」與「fast weight 就是現成 MLP 的投影矩陣」。（原版更新規則見 [TTT-Linear / TTT-MLP](ttt-linear-mlp.html)。）

## 實驗結果
作為 in-place 增強，abstract 稱「enables a 4B-parameter model to achieve superior performance on tasks with contexts up to 128k」（Qwen3-4B-Base，經約 20B-token 32k + 約 15B-token 128k 持續訓練）；從頭預訓練（最長 32k 語料）時穩定勝過其他 TTT 類方法。消融顯示 state size、chunk size、fast-weight 目標各自關鍵。定位：邁向 LLM 持續學習的一步。限制：主要驗證在 ≤128k。

## 使用建議
若你想在現成 LLM 上加 TTT 而不重訓：In-Place TTT 是目前最低成本路徑。**門檻**：需先驗證它在你的 base model（如 Qwen3）與目標 context 上的增益是否複現。
