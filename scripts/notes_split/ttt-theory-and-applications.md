---
slug: ttt-theory-and-applications
title: TTT 的理論還原與應用擴散（視覺、3D/4D、長上下文 LLM）
cluster: Architectures & Efficiency
kind: topic
tags: [test-time-training, linear-attention, in-context-learning, vit, 3d-reconstruction, long-context]
sources: [ttt-sequence-modeling.md]
arxiv: ["2602.21204", "2503.11842", "2509.25741", "2512.01643", "2509.26645", "2604.07350", "2602.20160", "2512.13898", "2603.06642", "2511.07343"]
related: ["ttt-sequence-modeling-overview", "ttt-linear-mlp", "lact-large-chunk-ttt"]
---
# TTT 的理論還原與應用擴散（2025–2026）

屬於 [TTT 序列建模總覽](ttt-sequence-modeling-overview.html)。

## 理論
- **TTT 其實是 learned linear attention**（Junchen Liu 等，NVIDIA，「TTT with KV Binding Is Secretly Linear Attention」，2602.21204，2026-02-24）：分析顯示一大類 TTT（含多層 MLP、momentum）可改寫成 learned linear attention operator，挑戰「test-time 記憶」詮釋；附帶可平行化、可簡化架構的實用好處，並把 LaCT、ViTTT 改寫成線性注意力形式。（正向的「TTT 為母框架」視角見 [TTT-Linear / TTT-MLP](ttt-linear-mlp.html)。）
- **可證明的 ICL 增益**：2503.11842（TTT provably improves Transformers as in-context learners）、2509.25741（TTT 增強非線性函數的 in-context learning，single-index model 風險上界）。

## 視覺
- **ViT³**（Dongchen Han 等，清華 + 阿里，2512.01643，CVPR 2026 Oral）：系統化六條視覺 TTT 設計準則，純 TTT 架構達線性複雜度，高解析度下對 softmax attention 有 4.6× 加速、90.3% 記憶體節省。

## 3D/4D 重建
- TTT3R（2509.26645，training-free，用 alignment confidence 導出 closed-form learning rate，pose 估計 2× 改進、20 FPS/6GB）。
- Fast Spatial Memory / LaCET（2604.07350，用 Fisher-weighted elastic prior + EMA anchor 穩定 [LaCT](lact-large-chunk-ttt.html) 的 fast weight，解決 catastrophic forgetting）。
- tttLRM（2602.20160，用 LaCT block 做長上下文自回歸 3D 重建）。

## 長上下文 LLM
- qTTT（2512.13898，Meta/Harvard/OpenAI，只對 query 投影做梯度更新以對抗 score dilution，15+ 長上下文 benchmark 上常勝過 thinking tokens）。
- SR-TTT（2603.06642，用 surprisal 把高驚奇 token 路由到 exact-attention residual cache，解 TTT 的 needle 失敗）。注意：SR-TTT 僅在 15.8M 參數、2048 context 驗證，能否放大到十億級未知；引用其結論時須註明規模限制。

## 效率/更新規則
- TNT（2511.07343，改進 Titans 的 chunkwise 訓練，最高 17× 加速）。
