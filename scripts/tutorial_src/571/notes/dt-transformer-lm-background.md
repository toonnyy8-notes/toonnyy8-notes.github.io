---
slug: dt-transformer-lm-background
title: Transformer 與語言模型背景（Decision Transformer 的架構來源）
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [transformer, GPT, language-model, attention]
arxiv: ["2106.01345", "1706.03762", "2005.14165", "2102.12092", "1607.06450"]
sources: ["2106.01345 §1, §2.2, §3", "arXiv abstract 1706.03762", "arXiv abstract 2005.14165", "arXiv abstract 2102.12092", "arXiv abstract 1607.06450"]
related: ["dt-lineage-overview", "dt-method-and-results"]
---
# Transformer 與語言模型背景

Decision Transformer（arXiv 2106.01345）在 §1 開頭引用 Transformer [1]、GPT-3 [2] 與 DALL-E [3]，作為「Transformer 能在大規模下建模高維分布」的證據，並以此提出問題：這類模型能否取代傳統 RL 演算法。

## 論文引用的前身工作

| 工作 | arXiv | 作者（年） | 在論文中的角色 |
| --- | --- | --- | --- |
| Attention Is All You Need | 1706.03762 | Vaswani 等（2017） | §2.2 的 self-attention 定義；論文以 query 與 key 的點積說明 Transformer 能形成「狀態與報酬的關聯」 |
| Language Models are Few-Shot Learners | 2005.14165 | Brown 等（2020） | §1 引為語言模型 zero-shot 泛化的例子；§6.2 引為「序列建模可擴展」的依據 |
| Zero-Shot Text-to-Image Generation | 2102.12092 | Ramesh 等（2021） | §1 引為分布外影像生成的例子 |
| Layer Normalization | 1607.06450 | Ba、Kiros、Hinton（2016） | §3 中 token embedding 之後使用 |

GPT [9]（Radford 等，2018，*Improving language understanding by generative pre-training*）沒有 arXiv 版本，本筆記不另行查證其內容；論文只說明它在 Transformer 上加入 causal self-attention mask，使模型能自迴歸生成。

## 與 Decision Transformer 的關係

論文 §2.2 說明：GPT 把 self-attention 的加總與 softmax 限制在前面的 token（j ∈ [1, i]）。Decision Transformer 直接使用這個架構，只改變輸入：把 returns-to-go、狀態與動作三種模態各自線性嵌入，再加上每個時間步一個的位置嵌入。細節見 [Decision Transformer 的方法與實驗結果](dt-method-and-results.html)。

來源說明：表中各項的內容取自該篇論文 abstract 與本論文對它的描述；整體脈絡見 [研究脈絡總覽](dt-lineage-overview.html)。
