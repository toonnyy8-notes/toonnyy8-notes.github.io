---
slug: lottery-ticket-redundancy-scaffold
title: 樂透票假說與「冗餘是否為必要的訓練鷹架」
cluster: Interpretability & Theory
kind: topic
tags: [lottery-ticket, overparameterization, initialization, weight-sharing]
sources: [weight-sharing-lora.md]
arxiv: ["1803.03635", "2510.25741"]
related: ["weight-sharing-recursive-overview", "relaxed-recursive-transformer", "recursive-transformer-followups"]
---
# 樂透票假說與「冗餘是否為必要的訓練鷹架」

屬於 [層間權重共享總覽](weight-sharing-recursive-overview.html)。

- Lottery Ticket Hypothesis（Frankle & Carbin, arXiv:1803.03635, ICLR 2019）：稠密網路含可獨立訓到全精度的稀疏子網（winning ticket）， [alphaXiv](https://www.alphaxiv.org/overview/1803.03635v5) 但從頭訓練該稀疏架構通常失敗——關鍵是初始化，過參數化提供「找到好初始化子網」的搜尋空間。 [MIT DSpace](https://dspace.mit.edu/bitstream/handle/1721.1/129953/1803.03635.pdf?sequence=2&isAllowed=y) stabilized/late-rewinding 版本顯示需先讓網路建立穩定結構（約 0.1–7% 訓練步）再剪枝。

## 對權重共享的推論
這強烈支持 Bae et al.「先訓練大模型再轉換」優於「從頭訓遞迴」的策略選擇——[RRT](relaxed-recursive-transformer.html) 的 pretrained-init vs random-init 差約 14.1 個百分點，恰是「冗餘作為訓練期鷹架」的實證。Ouro 從頭訓 LoopLM 成功，說明從頭訓可行但需 7.7T token 的巨量規模與精心穩定化（論文亦提到高 recursion 時訓練不穩，見 [後續工作](recursive-transformer-followups.html)）。 [Hugging Face](https://huggingface.co/papers/2510.25741)

**綜合證據傾向：訓練期冗餘/過參數化是有用的鷹架，轉換式路線在中小規模的資料效率明顯更高。**（此為基於現有證據的推論，非單一論文直接定論。）

## 矛盾並陳
關於「過參數化為何有效」，樂透票假說（好初始化子網）與 escape-dimension 觀點並存且互相質疑，本筆記未選邊。
