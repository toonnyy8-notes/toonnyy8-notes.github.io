---
slug: idql-evaluation-and-online-finetuning
title: IDQL 的實驗：D4RL 比較、單一超參數設定與線上微調
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [d4rl, antmaze, locomotion, online-finetuning, hyperparameter-sensitivity]
arxiv: ["2304.10573", "2004.07219", "2006.04779", "2110.06169", "2208.06193", "2106.06860", "2106.01345", "2209.14548", "2301.02328", "2302.02948", "2303.05479"]
sources: ["arXiv:2304.10573 §6, 表 1–3", "arXiv 摘要 2004.07219、2302.02948、2303.05479"]
related: ["offline-rl-benchmarks-and-baselines", "idql-implicit-policy-and-critic-choice", "diffusion-behavior-model-policy-extraction", "diffusion-bc-ddpm-and-network-design", "diffusion-policy-class"]
---
# IDQL 的實驗

基準為 D4RL（Fu et al., arXiv 2004.07219）。基準與基線的一般說明見 [本論文使用的離線 RL 基準與比較方法](offline-rl-benchmarks-and-baselines.html)。

## 比較設定

主要比較 CQL（2006.04779）、IQL（2110.06169）、DQL（2208.06193）。兩種協定：「-A」為報告的結果（不限調參量）；「-1」為每個領域只能調一個超參數（IDQL 的 locomotion 用 τ=0.7，antmaze 用 τ=0.9）。表 2 另含 %BC、DT（2106.01345）、TD3+BC（2106.06860）、EQL（2301.02328）、SfBC（2209.14548）。

## 表 1 的總分（原文數字）

| 方法 | -A 總分 | -1 總分 | 訓練時間 |
| --- | --- | --- | --- |
| CQL | 1002.1 | 854.3 | 80m |
| IQL | 1070.4 | 997.9 | 20m |
| DQL | 1209.7 | 957.4 | 240m |
| IDQL | 1213.8 | 1150.9 | 40–60m |

論文結論：locomotion 與最佳先前方法相當，antmaze 優於先前方法；-1 設定下 IDQL 只小幅下降，antmaze 比 IQL 高約 70 分。critic 訓練與擴散模型分開，因此訓練時間接近 IQL，遠低於 SfBC（785m）與 DQL（240m）。

## 線上微調（表 3）

比較 Cal-QL（2303.05479）、RLPD（2302.02948）、IQL。IDQL-Max 凍結行為策略、只微調價值網路；IDQL-Imp 微調所有網路。antmaze 總分：IQL 408.2 → 473.7，IDQL-Max 459.9 → 559.1，IDQL-Imp 462 → 548.2，RLPD 0.0 → 571.0。

## 限制（§7）

動作維度小的環境（如 locomotion）有過擬合問題，dropout 可緩解但會影響其他環境；Adroit 的線上微調表現不佳，作者推測過度預訓練有害。

critic 損失的消融見 [廣義 IQL 的隱含 actor 與 critic 損失選擇](idql-implicit-policy-and-critic-choice.html)，擴散模型設計見 [擴散行為複製、DDPM 與 LN_Resnet](diffusion-bc-ddpm-and-network-design.html)，抽取演算法見 [以擴散行為模型做策略抽取](diffusion-behavior-model-policy-extraction.html)；DQL 見 [擴散策略作為表達力強的策略類別](diffusion-policy-class.html)。
