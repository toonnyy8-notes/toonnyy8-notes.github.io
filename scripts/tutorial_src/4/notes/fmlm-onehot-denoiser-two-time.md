---
slug: fmlm-onehot-denoiser-two-time
title: FMLM 的設計：one-hot flow、CE denoiser 與 two-time denoiser
cluster: Diffusion Language Models
kind: topic
tags: [fmlm, flm, one-hot, cross-entropy, two-time-denoiser, simplex]
arxiv: ["2602.16813", "2406.07507", "2505.18825", "2510.22510", "2506.10892"]
sources: [paper-2602.16813-section-3-4]
related: ["fmlm-lineage-overview", "fmlm-flow-matching-interpolant", "fmlm-flow-map-distillation-lineage", "fmlm-discrete-diffusion-factorization", "cdlm-training-objective", "cdlm-noise-schedule"]
---
# FMLM 的設計：one-hot flow、CE denoiser 與 two-time denoiser

屬於 [Flow Map Language Models 的研究脈絡總覽](fmlm-lineage-overview.html)。以下內容全部取自本文第 3、4 節。

## 逐層堆疊

1. **表示**：one-hot 編碼加 argmax 解碼，見 [連續空間的文字擴散](fmlm-continuous-text-diffusion.html)。
2. **FLM**：在該表示上做 flow matching，網路學 denoiser D_t。引理 3.1：最佳 denoiser 是逐 token 後驗 p^l_{1|t}(· | I_t = x)，位於 simplex 上，所以輸出層用逐 token softmax，訓練目標為交叉熵（式 (12)）。命題 3.2：交叉熵最小化等價於最小化對真實後驗的 KL，並給出 W2 誤差界。
3. **FMLM**：把 FLM 蒸餾為 flow map。因為 flow map X_{s,t} 會離開 simplex，本文定義 two-time denoiser δ_{s,t}(x) = x + (1 − s) v_{s,t}(x)。命題 3.3：δ 位於 simplex，在對角上等於 D_s，滿足半群條件，並能還原 X_{s,t}（式 (19)）。
4. **KL 半群目標**（命題 3.4）：以 KL 取代平方誤差，對角項錨定到預訓練 D̂，非對角項用半群教師。

## 與離散擴散的關係

離散模型也學 p^l_{1|t}，但用它做祖先取樣，需要整個聯合分佈而有因子化誤差。連續 flow 用同一個後驗推出精確速度（式 (9)、(11)），因此能被精確蒸餾為少步生成器。命題 3.5 說明離散設定沒有可用的樣本層級決定性映射。

## 時間重參數化

訓練時改變時間取樣分佈、推論時改變時間格點。動機是 one-hot 的 token 決定集中在 t 接近 1 的窄區間，以解碼錯誤率 P_e(t) 衡量。這個指標沿用自 Duo（2506.10892）與 CANDI（2510.22510）。排程的比較見 [連續擴散語言模型的噪聲／時間排程](cdlm-noise-schedule.html)；MSE 到 CE 的目標變化見 [連續擴散語言模型的訓練目標與參數化](cdlm-training-objective.html)。
