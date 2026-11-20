---
slug: odd-token-kl-direction-gradients
title: Token 層級的正向與反向 KL：梯度形式與對 rollout policy 的敏感度
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [forward-KL, reverse-KL, mode-seeking, gradient, distillation]
arxiv: ["2609.35259", "2506.09477", "2306.13649"]
sources: ["2609.35259 §3, §5.1", "arXiv abstract 2506.09477"]
related: ["odd-lineage-hub", "odd-onpd-vs-offpd-setup", "odd-learning-rate-adam-clipping-rlvr", "knowledge-distillation-foundations"]
---
# Token 層級 KL 方向與梯度

屬於 [蒸餾動態研究的研究脈絡總覽](odd-lineage-hub.html)。

## 兩種方向
在固定前綴下，論文（arXiv:2609.35259 §3）定義
- 正向 KL：對 π_T 取期望的 KL(π_T ‖ π_S)，對教師有機率的 token 懲罰學生未覆蓋（mode-covering）。
- 反向 KL：對 π_S 取期望的 KL(π_S ‖ π_T)，對學生偏好、教師不看好的 token 懲罰較重（mode-seeking）。

慣例上 OnPD 配反向 KL、OffPD 配正向 KL。論文給出兩個理由：其一，序列層級 KL 的兩種鏈鎖分解恰好對應「教師 rollout 配正向 KL」與「學生 rollout 配反向 KL」；其二，單一樣本的對數比是對應方向 KL 的不偏估計。論文同時指出，當 KL 對整個詞彙表計算時，方向並不限制 rollout policy，且上述兩個理由都不蘊含慣例配對較易優化，因此把兩者視為獨立選擇。

## 梯度的不對稱（§5.1）
對學生 logit 的導數：
- 正向 KL 為 π_S(v) − π_T(v)，每個分量落在 [-1, 1]，在 Jacobian 有界時，兩種 rollout 造成的更新差異以前綴分布的總變差距離線性為界。
- 反向 KL 的導數以 π_S(v) 加權並含 log(π_S/π_T)，後者可任意大，因此不存在只依賴 rollout 距離與 Jacobian 的界；學生機率趨近 0 時導數也趨近 0，難以補回學生遺漏的教師 mode。

由此論文預期反向 KL 對 rollout policy 較敏感、較受益於 on-policy rollout；§5.2 以學生到教師之間連續的 rollout 光譜實驗驗證。

## 相關背景
- Tang and Munos（arXiv:2506.09477）：摘要指出在 LLM 的 RL 訓練中，把 KL 估計式當損失直接求導的常見實作不會得到正確的 KL 梯度，且部分實作忽略序列性而只得到部分梯度。本論文在不對取樣前綴求導時引用此文，並說明該設定下 OnPD 反向 KL 的更新一般不是序列層級反向 KL 的完整梯度。
- 梯度範數大且變動大的反向 KL 與梯度裁剪的關係見 [學習率、Adam、梯度裁剪與 RLVR](odd-learning-rate-adam-clipping-rlvr.html)。

KL 與蒸餾損失的基礎見 [知識蒸餾基礎](knowledge-distillation-foundations.html)；rollout policy 的定義見 [OnPD 與 OffPD 的定義](odd-onpd-vs-offpd-setup.html)。
