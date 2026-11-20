---
slug: odd-onpd-vs-offpd-setup
title: OnPD 與 OffPD 的定義：rollout policy 作為獨立變因
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [OnPD, OffPD, rollout-policy, distillation]
arxiv: ["2609.35259", "1503.02531", "1606.07947", "2306.13649", "2410.11325"]
sources: ["2609.35259 §2-3", "arXiv abstract 2306.13649", "arXiv abstract 2410.11325", "arXiv abstract 1606.07947"]
related: ["odd-lineage-hub", "knowledge-distillation-foundations", "on-policy-distillation-imitation", "odd-token-kl-direction-gradients"]
---
# OnPD 與 OffPD 的定義

屬於 [蒸餾動態研究的研究脈絡總覽](odd-lineage-hub.html)。

## 設定
論文（arXiv:2609.35259 §3）以學生 π_S 與教師 π_T 表示，輸出序列 y 的蒸餾目標為對各 token 位置的下一 token 分布差異 D(π_S(·|x, y<n) ‖ π_T(·|x, y<n)) 取平均，期望取自 rollout policy ρ 產生的序列。
- ρ = π_T：off-policy 蒸餾（OffPD），學生在固定的教師生成資料上訓練。
- ρ = π_S：on-policy 蒸餾（OnPD），學生自己產生軌跡，再向教師查詢各位置的分布。
- 為了計算可行，論文把取樣得到的軌跡視為固定，不對取樣過程求導（論文引 Agarwal et al., 2023 與 Tang and Munos, 2025）。

## 前身
- Hinton et al.（arXiv:1503.02531）與 Kim and Rush（arXiv:1606.07947）：論文以兩者描述 OffPD 的基礎，內容見 [知識蒸餾基礎](knowledge-distillation-foundations.html)。
- Agarwal et al.（arXiv:2306.13649）：論文以此作為 OnPD 的定義來源，內容見 [On-policy 蒸餾與線上模仿學習](on-policy-distillation-imitation.html)。
- Xu et al.（arXiv:2410.11325）Speculative Knowledge Distillation：摘要指出監督式蒸餾有訓練與推論的分布不一致，on-policy 蒸餾可能因學生樣本品質低而讓教師回饋不準，並提出學生與教師交錯取樣。論文將這類方法歸為在兩個策略的混合軌跡上訓練的混合式變體。

## 本論文的切入點
論文指出既有研究多半把 OnPD 與 SFT、RLVR 比較，同時改變了監督密度、獎勵與優化流程；很少有研究在其他訓練設定固定的條件下直接對比 OnPD 與較便宜的 OffPD。因此論文把 rollout policy 當作可單獨操作的變因。

KL 方向如何與 rollout policy 配對，見 [Token 層級 KL 方向與梯度](odd-token-kl-direction-gradients.html)。
