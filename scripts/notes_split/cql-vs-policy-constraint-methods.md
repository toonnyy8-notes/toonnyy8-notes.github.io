---
slug: cql-vs-policy-constraint-methods
title: CQL 與策略限制、不確定性、OPE 方法的比較
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [CQL, policy-constraint, BCQ, BEAR, BRAC, related-work]
arxiv: ["2006.04779", "1812.02900", "1906.00949", "1911.11361", "1910.00177", "1802.09477"]
sources: ["arXiv:2006.04779 §1, §4, §5", "arXiv abstract 1812.02900", "arXiv abstract 1906.00949", "arXiv abstract 1911.11361", "arXiv abstract 1910.00177"]
related: ["offline-rl-distribution-shift", "cql-algorithm-variants-and-guarantees", "cql-experiments-d4rl-atari"]
---
# CQL 與其他處理 OOD 動作的方法

CQL（arXiv:2006.04779）§5 把相關工作分成幾類，以下依該節整理。各方法要解決的問題見 [Offline RL 的分布偏移](offline-rl-distribution-shift.html)。

**策略限制類。** 把學到的策略限制在接近行為策略的範圍，距離以 KL（Jaques 等、BRAC arXiv:1911.11361、AWR arXiv:1910.00177、Siegel 等）、Wasserstein（BRAC）或 MMD（BEAR arXiv:1906.00949）衡量，然後只用該策略取樣的動作做 Bellman backup，或對價值加懲罰。BCQ（arXiv:1812.02900）也在此類，並被 CQL 引為需要行為策略模型的方法之一。SPIBB 對未見動作以行為策略 bootstrap。CQL 的批評：這些方法大多需要另外估計 `π_β(a|s)`，受限於估計準確度，資料來自多個來源時估計尤其困難。CQL 沒有這個需求。

**不確定性類。** 以不確定性估計判斷 Q 值是否可信，借用 online RL 探索的技術。CQL 引用文獻指出這類方法在 offline RL 中效果普遍不佳，原因是對不確定性估計的精度要求高。

**Robust MDP 與高信心策略改進。** CQL 認為這些方法在策略改進上偏保守；CQL 不對所有狀態動作對都低估 Q 值，預期較不保守。

**Off-policy evaluation。** 早期用 per-action importance sampling；較新的用 marginalized importance sampling 直接估計狀態分布比，變異較小但有偏。後者使用動態規劃，也會受 OOD 動作影響。CQL 的正則項透過 gap-expanding 性質直接處理 OOD 動作。

**Q 函數懲罰。** 先前的 Q 函數懲罰只用在帶示範的 online RL。Luo 等人以線性外插性質與學到的動態模型學保守價值函數；Kakade 與 Langford 的 CPI 在 online RL 中保守地改進策略。

**理論。** 定理 3.5、3.6 與 safe policy improvement 文獻相關，論文稱其界比 Petrik 等人的 ∞-norm 界更好。實驗中的對照見 [CQL 的實驗](cql-experiments-d4rl-atari.html)，CQL 本身的目標見 [min-max 形式與變體](cql-algorithm-variants-and-guarantees.html)。TD3（arXiv:1802.09477）在論文中作為防止 Q 值高估的標準做法被提及（取雙 Q 的最小值）。
