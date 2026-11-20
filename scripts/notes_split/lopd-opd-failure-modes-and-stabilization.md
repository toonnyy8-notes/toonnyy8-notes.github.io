---
slug: lopd-opd-failure-modes-and-stabilization
title: OPD 與 OPSD 的失敗模式與穩定化
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [on-policy-distillation, failure-modes, style-drift, information-leakage]
arxiv: ["2608.13040", "2604.00626", "2603.25562", "2604.13016", "2606.11709", "2604.03128", "2605.23493", "2604.10674"]
sources: ["arXiv abstract 2604.00626", "arXiv abstract 2603.25562", "arXiv abstract 2604.13016", "arXiv abstract 2606.11709", "arXiv abstract 2604.03128", "arXiv abstract 2605.23493", "arXiv abstract 2604.10674", "2608.13040 §3"]
related: ["on-policy-distillation-imitation", "odd-recent-opd-studies-2026", "lopd-privileged-context-substrates", "lopd-method-learnable-context-and-margin"]
---
# OPD 與 OPSD 的失敗模式與穩定化

先備：[On-policy 蒸餾與線上模仿學習](on-policy-distillation-imitation.html)。

## OPD 的一般問題
- LOPD 論文以 Song & Zheng（2026, arXiv:2604.00626，OPD 綜述）與 Li et al.（2026, arXiv:2604.13016，OPD 成敗條件）為 OPD 的引用依據，兩者以及 Fu et al.（arXiv:2603.25562）的內容見 [2026 年的 on-policy 蒸餾研究](odd-recent-opd-studies-2026.html)。

## OPSD 特有的問題
- RLCSD（Pan et al., 2026, arXiv:2606.11709）：摘要稱有特權脈絡的模型與無脈絡的模型之間的分布差距集中在風格 token 而非任務 token，稱為 privilege-induced style drift，並使輸出變短。
- Self-distilled RLVR（Yang et al., 2026, arXiv:2604.03128）：摘要稱只用特權老師的訊號會造成嚴重資訊洩漏與長期訓練不穩。
- EDGE-OPD（arXiv:2605.23493）：摘要稱特權資訊可能改變推理、降低一般能力。
- Skill-SD（arXiv:2604.10674）：摘要稱把 OPSD 直接與 RL 合併常導致訓練崩潰。

## 與 LOPD 的關係
LOPD 論文指出，若讓 composer 只最小化蒸餾損失，它可以把老師拉向學生（π^T 趨近 π^S），得到無資訊的脈絡。因此論文加入 privileged-margin 約束，把老師相對學生的 token 優勢與整條軌跡的驗證結果綁在一起，另以冷啟動與錨定項限制漂移。細節見 [LOPD 方法](lopd-method-learnable-context-and-margin.html)。
各類特權脈絡見 [特權脈絡的各種實體](lopd-privileged-context-substrates.html)。
