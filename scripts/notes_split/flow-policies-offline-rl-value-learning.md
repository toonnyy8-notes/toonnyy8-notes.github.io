---
slug: flow-policies-offline-rl-value-learning
title: Flow 策略與離線 RL 的價值學習：FQL、Flow Actor-Critic、Q-chunking 與 OGBench
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [flow-policy, offline-rl, FQL, OGBench, action-chunking]
arxiv: ["2502.02538", "2602.18015", "2507.07969", "2410.20092", "2304.10573", "2611.02160"]
sources: ["arXiv 摘要 2602.18015", "arXiv:2611.02160 §2、§6、附錄對 FQL、QC、OGBench 的使用", "2502.02538、2507.07969、2410.20092 僅依標題與 2611.02160 的使用方式"]
related: ["offline-rl-distribution-shift", "value-forcing-vf-ivf-algorithms", "diffusion-policy-class", "flow-matching-velocity-target", "cql-lineage-hub", "td3bc-lineage-overview"]
---
# Flow 策略與離線 RL 的價值學習

離線 RL 的核心困難是學到準確的價值函數，同時避免在資料分布外錯誤外推（2611.02160 §2；背景見 [Offline RL 的分布偏移](offline-rl-distribution-shift.html)）。該文把既有做法分為三類：保守懲罰（CQL 等，見 [CQL 研究脈絡](cql-lineage-hub.html)）、限制動作到資料支撐內（BCQ、BEAR、TD3+BC 一脈，見 [TD3+BC 研究脈絡](td3bc-lineage-overview.html)）、完全不查詢資料外樣本（IQL、IDQL、IVR）。

**表達力強的策略。** 資料集常為多峰分布，因此近年以 diffusion／flow 作為策略：
- IDQL（arXiv 2304.10573）以擴散策略作為 actor，見 [擴散策略類別](diffusion-policy-class.html)。
- Flow Q-Learning（Park 等，arXiv 2502.02538）：本文引用其 one-step flow 策略設定；floq 的實作也沿用此類策略。
- Flow Actor-Critic（Chae 等，arXiv 2602.18015）：摘要指出它同時以 flow 模型作 actor，並以 flow behavior proxy 構成保守 critic 正則項。
- Q-chunking（Li 等，arXiv 2507.07969）：對動作片段學習 critic；2611.02160 將其作為純量 critic 的 baseline（QC）。

**基準。** OGBench（Park 等，arXiv 2410.20092，離線 goal-conditioned RL 基準）提供 2611.02160 中 20 個任務。與本文最接近的設定是在子集資料上比較價值學習演算法（Xu 等 2023；Park 等 2026）；差別在於它們學純量回報，Value Forcing 學整個分布，詳見 [VF 與 IVF](value-forcing-vf-ivf-algorithms.html)。
