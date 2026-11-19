---
slug: value-forcing-lineage-hub
title: Value Forcing 的研究脈絡總覽：從 distributional RL 到每個 noise level 的 Bellman 一致性
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [value-forcing, distributional-rl, flow-matching, offline-rl, hub]
arxiv: ["2611.02160", "1707.06887", "1710.10044", "1806.06923", "2210.02747", "2209.03003", "2503.09817", "2509.06863", "2509.23087", "2510.22686", "2605.08515", "2510.07650", "2510.08218", "2605.08253", "2502.02538", "2304.10573", "2110.06169"]
sources: ["arXiv:2611.02160 §1, §2, §3, §4", "各引用論文的 arXiv 摘要"]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "distributional-rl-return-distributions", "flow-matching-velocity-target", "flow-based-return-critics", "flow-bellman-operator-all-noise-levels", "value-forcing-vf-ivf-algorithms", "flow-policies-offline-rl-value-learning", "diffusion-policy-class", "expectile-regression-value-learning", "iql-lineage-overview"]
---
# Value Forcing 的研究脈絡總覽

Value Forcing（arXiv 2611.02160）用 flow matching 學習回報（return）的整個分布，並在每一個 noise level 上要求 Bellman 一致性。理解它需要依序掌握下列背景，建議依此順序閱讀。

1. **價值函數與離線 RL 的基本記號**：[MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)、[Offline RL 的分布偏移](offline-rl-distribution-shift.html)。
2. **Distributional RL**：價值函數改為學習回報的分布，而不是期望值。見 [Distributional RL：回報分布與分布式 Bellman 方程](distributional-rl-return-distributions.html)。
3. **Flow matching**：用速度場把高斯噪聲傳送到目標分布，是本文表示回報分布的工具。見 [Flow matching 的速度目標](flow-matching-velocity-target.html)。
4. **以生成模型表示價值**：TD-Flow、floq、DFC、FlowCritic、FlowIQN 等，多數只在 flow 的終點（endpoint）施加 Bellman 目標。見 [以 flow matching 表示回報分布的 critic](flow-based-return-critics.html)。
5. **在 flow 路徑上施加 Bellman 一致性**：Value Flows、EVOR、PCBF 與本文的 flow Bellman operator。見 [Flow Bellman operator 與所有 noise level 的一致性](flow-bellman-operator-all-noise-levels.html)。
6. **演算法與策略抽取**：VF（actor-critic）與 IVF（implicit）；背景是 [IQL 的 expectile 回歸](expectile-regression-value-learning.html)、[IQL 研究脈絡](iql-lineage-overview.html) 與 flow／diffusion 策略。見 [VF 與 IVF](value-forcing-vf-ivf-algorithms.html)、[flow 策略與離線 RL 的價值學習](flow-policies-offline-rl-value-learning.html)、[擴散策略類別](diffusion-policy-class.html)。

時間軸見同資料夾的 timeline.json。
