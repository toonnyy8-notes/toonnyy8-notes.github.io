---
slug: value-forcing-vf-ivf-algorithms
title: VF 與 IVF：Value Forcing 的 actor-critic 與 implicit 兩種實例
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [value-forcing, VF, IVF, expectile, rejection-sampling, IDQL, IQL]
arxiv: ["2611.02160", "2110.06169", "2304.10573", "2303.15810", "2502.02538"]
sources: ["arXiv:2611.02160 §5、§6、附錄演算法與 baseline 說明", "arXiv 摘要 2110.06169、2304.10573、2303.15810、2502.02538"]
related: ["flow-bellman-operator-all-noise-levels", "expectile-regression-value-learning", "iql-lineage-overview", "flow-policies-offline-rl-value-learning", "diffusion-policy-class", "offline-rl-distribution-shift"]
---
# VF 與 IVF

兩個演算法都使用 [flow Bellman operator](flow-bellman-operator-all-noise-levels.html) 導出的 V 速度與 Q 速度損失，差別在策略改進的方式（2611.02160 §5）。

**VF（explicit，actor-critic）。** Q 值以對 noise 積分的速度估計：$Q(s,a)=\mathbb E_{z^0}\int_0^1 v_Q\,d\tau$（用 M 個噪聲樣本）。策略以 rejection sampling 取得：從 flow behavioral cloning 策略 $\pi_\beta$ 抽 K 個動作，取 Q 最大者（引用 Chen 等 2023）。此「從行為策略抽樣再挑選」的做法與 IDQL（Hansen-Estruch 等，arXiv 2304.10573，以 diffusion 策略作為 IQL 的 actor）同屬一個系譜，見 [擴散策略類別](diffusion-policy-class.html)。

**IVF（implicit）。** 沒有顯式 actor，訓練時只查詢資料集內的動作，把 V 速度損失改為 expectile 加權的 flow matching。背景是 IQL（Kostrikov 等，arXiv 2110.06169）以 expectile 回歸近似資料支撐內的最大值，見 [Expectile regression 與 value learning](expectile-regression-value-learning.html) 與 [IQL 研究脈絡](iql-lineage-overview.html)；2611.02160 同時引用 Sikchi 等 2024 的 Dual RL。論文理論結果：expectile 增大時，IVF 趨近資料集動作內的最佳值。in-sample 學習的另一代表為 Xu 等（arXiv 2303.15810，IVR），見 [離線 RL 的分布偏移](offline-rl-distribution-shift.html)。

**實作細節（依論文）。** 獎勵乘以 $(1-\gamma)$ 以正規化無限視野下折扣回報的尺度，論文 §6.3 以實驗檢驗其效果；訓練時交替更新 V 速度與 Q 速度，並使用 target 速度網路。

**評估。** 26 個長視野任務（20 個來自 OGBench，6 個來自 OCBench），在離線 RL 設定下比較，資料量越少增益越大；策略抽取比較對象含 Flow Q-Learning 一脈，見 [flow 策略與離線 RL 的價值學習](flow-policies-offline-rl-value-learning.html)。
