---
slug: flow-based-return-critics
title: 以 flow matching 表示價值與回報分布的 critic：TD-Flow、floq、DFC、FlowCritic、FlowIQN
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [flow-matching, distributional-rl, critic, endpoint-supervision]
arxiv: ["2503.09817", "2509.06863", "2509.23087", "2510.22686", "2605.08515", "2606.29820", "2611.02160"]
sources: ["各論文的 arXiv 摘要", "arXiv:2611.02160 §1、§2、附錄 baseline 說明對這些方法的描述", "2606.29820 僅依 2611.02160 的描述（未讀摘要細節）"]
related: ["distributional-rl-return-distributions", "flow-matching-velocity-target", "flow-bellman-operator-all-noise-levels", "value-forcing-vf-ivf-algorithms"]
---
# 以 flow matching 表示價值與回報分布的 critic

以生成模型表示價值的共同做法是：用 [flow matching](flow-matching-velocity-target.html) 學習從高斯噪聲到回報（或 Q 值）的映射，TD 目標則由下一個狀態的預測得到。

- **TD-Flow**（Farebrother 等，arXiv 2503.09817）：摘要指出它用機率路徑上的新 Bellman 方程搭配 flow matching，學習 geometric horizon model（預測未來狀態分布）。2611.02160 引用它作為「在不同 noise level 學習可提供更豐富監督」的先例。
- **floq**（Agrawalla 等，arXiv 2509.06863）：以速度場參數化 Q 函數，用 TD 目標訓練 flow matching；摘要動機是讓價值函數有 iterative computation 與中間運算的密集監督。2611.02160 描述它擬合的是純量 Q（對 M 個 bootstrap 樣本取平均）。
- **DFC**（Chen 等，arXiv 2509.23087）：Distributional Flow Critic，以 flow matching 建模完整的狀態-動作回報分布，搭配 flow 策略。
- **FlowCritic**（Zhong 等，arXiv 2510.22686）：以 flow matching 建模價值分布並生成樣本做價值估計。
- **FlowIQN**（Groom 等，arXiv 2605.08515）：摘要指出現有 flow critic 的來源-目標配對是任意的，與 Wasserstein 下的收縮性不一致；它在 mini-batch 內排序來源與 Bellman 目標樣本，近似單調最佳傳輸耦合（2611.02160 稱為 quantile-aligned coupling）。
- **Dual-Flow RL**（Li 等，arXiv 2606.29820）：2611.02160 將它歸為在 flow 終點施加 Bellman 一致性的一類。

**共同限制（依 2611.02160）。** DFC、FlowCritic、FlowIQN、Dual-Flow RL 與 floq 都在 flow 路徑的**終點**以去噪空間算出 Bellman 目標，並把它當作所有 noise level 的 flow matching 目標，因此其他 noise level 沒有各自的 Bellman 一致性監督。針對這點的後續做法見 [Flow Bellman operator 與所有 noise level 的一致性](flow-bellman-operator-all-noise-levels.html)。
