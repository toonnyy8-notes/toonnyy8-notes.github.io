---
slug: flow-bellman-operator-all-noise-levels
title: Flow Bellman operator：在所有 noise level 施加 Bellman 一致性
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [value-forcing, flow-bellman-operator, value-flows, PCBF, EVOR, contraction]
arxiv: ["2611.02160", "2510.07650", "2510.08218", "2605.08253", "2210.02747", "2209.03003"]
sources: ["arXiv:2611.02160 §1、§2、§4.1、§4.2、附錄 Table 1", "arXiv 摘要 2510.07650、2605.08253、2510.08218"]
related: ["distributional-rl-return-distributions", "flow-matching-velocity-target", "flow-based-return-critics", "value-forcing-vf-ivf-algorithms"]
---
# Flow Bellman operator：在所有 noise level 施加 Bellman 一致性

**問題。** 只在終點（$\tau=1$）施加 Bellman 目標的方法見 [flow-based critic](flow-based-return-critics.html)。2611.02160 指出已有三篇在路徑上加上更強限制，但各有缺陷：

- **Value Flows**（Dong 等，arXiv 2510.07650）與 **EVOR**（Espinosa-Dice 等，arXiv 2510.08218）：以速度場直接作為 TD bootstrap 目標，但把「乾淨」的 Bellman 方程原樣用在所有 noise level，包含 $\tau=0$ 的純噪聲。2611.02160 稱此為不正確的一致性。
- **PCBF**（Xu 等，arXiv 2605.08253）：以相同初始噪聲耦合兩條 flow 路徑，沿耦合路徑套用 Bellman 算子；2611.02160 指出耦合使邊際路徑本身缺乏 Bellman 一致性。其摘要也說明不要求各時間點的邊際滿足分布式 Bellman 不動點。

**Value Forcing 的做法（2611.02160 §4）。** 定義與 noise level 相關的 flow Bellman 算子，於噪聲水平 $\tau$：
$Z(s,a,\tau)\overset{d}{=}\tau r+\gamma Z(S',A',\tau)+(1-\tau)\sqrt{1-\gamma^2}\,N$，$N\sim\mathcal N(0,1)$。
- $\tau=0$：兩側都是標準高斯，需要高斯修正項 $\sqrt{1-\gamma^2}N$ 才能成立；省略它（如 Value Flows）會得到錯誤的方程。
- $\tau=1$：恢復一般的分布式 Bellman 方程。
- 對每個 $\tau$ 與 $p\in[1,\infty)$，算子在 $p$-Wasserstein 的上確界距離下是 $\gamma$-contraction，因此依 Banach 不動點定理，每個 noise level 都有唯一不動點。

**損失。** 利用 flow matching 把邊際速度表示為條件期望的性質（引用 Liu 等 2022〔arXiv 2209.03003〕與 Lipman 等 2022〔arXiv 2210.02747〕，見 [Flow matching 的速度目標](flow-matching-velocity-target.html)），導出 V 速度與 Q 速度互為目標的兩個損失，交替最小化，形式類似 Fitted Q-Evaluation。細節見 [VF 與 IVF](value-forcing-vf-ivf-algorithms.html)。
