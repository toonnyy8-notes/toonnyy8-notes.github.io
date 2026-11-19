---
slug: flowgrpo-flow-matching-backbone
title: Flow matching 與 rectified flow 作為生成骨幹
cluster: Diffusion & Generative Models
kind: topic
tags: [flow-matching, rectified-flow, sd3, velocity-field]
arxiv: ["2210.02747", "2209.03003", "2403.03206", "2505.05470"]
sources: ["paper.txt §1, §2, §3", "arXiv abstract 2210.02747", "arXiv abstract 2209.03003", "arXiv abstract 2403.03206"]
related: ["flowgrpo-lineage-hub", "flow-matching-velocity-target", "flow-ode-to-sde-for-rl"]
---
# Flow matching 與 rectified flow 作為生成骨幹

屬於 [Flow-GRPO 的研究脈絡總覽](flowgrpo-lineage-hub.html)。速度目標的一般說明見 [Flow matching 的速度目標](flow-matching-velocity-target.html)，這裡只寫 Flow-GRPO 用到的部分。

- **Flow Matching**（Lipman 等，2210.02747，2022）與 **Rectified Flow**（Liu、Gong、Liu，2209.03003，2022）：兩者直接回歸速度場，生成時積分 ODE。Flow-GRPO 依後者的框架。
- **Scaling Rectified Flow Transformers**（Esser 等，2403.03206，2024）：Stable Diffusion 3 的論文。Flow-GRPO 以 SD3.5-M 為主要實驗模型，並在附錄 C.4 加做 FLUX.1-Dev。

Flow-GRPO 採用的記號（論文 §3）：

- 加噪資料 $x_t=(1-t)x_0+t x_1$，$x_0$ 為資料，$x_1$ 為雜訊。
- 訓練目標 $\mathcal L=\mathbb E\|v-v_\theta(x_t,t)\|^2$，其中 $v=x_1-x_0$。
- 取樣為確定性 ODE $dx_t=v_t\,dt$。

論文的 Related Work 稱 flow matching 以較少的 ODE 步數達到有競爭力的 FID，且是近期影像與影片生成模型的主要選擇。確定性取樣在 RL 中造成的問題見 [ODE-to-SDE 轉換](flow-ode-to-sde-for-rl.html)。
