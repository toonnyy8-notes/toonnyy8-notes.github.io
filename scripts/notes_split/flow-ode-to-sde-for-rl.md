---
slug: flow-ode-to-sde-for-rl
title: Flow 模型 RL 的 ODE-to-SDE 轉換
cluster: Diffusion & Generative Models
kind: topic
tags: [ode-to-sde, flow-matching, stochastic-sampling, euler-maruyama, marginal-preserving]
arxiv: ["2505.05470", "2210.02747", "2209.03003", "2011.13456", "2303.08797", "2409.08861", "2503.19385", "2504.02407"]
sources: ["paper.txt §4.1, Appendix A", "paper.txt §2", "arXiv abstract 2503.19385", "arXiv abstract 2504.02407"]
related: ["flowgrpo-lineage-hub", "flowgrpo-diffusion-sde-background", "flowgrpo-design-choices-and-results"]
---
# Flow 模型 RL 的 ODE-to-SDE 轉換

屬於 [Flow-GRPO 的研究脈絡總覽](flowgrpo-lineage-hub.html)。

## 為什麼需要

GRPO 需要機率 $\pi_\theta(x_{t-1}\mid x_t,c)$ 與多樣的取樣。論文（§4.1）指出確定性 ODE $dx_t=v_t\,dt$ 有兩個問題：(1) 計算 $p(x_{t-1}\mid x_t,c)$ 需要散度估計，成本高；(2) 除了初始種子外沒有隨機性，探索不足，論文在 §5.3 報告降低隨機性會大幅降低訓練效率。

## 做法

構造一條 SDE，使其邊際分布在所有時間點與原 ODE 相同。形式（論文式 7）：

$dx_t=\big(v_t(x_t)-\tfrac{\sigma_t^2}{2}\nabla\log p_t(x_t)\big)dt+\sigma_t\,dw$

代入 rectified flow 的 $\alpha_t=1-t$、$\beta_t=t$，得 score 與速度的關係 $\nabla\log p_t(x)=-\tfrac{x}{t}-\tfrac{1-t}{t}v_t(x)$（式 27），因此（式 8）：

$dx_t=\big[v_t+\tfrac{\sigma_t^2}{2t}(x_t+(1-t)v_t)\big]dt+\sigma_t\,dw$

Euler–Maruyama 離散化（式 9）：

$x_{t+\Delta t}=x_t+\big[v_\theta+\tfrac{\sigma_t^2}{2t}(x_t+(1-t)v_\theta)\big]\Delta t+\sigma_t\sqrt{\Delta t}\,\epsilon$

論文取 $\sigma_t=a\sqrt{t/(1-t)}$，$a$ 為控制雜訊強度的純量，實驗用 $a=0.7$。此時 $\pi_\theta(x_{t-1}\mid x_t,c)$ 是各向同性高斯，與參考策略的 KL 有閉式：正比於 $\|v_\theta-v_{ref}\|^2$。

## 脈絡

理論來源見 [從 DDPM 到 score SDE 與 stochastic interpolants](flowgrpo-diffusion-sde-background.html)。同期工作：F5R-TTS（Sun 等，2504.02407，2025）對 flow-matching TTS 用 GRPO，但改為預測速度的高斯均值與變異數，需要重新訓練（論文描述）；Kim 等（2503.19385，2025）同樣用 SDE 隨機性，但目的是推論時擴展（論文描述）。$a$ 的消融見 [Denoising Reduction、KL 與實驗結果](flowgrpo-design-choices-and-results.html)。
