---
slug: flowgrpo-diffusion-sde-background
title: 從 DDPM 到 score SDE 與 stochastic interpolants
cluster: Diffusion & Generative Models
kind: topic
tags: [ddpm, ddim, score-sde, stochastic-interpolants, fokker-planck]
arxiv: ["2006.11239", "2010.02502", "2011.13456", "2303.08797", "2409.08861", "2505.05470"]
sources: ["paper.txt §2, Appendix A", "arXiv abstract 2011.13456", "arXiv abstract 2303.08797", "arXiv abstract 2409.08861"]
related: ["flowgrpo-lineage-hub", "ddpm-training-and-sampling-recap", "score-geometry-score-matching-sde", "flow-ode-to-sde-for-rl"]
---
# 從 DDPM 到 score SDE 與 stochastic interpolants

屬於 [Flow-GRPO 的研究脈絡總覽](flowgrpo-lineage-hub.html)。DDPM 的訓練與取樣見 [DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html)，score 與 SDE 的幾何見 [score matching 與 SDE](score-geometry-score-matching-sde.html)。

Flow-GRPO 的 Related Work 與附錄 A 引用下列工作，作為「ODE 與 SDE 可以有相同邊際分布」的理論基礎：

- **DDPM**（Ho、Jain、Abbeel，2006.11239，2020）：以離散步驟加噪並學習反向過程。
- **DDIM**（Song、Meng、Ermon，2010.02502，2020）：論文將其與 DDPM、score SDE 並列為擴散模型。
- **Score-Based Generative Modeling through SDEs**（Song 等，2011.13456，2020）：前向 SDE 對應一條 probability flow ODE。Flow-GRPO 附錄 A 說明其推導「受 SGM 中由 SDE 推 probability flow ODE 的方式啟發」，反向構造一條 SDE，使其邊際分布與 flow ODE 相同。
- **Stochastic Interpolants**（Albergo、Boffi、Vanden-Eijnden，2303.08797，2023）：論文稱其統一 flow 與 diffusion。
- **Adjoint Matching**（Domingo-Enrich 等，2409.08861，2024）：標題為以 memoryless stochastic optimal control 微調 flow 與 diffusion 模型；論文將其與 [28] 並列，稱「我們的工作建立在它們的理論基礎上」。

附錄 A 使用 Fokker–Planck 方程比較兩條過程的邊際密度演化，推導見 [ODE-to-SDE 轉換](flow-ode-to-sde-for-rl.html)。
