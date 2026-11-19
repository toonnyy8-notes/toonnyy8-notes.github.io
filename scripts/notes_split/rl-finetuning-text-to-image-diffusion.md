---
slug: rl-finetuning-text-to-image-diffusion
title: 文字生成圖像擴散模型的 RL 微調脈絡
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [t2i-alignment, rwr, dpo, ddpo, reward-backprop, preference]
arxiv: ["2505.05470", "1910.00177", "2304.06767", "2302.12192", "2305.18290", "2311.12908", "2402.10210", "2501.13918", "2309.17400", "2310.03739", "2305.13301", "2305.16381", "2502.01819", "1707.06347"]
sources: ["paper.txt §2 Alignment for T2I", "paper.txt Appendix C.1", "arXiv abstract of each listed ID"]
related: ["flowgrpo-lineage-hub", "flowgrpo-denoising-mdp-and-grpo", "flow-ode-to-sde-for-rl"]
---
# 文字生成圖像擴散模型的 RL 微調脈絡

屬於 [Flow-GRPO 的研究脈絡總覽](flowgrpo-lineage-hub.html)。論文 Related Work 把 T2I 對齊分成五類；下表只列已在 arXiv 核對標題的代表作。

| 類別 | 工作（arXiv，年） |
|---|---|
| 可微獎勵直接微調 | Prabhudesai 等（2310.03739，2023）；Clark 等（2309.17400，2023，Directly Fine-Tuning Diffusion Models on Differentiable Rewards） |
| Reward-weighted regression | AWR（Peng 等，1910.00177，2019）；Lee 等（2302.12192，2023）；RAFT（Dong 等，2304.06767，2023） |
| DPO 與變體 | DPO（Rafailov 等，2305.18290，2023）；Diffusion-DPO（Wallace 等，2311.12908，2023）；Self-Play Fine-Tuning of Diffusion Models（Yuan 等，2402.10210，2024）；Flow-DPO（Liu 等，2501.13918，2025） |
| PPO 式策略梯度 | DDPO（Black 等，2305.13301，2023）；DPOK（Fan 等，2305.16381，2023）；Score as Action（Zhao 等，2502.01819，2025） |
| 免訓練對齊 | 論文列為第五類（本筆記未逐篇核對） |

來源：分類為論文 §2 的說法，各作品內容取自其 arXiv 摘要標題，未重述細節。

## 與 Flow-GRPO 的關係

- 論文 §1 指出先前研究多把 RL 用在早期擴散模型，或對 flow 模型用離線 RL（DPO）；線上 RL 用於 flow matching 尚未被探索。
- 論文附錄 C.1 在 SD3.5-M 上比較 SFT、Flow-RWR、Flow-DPO 與其線上版本，並以 ODE-to-SDE 把 DDPO 改到 flow 模型，另比較 ReFL 與 ORW。結果：Flow-GRPO 在人類偏好任務上超過所有比較方法；DDPO 獎勵上升較慢且後期崩潰；ReFL 需要可微獎勵，GRPO 不需要。
- 論文指出 RWR 在其實驗中不隨訓練進步，與 DDPO 論文中對 RWR 的發現一致。

DDPO 使用的 MDP 見 [Denoising 作為 MDP，以及 Flow-GRPO 的目標函數](flowgrpo-denoising-mdp-and-grpo.html)；使 DDPO 類方法能用於 flow 模型的轉換見 [ODE-to-SDE 轉換](flow-ode-to-sde-for-rl.html)。
