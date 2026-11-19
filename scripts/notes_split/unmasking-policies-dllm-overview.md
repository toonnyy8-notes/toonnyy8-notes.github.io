---
slug: unmasking-policies-dllm-overview
title: 擴散語言模型的 unmasking 策略：從 heuristic 到強化學習總覽
cluster: Diffusion Language Models
kind: hub
tags: [dllm, masked-diffusion, unmasking, sampling, reinforcement-learning]
arxiv: ["2512.09106", "2107.03006", "2406.07524", "2502.09992", "2508.15487", "2503.09573", "2505.22618", "2504.12216", "2402.03300"]
sources: ["paper.txt §1, §2, §5", "arXiv API metadata"]
related: [mdm-discrete-diffusion-lineage, dllm-llada-dream-models, semi-ar-block-generation, confidence-thresholding-heuristics, rl-post-training-dllm-grpo, learned-unmasking-concurrent-work, unmasking-mdp-policy-design]
---
# 擴散語言模型的 unmasking 策略：從 heuristic 到強化學習總覽

「Learning Unmasking Policies for Diffusion Language Models」（arXiv 2512.09106）研究 masked diffusion 語言模型（dLLM）在每個取樣步驟要解開（unmask）哪些 token。論文把這個選擇視為序列決策問題，用強化學習訓練一個小型 policy。

閱讀順序如下，每一篇只談一個概念。

1. [Masked diffusion 的離散擴散脈絡](mdm-discrete-diffusion-lineage.html)：dLLM 的訓練目標與 mask 式前向過程從哪裡來。
2. [LLaDA 與 Dream：可比擬 AR 的 dLLM](dllm-llada-dream-models.html)：論文用作環境的兩個預訓練模型。
3. [Semi-AR 區塊生成](semi-ar-block-generation.html)：heuristic 目前依賴的生成設定。
4. [Confidence 門檻式 heuristic](confidence-thresholding-heuristics.html)：Fast-dLLM 與其後續方法，以及論文指出的缺點。
5. [dLLM 的強化學習後訓練與 GRPO](rl-post-training-dllm-grpo.html)：既有 RL 工作改的是模型本身；論文沿用 GRPO，但只訓練 policy。
6. [學習式 unmasking 的相關工作](learned-unmasking-concurrent-work.html)：與本論文同期、同樣學習取樣策略的方法。
7. [Unmasking 作為 MDP 與 policy 設計](unmasking-mdp-policy-design.html)：本論文的做法。

時間軸見 timeline.json，其中每筆都已用 arXiv API 核對標題、作者與編號。
