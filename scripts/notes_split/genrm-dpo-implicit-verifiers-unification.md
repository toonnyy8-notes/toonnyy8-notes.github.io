---
slug: genrm-dpo-implicit-verifiers-unification
title: DPO 與隱式驗證器：生成與驗證的統一
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [DPO, V-STaR, STaR, implicit-reward, unified-generation-verification]
arxiv: ["2408.15240", "2305.18290", "2402.06457", "2203.14465"]
sources: ["arXiv 摘要 2402.06457、2203.14465", "arXiv:2408.15240 §1、§5"]
related: ["genrm-lineage-hub", "genrm-yes-token-method", "genrm-best-of-n-discriminative-verifiers", "r1-rlhf-ppo-background", "r1-star-self-generated-reasoning"]
---
# DPO 與隱式驗證器：生成與驗證的統一

**DPO。** Rafailov 等（arXiv 2305.18290，Direct Preference Optimization: Your Language Model is Secretly a Reward Model）。GenRM §5：DPO 以政策的 logits 隱式表示獎勵，並以獎勵模型的損失訓練政策，藉此把生成（政策）與驗證（RM）合在一個模型。DPO 在 RLHF 流程中的位置見 [LLM 的 RL 前史：RLHF、PPO 與 GAE](r1-rlhf-ppo-background.html)。

**V-STaR。** Hosseini 等 2024（arXiv 2402.06457）。摘要：STaR 類自我改進方法會丟棄迭代中產生的錯誤解；V-STaR 同時利用正確與錯誤解，以 DPO 訓練驗證器，並在推論時於多個候選解中選一個。STaR 本身（Zelikman 等 2022，arXiv 2203.14465）見 [STaR](r1-star-self-generated-reasoning.html)。GenRM 作者之一 Hosseini 負責本文的判別式 RM 與 DPO 基線（作者貢獻段）。

**GenRM 的論點。** §5：這種模型綁定在推理任務上被觀察到錯誤外推與表徵退化；Pal 等 2024、Pang 等 2024、Setlur 等 2024、Yang 等 2024 各以額外技巧處理。Yang 等 2024 在另一個 head 上加輔助生成式 SFT 損失，訓練後丟棄，查詢 RM 時不產生文字。GenRM 的差別（§5）：損失較簡單（next-token prediction）、不需要參考政策、驗證表現明顯較好（圖 1、5）。

統一損失 L_GenRM = L_SFT(verify) + λ L_SFT(correct) 見 [GenRM 的 Yes token 方法](genrm-yes-token-method.html)。回到 [總覽](genrm-lineage-hub.html)。
