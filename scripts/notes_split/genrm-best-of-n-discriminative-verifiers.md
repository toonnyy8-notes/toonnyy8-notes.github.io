---
slug: genrm-best-of-n-discriminative-verifiers
title: Best-of-N 與判別式驗證器
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [best-of-n, verifier, reward-model, GSM8K, discriminative]
arxiv: ["2408.15240", "2009.01325", "2110.14168"]
sources: ["arXiv:2408.15240 §1、§2、§5", "arXiv 摘要 2009.01325、2110.14168"]
related: ["genrm-lineage-hub", "genrm-step-level-supervision-verifiers", "genrm-yes-token-method", "r1-cot-and-test-time-compute", "r1-rlhf-ppo-background"]
---
# Best-of-N 與判別式驗證器

**Best-of-N。** GenRM 論文 §2：對一道測試題，從生成器 LLM 取樣 N 個候選解，再由學習得到的驗證器（或獎勵模型）打分，選分數最高者作為最終答案。論文 §1 引用 Charniak and Johnson (2005) 與 Cobbe 等 (2021) 作為此策略的來源；驗證器越準，選到正確解的機率越高。

**Cobbe 等 2021（arXiv 2110.14168，Training Verifiers to Solve Math Word Problems）。** 摘要：提出 GSM8K（8.5K 題小學數學應用題），並提出訓練驗證器判斷模型完成內容的正確性；測試時產生多個候選解，選驗證器排名最高者。GenRM 的主要實驗資料即 GSM8K。

**Stiennon 等 2020（arXiv 2009.01325，Learning to summarize from human feedback）。** 摘要：蒐集人類對摘要的比較資料，訓練模型預測人類偏好的摘要，再把該模型當作獎勵函數微調摘要策略。GenRM §5 把它列為「預測兩個解之間偏好」的判別式 RM。

**判別式驗證器的定義。** GenRM §2、式 (3)：以二元交叉熵微調 LLM 作分類器，輸出分數 r_θ(x,y) = sigmoid(z_cls)，z_cls 是特殊 token cls 的 logit；訓練資料為正確與錯誤解的平衡混合。論文指出這種做法沒有使用 LLM 的文字生成能力，因此無法直接使用 CoT 與推論時運算。這是 GenRM 要改進的基線，細節見 [GenRM 的 Yes token 方法](genrm-yes-token-method.html)。

RLHF 的獎勵模型脈絡見 [LLM 的 RL 前史：RLHF、PPO 與 GAE](r1-rlhf-ppo-background.html)；Best-of-N 與 verifier 在推論時運算中的位置見 [思維鏈、驗證器與推論時運算](r1-cot-and-test-time-compute.html)。回到 [總覽](genrm-lineage-hub.html)。
