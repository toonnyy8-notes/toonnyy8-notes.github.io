---
slug: idql-lineage-hub
title: IDQL 的研究脈絡總覽：從 IQL 的隱含價值學習到擴散策略抽取
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [offline-rl, iql, diffusion-policy, actor-critic, policy-extraction]
arxiv: ["2304.10573", "2110.06169", "1910.00177", "2301.02328", "2303.15810", "2209.14548", "2206.00695", "2208.06193", "2006.11239", "2004.07219"]
sources: ["arXiv:2304.10573 §1, §2, §3", "各被引論文的 arXiv 摘要"]
related: ["rl-mdp-q-learning-basics", "offline-rl-distribution-shift", "iql-lineage-overview", "expectile-regression-value-learning", "awr-advantage-weighted-policy-extraction", "idql-implicit-policy-and-critic-choice", "in-sample-implicit-value-regularization-eql-ivr", "diffusion-behavior-model-policy-extraction", "diffusion-policy-class", "diffusion-bc-ddpm-and-network-design", "ddpm-training-and-sampling-recap", "trajectory-diffusion-planning", "idql-evaluation-and-online-finetuning"]
---
# IDQL 的研究脈絡總覽

IDQL（arXiv 2304.10573）處理的問題是：IQL 以 expectile regression 學到的 Q 函數，對應哪一個策略並不明確，而 IQL 用單峰高斯策略加 AWR 抽取策略，可能無法近似該策略。論文的做法有兩部分：把 IQL 的價值損失推廣到任意凸損失，推導出對應的隱含（implicit）actor；再用擴散模型表示行為策略，以 critic 算出的權重重新取樣，得到所需的策略。

## 閱讀順序

1. 基本記號與離線設定：[MDP、Q 函數與 Bellman backup](rl-mdp-q-learning-basics.html)、[離線 RL 的分布位移問題](offline-rl-distribution-shift.html)。
2. IQL 本身：[IQL 的研究脈絡總覽](iql-lineage-overview.html)、[Expectile regression 與價值學習](expectile-regression-value-learning.html)。
3. 原本的策略抽取方式：[Advantage-weighted regression 與策略抽取](awr-advantage-weighted-policy-extraction.html)。
4. 本論文的理論：[廣義 IQL 的隱含 actor 與 critic 損失選擇](idql-implicit-policy-and-critic-choice.html)。
5. 同期對 IQL 的推廣：[EQL 與 IVR：指數型 critic 與隱含價值正則化](in-sample-implicit-value-regularization-eql-ivr.html)。
6. 擴散模型作為策略：[擴散策略作為表達力強的策略類別](diffusion-policy-class.html)、[以擴散行為模型做策略抽取：SfBC、ARQ 與 IDQL](diffusion-behavior-model-policy-extraction.html)。
7. 擴散模型的訓練與網路設計：[擴散行為複製、DDPM 與 LN_Resnet](diffusion-bc-ddpm-and-network-design.html)。
8. 實驗：[IDQL 的實驗與線上微調](idql-evaluation-and-online-finetuning.html)。

DDPM 的完整推導見既有筆記 [DDPM 的訓練與取樣流程](ddpm-training-and-sampling-recap.html)，軌跡層級的擴散方法見 [軌跡層級擴散規劃（Diffuser）](trajectory-diffusion-planning.html)，不在此重寫。時間線見 timeline.json。
