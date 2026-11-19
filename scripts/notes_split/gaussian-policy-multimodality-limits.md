---
slug: gaussian-policy-multimodality-limits
title: Gaussian、CVAE 與 MMD 策略的多峰限制與 2D bandit 示例
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [behavior-cloning, multimodal, gaussian-policy, cvae, mmd, bandit]
arxiv: ["2208.06193", "1812.02900", "1906.00949", "2106.06860", "2110.06169", "1911.11361", "2206.11251", "2206.00695"]
sources: ["paper.txt §1, §2, §4, Appendix H", "arXiv API 驗證各篇題名與作者；對 BCQ、BEAR、TD3+BC、BRAC、IQL、Behavior Transformers、Goo & Niekum 的描述依 paper.txt 的敘述，未另讀其摘要"]
related: ["batch-constrained-policy-constraint-methods", "td3bc-algorithm-and-d4rl", "iql-algorithm-and-d4rl-evaluation", "offline-rl-distribution-shift", "diffusion-policy-class", "diffusion-ql-loss-coupling"]
---
# 多峰行為策略與既有策略類別的限制

論文（arXiv 2208.06193）的出發點是：離線資料常由多個策略混合收集，真實行為策略可能是多峰、偏態，或各動作維度相依；多數策略正則化方法使用的策略類別無法準確表示它。背景問題（OOD 動作的 Q 值高估）見 [Offline RL 的分布偏移與 OOD 動作造成的 Q 值高估](offline-rl-distribution-shift.html)。

## 論文所述的兩個原因

1. 策略類別表達力不足：多半是以網路輸出平均與對角共變異數的 Gaussian。以最小化 KL(資料‖策略) 擬合雙峰資料時，會產生 mode-covering，把高密度放在兩峰之間，而該處正是資料的低密度區（論文引用 Shafiullah et al., 2022，arXiv 2206.11251 作為 Gaussian 難以建模多峰行為的依據）。
2. 正則化方式不合適：KL 需要明確的密度值；MMD 每個狀態需要多個動作樣本。兩者都得先訓練行為複製策略，再把新策略拉向它，因此複製品質差會誤導策略改善。

## 論文描述的各方法

- BC-MLE：對角 Gaussian，最大化 `log π_θ(a|s)`。TD3+BC（arXiv 2106.06860）直接加 BC 項；IQL（arXiv 2110.06169）用 in-sample 的優勢函數加權 log 項。見 [TD3+BC 演算法與 D4RL 評估](td3bc-algorithm-and-d4rl.html)、[IQL 演算法、超參數與 D4RL 評估](iql-algorithm-and-d4rl-evaluation.html)。
- BC-CVAE：以條件 VAE 建模。BCQ（arXiv 1812.02900）用 CVAE 近似行為策略，再訓練偏差模型朝高 Q 值調整。
- BC-MMD：BEAR（arXiv 1906.00949）以 CVAE 複製行為，並以 MMD 限制策略接近它，使用 Tanh-Gaussian 策略。BRAC（arXiv 1911.11361）比較了 KL、MMD、Wasserstein 對偶等正則化。
- BCQ 與 BEAR 屬兩步驟正則化。這些方法的整理見 [批次約束與策略約束方法](batch-constrained-policy-constraint-methods.html)。
- Goo & Niekum（2022，arXiv 2206.00695）強調離線 RL 需要顯式行為複製；論文將其列為相關工作。

## 2D bandit 示例（論文 §4）

動作 `a ∈ [−1,1]²`，資料 `M=10000` 筆，來自四個 Gaussian 的等比混合，中心 `(0,0.8)、(0.8,0)、(0,−0.8)、(−0.8,0)`，標準差 `(0.05,0.05)`。

- 行為複製：擴散模型涵蓋四個模式；BC-MLE 以單一大變異數 Gaussian 覆蓋，高密度落在低密度資料區；CVAE 抓到四個峰但在峰之間給高密度，且不同隨機種子有時漏峰；MMD 的 Tanh-Gaussian 密度靠近邊界線。
- 加入獎勵後（每個峰的獎勵平均 0.0、1.5、3.0、5.0，標準差 0.5）：TD3+BC 因行為策略把多數密度放在無資料區而無法收斂到最佳峰；BCQ 的動作落在 CVAE 找到的四個對角；Diffusion-QL 找到高獎勵的行為模式。

Appendix H 另測了以混合密度網路（Bishop, 1994）取代 TD3+BC 確定性 actor 的 Gaussian mixture 策略作為額外基準。
