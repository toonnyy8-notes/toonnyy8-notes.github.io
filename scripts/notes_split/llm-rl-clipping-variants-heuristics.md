---
slug: llm-rl-clipping-variants-heuristics
title: LLM RL 中的 clipping 變體：DAPO、DCPO、Clip-Cov、SAPO、CISPO
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [DAPO, DCPO, Clip-Cov, SAPO, CISPO, entropy-collapse, clipping]
arxiv: ["2610.13015", "2503.14476", "2509.02333", "2505.22617", "2511.20347", "2506.13585"]
sources: ["arXiv abstract 2503.14476", "arXiv abstract 2509.02333", "arXiv abstract 2505.22617", "arXiv abstract 2511.20347", "arXiv abstract 2506.13585", "arXiv:2610.13015 §4.1"]
related: ["ppo-clip-tv-single-sample-proxy", "sequence-level-ratio-gspo-gmpo", "gppo-design-space-unification", "importance-sampling-corrections-llm-rl", "gppo-lineage-hub"]
---
# LLM RL 中的 clipping 變體

屬於 [GPPO 的研究脈絡總覽](gppo-lineage-hub.html)。GPPO 論文稱這些方法大多由特定的經驗觀察驅動，造成對 trust-region 約束的零散修改。以下依各論文摘要整理。

- **DAPO**（Yu et al., 2025，arXiv:2503.14476）：提出 Decoupled Clip 與 Dynamic sAmpling Policy Optimization，並開源大規模 RL 系統，摘要稱以 Qwen2.5-32B base 在 AIME 2024 達 50 分。GPPO 論文將其不對稱 clipping（ε_low、ε_high）解讀為 KL 散度加單樣本 MC（見 [GPPO 的設計空間](gppo-design-space-unification.html)）。
- **DCPO**（Yang et al., 2025，arXiv:2509.02333）：摘要指出 GRPO 的固定 clipping 範圍會導致零梯度，提出依 token 先驗機率調整 clipping 範圍的動態 clipping。GPPO 論文引用它的訓練設定（雙重 clipping，下限 0.5、上限 10，移除 KL 懲罰）。
- **Clip-Cov**（Cui et al., 2025，The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models，arXiv:2505.22617）：摘要指出 RL 訓練中策略熵急遽下降並伴隨表現飽和，作者建立熵 H 與表現 R 的經驗關係 R = −a·e^H + b。GPPO 論文將 Clip-Cov 描述為「clip 高共變異數 token」的基線。
- **SAPO**（Gao et al., 2025，Soft Adaptive Policy Optimization，arXiv:2511.20347）：摘要稱以平滑、受溫度控制的 gate 取代 hard clipping，並指出 token 級重要性比值變異大，在 MoE 模型更嚴重。GPPO 論文歸類為以軟 gating 保留原本被 clip 的梯度。
- **CISPO**（MiniMax et al., 2025，MiniMax-M1，arXiv:2506.13585）：GPPO 論文將 CISPO 歸類為以重要性權重 clipping 保留原本被 clip 的梯度；該論文的摘要（MiniMax-M1 模型）未描述 CISPO 細節，此處以 GPPO 論文的描述為準。

這些方法在 GPPO 實驗中作為對照（8 個基準的平均，數字取自論文 Table 1）：數學平均 SAPO 35.0、Clip-Cov 31.5、CISPO 30.5，程式平均 SAPO 53.5、Clip-Cov 53.0、CISPO 51.3。

另一組方法改變取樣單位而非 clipping 形狀，見 [序列層級的比值：GSPO 與 GMPO](sequence-level-ratio-gspo-gmpo.html)。clipping 作為 importance sampling 修正的角色見 [LLM RL 中的 importance sampling 修正](importance-sampling-corrections-llm-rl.html)。
