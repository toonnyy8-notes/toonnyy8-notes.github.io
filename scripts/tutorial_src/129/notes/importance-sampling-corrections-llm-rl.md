---
slug: importance-sampling-corrections-llm-rl
title: LLM RL 中的 importance sampling 修正：TIS、MIS、TOPR、GSPO、DPPO
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [importance-sampling, TIS, MIS, TOPR, GSPO, DPPO, clipping]
arxiv: ["2609.20807", "2503.14286", "2507.18071", "2510.18855", "2602.04879", "2603.19470", "2506.13585", "2409.19256"]
sources: ["arXiv:2609.20807 §2.2, §B.2, Table 1", "arXiv abstract 2503.14286", "arXiv abstract 2507.18071", "arXiv abstract 2602.04879", "arXiv abstract 2603.19470", "arXiv abstract 2510.18855"]
related: ["training-inference-mismatch-sources", "ppo-grpo-dapo-llm-rl-objectives", "score-centering-drift-decomposition"]
---
# LLM RL 中的 importance sampling 修正

**基本式。** rollout 來自 sampler q 而非 trainer p 時，可用重要性比 $r = p/q$ 精確修正：$\nabla E_p[R] = E_q[(p/q)\,R\,\nabla\log p]$（本文 Eq. 2）。實務上逐 token 套用（Zheng et al., 2025）。代價是變異增加：稀有 token 的比值可能很大（Ionides, 2008；Owen, 2013）。因此實務方法都對比值設上限，本文指出這必然引入偏差，也就是重新帶入 drift。

**Table 1 的格子。** 本文把常用方法整理成一個格子，維度為 level（token 或 sequence）、統計量、Low/High 區間、區間外處理（clip 或 drop）、被修正的優勢正負號。

- **TIS**（Yao et al., 2025）與 CISPO（MiniMax, arXiv 2506.13585）：token 比，上限 2.0，clip。
- **MIS**（IcePop，Ling Team, arXiv 2510.18855）：token 比，區間 0.5 至 5.0，區間外 drop。
- **GSPO**（Zheng et al., arXiv 2507.18071）：摘要指出其以序列似然定義重要性比並做序列層級 clipping；本文使用 token 比的幾何平均。
- **TOPR**（Le Roux et al., arXiv 2503.14286）：摘要指出其使用不對稱、逐步縮小（tapered）的 IS 變體。
- **DPPO**（Qi et al., arXiv 2602.04879）：本文指出它以二元 total variation 為基礎設定 mask 區間，但目標函數仍使用 IS 比。其摘要主張 PPO 的 ratio clipping 不適合大詞彙表。
- **Ye et al.（2026，ALP，arXiv 2603.19470）**：在 trainer 隱藏狀態注入可學習擾動，以擾動後的策略作為比值分子，降低重尾；本文指出它仍依賴 IS。

**本文的立場。** 這些方法都屬於同一個格子，score centering 不使用比值、mask 或 clip，是以加法修正的另一類方法（見 [Score centering：drift 與 signal 的分解](score-centering-drift-decomposition.html)）。實驗結果：在高斯權重雜訊與量化下 SC、TIS、MIS 表現最佳；PPO 與 DAPO 能撐過 staleness 但在量化與權重雜訊下崩潰，本文解釋為其 clip 是為策略移動產生的比值而設計，數值誤差不在此列。TIM 的來源見 [訓練與推論不一致的來源](training-inference-mismatch-sources.html)，PPO 與 DAPO 目標見 [PPO、GRPO、DAPO 與 LLM 的 RL 目標](ppo-grpo-dapo-llm-rl-objectives.html)。
