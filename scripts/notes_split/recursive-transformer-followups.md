---
slug: recursive-transformer-followups
title: 遞迴 Transformer 後續工作：MoR、Ouro、MoL、CoLa 與 recurrent-depth
cluster: Architectures & Efficiency
kind: topic
tags: [mixture-of-recursions, ouro, looplm, recurrent-depth, test-time-compute, mixture-of-lora]
sources: [weight-sharing-lora.md]
arxiv: ["2507.10524", "2510.25741", "2512.12880", "2507.07996", "2502.05171"]
related: ["weight-sharing-recursive-overview", "relaxed-recursive-transformer", "stages-of-inference-weight-sharing", "depth-redundancy-adjacent-methods", "looped-transformers-reasoning"]
---
# 遞迴 Transformer 後續工作：MoR、Ouro、MoL、CoLa 與 recurrent-depth（2025–2026）

屬於 [層間權重共享總覽](weight-sharing-recursive-overview.html)。前身見 [RRT](relaxed-recursive-transformer.html)。

- **Mixture-of-Recursions（MoR）**（Bae, Kim, Bayat 等，arXiv:2507.10524, NeurIPS 2025）：統一「參數共享 + per-token 自適應深度 + recursion-wise KV caching」。135M–1.7B 規模下形成新 Pareto 前緣；>360M 時以約 1/3 唯一參數 match 或超越 vanilla； [alphaXiv](https://www.alphaxiv.org/abs/2507.10524v1) expert-choice routing（42.6%）優於 token-choice（40.0%）； [arXiv](https://arxiv.org/html/2507.10524v1) MoR-4 搭配 continuous depth-wise batching 達最高 2.06× 吞吐。 [alphaXiv](https://www.alphaxiv.org/abs/2507.10524v1) 135M 最小尺寸時 MoR 略遜 vanilla（遞迴瓶頸）。 [arXiv](https://arxiv.org/pdf/2507.10524)
- **Ouro / LoopLM**（Zhu et al., arXiv:2510.25741, 2025，33 位作者含 Bengio）：從頭預訓練 1.4B/2.6B LoopLM、7.7T token，配 entropy-regularized 自適應深度。 [arXiv](https://arxiv.org/abs/2510.25741) 摘要原文：「Ouro 1.4B and 2.6B models enjoy superior performance that match the results of up to 12B SOTA LLMs across a wide range of benchmarks」；Ouro-1.4B 在 GSM8K 達 78.92（vs 4B 基線 72.86）、MATH500 82.40（vs 59.60）、BBH 71.02（vs 70.95）。關鍵發現：recurrence「不增加原始知識儲存（looped 與非 looped 皆約每參數 2 bits），但大幅提升知識操作/多跳推理能力」（原文：「this advantage stems not from increased knowledge capacity, but from superior knowledge manipulation capabilities」），且「yields reasoning traces more aligned with final outputs than explicit CoT」——直接支持「能力提升侷限在推理/操作、而非容量」的論點。論文亦提到高 recursion 時訓練不穩。 [Hugging Face](https://huggingface.co/papers/2510.25741)
- **Improving Recursive Transformers with Mixture of LoRAs（MoL）**（Nouriborji, Rohanian, Rohanian, arXiv:2512.12880, 2025）：在共享 FFN 內插入 token-conditional LoRA experts（ModernALBERT, 50M–120M）； [arXiv](https://arxiv.org/abs/2512.12880) GLUE 上 ModernALBERT 81.94 > RRT 80.95（同初始化/目標）；MoL 8-expert/top-2 為 77.24 > MoA 76.87； [The Moonlight](https://www.themoonlight.io/en/review/improving-recursive-transformers-with-mixture-of-loras) 並提出把多個 expert 合併成單一 adapter 以利部署。 [arXiv](https://arxiv.org/abs/2512.12880)
- **Skip a Layer or Loop it?（CoLa）**（Li, Li, Zhou, arXiv:2507.07996, 2025）：以 MCTS 在測試時對預訓練 LLM 逐層 skip/repeat/重排。 [GitHub](https://github.com/AkihikoWatanabe/paper_notes/issues/2610) >75% 原本答對的樣本可找到更短 CoLa；>60% 原本答錯的可找到答對的 CoLa—— [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250707996L/abstract) 同時支持「冗餘可壓縮」與「重複層可提升」。
- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**（Geiping et al., arXiv:2502.05171, NeurIPS 2025）：Huginn-3.5B、800B token，推理時最多 50 loops、等效約 50B 參數運算量，數學/程式碼受益最大。把 loop 數當成 test-time compute 旋鈕。

## 潛在發展方向
1. **per-token 動態遞迴深度**
   - 依據 + 證據：MoR（已實作，Pareto 前緣、2.06× 吞吐）、Ouro 的 entropy-regularized 深度、CoLa（>60% 錯題可靠重複層救回）。
   - 假設：把 per-token 深度路由與四階段（見 [四階段假說](stages-of-inference-weight-sharing.html)）對齊（簡單 token 少 loop、需多跳推理 token 多 loop）。
   - 風險：router 訓練不穩、token-choice 明顯遜於 expert-choice；小模型（135M）遞迴瓶頸使 MoR 落後 vanilla。
2. **遞迴 + MoE 組合**
   - 依據 + 證據：MoEUT（首個共享層在 LM 上勝標準 Transformer，見 [相鄰路線](depth-redundancy-adjacent-methods.html)）、MoL（ModernALBERT 81.94 > RRT 80.95）、Tying the Loop（tied expert layers）。
   - 假設：MoE 補回共享砍掉的參數容量、遞迴提供深度，兩者解決彼此的短板（共享缺容量、MoE 缺深度 inductive bias）。
   - 風險：serving 複雜度疊加（multi-LoRA/multi-expert + 遞迴路由）；容量回升可能侵蝕資源效率初衷。
3. **遞迴 × test-time compute / latent reasoning**
   - 依據 + 證據：Geiping recurrent-depth（loop 數即 compute 旋鈕，Huginn 等效 50B）、Saunshi 深度-迭代等價（見 [looped transformers](looped-transformers-reasoning.html)）、Ouro 知識操作提升。
   - 假設：把 loop 深度當成與 CoT token 並列的第三條 test-time scaling 軸；對數學/程式碼受益最大。
   - 風險：每多一 loop = 一次完整 forward，序列延遲線性上升；深 loop 訓練不穩（Ouro、mechanistic 分析皆提及）；path-independence/不動點是否總成立仍未定。
