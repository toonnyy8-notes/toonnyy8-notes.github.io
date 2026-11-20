---
slug: lopd-lineage-hub
title: LOPD（Latent On-Policy Self-Distillation）的研究脈絡總覽
cluster: RL, Reasoning & Self-Distillation
kind: hub
tags: [LOPD, OPSD, on-policy-distillation, latent-context]
arxiv: ["2608.13040", "2402.03300", "2601.18734", "2601.20802", "2604.10674", "2412.06769", "2509.24704"]
sources: ["2608.13040 §1", "2608.13040 §2 Related Work", "2608.13040 reference list"]
related: ["rlvr-grpo-credit-assignment", "ppo-grpo-dapo-llm-rl-objectives", "knowledge-distillation-foundations", "on-policy-distillation-imitation", "context-distillation-self-distillation", "sdpo-lineage-overview", "lopd-opsd-self-teacher-privileged-context", "lopd-privileged-context-substrates", "lopd-opd-failure-modes-and-stabilization", "lopd-latent-reasoning-continuous-tokens", "lopd-latent-memory-for-agents", "lopd-method-learnable-context-and-margin"]
---
# LOPD 的研究脈絡總覽

LOPD（arXiv:2608.13040）處理的問題是：on-policy self-distillation（OPSD）的老師需要「特權脈絡」（privileged context，例如答案、回饋、技能、成功軌跡），現有方法都由設計者手工指定這個脈絡的形式。LOPD 把脈絡改成由經驗檢索並壓縮成連續 latent token，與學生一起端到端訓練。讀懂它需要依序理解下列脈絡。

1. 背景：RLVR 只有稀疏的結果獎勵，GRPO 是常用基線（DeepSeekMath，arXiv:2402.03300）。見 [RLVR 與信用分配瓶頸](rlvr-grpo-credit-assignment.html)、[PPO、GRPO、DAPO 的目標函數](ppo-grpo-dapo-llm-rl-objectives.html)。
2. 蒸餾的三個基礎：[知識蒸餾基礎](knowledge-distillation-foundations.html)、[On-policy 蒸餾與線上模仿學習](on-policy-distillation-imitation.html)、[Context distillation 與自蒸餾](context-distillation-self-distillation.html)。SDPO 這條線的總覽見 [SDPO 研究脈絡總覽](sdpo-lineage-overview.html)。
3. [OPSD：同一模型帶特權脈絡當老師](lopd-opsd-self-teacher-privileged-context.html)：2026 年初的 OPSD、π-Distill、SDFT、OPCD。
4. [特權脈絡的各種實體](lopd-privileged-context-substrates.html)：答案、回饋、自我修正、技能、同儕 rollout、檢索證據；這些都是 LOPD 論文所稱「設計者指定」的 substrate。
5. [OPD 與 OPSD 的失敗模式與穩定化](lopd-opd-failure-modes-and-stabilization.html)：解釋 LOPD 為何加入 privileged-margin 約束。
6. [Latent reasoning 與連續 token](lopd-latent-reasoning-continuous-tokens.html) 與 [Latent memory](lopd-latent-memory-for-agents.html)：LOPD 的 latent 脈絡借用這兩條線的形式。
7. [LOPD 方法](lopd-method-learnable-context-and-margin.html)：composer、reverse KL、privileged-margin 與對偶變數。

時間順序的文獻清單見 timeline.json。前人工作的內容來自該篇 arXiv 摘要或 LOPD 論文的描述，各筆記的 `sources` 欄位逐一標示。
