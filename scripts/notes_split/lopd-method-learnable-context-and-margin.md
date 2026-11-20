---
slug: lopd-method-learnable-context-and-margin
title: LOPD 方法：可學習的 latent 脈絡與 privileged-margin
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [LOPD, QFormer, reverse-KL, privileged-margin, dual-variable]
arxiv: ["2608.13040", "2601.18734", "2601.20802", "2604.10674", "2601.19897", "2402.03300"]
sources: ["2608.13040 §3", "2608.13040 §4.1", "2608.13040 Appendix A.1-A.4"]
related: ["lopd-opsd-self-teacher-privileged-context", "lopd-privileged-context-substrates", "lopd-opd-failure-modes-and-stabilization", "lopd-latent-reasoning-continuous-tokens", "lopd-latent-memory-for-agents"]
---
# LOPD 方法

本筆記只整理論文本身的設計，作為前幾篇脈絡的落點。

## 流程
1. 學生 π^S 從任務與互動歷史取樣軌跡 τ，取得驗證獎勵 r，A = 2r − 1。
2. 檢索：經驗庫只存訓練集上成功的軌跡（observation-lite 格式）；以 Qwen3-Embedding-8B 與 FAISS 取 top-3（附錄 A.3）。
3. Composer：凍結 backbone 加 LoRA 作 encoder，QFormer 風格的壓縮器（學習 query）把每筆經驗壓成 K=32 個 latent token，共 96 個，組成 c_φ。
4. 老師是同一模型的凍結副本 θ̄，在學生走過的每個前綴上條件於 c_φ 重新評分。梯度經凍結網路回傳到 c_φ。
5. 蒸餾損失為 reverse KL，使用老師 top-M（M=20）加尾部合併的分布。
6. 推論時只保留學生，不需要檢索、composer 或 latent 脈絡。

## privileged-margin 目標
每個 token 的 privilege 為 δ = log π^T(a) − sg[log π^S(a)]，Δ(φ) 為 A(τ)·δ 的遮罩平均。目標式為 min_{θ,φ} max_{β≥0} L_distill + β(m − Δ) + λ‖c_φ − sg[c_φ0]‖²，m=0.05，β 以 β ← [β + η(m − Δ)]₊ 更新。若脈絡退化使 π^T → π^S，則 Δ → 0 < m，對偶懲罰啟動。動機見 [OPD 與 OPSD 的失敗模式與穩定化](lopd-opd-failure-modes-and-stabilization.html)。

## 冷啟動與實驗設定
Composer 先以成功軌跡做 next-token NLL 冷啟動（附錄 A.2）。實驗使用 Qwen3-4B/8B 與 Olmo3-7B，基線包含 GRPO、SDFT、OPSD、SDPO、Skill-SD；這些基線的脈絡對照見 [OPSD：同一模型帶特權脈絡當老師](lopd-opsd-self-teacher-privileged-context.html) 與 [特權脈絡的各種實體](lopd-privileged-context-substrates.html)。latent token 的來源脈絡見 [Latent reasoning](lopd-latent-reasoning-continuous-tokens.html) 與 [Latent memory](lopd-latent-memory-for-agents.html)。
