---
slug: odd-strong-to-weak-distillation-practice
title: 強到弱蒸餾在後訓練中的使用與本論文的模型選擇
cluster: RL, Reasoning & Self-Distillation
kind: topic
tags: [strong-to-weak, post-training, Llama-3, Qwen2.5, model-reports]
arxiv: ["2609.35259", "2404.14219", "2507.20534", "2601.02780", "2505.09388", "2606.19348", "2407.21783", "2412.15115"]
sources: ["2609.35259 §1-2, §4.1", "arXiv abstract 2407.21783 (title only)", "arXiv listing 2404.14219, 2507.20534, 2601.02780, 2505.09388, 2606.19348, 2412.15115 (title only)"]
related: ["odd-lineage-hub", "odd-onpd-vs-offpd-setup", "knowledge-distillation-foundations"]
---
# 強到弱蒸餾的實務背景

屬於 [蒸餾動態研究的研究脈絡總覽](odd-lineage-hub.html)。論文（arXiv:2609.35259）寫道，強到弱蒸餾已成為現代後訓練流程中常見的作法，用較強的教師把推理行為轉移給較弱的學生。

## 論文所引的實例（僅確認標題與作者，內容依論文敘述）
- Phi-3 技術報告（Abdin et al., 2024，arXiv:2404.14219）。
- Kimi K2（Bai et al., 2025，arXiv:2507.20534）。
- MiMo-V2-Flash 技術報告（Xiao et al., 2026，arXiv:2601.02780），論文同時把它列為 on-policy 蒸餾的使用者。
- Qwen3 技術報告（Yang et al., 2025a，arXiv:2505.09388）與 DeepSeek-V4（DeepSeek-AI et al., 2026，arXiv:2606.19348），論文以其說明蒸餾在新一代模型訓練流程中的地位。
- 論文另引 Kimi k3（Bai et al., 2026），未在 arXiv 查得，未查證。

## 本論文的模型選擇（§4.1）
- 主要實驗：Llama-3.1-8B 教師蒸餾到 Llama-3.2-1B 學生（Grattafiori et al., 2024，arXiv:2407.21783）；附錄 A.5 為 Qwen2.5-7B 教師與 Qwen2.5-1.5B 學生（Yang et al., 2025b，arXiv:2412.15115）。
- 師生取自同一模型家族，以共用 tokenizer 與詞彙表，才能直接比較 token 層級分布。
- 每個任務另訓練教師並凍結，以確保足夠的能力差距。
- 任務為 MedReason（醫學）、Science（科學）與 Countdown（算術），訓練為全參數微調 150 步，每組設定三個隨機種子。

蒸餾損失的基礎見 [知識蒸餾基礎](knowledge-distillation-foundations.html)。

OnPD 與 OffPD 的定義見 [OnPD 與 OffPD 的定義](odd-onpd-vs-offpd-setup.html)。
