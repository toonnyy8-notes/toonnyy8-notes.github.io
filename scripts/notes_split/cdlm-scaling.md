---
slug: cdlm-scaling
title: 連續擴散語言模型的規模化策略與大模型結果（Sigma、Lafa 等）
cluster: Diffusion Language Models
kind: topic
tags: [scaling, ar-initialization, block-diffusion, self-speculative-decoding, sigma, lafa]
sources: [flow-matching-lm-2026.md]
arxiv: ["2605.07820", "2605.18530", "2610.02665", "2610.28559", "2606.27617", "2610.00894"]
related: [continuous-flow-lm-overview, flow-map-few-step-distillation, cdlm-representation-space, cdlm-noise-schedule]
---
# 連續擴散語言模型的規模化策略與大模型結果（Sigma、Lafa 等）

屬於 [2026 年連續 flow 語言模型總覽](continuous-flow-lm-overview.html)。

| 模型 | 規模 | 初始化 / 資料 | 結構 | 主要結果 |
|---|---|---|---|---|
| Scaling CFM（2605.07820） | 1.7B | 從頭訓練，2.1T tokens，Nemotron，\|V\|≈100k | 全序列 one-hot flow | 1024 步 Gen PPL 58.66，AR 45.33；8 步 MAUVE 0.88，AR 0.70 |
| RePlaid（2605.18530） | 對齊現代架構 | — | embedding 擴散 | 追上 AR 的算力倍數約 20×（有 SC）／27×（無 SC）；MDLM 約 14×、Duo 約 22×；OWT PPL 22.1 |
| Sigma-3B / 8B（2610.02665） | 3.8B / 8.5B（總參數） | 由 1T tokens 預訓練的內部 AR baseline 初始化；再訓練 300B tokens；約 9,600 / 19,200 GB200 GPU 小時 | block 大小 32，16 維 embedding，AR 輔助 loss | 見下方說明 |
| Lafa（2610.28559） | 0.5B / 7B / 8B | Qwen2.5-Instruct（0.5B, 7B）、Qwen3-8B Base；少於 30B tokens | 前綴因果注意力 + block 內雙向非同步 flow；loss = AR CE + 去噪 CE + embedding MSE | 7B 吞吐量為 AR 的 2.02–2.54× |
| MLFM（2606.27617） | 以 1028M SMDM 為骨幹 | LoRA 319M 可訓練參數，約 100B 位置 | 遮罩 + 連續 interpolant | GSM8K 31.24（SMDM 58.5，經 GSM8K 40 epoch 微調）；MT-Bench 2.27（SMDM 1.60） |
| ClockDLM（2610.00894） | 小規模 | — | 位置相依時鐘、sliding window、KV cache | 連續 DLM 中 OWT likelihood 最佳；加 Cache Grab 後 GSM8K 超過 AR |

（Sigma：[arxiv](https://arxiv.org/pdf/2610.02665v1)；Lafa：[arxiv](https://arxiv.org/abs/2610.28559)；MLFM：[arxiv](https://arxiv.org/pdf/2606.27617)；ClockDLM：[arxiv](https://arxiv.org/html/2610.00894)）

## Sigma 的結果
- **Sigma 3B base：** 平均 64.34，NLD-3B 為 64.74；GSM8K 84.50 vs 83.40。
- **Sigma 8B base：** 平均 69.67，NLD-8B 為 70.22，Qwen3-8B 為 68.91。
- **Sigma 8B-Instruct：** 平均 59.27，NLD 為 59.92。
- **Sigma 的 pass@k：** MBPP 上 k=256 時 91.8% vs 87.6%。作者把這歸因於 embedding 空間 steering 提高了多樣性。
- **大模型（diffusion 模式）：** Sigma-8B base：GSM8K 89.07、HumanEval 61.40、MBPP 64.78。Sigma-8B-Instruct：GSM8K 92.49、MMLU 77.46、IFEval 79.14、GPQA 44.70。MMLU（likelihood 評分）：Sigma-8B 73.96，NLD-8B 76.01。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- 少步表現（DDIM-4 平均 7.74）見 [少步蒸餾](flow-map-few-step-distillation.html)；Sigma 的 16 維 embedding 未正則化時塌縮到約 4 維，見 [表示空間](cdlm-representation-space.html)。

## Lafa 的設計
- **Lafa 的 macro-average：** 有二手報導稱 8B macro-average 為 63.1%，AR SFT 為 56.8%，Efficient-DLM 為 63.5%。 [aiweekly](https://aiweekly.co/alerts/lafa-flow-language-model-hits-254x-ar-throughput-at-7b-scale) 此數字未能從原文表格直接核對，引用時應回查原文。
- 多 token 預測效果與 flow 草稿加 AR 驗證的做法見 [自我推測解碼](lafa-self-speculative-decoding.html)。 [arxiv](https://arxiv.org/abs/2610.28559)
- 與既有做法的對照：論文 §1 寫明其少於 30B tokens 的訓練「cheaper than comparable adaptation of discrete diffusion models that use up to 500B tokens」。
- 學到的 per-token time 呈由左到右順序，見 [噪聲／時間排程](cdlm-noise-schedule.html)。

## 規模化路線的共同點
- 從 AR 權重初始化（Sigma、Lafa、ELF-REG 使用 AR 表示對齊、PlaidQ）。
- block 化或位置相依時間，以取得 KV cache 與可變長度生成（Sigma、Lafa、Clock Diffusion）。
- 保留 AR 輔助 loss。
- 這些做法使連續模型的訓練成本與離散擴散 LLM 相當。但 Sigma 與 Lafa 的品質基準都是同算力的離散擴散或 AR SFT，不是前沿 AR 模型。
