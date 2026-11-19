---
slug: continuous-flow-lm-overview
title: 2026 年連續 flow / flow matching 語言模型總覽
cluster: Diffusion Language Models
kind: hub
tags: [continuous-diffusion-lm, flow-matching, flow-map, overview]
sources: [flow-matching-lm-2026.md]
arxiv: ["2610.02665", "2602.16813", "2605.18530", "2610.28559", "2606.08417", "2607.00588", "2605.07820", "2602.12233", "2604.09784", "2604.11748", "2605.10938", "2605.11125", "2606.24773", "2606.27617", "2607.00714", "2608.23551", "2608.22244", "2609.40235", "2609.29102", "2609.04531", "2610.00497", "2610.00894", "2610.13834", "2610.12898", "2610.22330", "2406.07507", "2505.18825"]
related: ["cdlm-training-objective", "cdlm-representation-space", "cdlm-noise-schedule", "self-conditioning-flow-lm", "flow-map-few-step-distillation", "cdlm-scaling", "cdlm-open-problems", "cdlm-likelihood-estimation", "cdlm-architecture-recipes", "hybrid-continuous-discrete-lm", "cdlm-benchmarks-and-eval"]
---
# 2026 年連續 flow / flow matching 語言模型總覽

2026 年連續語言模型的性能提升有幾個來源，各來源都有消融實驗支撐：(1) 把訓練目標由 MSE 回歸改為以 softmax 去噪器預測乾淨 token 的 cross-entropy；(2) 把時間或噪聲排程改成依「解碼錯誤率」或資訊量均勻分配；(3) 用 self-conditioning，並修正它在推論時的偏差；(4) 用 flow map 自蒸餾換取少步生成；(5) 規模化時從預訓練 AR 模型初始化，並採 block 化與 KV cache。資料顯示，這些改動使連續模型在小規模基準上追平離散擴散，在 1–8 步的少步區間有明確優勢。在 3B–8B 規模，連續模型的表現已追平同算力的離散擴散，但仍未超過 AR 模型，評估指標本身也有被操縱的問題。 [arxiv](https://arxiv.org/pdf/2610.02665v1)

## TL;DR

- **改了什麼：** 主要修改有五項：
  - 訓練目標由 MSE / ε 預測改為 CE 去噪器（x-prediction + softmax）。見 [訓練目標與參數化](cdlm-training-objective.html)。
  - 表示空間由低維可學 embedding 改為 one-hot、超球面、凍結 contextual embedding 或 16 維 unit-norm embedding，並同時處理 embedding collapse。見 [表示空間與 decoder 問題](cdlm-representation-space.html)。
  - 排程依解碼錯誤率或 loss 分佈重新參數化。見 [噪聲／時間排程](cdlm-noise-schedule.html)。
  - self-conditioning 的作用被重新分析並修正。見 [self-conditioning](self-conditioning-flow-lm.html)。
  - 以 flow map（semigroup / Eulerian KL 自蒸餾）取代多步積分。見 [flow map 少步蒸餾](flow-map-few-step-distillation.html)。
- **效果到哪裡：**
  - FLM/FMLM（arXiv 2602.16813）在 LM1B/OWT 上與 MDLM、Duo 打平；一步生成勝過離散擴散蒸餾後的 8 步結果。 [arxiv](https://arxiv.org/html/2602.16813v2) [takara](https://tldr.takara.ai/p/2602.16813)
  - RePlaid（2605.18530）摘要稱其與 AR 的算力差距「compute gap of only 20× compared to autoregressive models」，並勝過 Duo；Plaid 約 64× 的對照數字未見於該摘要。
  - Sigma（2610.02665）摘要稱在 GSM8K、Minerva、HumanEval、MBPP 等基準上「achieves competitive performance with discrete models」；3B/8B 的平均分數與同算力的 Nemotron-Labs-Diffusion 相差不到 1 分。
  - Lafa（2610.28559）以少於 30B tokens 從 Qwen 改造到 8B。以 self-speculative decoding，摘要稱達到「2.54× the throughput of an AR model at comparable quality」，圖 1 說明 7B 為 2.02–2.54×。詳見 [規模化策略](cdlm-scaling.html)。
- **尚未解決：**（詳見 [未解決問題](cdlm-open-problems.html)）
  - Gen PPL 可被重複內容與無參數取樣器操縱（2606.08417、2607.00588）。
  - 大模型的少步品質仍明顯下降：Sigma 摘要稱連續軌跡在低 NFE 時「graceful degradation」，但其表 8 顯示 DDIM-4 時平均分數僅 7.74。
  - one-hot 表示的記憶體成本隨詞彙量線性增加。 [arxiv](https://arxiv.org/pdf/2605.07820)
  - 推理任務在低溫解碼下仍落後離散擴散與 AR。

## 時間脈絡與主要團隊

這波研究集中在 2026 年 2 月至 10 月。

| 時間 | 論文（arXiv） | 團隊 |
|---|---|---|
| 2026-02 | FLM/FMLM（2602.16813） | KAIST + CMU（Lee, Kim, Boffi 等） |
| 2026-02 | Categorical Flow Maps（2602.12233） | Amsterdam / Oxford（Roos, Davis, van de Meent, Welling 等） |
| 2026-04 | Discrete Flow Maps（2604.09784） | Harvard / Oxford / MIT / NYU（Potaptchik, Albergo 等） |
| 2026-04 | LangFlow（2604.11748） | UIUC（Ge Liu 組） |
| 2026-05 | ELF（2605.10938） | MIT（Kaiming He 等） |
| 2026-05 | S-FLM（2605.11125） | EPFL（Deschenaux & Gulcehre） |
| 2026-05 | Scaling Categorical Flow Maps（2605.07820） | Apple + Oxford |
| 2026-05 | RePlaid（2605.18530） | NVIDIA + Cornell |
| 2026-06 | FMLM+（2606.24773） | CMU + KAIST |
| 2026-06 | Masked Language Flow Models（2606.27617） | Oxford |
| 2026-06 | Hacking Generative Perplexity（2606.08417） | Franca & Tong |
| 2026-07 | FMLM⋆ fixed-point flows（2607.00714） | — |
| 2026-07 | Low Perplexity is Repetition（2607.00588） | — |
| 2026-08 | ConvergeFlow（2608.23551） | CUHK + Michigan |
| 2026-08 | Untied Self-Conditioning（2608.22244） | — |
| 2026-09 | DMD for CDLMs（2609.40235） | — |
| 2026-09 | ELF-REG（2609.29102） | — |
| 2026-09 | PlaidQ（2609.04531） | 已撤稿 |
| 2026-10 | Gumbel Straight Flow（2610.00497） | Google DeepMind Amsterdam |
| 2026-10 | Clock Diffusion（2610.00894） | NVIDIA + Cornell |
| 2026-10 | Sigma（2610.02665） | NVIDIA + Cornell + Georgia Tech |
| 2026-10 | Design Space（2610.13834） | Apple + Oxford |
| 2026-10 | Predictive-State Flows（2610.12898） | — |
| 2026-10 | LangDiT（2610.22330） | — |
| 2026-10 | Lafa（2610.28559） | Chavan, Lee, Bamba, Yoo, Hong, Raghunathan, Kim（arXiv 頁面未列機構） |

（表中 ELF、MLFM、Sigma 的來源：[arxiv](https://arxiv.org/pdf/2605.10938)、[arxiv](https://arxiv.org/pdf/2606.27617)、[arxiv](https://arxiv.org/pdf/2610.02665v1)）

研究集中出現有三個可追溯的原因。

- **flow map 理論成熟。** Boffi、Albergo、Vanden-Eijnden 在 2024–2025 年發展了 flow map matching / self-distillation 理論（2406.07507、2505.18825）。FLM、CFM、DFM 三篇幾乎同時把它移植到 one-hot / simplex 上。三篇互相引用為 concurrent work。
- **因子化誤差論點成為共同動機。** FLM 用 new-york / san-diego 的玩具例子說明：離散擴散每步獨立取樣各 token，少步時會產生 new-diego 這類錯誤組合。連續 flow 以確定性 ODE 聯合搬運整個序列，沒有這個問題。 [arxiv](https://arxiv.org/html/2602.16813v2)
- **影像擴散的工具可以直接沿用。** 例如 x-prediction、logit-normal 時間取樣、CFG、REPA、Muon、DMD。ELF 由影像擴散團隊提出。 [arxiv](https://arxiv.org/pdf/2605.10938)

注意：PlaidQ（2609.04531，Duke/Tsinghua，0.7B 程式碼模型）在 alphaXiv 頁面標示 v2 已撤稿，理由是「institutional disclosure requirements」。該文報告的 16 步 HumanEval pass@10 31.78、一步 pass@1 7.07 不應作為已確認結果引用。 [alphaxiv](https://www.alphaxiv.org/abs/2609.04531)

## 判斷：性能提升的主要來源

依證據強度排序：
1. 把去噪器輸出限制在 simplex / 凸包，並以 CE / KL 訓練（FLM、DFM 消融；LangFlow 的 collapse 分析）。→ [訓練目標](cdlm-training-objective.html)
2. 依解碼錯誤率或 loss 重新分配時間（FLM、Scaling CFM 的 λ 掃描；LangFlow 的 Gen PPL 約 1000 → 154）。→ [排程](cdlm-noise-schedule.html)
3. self-conditioning（LangFlow 154 → 81.5）及其推論修正（Untied SC：531 → 62）。→ [self-conditioning](self-conditioning-flow-lm.html)
4. flow map 自蒸餾（OWT 一步 Gen PPL：FMLM 168、GSF 115，離散蒸餾 370–5743）。→ [少步蒸餾](flow-map-few-step-distillation.html)
5. 規模化時以 AR 初始化並採 block 結構（Sigma、Lafa）。→ [規模化](cdlm-scaling.html)

這些改動多數在 CDCD（2022）已有雛形：CE 去噪器與 time warping 都在其中。2026 年的不同之處有三點：把這些設計放進 flow matching / flow map 框架並做系統消融；表示空間改為 one-hot 或受約束的 embedding；有了少步蒸餾工具。

**哪些主張可視為已確立。** 小規模（約 170M）下，連續 flow 在少步區間勝過離散擴散蒸餾，這一點有多篇獨立論文重現（FMLM、DFM、GSF、ELF+PD）。「連續擴散全面勝過離散擴散」這個說法目前沒有證據支持：
- Sigma 在 3B/8B 規模上是持平，不是勝出。
- 推理任務在低溫下仍落後。
- MCQA 的 likelihood 評分也落後。

## 建議閱讀順序

1. [訓練目標與參數化：從 MSE 到 CE 去噪器](cdlm-training-objective.html)
2. [表示空間與 rounding / decoder 問題](cdlm-representation-space.html)
3. [噪聲／時間排程](cdlm-noise-schedule.html)
4. [self-conditioning 的作用與理論解釋](self-conditioning-flow-lm.html)
5. [SDE vs ODE 與 likelihood 估計](cdlm-likelihood-estimation.html)
6. [flow map / consistency / 分佈匹配的少步生成](flow-map-few-step-distillation.html)
7. [架構與訓練配方](cdlm-architecture-recipes.html)
8. [規模化策略與大模型結果](cdlm-scaling.html)
9. [混合連續—離散方法](hybrid-continuous-discrete-lm.html)
10. [基準結果對照與評估方法的限制](cdlm-benchmarks-and-eval.html)
11. [未解決問題與後續方向](cdlm-open-problems.html)
