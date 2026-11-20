---
slug: duo-consistency-distillation-to-discrete
title: 從 consistency distillation 到 Discrete Consistency Distillation（DCD）
cluster: Diffusion Language Models
kind: topic
tags: [consistency-models, distillation, few-step, sdtt, dcd]
arxiv: ["2506.10892", "2303.01469", "2310.14189", "2202.00512", "2410.12557", "2410.21035"]
sources: ["arXiv:2506.10892 §1、§2.3、§4.2、§5.2、§6", "各論文 arXiv 摘要"]
related: ["duo-lineage-hub", "duo-gaussian-diffusion-foundations", "duo-argmax-duality-theory", "duo-self-correction-and-correctors", "fmlm-flow-map-distillation-lineage", "fewer-steps-sampling-acceleration"]
---
# 從 consistency distillation 到 DCD

屬於 [Duo 研究脈絡總覽](duo-lineage-hub.html)。

## 連續擴散側的前人
- Salimans 與 Ho 2022，*Progressive Distillation for Fast Sampling of Diffusion Models*（arXiv:2202.00512）：逐輪減半步數的蒸餾。Duo 的「N 輪、每輪步長 δ 加倍」流程與此相近，此對應為整理者的觀察，非論文明述。
- Song 等人 2023，*Consistency Models*（arXiv:2303.01469）與 Song 與 Dhariwal 2023，*Improved Techniques for Training Consistency Models*（arXiv:2310.14189）：Duo §2.3 描述為「在雜訊分佈與資料分佈之間定義雙射映射」，建立在 PF-ODE 的確定性取樣器上。
- Frans 等人 2024，*One Step Diffusion via Shortcut Models*（arXiv:2410.12557）：Duo 與 consistency 並列，作為依賴確定性映射的少步方法。
這些方法需要確定性的雜訊到資料映射，Duo 稱 MDM 因確定性先驗而缺少此性質。

## 離散側的前人
Deschenaux 與 Gulcehre 2024，*Beyond Autoregression: Fast LLMs via Self-Distillation Through Time*（SDTT，arXiv:2410.21035）：沿隨機軌跡蒸餾 MDM；Duo 稱其少步表現嚴重退化，並以 MDLM + SDTT 為蒸餾基線。

## DCD（§4.2）
1. 取乾淨資料 x 與 Gaussian 雜訊 ε，沿 PF-ODE 軌跡取 α̃_t x + √(1 − α̃_t²) ε，再以 arg max 投影，得到 Deterministic Discrete Trajectory。
2. 取相鄰一對 (z_s, z_t)，學生看 z_t，教師看 z_s，最小化 KL(x_θ(z_t, t) ‖ x_θ⁻(z_s, s))。
3. 共 N 輪，每輪 M 步，結束後以學生權重更新教師，δ 加倍。論文以未平均的去噪模型權重為教師，並報告優於 EMA 教師（Fig. 10）。

## 論文報告的結果
OWT 上 N = 5、起始 δ = 1/512、每 M = 10K 步加倍。Ancestral 取樣器在 16 步達到基底模型 1024 步的 Gen PPL（64 倍）；加上 Greedy-Tail 為 8 步（128 倍），熵略降。T ≤ 32 時，蒸餾後的 Duo 優於以 SDTT 蒸餾的 MDLM；NFE 較高時 MDLM 較好。站內另有連續 flow 語言模型的少步蒸餾整理：[Flow map 與少步蒸餾的發展](fmlm-flow-map-distillation-lineage.html)、[減少取樣步數的相關工作](fewer-steps-sampling-acceleration.html)。
