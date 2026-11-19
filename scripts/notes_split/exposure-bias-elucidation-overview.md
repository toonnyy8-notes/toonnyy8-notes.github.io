---
slug: exposure-bias-elucidation-overview
title: 擴散模型曝光偏差分析與 Epsilon Scaling：研究脈絡總覽
cluster: Diffusion & Generative Models
kind: hub
tags: [exposure-bias, epsilon-scaling, diffusion, lineage]
arxiv: ["2308.15321", "2006.11239", "2102.09672", "2201.06503", "1511.06732", "2301.11706", "2305.15583", "2308.05021"]
sources: [2308.15321]
related: [diffusion-exposure-bias, teacher-forcing-and-scheduled-sampling, ddpm-training-and-sampling-recap, diffusion-sampling-variance-schedule, exposure-bias-prior-mitigations, prediction-error-variance-analysis, epsilon-scaling-method, eps-prediction-ddpm-score, diffusion-forcing]
---
# 擴散模型曝光偏差分析與 Epsilon Scaling：研究脈絡總覽

本篇整理 *Elucidating the Exposure Bias in Diffusion Models*（Ning 等人，ICLR 2024，arXiv:2308.15321）所依賴的研究脈絡，依閱讀順序連到各主題筆記。曝光偏差的一般定義見 [擴散模型的曝光偏差](diffusion-exposure-bias.html)，本脈絡不重複撰寫。

## 閱讀順序
1. [曝光偏差在序列生成中的來源](teacher-forcing-and-scheduled-sampling.html)：Ranzato 等人（2015）與 Schmidt（2019）。本論文以這兩篇作為「曝光偏差」一詞的出處。
2. [DDPM 的反向取樣公式](ddpm-training-and-sampling-recap.html)：Sohl-Dickstein、Ho 等人、Song & Ermon、Song 等人的 SDE 與 DDIM。本論文第 2 節以 DDPM 的後驗與 ε 預測為起點。另可參考既有筆記 [ε-prediction 與 DDPM、score-based 模型](eps-prediction-ddpm-score.html)。
3. [取樣變異數與 noise schedule](diffusion-sampling-variance-schedule.html)：Nichol & Dhariwal、Analytic-DPM、EDM。本論文用 β̃_t 作為取樣變異數下界，並以 Analytic-DPM 的高斯近似建模預測誤差。
4. [曝光偏差的既有緩解方法](exposure-bias-prior-mitigations.html)：Ning 等人 2023（DDPM-IP）、Li 等人 2023a（shifted time steps）、Li & van der Schaar 2023、PFGM++。
5. [預測誤差與取樣變異數分析](prediction-error-variance-analysis.html)：本論文第 3 節的分析與度量 δ_t。
6. [Epsilon Scaling 方法與結果](epsilon-scaling-method.html)：本論文第 4、5 節。

## 與其他筆記的關係
Diffusion Forcing 等序列生成方法也處理訓練與推論的輸入落差，見 [Diffusion Forcing](diffusion-forcing.html)。
