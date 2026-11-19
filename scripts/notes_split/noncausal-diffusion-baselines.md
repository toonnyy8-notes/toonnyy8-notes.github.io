---
slug: noncausal-diffusion-baselines
title: 非因果擴散對照組：DiffWave、WaveGrad、AudioLDM、CSDI、SSSD、Diffusion-TS
cluster: Diffusion & Generative Models
kind: topic
tags: [diffwave, wavegrad, audioldm, csdi, sssd, diffusion-ts, non-autoregressive, bidirectional]
sources: [causal-attention-diffusion.md]
arxiv: ["2009.09761", "2009.00713", "2301.12503", "2107.03502", "2208.09399", "2403.01742"]
related: ["causal-attention-diffusion-overview", "diffusion-forcing", "streaming-flow-matching-speech"]
---
# 非因果擴散對照組：DiffWave、WaveGrad、AudioLDM、CSDI、SSSD、Diffusion-TS

屬於 [Causal Attention × Diffusion 總覽](causal-attention-diffusion-overview.html)。

## 核心張力：因果性會傷害純擴散的品質
DiffWave 作者在論文 footnote 中明確報告：「Empirically, we found the causal dilated convolution architecture leads to much worse audio quality in DiffWave」，並在正文說明採用雙向膨脹卷積架構「because there is no autoregressive generation constraint」。這是整個領域的關鍵發現：當不需要串流時，純擴散模型放棄因果性以換取品質。

## 音訊/語音
- **DiffWave**（Kong, Ping, Huang, Zhao, Catanzaro, ICLR 2021，arXiv 2009.09761）：非 AR、雙向膨脹卷積；原論文 verbatim「It matches the strong WaveNet vocoder in terms of speech quality (MOS: 4.44 vs. 4.43), while synthesizing orders of magnitude faster」（small DiffWave 2.64M 參數、22.05 kHz 達 MOS 4.35）。明確發現因果膨脹卷積會大幅降低音質。
- **WaveGrad**（Chen et al., ICLR 2021，arXiv 2009.00713）：非 AR，6 步即可生成高保真音訊；架構源於 GAN-TTS 生成器（UBlock），無因果注意力，宣稱可匹配強 likelihood-based AR baseline。
- **AudioLDM / AudioLDM 2**（Liu et al., ICML 2023，arXiv 2301.12503）：Stable Diffusion U-Net + cross-attention（文字條件），自注意力為非因果；AudioLDM 在單 GPU 上以 AudioCaps 訓練達 SOTA Fréchet 距離。
- **NaturalSpeech 2、E1 TTS、其他 non-AR diffusion TTS**：以 duration predictor 取代 AR 對齊，皆非因果。

## 時間序列
- **CSDI**（Tashiro, Song, Song, Ermon, NeurIPS 2021，arXiv 2107.03502）：條件式 score-based 擴散補值，骨幹採 DiffWave 架構 + 2D 注意力（時間 Transformer + 特徵 Transformer）。注意力為**非因果全域**，因果性僅來自 conditional masking（觀測 vs 待補）。原論文摘要：「On healthcare and environmental data, CSDI improves by 40-70% over existing probabilistic imputation methods（會議版作 40-65%）... deterministic imputation by CSDI reduces the error by 5-20%」，評測於 PhysioNet Challenge 2012 與北京空氣品質（PM2.5）資料集。
- **SSSD**（Lopez Alcaraz & Strodthoff, TMLR 2022，arXiv 2208.09399）：以 S4 結構化狀態空間模型取代注意力捕捉長程依賴，僅對待補區域注入擴散噪聲；在 blackout-missing 與 forecasting 情境表現突出。
- **Diffusion-TS**（Yuan & Qiao, ICLR 2024，arXiv 2403.01742）：encoder-decoder Transformer + 季節-趨勢分解 + Fourier loss，直接重建樣本而非噪聲；明確自稱「non-autoregressive diffusion model」，無因果遮罩。

這三者證明了**純擴散時間序列生成的主流選擇是非因果架構**。若「無 AR」定義為「單次全序列 forward」，只有 CSDI/SSSD/Diffusion-TS/DiffWave 符合，但它們沒有因果注意力。因果路線見 [Diffusion Forcing](diffusion-forcing.html)、[串流 flow matching](streaming-flow-matching-speech.html)。
