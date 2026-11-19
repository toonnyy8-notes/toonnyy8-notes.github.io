---
slug: causal-attention-diffusion-overview
title: Causal Attention 與 Diffusion 結合於連續序列生成（音訊/語音、時間序列）總覽
cluster: Diffusion & Generative Models
kind: hub
tags: [causal-attention, streaming, audio, time-series, overview]
sources: [causal-attention-diffusion.md]
arxiv: ["2407.01392", "2412.10117", "2509.20846"]
related: ["noncausal-diffusion-baselines", "diffusion-forcing", "rolling-diffusion", "streaming-flow-matching-speech", "causvid-self-forcing"]
---
# Causal Attention 與 Diffusion 結合於連續序列生成（音訊/語音、時間序列）總覽

## TL;DR
- **「純 Diffusion + Causal Attention」是一個真實但狹窄的研究縫隙**：絕大多數成功的非自回歸（non-AR）擴散音訊／時間序列模型（DiffWave、WaveGrad、AudioLDM、CSDI、SSSD、Diffusion-TS）刻意使用**雙向（非因果）**架構，因為因果遮罩會降低品質（見 [非因果擴散對照組](noncausal-diffusion-baselines.html)）；真正融入因果性的，幾乎都是為了**串流／即時生成**而把擴散「自回歸化」的混合範式（Diffusion Forcing、Rolling Diffusion、CosyVoice 2、StreamFlow、Self-Forcing）。
- **代表性案例**：時間序列以 [Diffusion Forcing](diffusion-forcing.html)（NeurIPS 2024，因果 RNN/GRU 骨幹）與 [Rolling Diffusion](rolling-diffusion.html) 為核心；語音／音訊以 CosyVoice 2 的 chunk-aware causal flow matching、StreamFlow 的 block-wise causal attention mask 為核心（見 [串流 flow matching](streaming-flow-matching-speech.html)）；視訊（機制可遷移到音訊）以 [Self-Forcing](causvid-self-forcing.html) 的因果注意力 + rolling KV cache 為核心。
- **務實建議**：若你要的是「純 diffusion、無 AR、又要因果」，最接近的是 **Diffusion Forcing**（透過 per-token 獨立噪聲等級實現因果而非逐步解碼）與 **Rolling Diffusion**（滑動窗口噪聲排程）；但須注意這些方法在「token/frame 軸」上仍帶有滾動式推理，學術上常被歸為 AR-diffusion 混合，純粹主義者需謹慎界定。

## Key Findings
1. **核心張力：因果性會傷害純擴散的品質。** DiffWave 作者明確報告因果膨脹卷積大幅降低音質（細節見 [非因果擴散對照組](noncausal-diffusion-baselines.html)）。當不需要串流時，純擴散模型放棄因果性以換取品質。
2. **時間序列主流擴散模型多為非因果。** CSDI、SSSD、Diffusion-TS 都是非 AR、非因果。
3. **真正的「因果 + 擴散」集中在串流範式。** Diffusion Forcing（arXiv 2407.01392）是最重要的架構創新：訓練擴散模型對每個 token 賦予獨立噪聲等級，用因果骨幹做「下一 token 預測」但無需完全擴散過去 token。
4. **語音串流以 causal block attention 為主流機制。** CosyVoice 2（arXiv 2412.10117）的 chunk-aware causal flow matching 在訓練時隨機抽樣四種遮罩（non-causal、full-causal、chunk-M、chunk-2M）；StreamFlow（NeurIPS 2025）用 block-wise guided attention mask。
5. **TimeGrad 是必須排除的 AR baseline。** TimeGrad（ICML 2021）以 RNN（LSTM/GRU）逐時步自回歸地條件化擴散，是明確的 AR-diffusion，須排除於「純擴散」範疇（數字見 [Diffusion Forcing](diffusion-forcing.html)）。

## 問題界定：三種「因果性」的混淆
研究這個題目首先要釐清「causal」在文獻中有三種不同意涵，極易混淆：
1. **因果注意力遮罩（causal attention masking）**：架構層級，限制 token t 只能注意 ≤ t 的 token。這是本主題的核心。
2. **因果推論（causal inference, Pearl's ladder）**：如 CaTSG（arXiv 2509.20846）的 backdoor adjustment、counterfactual generation，與架構因果性無關。
3. **時間序列因果發現（causal discovery）**：如 CausalDiffusion／DiffCATS，生成帶有 ground-truth 因果圖的序列。

一個關鍵的負面發現是：**對於不需串流的純擴散生成，因果遮罩通常有害**，因為擴散的優勢正是全序列雙向去噪。因此真正有意義的「因果 + 擴散」研究幾乎都源於**串流／即時／長序列外推**的需求。

## 架構比較表

| 模型 | 年份/venue | 領域 | 骨幹 | 因果機制 | 是否純擴散/無AR | 關鍵指標 | 對AR比較 |
|---|---|---|---|---|---|---|---|
| **Diffusion Forcing** | NeurIPS 2024 | 時間序列/規劃/視訊 | 因果 RNN(GRU)+U-Net | per-token獨立噪聲(noise as masking)；CDF因果結構 | 擴散，但token軸帶滾動推理（混合） | CRPS_sum: Solar 0.289, Electricity 0.023 | 勝TimeGrad多數；與ScoreGrad打平 |
| **Rolling Diffusion** | ICML 2024 | 視訊/流體/天氣 | U-Net/DiT | 滑動窗口漸進噪聲排程 | 擴散，軟自回歸 | 複雜動態下優於標準擴散 | ERDM勝conditional AR-EDM |
| **CSDI** | NeurIPS 2021 | 時間序列補值 | DiffWave+2D注意力 | 無時間因果（conditional masking） | 純擴散，非AR | CRPS改善40–70%, MAE改善5–20% | 勝AR/unconditional baseline |
| **SSSD** | TMLR 2022 | 時間序列補值/預測 | S4狀態空間 | 無（S4可因果但此處雙向補值） | 純擴散，非AR | blackout情境SOTA | 勝先前方法 |
| **Diffusion-TS** | ICLR 2024 | 時間序列生成 | enc-dec Transformer | 無因果遮罩 | 純擴散，非AR | 多項SOTA | — |
| **DiffWave** | ICLR 2021 | 波形 | 雙向膨脹卷積 | 刻意避免因果（因果大幅降品質） | 純擴散，非AR | MOS 4.44 | 匹配WaveNet(4.43) |
| **WaveGrad** | ICLR 2021 | 波形 | GAN-TTS式UBlock | 無 | 純擴散，非AR | 6步高保真 | 匹配AR baseline |
| **AudioLDM** | ICML 2023 | 文字→音訊 | SD U-Net+cross-attn | 無（非因果自注意力） | 純擴散，非AR | SOTA FD | — |
| **CosyVoice 2** | 2024 | 串流TTS | LLM+causal flow matching DiT | chunk-aware causal attention mask（4種） | flow matching；LM部分為AR | WER 2.47%, SS 0.745 | 串流近乎無損（作者自報） |
| **StreamFlow** | NeurIPS 2025 | 串流語音解碼 | Scale-DiT flow matching(OT-CFM) | block-wise causal attention mask | flow matching，非token-AR | PESQ/ViSQOL優於串流baseline | 接近非串流 |
| **Self-Forcing** | 2025 | 視訊（可遷移音訊） | 因果DiT+rolling KV cache | causal attention mask | 擴散蒸餾為AR | 次秒延遲即時 | 匹配/超越非因果擴散 |
| **TimeGrad（排除）** | ICML 2021 | 時間序列預測 | RNN+擴散 | RNN逐時步AR | **AR擴散，須排除** | CRPS_sum: Solar 0.287 | 是baseline本身 |

## 建議
1. **若目標是嚴格「純擴散 + 因果 + 無 AR」**：最佳起點是 **Diffusion Forcing**（理論上以 noise-as-masking 取代逐步解碼）與 **Rolling Diffusion**（漸進噪聲排程）。但務必在論文中明確界定：這些方法在 token/frame 軸上仍有滾動推理，學界常稱其為「AR-diffusion 混合」或「diffusion-forcing 範式」。若你的「無 AR」定義是「無逐步離散 token 解碼且無 teacher forcing」，它們符合；若定義是「單次全序列 forward」，則只有 CSDI/SSSD/Diffusion-TS/DiffWave 符合，但那些沒有因果注意力。
2. **音訊/語音方向**：見 [串流 flow matching](streaming-flow-matching-speech.html)。
3. **時間序列方向（明確的研究空白）**：見 [Diffusion Forcing](diffusion-forcing.html) 中「以 causal-masked Transformer 替換 RNN 骨幹」。
4. **評估基準與門檻**：音訊用 MOS/UTMOS/PESQ/ViSQOL + FAD；時間序列用 CRPS_sum/MAE/RMSE。決策門檻：若因果版本相對非因果版本品質下降 < 5%（如 CosyVoice 2 宣稱串流「近乎無損」），則因果化值得；若下降顯著（如 DiffWave 的因果卷積導致音質大幅下降），則應放棄因果改走雙向 + 後處理串流。
5. **若研究動機是即時/串流**：因果是剛需，採 Self-Forcing 式的「雙向教師蒸餾為因果學生 + rolling KV cache」是當前最強範式，可遷移到音訊 latent。若動機僅是離線生成品質，**不要**引入因果注意力。

## 一般注意事項
- **「純擴散 + 因果注意力 + 連續序列」的交集本身很小**，且多數成員嚴格說是「AR-diffusion 混合」（Diffusion Forcing、Rolling Diffusion、Self-Forcing、CosyVoice 2）。完全不帶任何滾動/區塊自回歸、又用因果注意力遮罩、又是連續資料的「純粹」案例，在現有文獻中幾乎不存在——這反映了一個真實的領域結構，而非搜尋遺漏。
- **Flow matching vs diffusion**：CosyVoice 2、StreamFlow、BinauralFlow 屬 flow matching（conditional flow matching / OT-CFM），是擴散的近親但技術上有別，若報告需嚴格限定 DDPM/score-based 須另作說明。
- **多義詞「causal」**：時間序列文獻大量出現的 CaTSG、CausalDiffusion、DiffCATS 是 Pearl 因果推論／因果發現，與因果注意力無關，不應誤納。
- 部分數據（如 CosyVoice 2 的 WER/SS、StreamFlow 的相對優勢）來自論文作者自報，缺乏獨立第三方複現；應視為作者宣稱。
- 本領域進展極快（多篇 2025–2026 arXiv 預印本如 OmniForcing、Hallo-Live、Forcing-KV、ERDM），部分未經同行評審，引用時需標註預印本狀態。

## 建議閱讀順序
1. [非因果擴散對照組：DiffWave、WaveGrad、AudioLDM、CSDI、SSSD、Diffusion-TS](noncausal-diffusion-baselines.html)
2. [Diffusion Forcing](diffusion-forcing.html)
3. [Rolling Diffusion](rolling-diffusion.html)
4. [串流 flow matching（CosyVoice 2、StreamFlow 等）](streaming-flow-matching-speech.html)
5. [CausVid 與 Self-Forcing](causvid-self-forcing.html)
