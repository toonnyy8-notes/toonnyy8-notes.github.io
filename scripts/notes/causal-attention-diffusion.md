# 深度分析：Causal Attention 與 Diffusion Model 結合於連續性序列生成（音訊/語音與時間序列）

## TL;DR
- **「純 Diffusion + Causal Attention」是一個真實但狹窄的研究縫隙**：絕大多數成功的非自回歸（non-AR）擴散音訊／時間序列模型（DiffWave、WaveGrad、AudioLDM、CSDI、SSSD、Diffusion-TS）刻意使用**雙向（非因果）**架構，因為因果遮罩會降低品質；真正融入因果性的，幾乎都是為了**串流／即時生成**而把擴散「自回歸化」的混合範式（Diffusion Forcing、Rolling Diffusion、CosyVoice 2、StreamFlow、Self-Forcing）。
- **代表性案例**：時間序列以 Diffusion Forcing（NeurIPS 2024，因果 RNN/GRU 骨幹）與 Rolling Diffusion 為核心；語音／音訊以 CosyVoice 2 的 chunk-aware causal flow matching、StreamFlow 的 block-wise causal attention mask 為核心；視訊（機制可遷移到音訊）以 Self-Forcing 的因果注意力 + rolling KV cache 為核心。
- **務實建議**：若你要的是「純 diffusion、無 AR、又要因果」，最接近的是 **Diffusion Forcing**（透過 per-token 獨立噪聲等級實現因果而非逐步解碼）與 **Rolling Diffusion**（滑動窗口噪聲排程）；但須注意這些方法在「token/frame 軸」上仍帶有滾動式推理，學術上常被歸為 AR-diffusion 混合，純粹主義者需謹慎界定。

## Key Findings

1. **核心張力：因果性會傷害純擴散的品質。** DiffWave 作者在論文 footnote 中明確報告：「Empirically, we found the causal dilated convolution architecture leads to much worse audio quality in DiffWave」，並在正文說明採用雙向膨脹卷積架構「because there is no autoregressive generation constraint」。這是整個領域的關鍵發現：當不需要串流時，純擴散模型放棄因果性以換取品質。

2. **時間序列主流擴散模型多為非因果。** CSDI（NeurIPS 2021）使用 2D 注意力（時間 Transformer + 特徵 Transformer），其時間注意力是非因果的全域注意力，透過條件遮罩（conditional masking）區分觀測值與待補值，而非時間因果遮罩。SSSD（TMLR 2022）以 S4 結構化狀態空間取代注意力；Diffusion-TS（ICLR 2024）用 encoder-decoder Transformer 做全序列重建。這些都是非 AR、非因果。

3. **真正的「因果 + 擴散」集中在串流範式。** Diffusion Forcing（arXiv 2407.01392）是最重要的架構創新：訓練擴散模型對每個 token 賦予獨立噪聲等級，用因果骨幹做「下一 token 預測」但無需完全擴散過去 token。

4. **語音串流以 causal block attention 為主流機制。** CosyVoice 2（arXiv 2412.10117）的 chunk-aware causal flow matching 在訓練時隨機抽樣四種遮罩（non-causal、full-causal、chunk-M、chunk-2M）；StreamFlow（NeurIPS 2025）用 block-wise guided attention mask。

5. **TimeGrad 是必須排除的 AR baseline。** TimeGrad（ICML 2021）以 RNN（LSTM/GRU）逐時步自回歸地條件化擴散，是明確的 AR-diffusion，須排除於「純擴散」範疇。

## Details

### 一、問題界定：三種「因果性」的混淆

研究這個題目首先要釐清「causal」在文獻中有三種不同意涵，極易混淆：

1. **因果注意力遮罩（causal attention masking）**：架構層級，限制 token t 只能注意 ≤ t 的 token。這是本報告的核心。
2. **因果推論（causal inference, Pearl's ladder）**：如 CaTSG（arXiv 2509.20846）的 backdoor adjustment、counterfactual generation，與架構因果性無關。
3. **時間序列因果發現（causal discovery）**：如 CausalDiffusion／DiffCATS，生成帶有 ground-truth 因果圖的序列。

本報告聚焦第 1 種。一個關鍵的負面發現是：**對於不需串流的純擴散生成，因果遮罩通常有害**，因為擴散的優勢正是全序列雙向去噪。因此真正有意義的「因果 + 擴散」研究幾乎都源於**串流／即時／長序列外推**的需求。

### 二、時間序列領域

#### Diffusion Forcing（最核心案例）
- **論文**：Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, Vincent Sitzmann，"Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion"，NeurIPS 2024，arXiv 2407.01392。
- **核心架構**：主要實作用的是**因果 RNN（GRU）骨幹**，非因果 Transformer。論文原文：「We implement DF for sequence generation as Causal Diffusion Forcing (CDF), in which future tokens depend on past ones via a causal architecture. We train the model to denoise all tokens of a sequence at once, with an independent noise level」，並註明「Our current causal implementation is based on a small RNN」。Transformer 版本（含 causal attention mask）僅在 Appendix B.1 作為未來工作討論。架構細節：U-Net 的輸出餵入 GRU，GRU 隱藏狀態 z_{t-1} 攜帶過去資訊，以馬可夫方式更新；非空間資料則以 residual MLP 取代 U-Net。
- **為何不需 AR 逐步解碼**：關鍵創新是「noise as masking」——每個 token 有獨立隨機噪聲等級，零噪聲=乾淨（已知），高噪聲=被遮罩（待生成）。因此模型可以同時是全序列擴散與下一 token 預測，不必逐 token 順序解碼，同時又能做長程 guidance（這是純 next-token 模型做不到的）。
- **實驗結果（CRPS_sum，越低越好，五次運行均值±標準差）**：Exchange 0.003±0.001、Solar 0.289±0.002、Electricity 0.023±0.001、Traffic 0.040±0.004、Taxi 0.075±0.002、Wikipedia 0.085±0.007。
- **與 AR baseline 比較**：主要對手 TimeGrad（AR 擴散，teacher forcing）：Exchange 0.006、Solar 0.287、Electricity 0.021、Traffic 0.044、Taxi 0.114、Wikipedia 0.049。Diffusion Forcing 在多數資料集勝出，但與 ScoreGrad sub-VP SDE「整體打平」，且在 Wikipedia 上僅排第四。規劃任務（D4RL Maze2D/Multi2D）Diffusion Forcing 單任務平均 reward 141.7 對 Diffuser 119.5、多任務 146.2 對 129.4；真實 Franka 機器人模仿學習成功率 80%（遮擋/噪聲下降至 76%，對照 next-frame diffusion 48%）。
- **局限**：主實作仍是 RNN（Transformer 因果版僅為未來工作）；視訊實驗無量化指標（僅定性宣稱可穩定 rollout 至 1000+ frames，訓練 72 frames 可外推至 180 甚至 2000 frames）；CSDI 並非其 baseline；作者自承時間序列非其核心應用，episode reward 指標可能對 Diffuser 有利而不完全反映規劃品質。

#### Rolling Diffusion Models
- **論文**：Ruhe, Heek, Salimans, Hoogeboom，"Rolling Diffusion Models"，arXiv 2402.09470（ICML 2024）。
- **核心架構**：滑動窗口去噪——將全域擴散時間重參數化為「frame-dependent local time」，越靠未來的 frame 賦予越高噪聲，反映未來的不確定性遞增。這隱含了時間因果結構（過去比未來更確定／更乾淨）。
- **為何不需 AR**：透過漸進噪聲排程而非逐步解碼來實現時間方向性；但推理時以滑動窗口前進，本質上是一種「軟自回歸」。
- **應用與結果**：在 Kinetics-600 視訊預測與混沌流體動力學預測上，當時間動態複雜時優於標準擴散。延伸工作 Elucidated Rolling Diffusion Models（ERDM，NVIDIA，arXiv 2506.20024）將 EDM 的預條件/取樣器原則移植到 rolling 設定，用於 2D Navier-Stokes 與 ERA5 全球天氣預報（1.5°），優於 conditional autoregressive EDM baseline。
- **局限**：當時間動態簡單時優勢不明顯；推理仍帶滾動性質。

#### CSDI、SSSD、Diffusion-TS（對照組：非因果純擴散）
- **CSDI**（Tashiro, Song, Song, Ermon, NeurIPS 2021，arXiv 2107.03502）：條件式 score-based 擴散補值，骨幹採 DiffWave 架構 + 2D 注意力（時間 Transformer + 特徵 Transformer）。注意力為**非因果全域**，因果性僅來自 conditional masking（觀測 vs 待補）。原論文摘要：「On healthcare and environmental data, CSDI improves by 40-70% over existing probabilistic imputation methods（會議版作 40-65%）... deterministic imputation by CSDI reduces the error by 5-20%」，評測於 PhysioNet Challenge 2012 與北京空氣品質（PM2.5）資料集。
- **SSSD**（Lopez Alcaraz & Strodthoff, TMLR 2022，arXiv 2208.09399）：以 S4 結構化狀態空間模型取代注意力捕捉長程依賴，僅對待補區域注入擴散噪聲；在 blackout-missing 與 forecasting 情境表現突出。
- **Diffusion-TS**（Yuan & Qiao, ICLR 2024，arXiv 2403.01742）：encoder-decoder Transformer + 季節-趨勢分解 + Fourier loss，直接重建樣本而非噪聲；明確自稱「non-autoregressive diffusion model」，無因果遮罩。
- 這三者證明了**純擴散時間序列生成的主流選擇是非因果架構**。

### 三、音訊/語音領域

#### 對照組：純非因果擴散聲碼器／TTS
- **DiffWave**（Kong, Ping, Huang, Zhao, Catanzaro, ICLR 2021，arXiv 2009.09761）：非 AR、雙向膨脹卷積；原論文 verbatim「It matches the strong WaveNet vocoder in terms of speech quality (MOS: 4.44 vs. 4.43), while synthesizing orders of magnitude faster」（small DiffWave 2.64M 參數、22.05 kHz 達 MOS 4.35）。**明確發現因果膨脹卷積會大幅降低音質**（見 Key Finding 1）。
- **WaveGrad**（Chen et al., ICLR 2021，arXiv 2009.00713）：非 AR，6 步即可生成高保真音訊；架構源於 GAN-TTS 生成器（UBlock），無因果注意力，宣稱可匹配強 likelihood-based AR baseline。
- **AudioLDM / AudioLDM 2**（Liu et al., ICML 2023，arXiv 2301.12503）：Stable Diffusion U-Net + cross-attention（文字條件），自注意力為非因果；AudioLDM 在單 GPU 上以 AudioCaps 訓練達 SOTA Fréchet 距離。
- **NaturalSpeech 2、E1 TTS、其他 non-AR diffusion TTS**：以 duration predictor 取代 AR 對齊，皆非因果。

#### 真正融入因果性的串流案例
- **CosyVoice 2**（Du et al., Alibaba, arXiv 2412.10117）：chunk-aware **causal** flow matching decoder（擴散家族的 flow matching）。訓練時隨機抽樣四種注意力遮罩：non-causal（離線）、full-causal（僅過去、極低延遲）、chunk-M、chunk-2M（過去+少量未來）。這實現「單一模型同時支援串流與非串流」，並以多遮罩訓練達成隱式自蒸餾；前身用 look-ahead 卷積提供有限未來資訊。LibriSpeech test-clean WER 2.47%、speaker similarity 0.745，論文宣稱串流模式「virtually lossless synthesis quality in the streaming mode」（作者自報）。
- **StreamFlow**（Dake Guo, Jixun Yao, Linhan Ma, He Wang, Lei Xie，提交 2025-06-30，arXiv 2506.23986，NeurIPS 2025）：streaming flow matching + DiT。原文：「we propose StreamFlow, a novel neural architecture that facilitates streaming flow matching with diffusion transformers (DiT)... we introduce block-wise attention masks that enable the current block to receive information from the previous or subsequent block」。序列切塊，當前塊只能接收前/後鄰塊資訊；不同 DiT 層的遮罩階層式組合以調控感受野，採 OT-CFM（optimal-transport conditional flow matching）。並提出 Scale-DiT（adaptive layernorm + learnable residual scaling）。訓練用「causal noising」：固定過去 context、遮罩串流窗口以學習嚴格因果表徵。客觀指標（PESQ、STOI、ViSQOL、UTMOS）優於其他串流方法，接近非串流方法。
- **Causal Diffusion Models for Generalized Speech Enhancement / Stream.FM**（語音增強/復原）：frame-causal flow-based 模型，演算法延遲 32 ms、總延遲 48 ms，causal U-Net + causal convolution（僅在頻率軸降採樣以降延遲），無 look-ahead。
- **BinauralFlow**（arXiv 2505.22865）：causal U-Net + conditional flow matching 做串流雙耳語音合成，將雙耳渲染視為生成而非回歸問題。

#### 視訊/音視訊（機制可直接遷移到音訊 latent）
- **Self-Forcing**（Huang et al., arXiv 2506.08009）：將雙向 DiT 蒸餾為**因果注意力遮罩** + rolling KV cache 的自回歸擴散，達到次秒延遲即時串流，品質匹配甚至超越較慢的非因果擴散。基於 Wan2.1-T2V-1.3B + Flow Matching；初始化沿用 CausVid 的 causal attention masking 微調，訓練時 rollout 自生成 context 以縮小 train-test gap。
- **OmniForcing**（arXiv 2603.11647）、**Hallo-Live**（arXiv 2604.23632）：音視訊串流，使用 asymmetric block-causal alignment、Audio Sink Token、Future-Expanding Attention（讓 video block 存取同步音訊 + 短程未來音素線索以解決因果生成的發音延遲）。
- **Streaming Co-Speech Gestures via Accelerated Rolling Diffusion**（arXiv 2503.10488）：把 Rolling Diffusion 用於串流手勢生成（連續運動序列），以結構化漸進噪聲排程實現連續長序列生成。

### 四、架構比較表

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

## Recommendations

1. **若目標是嚴格「純擴散 + 因果 + 無 AR」**：最佳起點是 **Diffusion Forcing**（理論上以 noise-as-masking 取代逐步解碼）與 **Rolling Diffusion**（漸進噪聲排程）。但務必在論文中明確界定：這些方法在 token/frame 軸上仍有滾動推理，學界常稱其為「AR-diffusion 混合」或「diffusion-forcing 範式」。若你的「無 AR」定義是「無逐步離散 token 解碼且無 teacher forcing」，它們符合；若定義是「單次全序列 forward」，則只有 CSDI/SSSD/Diffusion-TS/DiffWave 符合，但那些沒有因果注意力。

2. **音訊/語音方向**：採用 **CosyVoice 2 的 chunk-aware causal flow matching** 或 **StreamFlow 的 block-wise causal attention mask** 作為架構藍本——這是目前唯一在連續音訊上同時做到「擴散家族 + 顯式因果注意力遮罩 + 高品質」的成熟路線。注意 flow matching 是擴散的連續正規化流推廣，若審稿人嚴格區分「diffusion vs flow matching」需先界定。

3. **時間序列方向（明確的研究空白）**：若需因果，把 **Diffusion Forcing 的因果 RNN 骨幹替換為 causal-masked Transformer**（其 Appendix B.1 已勾勒方向但未實作）是一個明確的貢獻點。Benchmark 用 CRPS_sum，對照 TimeGrad（AR）、ScoreGrad、CSDI。

4. **評估基準與門檻**：音訊用 MOS/UTMOS/PESQ/ViSQOL + FAD；時間序列用 CRPS_sum/MAE/RMSE。決策門檻：若因果版本相對非因果版本品質下降 < 5%（如 CosyVoice 2 宣稱串流「近乎無損」），則因果化值得；若下降顯著（如 DiffWave 的因果卷積導致音質大幅下降），則應放棄因果改走雙向 + 後處理串流。

5. **若研究動機是即時/串流**：因果是剛需，採 Self-Forcing 式的「雙向教師蒸餾為因果學生 + rolling KV cache」是當前最強範式，可遷移到音訊 latent。若動機僅是離線生成品質，**不要**引入因果注意力。

## Caveats
- **「純擴散 + 因果注意力 + 連續序列」的交集本身很小**，且多數成員嚴格說是「AR-diffusion 混合」（Diffusion Forcing、Rolling Diffusion、Self-Forcing、CosyVoice 2）。完全不帶任何滾動/區塊自回歸、又用因果注意力遮罩、又是連續資料的「純粹」案例，在現有文獻中幾乎不存在——這反映了一個真實的領域結構，而非搜尋遺漏。
- **Flow matching vs diffusion**：CosyVoice 2、StreamFlow、BinauralFlow 屬 flow matching（conditional flow matching / OT-CFM），是擴散的近親但技術上有別，若報告需嚴格限定 DDPM/score-based 須另作說明。
- **多義詞「causal」**：時間序列文獻大量出現的 CaTSG、CausalDiffusion、DiffCATS 是 Pearl 因果推論／因果發現，與因果注意力無關，不應誤納。
- **Diffusion Forcing 的時間序列結果有限**：作者自承時間序列非核心應用，與 ScoreGrad 打平而非全面超越，且未與 CSDI 直接比較；視訊實驗無量化指標。
- 部分數據（如 CosyVoice 2 的 WER/SS、StreamFlow 的相對優勢）來自論文作者自報，缺乏獨立第三方複現；應視為作者宣稱。
- 本領域進展極快（多篇 2025–2026 arXiv 預印本如 OmniForcing、Hallo-Live、Forcing-KV、ERDM），部分未經同行評審，引用時需標註預印本狀態。