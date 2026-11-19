# 時序 Diffusion / Flow Matching 的長序列上下文一致性：機制、理論與對 Zero-Shot Diffusion Source Separation + SMC 的建議

## TL;DR
- **訓練時最強的一致性機制是 Diffusion Forcing（arXiv:2407.01392，per-token 獨立噪聲）與其視覺後繼 Self-Forcing（arXiv:2506.08009）/ CausVid（arXiv:2412.07772）；推論時（training-free，最適合 zero-shot inverse problem）最可轉移的是 Duet Singing（arXiv:2311.07345）的 overlapping-segment autoregressive posterior sampling、MultiDiffusion（arXiv:2302.08113）的重疊區平均、以及 FIFO-Diffusion（arXiv:2405.11473）的 diagonal denoising。**
- **關鍵研究缺口：所有已發表的 SMC + diffusion 方法都沿「diffusion 噪聲軸」做粒子濾波（FPS、MCGdiff、TDS、SMCDiff）；所有沿「時間軸」的分塊/自回歸 diffusion 生成都用單軌跡 rollout，不做粒子重採樣。沿 audio 時間軸做 block-wise SMC + diffusion prior 目前無前例——這正是使用者可以佔據的原創空間。**
- **對使用者的建議：以 Duet 式 overlapping-segment autoregressive posterior sampling 為 backbone，將「跨段一致性」重寫為沿時間軸的 Feynman-Kac / 粒子濾波問題（block-wise SMC），用 speaker/instrument embedding 或前段重疊區作為 twisting potential 重加權粒子，可同時解決 identity drift 與 error accumulation，且完全 training-free、與現有 UnDiff/MSDM/ArrayDPS/DGMO prior 相容。**

---

## Key Findings

1. **兩種「序貫性」必須區分**：diffusion 有兩個時間軸——(a) 去噪軸（noise level t: T→0），(b) 資料/序列軸（audio time n）。既有 SMC-diffusion 文獻（FPS、MCGdiff、TDS、SMCDiff）全部沿 (a) 做濾波；長序列生成問題本質上是沿 (b) 的外推。使用者提的 block-wise SMC 若沿 (b)，屬於文獻空白。

2. **訓練時機制**：Diffusion Forcing、Rolling Diffusion、CausVid、Self-Forcing 皆需重訓，但能穩定外推超過訓練長度；核心共通點是「乾淨的過去 context 以零噪聲餵入，未來以較高噪聲」，避免 autoregressive rollout 的分布偏移。

3. **推論時機制（training-free）**：MultiDiffusion / Gen-L-Video（重疊窗平均）、FIFO-Diffusion（對角去噪佇列）、outpainting/inpainting continuation、Duet 的 autoregressive posterior sampling，都不需重訓，直接可套在既有 prior 上；這是 zero-shot inverse problem 設定唯一可用的一類。

4. **flow matching 端**：CosyVoice 2 的 chunk-aware causal flow matching、F5-TTS/E2-TTS 的 infilling 條件、MeanFlow 的一步生成，是 flow 端對應技術；但多屬條件生成，尚未見用於 posterior sampling。

5. **與分離的連結**：僅 Duet Singing（arXiv:2311.07345）明確在 posterior sampling 設定下處理跨段 identity 一致性；MSDM、ArrayDPS、UnDiff、DGMO 皆未系統性處理長序列跨段漂移，多在固定片段上運作。

---

## Details

### 一、訓練時的一致性機制（需重訓）

**Diffusion Forcing（Chen et al., NeurIPS 2024, arXiv:2407.01392）**
核心機制：訓練時對序列中每個 token（frame）指派「獨立、隨機的噪聲等級」，使模型能接受乾淨（零噪聲）的過去 frame 作為 context 而不產生分布偏移，達成「stable, arbitrarily long rollouts」。它統一了 next-token prediction（可變長度）與 full-sequence diffusion（可全局引導）。論文證明其優化的是「所有子序列聯合分布 likelihood 的變分下界」。實驗顯示能 rollout 超過訓練長度而 baseline 發散。
- 是否重訓：**是**（新訓練範式）
- 能否 posterior sampling：架構原生支援 guidance，理論上可導入 likelihood 引導，但原文聚焦生成與 planning，未做 inverse problem
- 計算成本：訓練成本與標準 diffusion 相當；推論可用 causal 架構 + KV cache

**Rolling Diffusion（Ruhe et al., 2024, arXiv:2402.09470）**
核心機制：sliding window 去噪，對窗內較晚（較遠未來）的 frame 指派較高噪聲，反映「未來不確定性遞增」的 arrow-of-time inductive bias（SNR(τ_{w+1}) < SNR(τ_w)）。用線性 reparameterization 使同一 inductive bias 可在任意窗位置重用，支援無限展開。在 Kinetics-600 長影片預測與混沌流體預測上優於標準 diffusion。已被延伸到音訊-影片聯合生成（RFLAV, arXiv:2503.08307 的 Rolling Flow Matching）。
- 是否重訓：**是**
- 能否 posterior sampling：未見，但 sliding window + 遞增噪聲的結構與音訊時間軸天然契合
- 計算成本：與窗大小成正比，比 full-sequence 便宜

**CausVid（Yin et al., CVPR 2025, arXiv:2412.07772）**
核心機制：把雙向 diffusion teacher 蒸餾成 causal（block-wise causal attention）autoregressive student，用 asymmetric distillation + DMD 把 50 步蒸成 4 步。相較雙向 baseline，初始延遲從 219.2 秒降到 1.3 秒（160× 加速）、吞吐量從 0.6 提升到 9.4 FPS（16×），達成串流生成。在全部 946 條 VBench-Long prompt、16 項指標上取得「最高總分 84.27」，為當時所有已驗證 VBench 提交中第一。關鍵是「asymmetric distillation」緩解 autoregressive error accumulation，使短 clip 訓練能長時生成。
- 是否重訓：**是**（需蒸餾）
- 能否 posterior sampling：可 zero-shot 做 image-to-video（因 causal 設計），但非 inverse-problem posterior sampling
- 計算成本：訓練貴（蒸餾），推論極快（few-step + KV cache）

**Self-Forcing（Huang et al., NeurIPS 2025 Spotlight, arXiv:2506.08009）**
核心機制：解決 exposure bias——訓練時就用「自己生成的前綴」做 autoregressive rollout（帶 KV cache），而非用 ground-truth context，並用 video-level holistic loss 監督整段。指出 CausVid 的缺陷：其 student 在 diffusion forcing 下用 ground-truth context，導致 concat 後不成有效影片。加 rolling KV cache 達次秒延遲、單 GPU 即時串流。
- 是否重訓：**是**
- 能否 posterior sampling：否（生成導向）
- 計算成本：訓練用 few-step + stochastic gradient truncation 平衡；推論即時
- 後繼：Causal Forcing、Rolling Forcing（arXiv:2509.25161）、LongLive 等一系列

**長上下文架構層面**：
- causal attention + KV cache（CausVid、Self-Forcing）
- SSM/Mamba 作為 backbone：Mamba（arXiv:2312.00752）在音訊波形建模上優於 SaShiMi/Transformer——原文於 4 小時 YouTubeMix 16kHz 鋼琴資料上（訓練長度 8,192→約 10⁶ samples）指出「Mamba 與 SaShiMi(S4+MLP) baseline 皆隨 context 增長而穩定改善；Mamba 全程更佳，且差距在較長長度時擴大」，線性複雜度適合長序列；已有 DiffuMamba、Video Diffusion State Space Models 等把 Mamba 當 diffusion backbone。對 16kHz 長音訊特別有吸引力。

**Flow matching 端**：
- **CosyVoice 2（arXiv:2412.10117）**：chunk-aware causal flow matching decoder，統一 streaming/non-streaming，token overlap 機制維持跨 chunk 連續性
- **F5-TTS（arXiv:2410.06885）/ E2-TTS**：非自回歸 flow matching + DiT，用 infilling（把參考音訊當條件延伸）
- **MeanFlow（Geng, Deng, Bai, Kolter & He, arXiv:2505.13447）**：學 average velocity 而非 instantaneous velocity；MeanFlow-XL/2（676M 參數、240 epochs）「以單次函數評估（1-NFE）在 ImageNet 256×256 從頭訓練達 FID 3.43」（2-NFE 為 FID 2.93）；已延伸到語音增強（MeanFlowSE, arXiv:2509.14858）。對降低 posterior sampling 每步成本有潛力

### 二、推論時的一致性機制（training-free，zero-shot 可用）

**Overlap-add / crossfade**：最簡單，但在分離的相位不連續處會產生 artifact，且無法傳遞 source identity。

**MultiDiffusion（Bar-Tal et al., ICML 2023, arXiv:2302.08113）**
核心機制：把多個重疊 crop 的 diffusion 路徑用最小平方（重疊區平均）綁定成單一一致生成——`Ψ(J_t,z)=argmin Σ‖F_i(J)−Φ(F_i(J),z)‖²`。原用於 panorama（H×4W），從同一初始噪聲融合原本不相關的路徑成連貫全景。可直接轉移到音訊時間軸（把 crop 換成時間窗）。
- 是否重訓：**否**
- 能否 posterior sampling：可與 guidance 結合
- 計算成本：與窗數線性，重疊增加成本

**Gen-L-Video（arXiv:2305.18264）/ FreeNoise / FreeLong**：training-free 用 sliding-window 合併重疊子段（temporal co-denoising）擴展短影片模型到數百 frame，保持一致性；FreeNoise 用 window-based temporal attention + noise rescheduling。

**FIFO-Diffusion（Kim et al., NeurIPS 2024, arXiv:2405.11473）**
核心機制：diagonal denoising——維持一個佇列，frame 噪聲等級沿佇列遞增；每步 dequeue 頭部全乾淨 frame、enqueue 尾部新噪聲。加 latent partitioning 減少訓練-推論 gap、lookahead denoising 利用前向參考。training-free，理論上無限長，記憶體恆定。
- 是否重訓：**否**
- 能否 posterior sampling：未見，但佇列結構與 streaming inverse problem 契合
- 計算成本：恆定記憶體，可多 GPU 平行

**Duet Singing Voices Separation（Yu et al., 2023, arXiv:2311.07345）— 與使用者最相關**
核心機制：發現單一 unconditional 單歌手 DM 做 posterior sampling 時，分離出的 singer identity 會「每隔一小段就從一個歌手切換到另一個」（因為 prior 只提供隱式音色連貫與音高分布）。解法：把混音切成 overlapping segments，**sequentially/autoregressive** 做 posterior sampling，將「當前段混音 + 前一段已分離結果的重疊部分」當條件，等於把前段結果當 inpainting condition 延伸。用 Mariani et al. 的 weakly-supervised posterior score。在 MedleyVox 資料集（Jeon et al. 2023；來自 23 首 MedleyDB v1+v2 歌曲的 381 段 / 約 1.1 小時）以不同 overlap ratio 測試，「the proposed method outperforms naive posterior sampling baseline」。
- 是否重訓：**否**（用預訓練單源 DM）
- 能否 posterior sampling：**是**——這正是 inverse problem 設定
- 計算成本：段數 × 每段 posterior sampling；overlap 越大越貴但越連貫
- 缺點：單軌跡貪婪傳遞，早期錯誤會累積（無回溯機制）；仍可能漂移

**Latent 空間長序列**：Stable Audio（arXiv:2402.04825）用 timing conditioning（duration + start time embedding）讓 latent diffusion 生成到訓練窗長度的變長音訊，解決「隨機 crop 導致樂句中間開始/結束」問題；「Stable Audio is capable of rendering stereo signals of up to 95 sec at 44.1kHz in 8 sec on an A100 GPU (40GB VRAM)」。Long-form music generation（arXiv:2404.10301）進一步延伸。InfiniteAudio、FreeAudio（arXiv:2507.08557）做 training-free 長音訊。

### 三、理論觀點

**為何固定長度訓練的 prior 外推失敗**：
- 訓練分布 vs 測試長度 mismatch：模型只見過固定窗內的統計，長序列的全局結構（如整首歌的 identity）不在訓練分布中。
- positional encoding 外推：LongDiff（CVPR 2025）分析指出，序列變長時模型「無法準確區分 frame 相對位置」（positional ambiguity）導致時序不連貫，以及「temporal correlation entropy 隨長度下降」（information dilution）。
- FIFO 的訓練-推論 gap：模型訓練時所有 frame 同噪聲等級，diagonal denoising 卻讓 frame 有不同噪聲，故需 latent partitioning 緩解。

**Autoregressive error accumulation / drift 與解法**：
- 問題：exposure bias——訓練用 ground-truth context，推論用自己的不完美輸出，誤差逐段放大。
- 解法譜系：Diffusion Forcing（乾淨 context 零噪聲餵入避免偏移）→ CausVid（asymmetric distillation）→ Self-Forcing（訓練時就自 rollout + video-level loss）→ rollout/self-resampling training。

**兩種序貫性的關係（對 SMC 至關重要）**：
- 沿 diffusion 噪聲軸的 SMC：FPS（Dou & Song, ICLR 2024, arXiv:2308.14409）證明「Bayesian posterior sampling 是 Bayesian filtering 的特例」，把去噪鏈 t=N→0 當 state-space 濾波問題，粒子數→∞ 時漸近精確。MCGdiff（Cardoso et al., ICLR 2024 oral, arXiv:2308.07983）把恢復問題寫成「從前向 diffusion 改編的 Feynman-Kac model」，SMC 沿噪聲等級求解。TDS（Wu, Trippe et al., NeurIPS 2023, arXiv:2306.17775）用 twisting function 做 reconstruction-guidance proposal，SMC 沿去噪步驟。SMCDiff（Trippe et al., ICLR 2023, arXiv:2206.04119）同理。
- 沿資料/時間軸的序貫：autoregressive 生成、chunk-by-chunk、FIFO 佇列——但這些**都是單軌跡確定性 rollout，不做粒子重採樣**。
- **關鍵洞見**：這兩條線從未被結合。沿時間軸做 SMC（粒子=不同的分離假設，沿 audio 時間塊傳播並依與觀測/前段一致性重加權），用 diffusion prior 作 proposal，是空白。

### 四、與 source separation / inverse problem 的連結與研究缺口

**已用於 posterior sampling 的一致性技術**：
- 明確者僅 Duet（overlapping segment + autoregressive posterior sampling）。
- MSDM（Mariani et al., ICLR 2024, arXiv:2302.02257）：學 4 源（Bass/Drums/Guitar/Piano）聯合分布，用 Dirac likelihood 分離；但在固定片段上，跨長序列一致性靠聯合 prior 隱式維持，未系統處理漂移。後繼有 MSG-LD（arXiv:2409.12346）、MGE-LDM（arXiv:2505.23305）等 latent 版本。
- ArrayDPS（Xu et al., ICML 2025, arXiv:2505.05657）：多通道 blind separation，用 DPS + IVA 初始化估計 relative RIR；array-agnostic、unsupervised，SDR 上超越所有 unsupervised baseline 並可比擬 supervised，但一致性來自空間 RTF 而非時間軸機制。
- UnDiff（Iashchenko et al., 2023）：首個通用音訊 inverse problem diffusion 框架（含分離、bandwidth extension、vocoding、declipping），但論文自陳分離有 failure case。
- DGMO（Lee et al., Interspeech 2025, arXiv:2506.02858）：training-free，用預訓練 audio diffusion 做 spectrogram mask 優化（test-time optimization）解 language-queried 分離，非 posterior sampling，且逐段處理。相關的 ZeroSep（arXiv:2505.23625）用 diffusion inversion 做 zero-shot 分離。

**分段 posterior sampling 維持 source identity 的現有做法與優缺點**：
| 做法 | 優點 | 缺點 |
|---|---|---|
| overlapping segments + AR conditioning（Duet） | training-free、直接、可人在環 | 單軌跡貪婪、早期錯誤累積、無回溯 |
| speaker/instrument embedding 條件化 | 明確錨定 identity | 需要 embedding 模型，且 embedding 需在 zero-shot 下可靠 |
| joint prior（MSDM） | 跨源相關性強 | 需 joint 訓練、源數固定、不可任意泛化 |

**block-wise / 沿時間軸 SMC 是否有前例**：
經專門調查，**沒有**。所有 SMC-diffusion 沿噪聲軸；TimeGrad（Rasul et al., ICML 2021, arXiv:2101.12072）沿時間軸自回歸但每步只採單一值向前 rollout（無 SMC、易累積誤差）；FIFO/Rolling/block video diffusion（如 BlockVid、SANA-Video、Rolling Forcing）沿時間軸但用 KV-cache/噪聲排程/蒸餾，無粒子重採樣。DiffPF（arXiv:2507.15716）則是相反關係——把 diffusion 放進傳統粒子濾波的狀態更新，非「用 SMC 沿時間重採樣 diffusion 生成的訊號塊」。這是明確的研究缺口。

**video 技術轉移到音訊分離的可行性與缺口**：
- Diffusion Forcing → 音訊：per-frame 獨立噪聲天然適合把「已分離的前段」當零噪聲 context，但需重訓 prior，與 zero-shot 目標衝突。
- Rolling Diffusion → 音訊：sliding window 遞增噪聲適合 streaming 分離；已有 RFLAV 用於音訊-影片，但未用於 posterior sampling。
- FIFO diagonal denoising → 音訊 streaming inverse problem：training-free，最可能直接轉移。
- 缺口：以上都是生成設定，尚無人把它們與 posterior sampling / likelihood guidance 結合做長音訊分離。

---

## Recommendations

**分階段建議：**

**階段 1（低風險，立即可做）**：以 Duet（arXiv:2311.07345）的 overlapping-segment autoregressive posterior sampling 為 baseline，套在你現有的 UnDiff/MSDM 式 prior 上。用重疊區 inpainting 條件傳遞 identity。這是 training-free、已驗證有效的起點。benchmark：跨段 SI-SDR、identity switch rate（可用 speaker/instrument verification embedding 的段間相似度量化 drift）。

**階段 2（核心創新，佔據空白）**：把階段 1 的單軌跡改為**沿時間軸的 block-wise SMC**。具體：
- 粒子 = 對「到目前為止已分離序列」的多個假設；
- 沿 audio 時間塊（非噪聲軸）傳播：每個新塊用 diffusion prior 做 posterior sampling 作為 proposal；
- twisting/reweight potential = 前段重疊區一致性 + （可選）speaker/instrument embedding 相似度 + 混音重建 likelihood；
- resample 淘汰漂移的粒子。
這同時解決 identity drift（重加權偏好一致粒子）與 error accumulation（多假設避免早期錯誤鎖死），且理論上粒子數→∞ 漸近正確（延續 FPS 的 Feynman-Kac 框架，但改到時間軸）。這是文獻空白，具原創性。可將 TDS 的 twisting 思想從噪聲軸「借」到時間軸作為理論骨架。

**階段 3（若需即時/長音訊）**：引入 FIFO-Diffusion（arXiv:2405.11473）式 diagonal denoising 佇列或 Rolling Diffusion 式遞增噪聲，做恆定記憶體的 streaming block-wise SMC。若願意重訓 prior，考慮 Diffusion Forcing / Self-Forcing 式訓練讓 prior 原生支援乾淨 context 條件化；若需降低每步成本，考慮 MeanFlow 式一步 proposal。

**與 Duet AR posterior sampling 相比的優劣**：
- 優：block-wise SMC 保留多假設與回溯能力（resample），對 monotimbral（同音色）源的 identity drift 更 robust；有漸近正確性保證。
- 劣：計算成本 = 粒子數 × 段數 × 每段 sampling，比 Duet 單軌跡貴 N_particles 倍；需設計低方差 proposal 與有效 potential，否則粒子退化（weight collapse）。緩解：少粒子 + GPU 平行（如 LD-SMC, arXiv:2502.05908 所述）。

**會改變建議的門檻**：若階段 1 的 identity switch rate 已足夠低（如 < 5%），則 SMC 的邊際效益可能不值其成本，應改投資 better likelihood/guidance；若 drift 嚴重且 embedding 條件化無效，則 SMC 的多假設優勢最大化，值得全力投入階段 2。

---

## Caveats
- **時效與版本**：本報告多篇為 2024–2026 的快速演進領域（video streaming diffusion 尤甚），部分後繼工作（Causal Forcing、Rolling Forcing、LongLive 等）仍在 arXiv 階段，結論可能隨新工作變動。
- **負面發現的信心度**：「沿時間軸 SMC + diffusion prior 無前例」是基於廣泛但非窮盡的搜尋（子代理已交叉檢查 TDS/FPS/MCGdiff/SMCDiff/TimeGrad/FIFO/block-video 等）；建議投稿前再做一次針對性檢索確認，尤其 2025–2026 的 inference-time-scaling 文獻。
- **posterior sampling 可轉移性**：多數 video 一致性技術是生成導向，轉到 inverse-problem posterior sampling 的可行性多為推論，缺乏直接實驗證據，屬研究假設而非已證事實。
- **評估指標爭議**：generative separation 的 BSS-Eval（SDR/SIR/SAR）對相位與非線性失真敏感、對生成式系統可能不可靠；量化 identity drift 尚無標準指標，需自行設計（如 embedding 段間相似度）。
- **MeanFlow/一步生成用於 posterior sampling**：目前主要在無條件生成與語音增強，用於 inverse problem 的引導穩定性未經充分驗證。
- **FPS arXiv 編號**：不同來源對 Dou & Song 的 FPS 編號略有出入（本報告採 2308.14409 / OpenReview id=tplXNcHZs1），引用前建議再核對。