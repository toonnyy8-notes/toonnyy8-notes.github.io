# 2026 年連續 flow / flow matching 語言模型研究近況，以及與 2022–2024 年連續擴散語言模型的差異

2026 年連續語言模型的性能提升有幾個來源，各來源都有消融實驗支撐：(1) 把訓練目標由 MSE 回歸改為以 softmax 去噪器預測乾淨 token 的 cross-entropy；(2) 把時間或噪聲排程改成依「解碼錯誤率」或資訊量均勻分配；(3) 用 self-conditioning，並修正它在推論時的偏差；(4) 用 flow map 自蒸餾換取少步生成；(5) 規模化時從預訓練 AR 模型初始化，並採 block 化與 KV cache。資料顯示，這些改動使連續模型在小規模基準上追平離散擴散，在 1–8 步的少步區間有明確優勢。在 3B–8B 規模，連續模型的表現已追平同算力的離散擴散，但仍未超過 AR 模型，評估指標本身也有被操縱的問題。 [arxiv](https://arxiv.org/pdf/2610.02665v1)

## TL;DR

- **改了什麼：** 主要修改有五項：
  - 訓練目標由 MSE / ε 預測改為 CE 去噪器（x-prediction + softmax）。
  - 表示空間由低維可學 embedding 改為 one-hot、超球面、凍結 contextual embedding 或 16 維 unit-norm embedding，並同時處理 embedding collapse。
  - 排程依解碼錯誤率或 loss 分佈重新參數化。
  - self-conditioning 的作用被重新分析並修正。
  - 以 flow map（semigroup / Eulerian KL 自蒸餾）取代多步積分。
- **效果到哪裡：** 
  - FLM/FMLM（arXiv 2602.16813）在 LM1B/OWT 上與 MDLM、Duo 打平；一步生成勝過離散擴散蒸餾後的 8 步結果。 [arxiv](https://arxiv.org/html/2602.16813v2) [takara](https://tldr.takara.ai/p/2602.16813)
  - RePlaid（2605.18530）摘要稱其與 AR 的算力差距「compute gap of only 20× compared to autoregressive models」，並勝過 Duo；Plaid 約 64× 的對照數字未見於該摘要。
  - Sigma（2610.02665）摘要稱在 GSM8K、Minerva、HumanEval、MBPP 等基準上「achieves competitive performance with discrete models」；3B/8B 的平均分數與同算力的 Nemotron-Labs-Diffusion 相差不到 1 分。
  - Lafa（2610.28559）以少於 30B tokens 從 Qwen 改造到 8B。以 self-speculative decoding，摘要稱達到「2.54× the throughput of an AR model at comparable quality」，圖 1 說明 7B 為 2.02–2.54×。
- **尚未解決：**
  - Gen PPL 可被重複內容與無參數取樣器操縱（2606.08417、2607.00588）。
  - 大模型的少步品質仍明顯下降：Sigma 摘要稱連續軌跡在低 NFE 時「graceful degradation」，但其表 8 顯示 DDIM-4 時平均分數僅 7.74。
  - one-hot 表示的記憶體成本隨詞彙量線性增加。 [arxiv](https://arxiv.org/pdf/2605.07820)
  - 推理任務在低溫解碼下仍落後離散擴散與 AR。

## 一、時間脈絡與主要團隊

這波研究集中在 2026 年 2 月至 10 月。

| 時間 | 論文（arXiv） | 團隊 |
|---|---|---|
| 2026-02 | FLM/FMLM（2602.16813） | KAIST + CMU（Lee, Kim, Boffi 等） |
| 2026-02 | Categorical Flow Maps（2602.12233） | Amsterdam / Oxford（Roos, Davis, van de Meent, Welling 等） |
| 2026-04 | Discrete Flow Maps（2604.09784） | Harvard / Oxford / MIT / NYU（Potaptchik, Albergo 等） |
| 2026-04 | LangFlow（2604.11748） | UIUC（Ge Liu 組） |
| 2026-05 | ELF（2605.10938） | MIT（Kaiming He 等） | [arxiv](https://arxiv.org/pdf/2605.10938)
| 2026-05 | S-FLM（2605.11125） | EPFL（Deschenaux & Gulcehre） |
| 2026-05 | Scaling Categorical Flow Maps（2605.07820） | Apple + Oxford |
| 2026-05 | RePlaid（2605.18530） | NVIDIA + Cornell |
| 2026-06 | FMLM+（2606.24773） | CMU + KAIST |
| 2026-06 | Masked Language Flow Models（2606.27617） | Oxford | [arxiv](https://arxiv.org/pdf/2606.27617)
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
| 2026-10 | Sigma（2610.02665） | NVIDIA + Cornell + Georgia Tech | [arxiv](https://arxiv.org/pdf/2610.02665v1)
| 2026-10 | Design Space（2610.13834） | Apple + Oxford |
| 2026-10 | Predictive-State Flows（2610.12898） | — |
| 2026-10 | LangDiT（2610.22330） | — |
| 2026-10 | Lafa（2610.28559） | Chavan, Lee, Bamba, Yoo, Hong, Raghunathan, Kim（arXiv 頁面未列機構） |

研究集中出現有三個可追溯的原因。

- **flow map 理論成熟。** Boffi、Albergo、Vanden-Eijnden 在 2024–2025 年發展了 flow map matching / self-distillation 理論（2406.07507、2505.18825）。FLM、CFM、DFM 三篇幾乎同時把它移植到 one-hot / simplex 上。三篇互相引用為 concurrent work。
- **因子化誤差論點成為共同動機。** FLM 用 new-york / san-diego 的玩具例子說明：離散擴散每步獨立取樣各 token，少步時會產生 new-diego 這類錯誤組合。連續 flow 以確定性 ODE 聯合搬運整個序列，沒有這個問題。 [arxiv](https://arxiv.org/html/2602.16813v2)
- **影像擴散的工具可以直接沿用。** 例如 x-prediction、logit-normal 時間取樣、CFG、REPA、Muon、DMD。ELF 由影像擴散團隊提出。 [arxiv](https://arxiv.org/pdf/2605.10938)

注意：PlaidQ（2609.04531，Duke/Tsinghua，0.7B 程式碼模型）在 alphaXiv 頁面標示 v2 已撤稿，理由是「institutional disclosure requirements」。該文報告的 16 步 HumanEval pass@10 31.78、一步 pass@1 7.07 不應作為已確認結果引用。 [alphaxiv](https://www.alphaxiv.org/abs/2609.04531)

## 二、舊方法（2022–2024）與新方法的逐項比較

### 2.1 訓練目標與參數化

**舊做法**
- Diffusion-LM（Li et al. 2022）在可學 embedding 上以 MSE 預測 x₀，最後一步用 rounding 把向量對回 token。
- Plaid（Gulrajani & Hashimoto 2023）使用 VDM 式 ELBO。
- CDCD（Dieleman et al. 2022）已經用 CE 訓練 softmax 去噪器（score interpolation）。但 CDCD 之後的工作多數沒有延續這條路線。

**新做法與證據**
- **FLM（2602.16813）**
  - 在 one-hot 上做線性 interpolant。去噪器輸出逐 token 的 softmax，以 CE 訓練。 [arxiv](https://arxiv.org/html/2602.16813v2)
  - 去噪器的最佳解是 E[x₁|x_t]，即每個位置在 simplex 上的後驗。
  - 消融：CE 去噪器優於 velocity MSE，one-hot 優於可學 embedding，Euclidean 優於 Riemannian / simplex 擴散。
- **DFM（2604.09784）** 指出 flow map 的「mean denoiser」ψ_{s,t} 是時間加權的條件期望，因此恆落在 simplex 上。所以 semigroup、Lagrangian、Eulerian 三種一致性條件都能寫成 KL / CE 目標，取代 Euclidean L² 回歸。 [arxiv](https://arxiv.org/pdf/2604.09784)
- **LangFlow（2604.11748）** 用 Bregman divergence 把 embedding 空間的擴散連到 flow matching，從而導出 CE 目標。 [arxiv](https://arxiv.org/html/2604.11748v3) 它並指出 Plaid 式 MSE 會使 embedding collapse（NND 0.058 vs 0.877）。
- **ELF（2605.10938）** 採混合做法：
  - 中間步以 x-prediction + MSE 去噪；最後一步由同一網路以 CE 解碼成 token。 [arxiv](https://arxiv.org/pdf/2605.10938)
  - 論文說明選 x-prediction 而非 v-prediction 的理由：x-prediction 能讓去噪與解碼共用權重。 [arxiv](https://arxiv.org/pdf/2605.10938)
- **ConvergeFlow（2608.23551）** 是反例，只用 MSE：
  - 把資料預測器限制在 token embedding 的凸包內（softmax 加權平均）。 [arxiv](https://arxiv.org/html/2608.23551v1)
  - 證明在正則條件下，即使預測器有誤差，ODE 仍會收斂到合法的 token embedding，因此不需要 CE 監督的解碼器。 [arxiv](https://arxiv.org/html/2608.23551v1)
  - OWT 結果為「與現有連續與離散擴散 LM 相當」。 [arxiv](https://arxiv.org/html/2608.23551v1)

**結論**
- 實驗支持的部分：CE 目標，或至少把預測器的輸出限制在 simplex / 凸包，是性能提升的主因之一。
- ConvergeFlow 的結果顯示，必要條件是參數化把輸出限制在合法集合，而不是 CE 本身。
- RePlaid 報告：在 learned embedding 上加輔助 CE，PPL 反而由 22.1 升到 26.1。CE 的效益取決於表示空間，不是普遍成立。

### 2.2 表示空間與 rounding / decoder 問題

| 表示 | 代表論文 | 主要優點 | 主要成本／問題 |
|---|---|---|---|
| 可學低維 embedding（舊） | Diffusion-LM, Plaid | 維度低 | embedding collapse、rounding 誤差 |
| one-hot（R^{\|V\|}） | FLM, CFM, DFM, GSF | 去噪器輸出即 simplex，可直接用 CE / KL | L×\|V\| 張量，記憶體隨詞彙量線性增加 |
| 單位超球面 S^{d−1} | S-FLM（2605.11125） | 不需實體化 one-hot，訓練成本接近離散擴散 | 需截斷噪聲區間 | [arxiv](https://arxiv.org/pdf/2605.11125v3)
| 凍結 contextual embedding（T5） | ELF | 語義結構好、訓練 token 少 | 依賴外部編碼器 | [arxiv](https://arxiv.org/pdf/2605.10938)
| 16 維 unit-norm embedding | Sigma（2610.02665） | 參數占比 <1% | 未正則化時塌縮到約 4 個有效維度 | [arxiv](https://arxiv.org/pdf/2610.02665v1)
| norm 固定、只學方向 | Lafa（2610.28559） | 保留預訓練 AR embedding 的 norm | — |
| VAE / latent | TextLDM（2605.07748）、PSF（2610.12898） | 擴散對象維度低 | 需額外解碼器 |

**one-hot 的成本**
- Scaling CFM（2605.07820）的限制章節寫明：L=2048、|V|≈100k 時「manageable given sufficient hardware (256 H100s)」，但成本隨序列長度與詞彙量線性增加。作者以 Cut Cross-Entropy 吸收這個成本，並未解決它。
- S-FLM 以此為動機改用超球面，並以 SLERP 插值。 [arxiv](https://arxiv.org/pdf/2605.11125v3)

**S-FLM 的噪聲截斷**
- 在高維球面上，embedding 分得很開，後驗很快變成 one-hot。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
- 作者推導出截斷上界 α⋆(δ) ≈ (2/π)·arcsin(√(2 log(2(|V|−1)/δ)/d))，只在高噪聲段訓練。 [arxiv](https://arxiv.org/pdf/2605.11125v3)

**decoder 問題的三種處理方式**
- ELF 讓最後一步由同一網路做 CE 解碼。 [arxiv](https://arxiv.org/pdf/2605.10938)
- Du & Ma（2606.08810）以 decoder basin / margin 分析並提出 BGEE early exit。
- ConvergeFlow 證明收斂性，從而取消 decoder。 [arxiv](https://arxiv.org/html/2608.23551v1)

### 2.3 噪聲 / 時間排程

| 做法 | 論文 | 內容 | 證據 |
|---|---|---|---|
| time warping（舊） | CDCD（2022） | 依 loss 的 CDF 重新分配時間取樣 | — |
| 解碼錯誤率排程 | FLM、DFM、GSF | 選 β(t) 使 P(argmax x_t ≠ argmax x₁) 隨 t 近似線性變化 | 見下方說明 |
| 混合排程 | DFM、Scaling CFM | α_t=(1−λ)t+λγ_t | 見下方說明 |
| 可學 Gumbel / logNSR 排程 | LangFlow | information-uniform 原則 | Gen PPL 約 1000 → 154 |
| 截斷 + 自適應排程 | S-FLM | 依 \|dL̂/dt\| 的反 CDF 每 50 步更新，參考 InfoNoise | GSM8K 7.7% → 11.1%（加入自適應） | [arxiv](https://arxiv.org/pdf/2605.11125v3)
| logit-normal 時間取樣 | ELF | 沿用影像 rectified flow | — | [arxiv](https://arxiv.org/pdf/2605.10938)
| 常數 per-timestep loss | RePlaid | 最小化 ELBO 估計變異數 | — |
| 逐位置時間 | Clock Diffusion（2610.00894）、Lafa、DFM | 位置相依時鐘 u^ℓ(t)；Lafa 以 loss 反 CDF 學 per-token time | — |

- **解碼錯誤率排程。** FLM 稱此舉「greatly improves training stability」。GSF 圖 4 顯示，在大詞彙量下 FLM 的解碼錯誤集中在 t≈0.8 附近，是一個急遽的相變，必須做時間重參數化。 [arxiv](https://arxiv.org/html/2610.00497)
- **混合排程的消融。** Scaling CFM 在 400M 規模掃 λ，最佳值約 0.75，λ=1（純錯誤率排程）時崩潰。DFM 使用 λ=0.9。
- **逐位置時間。** Lafa 學到的 per-token time 自然呈現由左到右的順序，論文寫明「this left-to-right bias is not manually imposed」。 [arxiv](https://arxiv.org/html/2610.28559)

### 2.4 self-conditioning 的作用與理論解釋

**舊做法：** Analog Bits（Chen et al. 2022）提出 self-conditioning，訓練時以 50% 機率把上一步的預測作為額外輸入。CDCD、Plaid 沿用此做法。

**新的實證**
- LangFlow：self-conditioning 使 LM1B Gen PPL 由 154 降到 81.5，PPL 由 49 降到 30。
- Design Space（2610.13834）的解釋：self-conditioning 修正 train/inference mismatch，也解決 token identifiability 問題。

**新的理論解釋**
- **FMLM⋆（2607.00714）** 證明帶 self-conditioning 的 flow LM 在做定點迭代（fixed-point iteration），並由此定義 fixed-point flow。這類 flow 有對應的 flow map，可以同時壓縮定點迭代與 flow。作者報告 OWT 一步與少步結果達到 SOTA。 [arxiv](https://arxiv.org/html/2607.00714) [arxiv](https://arxiv.org/pdf/2607.00714)
- **Untied Self-Conditioning（2608.22244）** 指出另一種推論時的不匹配：步長大時，solver 會耦合 latent 與 self-conditioning 兩條通路，而訓練時沒有這種耦合。 [arxiv](https://arxiv.org/pdf/2608.22244)
  - 方法是不需重訓、每步一次前向的取樣器：以 overlap transform 衰減冗餘方向，並以 solver 修正把預測移向步內平均。 [arxiv](https://arxiv.org/html/2608.22244)
  - 8 NFE 時，LangFlow 的 OWT Gen PPL 由 531 降到約 62；ELF-B 由 71 降到約 43。 [awesomepapers](https://awesomepapers.io/llm-papers/papers/2608.22244)
- **Low Perplexity is Repetition（2607.00588）** 發現 self-conditioning 迴路沿單一方向形成收縮吸引子，使輸出偏向重複內容。 [arxiv](https://arxiv.org/abs/2607.00588)
  - 去除重複後，ELF-B 的 Gen PPL 由 19.5 升為 27.7。 [arxiv](https://arxiv.org/abs/2607.00588)
  - 在 Gen PPL 上，最小的 ELF 模型因重複最多而得到最佳分數。 [arxiv](https://arxiv.org/abs/2607.00588)
  - 作者提出 ACE：每步從回饋中減去該方向。 [arxiv](https://arxiv.org/abs/2607.00588)

**結論**
- self-conditioning 是性能提升的主要來源之一。
- 它同時是低 entropy / 重複問題的來源之一。
- 以 Gen PPL 報告的 self-conditioning 增益，有一部分來自重複內容。

### 2.5 SDE vs ODE 與 likelihood 估計

- **舊做法：** Plaid 使用 VDM ELBO，可以直接報告 PPL 上界。
- **確定性 ODE 取樣：** FLM 系列以確定性 ODE 取樣，論點是 ODE 能表示序列的疊加並避免逐 token 祖先取樣。
- **ODE-based NLL 上界：** LangFlow 用它報告 PPL：LM1B 30.0、OWT 24.6。
- **Scaling CFM 的半離散 ELBO：**
  - 用 Pinsker 不等式把 L² 擴散損失上界為 CE，從而在 MCQA 上評分。
  - 作者表示 bound 的緊度未知。
  - 在 6 項 MCQA 中，1.7B CFM 低於同資料同算力的 MDLM / Duo（例如 ARC-E 47.1 vs 50.5 / 53.4），BoolQ 42.5 甚至低於機率水準 50.4。
  - 作者表示透過反解 ODE 計算精確 likelihood 在 L×|V| 維度下過於昂貴。
- **Sigma** 使用 VDM ELBO，並以 DDPM（SDE）與 DDIM（ODE）兩種取樣器報告結果。DDPM-256 品質最好。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- **Clock Diffusion** 的 ClockDLM 以 sliding window 變分界，在 OWT 上取得連續 DLM 中最佳的 likelihood bound，並勝過 block 半自迴歸離散擴散（BD3LM）。 [arxiv](https://arxiv.org/html/2610.00894)

### 2.6 少步生成：flow map / consistency / 分佈匹配蒸餾

離散擴散的蒸餾受因子化誤差限制。DFM 與 FMLM 都論證，離散更新在大步長下函數形式不正確，連續 flow 的 flow map 是唯一存在的。 [arxiv](https://arxiv.org/pdf/2604.09784) [arxiv](https://arxiv.org/html/2602.16813v2)

**蒸餾方法**

| 方法 | 論文 | 一致性條件 / 目標 | 備註 |
|---|---|---|---|
| PSD（semigroup + KL） | FMLM, DFM, Scaling CFM | X_{u,t}∘X_{s,u}=X_{s,t}，以 KL 監督 | 大規模下穩定 |
| ESD（Eulerian logit） | DFM | logit 空間的 Eulerian 恆等式 | 小規模最佳；OWT 1 步崩潰（entropy 0.26）；1.7B 規模下全部發散 |
| 端點一致性 | CFM（2602.12233） | endpoint consistency | — |
| Gumbel 耦合 + AR teacher | GSF（2610.00497） | AR 的 Gumbel-max 耦合給出不相交的直線路徑 | 見下文 |
| 分佈匹配（reverse KL） | DMD for CDLMs（2609.40235） | Simplex-DMD（pathwise）、Reinforce-DMD（REINFORCE + density ratio） | — | [arxiv](https://arxiv.org/abs/2609.40235)
| 漸進蒸餾 | ELF+PD | — | 見下表 |

- **GSF** 證明在無限容量假設下，AR teacher 的 Gumbel-max 耦合給出的最佳路徑直線度 S=0。這同時讓所需的時間 warping 變得溫和許多。 [arxiv](https://arxiv.org/html/2610.00497)

**少步結果（Gen PPL / entropy，GPT-2 Large 評分，OWT L=1024）**

| 方法 | 1 步 | 2 步 | 4 步 | 8 步 |
|---|---|---|---|---|
| MDLM+SDTT | 1260.86 / 5.26 | 877.22 / 5.34 | 339.73 / 5.38 | — | [arxiv](https://arxiv.org/pdf/2604.09784)
| Duo+DCD | 5743.29 / 6.02 | 891.16 / 5.41 | 250.86 / 5.37 | — | [arxiv](https://arxiv.org/pdf/2604.09784)
| Duo+Di4C | 370.51 / 3.92 | 210.22 / 4.63 | 154.67 / 4.85 | — | [arxiv](https://arxiv.org/pdf/2604.09784)
| FMLM | 168.30 / 5.17 | 133.29 / 5.25 | 111.31 / 5.26 | 86.50 | [arxiv](https://arxiv.org/pdf/2605.10938)
| DFM（PSD） | 180.29 / 4.91 | 152.83 / 5.03 | 122.32 / 5.10 | 98.54 / 5.11 | [arxiv](https://arxiv.org/pdf/2604.09784)
| DFM（ESD） | 5.33 / 0.26（崩潰） | 108.91 / 5.15 | 77.08 / 5.27 | 62.98 / 5.23 | [arxiv](https://arxiv.org/pdf/2604.09784)
| GSF（AR teacher） | 114.9 / 5.07 | 105.6 / 5.30 | 99.7 / 5.32 | — | [arxiv](https://arxiv.org/html/2610.00497)
| ELF+PD | 136.10 / 5.26 | — | 34.33 / 5.16 | 23.18 / 5.07 | [arxiv](https://arxiv.org/pdf/2605.10938)

**讀表注意**
- 各論文的 teacher、資料與 tokenizer 不同。表中 ELF+PD 的 entropy 接近該文自設的 5.0 退化線，部分 Gen PPL 改善來自較低的多樣性。 [arxiv](https://arxiv.org/pdf/2605.10938)
- **GSF：** 一步的 Gradient Moment（GM）為 0.94，FMLM 為 1.18；作者重新實作的 FMLM 一步為 140.0 / 4.93。 [arxiv](https://arxiv.org/html/2610.00497)
- **Untied SC：** 在 ELF-B 上，8 NFE 的 Gen PPL 為 45.63，HiGS 為 56.90，官方取樣器為 74.18。 [arxiv](https://arxiv.org/html/2608.22244)
- **規模化後的少步結果：** Scaling CFM 的 1.7B 模型（2.1T tokens）在 PSD 蒸餾後，4 步 MAUVE 提升最多，token entropy 接近資料水準。但作者也寫明「Gen PPL remains high」。未蒸餾模型在 1–2 步時輸出退化（「the the the」），反而讓 Gen PPL 偏低。
- **Sigma：** DDIM-32 平均 60.45，DDIM-16 平均 55.17，DDIM-4 平均僅 7.74（DDPM-256 為 64.34）。Sigma 摘要稱連續軌跡在低 NFE 時「graceful degradation」，但上述數字顯示少步品質在大模型上仍明顯下降。

### 2.7 架構與訓練配方

- **DiT 骨幹：** 小規模工作多沿用 SEDD / MDLM 的 DiT（約 170M），以 adaLN 注入時間。FMLM 對 s、t 雙時間做 adaLN 條件化。
- **LangDiT（2610.22330）：**
  - 系統比較 Transformer 設計後發現，因果 LLM 或影像 DiT 的有效設計不一定適用於 CDLM。 [arxiv](https://arxiv.org/html/2610.22330v1)
  - 採用 QK normalization 等設計，learned embedding 的 row norm 設為 0.1√d。 [arxiv](https://arxiv.org/html/2610.22330v1)
  - 作為 LangFlow 的骨幹時，OWT Gen PPL 在 1024 NFE 為 21.72，在 128 NFE 為 44.72。 [arxiv](https://arxiv.org/abs/2610.22330)
- **S-arch（S-FLM）：** 參考 nGPT，使激活值保持在球面上。在 T=1 下，GSM8K 由 11.1% 提升到 12.4%。作者也寫明 S-arch 訓練較慢。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
- **ELF：**
  - logit-normal 時間取樣、training-time CFG、in-context conditioning、128 維 bottleneck、Muon 優化器。 [arxiv](https://arxiv.org/pdf/2605.10938)
  - 約 45.2B 有效訓練 tokens（OWT 5 epochs）；同篇論文稱其他 DLM 通常使用超過 500B。 [arxiv](https://arxiv.org/pdf/2605.10938)
  - 模型大小：ELF-B 105M、ELF-M 342M、ELF-L 652M。 [arxiv](https://arxiv.org/pdf/2605.10938)
  - WMT14 De-En BLEU 26.4（AR 25.2）；XSum ROUGE-1/2/L 36.0 / 12.2 / 27.8。 [arxiv](https://arxiv.org/pdf/2605.10938)
- **ELF-REG（2609.29102）：**
  - 以 REPA + REG 對齊凍結的 AR teacher 表示。
  - GSM8K 55.96%（64 NFE）、MATH-500 13.39%、HumanEval 22.56%。
- **CFG：**
  - DFM 的 block 條件生成中，ω 由 0 增至 2 時，Gen PPL 由 56.31 降到 30.98，entropy 由 5.20 降到 4.81。 [arxiv](https://arxiv.org/pdf/2604.09784)
  - Sigma 在 embedding 空間使用 CFG 與 score temperature，每次前向開銷少於 5%。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- **優化超參數：** Scaling CFM 在 400M 規模做了 702 + 324 組掃描，Gen PPL 與 MAUVE 主要受學習率與 weight decay 影響。β₂=0.95 優於較大值。Complete(d)P 超參數遷移反而使 Gen PPL 變差 35%。

### 2.8 規模化策略

| 模型 | 規模 | 初始化 / 資料 | 結構 | 主要結果 |
|---|---|---|---|---|
| Scaling CFM（2605.07820） | 1.7B | 從頭訓練，2.1T tokens，Nemotron，\|V\|≈100k | 全序列 one-hot flow | 1024 步 Gen PPL 58.66，AR 45.33；8 步 MAUVE 0.88，AR 0.70 |
| RePlaid（2605.18530） | 對齊現代架構 | — | embedding 擴散 | 追上 AR 的算力倍數約 20×（有 SC）／27×（無 SC）；MDLM 約 14×、Duo 約 22×；OWT PPL 22.1 |
| Sigma-3B / 8B（2610.02665） | 3.8B / 8.5B（總參數） | 由 1T tokens 預訓練的內部 AR baseline 初始化；再訓練 300B tokens；約 9,600 / 19,200 GB200 GPU 小時 | block 大小 32，16 維 embedding，AR 輔助 loss | 見下方說明 | [arxiv](https://arxiv.org/pdf/2610.02665v1)
| Lafa（2610.28559） | 0.5B / 7B / 8B | Qwen2.5-Instruct（0.5B, 7B）、Qwen3-8B Base；少於 30B tokens | 前綴因果注意力 + block 內雙向非同步 flow；loss = AR CE + 去噪 CE + embedding MSE | 7B 吞吐量為 AR 的 2.02–2.54× | [arxiv +2](https://arxiv.org/abs/2610.28559)
| MLFM（2606.27617） | 以 1028M SMDM 為骨幹 | LoRA 319M 可訓練參數，約 100B 位置 | 遮罩 + 連續 interpolant | GSM8K 31.24（SMDM 58.5，經 GSM8K 40 epoch 微調）；MT-Bench 2.27（SMDM 1.60） | [arxiv](https://arxiv.org/pdf/2606.27617)
| ClockDLM（2610.00894） | 小規模 | — | 位置相依時鐘、sliding window、KV cache | 連續 DLM 中 OWT likelihood 最佳；加 Cache Grab 後 GSM8K 超過 AR | [arxiv](https://arxiv.org/html/2610.00894)

- **Sigma 3B base：** 平均 64.34，NLD-3B 為 64.74；GSM8K 84.50 vs 83.40。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- **Sigma 8B base：** 平均 69.67，NLD-8B 為 70.22，Qwen3-8B 為 68.91。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- **Sigma 8B-Instruct：** 平均 59.27，NLD 為 59.92。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- **Sigma 的 pass@k：** MBPP 上 k=256 時 91.8% vs 87.6%。作者把這歸因於 embedding 空間 steering 提高了多樣性。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- **Lafa 的 macro-average：** 有二手報導稱 8B macro-average 為 63.1%，AR SFT 為 56.8%，Efficient-DLM 為 63.5%。 [aiweekly](https://aiweekly.co/alerts/lafa-flow-language-model-hits-254x-ar-throughput-at-7b-scale) 此數字未能從原文表格直接核對，引用時應回查原文。

**Lafa 的設計**
- 論文稱 flow 訓練有多 token 預測效果，能保留甚至提升 AR 能力。 [arxiv](https://arxiv.org/abs/2610.28559) [arxiv](https://arxiv.org/abs/2610.28559)
- 推論時以 flow 產生草稿，由同一模型的 AR 頭驗證（self-speculative decoding）。 [arxiv +2](https://arxiv.org/abs/2610.28559)
- 與既有做法的對照：論文 §1 寫明其少於 30B tokens 的訓練「cheaper than comparable adaptation of discrete diffusion models that use up to 500B tokens」。

**規模化路線的共同點**
- 從 AR 權重初始化（Sigma、Lafa、ELF-REG 使用 AR 表示對齊、PlaidQ）。
- block 化或位置相依時間，以取得 KV cache 與可變長度生成（Sigma、Lafa、Clock Diffusion）。
- 保留 AR 輔助 loss。
- 這些做法使連續模型的訓練成本與離散擴散 LLM 相當。但 Sigma 與 Lafa 的品質基準都是同算力的離散擴散或 AR SFT，不是前沿 AR 模型。

### 2.9 混合連續—離散方法

- **CANDI（Pynadath et al. 2025）：** 混合連續與離散，提出解碼錯誤率排程，為 FLM 等論文所沿用。
- **SemiCat（2610.13834）：**
  - 在 DoubleCat 上加入 CANDI 式的離散 reveal。
  - TinyGSM 達 49.94%（調過取樣器）。 [arxiv](https://arxiv.org/abs/2610.13834)
  - 1.7B 規模下，6 項 likelihood 評測中 4 項勝過離散擴散；無條件生成勝過 one-hot flow。 [arxiv](https://arxiv.org/abs/2610.13834)
  - 作者自述：隨機 reveal 使軌跡只是分段連續，較難套用連續加速工具。 [arxiv](https://arxiv.org/html/2610.13834)
- **FMLM+（2606.24773）：**
  - 在 FMLM 上加入遮罩式噪聲，乾淨位置的 t 固定為 1。
  - 證明 s=t=0 時的目標等同 MDM 目標，因此可從 MDM 初始化或蒸餾。
  - Posterior Refinement 依最後一步 flow map 輸出的 p_max 保留高信心 token，並對其他位置重新加噪。
  - 結果：GSM8K（TinyGSM 訓練）32 NFE 達 19.0%，MDLM 1024 NFE 為 18.0%；從 MDLM 初始化時，64 NFE 達 33.6%；無條件 Sudoku 9 NFE 達 87.1%。
- **MLFM（2606.27617）：** 交替做連續去噪與離散 unmask。CCFG + OTP 取樣器使 GSM8K 由 13.19 升到 31.24。 [arxiv](https://arxiv.org/pdf/2606.27617)
- **Predictive-State Flows（2610.12898）：**
  - 主張 diffusion prior 不應負責細粒度 token 選擇。 [arxiv](https://arxiv.org/abs/2610.12898)
  - 以 rectified flow 生成 K≪L 個 block 層級的 predictive state，由 block decoder 產生 token。 [arxiv](https://arxiv.org/abs/2610.12898) [arxiv](https://arxiv.org/html/2610.12898)

## 三、基準結果對照（附評估設定差異）

### 3.1 LM1B / OWT 多步

| 模型 | 資料 | PPL（上界） | Gen PPL | 設定 |
|---|---|---|---|---|
| LangFlow | LM1B | 30.0 | — | ODE NLL 上界 |
| LangFlow | OWT | 24.6 | 36.5 | — |
| LangFlow + LangDiT | OWT | — | 21.72 | 1024 NFE | [arxiv](https://arxiv.org/abs/2610.22330)
| LangFlow + LangDiT | OWT | — | 44.72 | 128 NFE | [arxiv](https://arxiv.org/abs/2610.22330)
| RePlaid | OWT | 22.1 | — | ELBO |
| ELF | OWT | — | 24 | 32 步，SDE，CFG=3；去除重複後 ELF-B 由 19.5 → 27.7 | [arxiv](https://arxiv.org/pdf/2605.10938)
| DFM 對角（未蒸餾） | OWT | — | 47.07 / entropy 5.12 | 1024 步 | [arxiv](https://arxiv.org/pdf/2604.09784)
| DFM ESD | OWT | — | 36.48 / 4.95 | 1024 步 | [arxiv](https://arxiv.org/pdf/2604.09784)

### 3.2 推理任務（TinyGSM 訓練、GSM8K 零樣本、約 170M）

| 模型 | GSM8K |
|---|---|
| AR（sample / greedy） | 53.9 / 63.3 | [arxiv](https://arxiv.org/pdf/2605.11125v3)
| MDLM / Duo（T=1） | 18.0 / 17.2 | [arxiv](https://arxiv.org/pdf/2605.11125v3)
| MDLM / Duo（T=0.1） | 約 33 / 約 36 | [arxiv](https://arxiv.org/pdf/2605.11125v3)
| FLM | 0.3 | [arxiv](https://arxiv.org/pdf/2605.11125v3)
| CANDI | 0.2 | [arxiv](https://arxiv.org/pdf/2605.11125v3)
| S-FLM（T=1，top-1 velocity） | 18.0 | [arxiv](https://arxiv.org/pdf/2605.11125v3)
| S-FLM（T=0.1） | 約 18 | [arxiv](https://arxiv.org/pdf/2605.11125v3)
| FMLM+ PR（32 NFE） | 19.0 |
| FMLM+ Init（64 NFE） | 33.6 |
| SemiCat（調過取樣器） | 49.94 | [arxiv](https://arxiv.org/abs/2610.13834)

- S-FLM 在 GSM8K 上的逐項消融：1.2 → 7.7（截斷）→ 8.1（不重新正規化）→ 11.1（自適應排程）→ 12.4（S-arch）→ 18.0（top-1）。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
- 在這個基準上，one-hot FLM 幾乎無法解題。 [arxiv](https://arxiv.org/pdf/2605.11125v3)

### 3.3 大模型（diffusion 模式）

- **Sigma-8B base：** GSM8K 89.07、HumanEval 61.40、MBPP 64.78。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- **Sigma-8B-Instruct：** GSM8K 92.49、MMLU 77.46、IFEval 79.14、GPQA 44.70。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
- **MMLU（likelihood 評分）：** Sigma-8B 73.96，NLD-8B 76.01。 [arxiv](https://arxiv.org/pdf/2610.02665v1)

### 3.4 評估方法的限制

- **Gen PPL 可被操縱。** Franca & Tong 的 Hacking Generative Perplexity（2606.08417）構造了零參數的簡單取樣器。這些取樣器在 LM1B 與 OWT 上以非退化的 entropy 取得 SOTA Gen PPL，輸出卻在構造上就不連貫。作者建議改用分佈距離指標。
- **重複內容會壓低 Gen PPL。** 2607.00588 顯示，帶 self-conditioning 的連續模型重複遠多於人類文本，而 Gen PPL 獎勵重複。 [arxiv](https://arxiv.org/abs/2607.00588) [arxiv](https://arxiv.org/html/2607.00588)
- **少步退化也會壓低 Gen PPL。** Scaling CFM 的未蒸餾模型在 1 步時的 Gen PPL 為 8.11，但輸出是「the the the」。DFM ESD 在 OWT 1 步時 Gen PPL 5.33、entropy 0.26。 [arxiv](https://arxiv.org/pdf/2604.09784)
- **替代指標。** S-FLM 改用有標準答案的任務（Sudoku、GSM8K），並畫出溫度掃描下的 Gen PPL–entropy 前緣。 [arxiv](https://arxiv.org/pdf/2605.11125v3) GSF 使用 Gradient Moment。 [arxiv](https://arxiv.org/html/2610.00497) Scaling CFM 使用 MAUVE⋆ 與 ΔPPL。
- **跨論文比較的限制。** Gen PPL 評分模型不一致：GPT-2 Large 與 Qwen2.5-7B 都有人用。tokenizer、序列長度與溫度設定也不同，跨論文的數字只能作為參考。

## 四、未解決問題與各論文自述的限制

1. **大模型少步品質。**
   - Sigma 自述在低步數下（DDIM-4/8）表現大幅下降，高步數時每 token 解碼仍比最佳化過的 AR 慢。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
   - Scaling CFM 自述蒸餾後的 Gen PPL 仍高，ESD 在大規模下不穩定。
2. **對 AR 的差距。**
   - RePlaid 的算力倍數約 20×。
   - Scaling CFM 的 1.7B 模型 Gen PPL 為 58.66，AR 為 45.33。
   - S-FLM 自述「a substantial gap to autoregressive decoding remains」。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
   - MLFM 在 GSM8K 上為 31.24，LLaMA-2 為 58.6。 [arxiv](https://arxiv.org/pdf/2606.27617)
3. **低 entropy 與重複。**
   - LangFlow 樣本的 entropy 偏低。
   - ELF 的低 Gen PPL 主要來自重複（2607.00588）。 [arxiv](https://arxiv.org/html/2607.00588v1)
   - JSD 與 forward KL 蒸餾在 entropy 與品質之間有取捨（Scaling CFM）。
4. **one-hot 的記憶體與計算成本。** 成本隨 |V| 與 L 線性增加（Scaling CFM、S-FLM）。超球面或低維 embedding 能避開這個問題，但又必須處理 embedding collapse：Sigma 未正則化時塌縮到約 4 維。 [arxiv](https://arxiv.org/pdf/2610.02665v1)
5. **likelihood 評估。** 半離散 ELBO 的緊度未知，CFM 在 MCQA 上低於以 ELBO 訓練的離散擴散。S-FLM 的截斷界假設 embedding 隨機分佈，作者稱它只是啟發式。 [arxiv](https://arxiv.org/pdf/2605.11125v3)
6. **超參數遷移。** 時間條件層不在 µP / Complete(d)P 的涵蓋範圍內（Scaling CFM）。
7. **混合方法與連續工具不相容。** SemiCat 的分段軌跡難以套用連續加速工具。 [arxiv](https://arxiv.org/html/2610.13834) FMLM+ 的 PR 固定已提交的 token，作者寫明無法回頭修正。
8. **實驗規模與重複次數。** S-FLM 每個設定只訓練一次（單次 TinyGSM 訓練超過 300 美元）。 [arxiv](https://arxiv.org/pdf/2605.11125v3) FMLM+ 自述實驗規模小、資料多為合成。
9. **研究誠信面。** PlaidQ 已撤稿； [alphaxiv](https://www.alphaxiv.org/abs/2609.04531) Lafa 的部分彙總數字目前只見於二手報導。 [aiweekly](https://aiweekly.co/alerts/lafa-flow-language-model-hits-254x-ar-throughput-at-7b-scale)

## 五、判斷與建議

- **性能提升的主要來源。** 依證據強度排序：
  1. 把去噪器輸出限制在 simplex / 凸包，並以 CE / KL 訓練（FLM、DFM 消融；LangFlow 的 collapse 分析）。
  2. 依解碼錯誤率或 loss 重新分配時間（FLM、Scaling CFM 的 λ 掃描；LangFlow 的 Gen PPL 約 1000 → 154）。
  3. self-conditioning（LangFlow 154 → 81.5）及其推論修正（Untied SC：531 → 62）。
  4. flow map 自蒸餾（OWT 一步 Gen PPL：FMLM 168、GSF 115，離散蒸餾 370–5743）。
  5. 規模化時以 AR 初始化並採 block 結構（Sigma、Lafa）。

  這些改動多數在 CDCD（2022）已有雛形：CE 去噪器與 time warping 都在其中。2026 年的不同之處有三點：把這些設計放進 flow matching / flow map 框架並做系統消融；表示空間改為 one-hot 或受約束的 embedding；有了少步蒸餾工具。
- **哪些主張可視為已確立。** 小規模（約 170M）下，連續 flow 在少步區間勝過離散擴散蒸餾，這一點有多篇獨立論文重現（FMLM、DFM、GSF、ELF+PD）。「連續擴散全面勝過離散擴散」這個說法目前沒有證據支持：
  - Sigma 在 3B/8B 規模上是持平，不是勝出。
  - 推理任務在低溫下仍落後。
  - MCQA 的 likelihood 評分也落後。
- **評估時的做法。** 引用任何 Gen PPL 數字時，應同時檢查 entropy、重複率（Rep-n）、評分模型與溫度，並優先參考 MAUVE、GSM8K、HumanEval 這類有標準答案或分佈距離的指標。
- **後續值得追蹤的方向：**
  - 解決大模型少步品質下降的方法（Sigma 的 SUSReg / PDD、DMD for CDLMs）。
  - 不需實體化 one-hot 的 CE 參數化（S-FLM、ConvergeFlow 的凸包預測器）。
  - Lafa 原文表格的逐項數字，以及它以 flow 草稿加 AR 驗證的吞吐量優勢能否在其他實作中重現。