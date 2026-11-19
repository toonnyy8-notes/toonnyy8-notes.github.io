# Latent Diffusion 兩大主線深度文獻彙整（2025 下半年 – 2026.09）

## TL;DR
- **Diffusability（可擴散性）研究已從「各說各話的單一代理指標」收斂成「多面向共識」**：Kling 團隊的系統性研究（arXiv:2606.03578）在 86 個 tokenizer × 4 個 backbone 上證實，語意可分性（LNC）、空間結構（SRSS）與新提出的 Velocity Irreducible Variance（VIV，與 gFID 的 Pearson=0.87）三者最能穩定預測 gFID，且 SRSS+LNC 聯合迴歸 R²=0.91；rFID 與 gFID 的錯配（optimization dilemma）本質是「高維 latent 難學」而非「壓縮不足」。
- **RAE（Representation Autoencoder，arXiv:2510.11690）以凍結 DINOv2/SigLIP2 + 訓練 ViT decoder 取代 VAE**，其 DiT^DH 在 ImageNet 256 達 1.51 FID（無 guidance）、1.13（256² 與 512² 皆用 AutoGuidance），並主張「應成為 DiT 新預設」；但它高度依賴 noise schedule shift、width≥token-dim、wide DDT head 等修正，且 latent 各向異性極高導致 CFG 與蒸餾困難——這正是 diffusability 研究中「語意可分性↑但 spectral anisotropy 帶來張力」的具體案例。
- **兩條路線的張力尚未解決，且出現 pixel-space 第三勢力反撲**：若 latent 價值主要來自語意先驗（RAE）而非壓縮，則 latent 層的必要性受質疑；但 DiffusionBench（arXiv:2606.24888）在統一框架重訓 JiT/PixNerd/PixelGen 等 pixel-space 方法時，於 80 epoch 未觀察到加速收斂，顯示「表徵先驗」目前仍需某種 encoder 承載。

---

## Key Findings

1. **錯配問題的本質**：VA-VAE/LightningDiT（CVPR 2025 Oral，arXiv:2501.01423）首次系統化命名「optimization dilemma」——tokenizer 維度（d）上升會改善 rFID 但惡化 gFID，因為高維無約束 latent 難以被 diffusion 學習。解法是用 vision foundation model 對齊（VF Loss）。
2. **六到七種 diffusability 觀點**：語意可分性、空間結構、仿射等變性、頻譜平滑度、流形連續性、分布均勻性、速度歧義（velocity ambiguity）。這些多半各自搭配一個自我解釋的代理指標，泛化性存疑。
3. **系統性裁決**：Zhong et al. 顯示 VIV、語意可分性、空間結構跨 backbone/tokenizer 家族最穩定；仿射等變性最弱（被剔除）；頻譜平滑度、分布均勻性、VIV 之間存在共線性，多數其他觀點近似正交。
4. **RAE 路線快速分化**：原始 RAE → RAEv2（Improved Baselines，arXiv:2605.18324，RAE 與 REPA 互補、self-guidance 取代 AutoGuidance）→ Scale-RAE（T2I，arXiv:2601.16208，相對 FLUX VAE 4.0×/4.6× 收斂加速）→ 蒸餾（Drifting Models，arXiv:2606.15553）→ 影片（V-RAE、VideoRAE）。
5. **評估指標本身在演進**：gFID 被批評為「Goodhart 化」，出現 FDr⁶（六個表徵空間平均，arXiv:2604.28190）、iFID（arXiv:2603.05630，Pearson/Spearman ≈0.85）、DiffusionBench（同時評 ImageNet 與 T2I）。
6. **理論支撐**：Wang & Pehlevan（NeurIPS 2025 Spotlight，arXiv:2503.03206）的頻譜偏置理論（τ∝λ⁻¹）為「高頻能量少 = 較好 diffusability」提供了機制解釋。

---

## Details

### 時間軸（研究線演進脈絡）

- **2024.10** REPA（arXiv:2410.06940，ICLR 2025 Oral）：對齊 diffusion 中間層與 DINOv2 特徵，SiT 訓練加速 >17.5×（在 <400K steps 內即匹配 SiT-XL 訓練 7M steps 的無 CFG 表現），最終 FID 1.42（w/ CFG + guidance interval）。
- **2025.01** VA-VAE / LightningDiT（arXiv:2501.01423，CVPR 2025 Oral）：命名 optimization dilemma；VF Loss；ImageNet-256 FID 1.35，64 epoch 達 2.11（>21× 加速，官方 repo 稱 21.8×）。
- **2025.02** EQ-VAE（arXiv:2502.09509，ICML 2025）：仿射等變正則化；DiT-XL 加速 ×7、REPA ×4。
- **2025.02** MAETok（arXiv:2502.03444，ICML 2025 Spotlight）：mask modeling AE；證明「不需要 variational」；ImageNet-512 gFID 1.69、76× 訓練加速、31× 推論吞吐，僅用 128 tokens。
- **2025.02** Improving Diffusability of Autoencoders（arXiv:2502.14831，Snap Research，ICML 2025）：頻譜分析找出 latent 高頻過量；scale equivariance 正則化；FID −19%、FVD ≥−44%。
- **2025.03** Wang & Pehlevan 頻譜偏置理論（arXiv:2503.03206，NeurIPS 2025 Spotlight）。
- **2025.10** RAE（arXiv:2510.11690，ICLR 2026）與「There is No VAE」（arXiv:2510.12586）同月出現，兩條對立路線正面對決。
- **2025.11–12** pixel-space 井噴：PixelDiT（arXiv:2511.20645）、iREPA（arXiv:2512.10794，ICLR 2026）、Prism Hypothesis（arXiv:2512.19693）、Scalable Visual Tokenizers（arXiv:2512.13687）。
- **2026.01–03** Scale-RAE（arXiv:2601.16208）、PixelGen（arXiv:2602.02493）、iFID（arXiv:2603.05630）、RAEv2（arXiv:2605.18324）。
- **2026.06** 系統性裁決與統一評測集中出現：Diffusing in the Right Space（arXiv:2606.03578）、DiffusionBench（arXiv:2606.24888）、RAE 蒸餾（arXiv:2606.15553）、Representation Fréchet Loss（arXiv:2604.28190）。
- **2026.07–08** 影片延伸：VideoRAE（arXiv:2607.14088）、V-RAE（arXiv:2608.13556）。

---

### 主題一：Latent Diffusability

#### 1. 錯配的來源
VA-VAE 的 Table 2 顯示：固定 compute budget 下，隨 tokenizer 維度上升 rFID 下降但 gFID 上升。作者論證這源自「學習無約束高維 latent space 的內在困難」。解法是在訓練 tokenizer 時就把 latent 對齊到 pre-trained vision foundation model。VF Loss 由兩項組成：marginal cosine similarity（L_mcos）與 marginal distance matrix similarity（L_mdms），並用 adaptive weighting 平衡與 reconstruction/KL loss。整合 LightningDiT 後 ImageNet-256 FID 1.35（rFID 0.28）。

#### 2. 各觀點、動機、做法、實證

**(a) 語意可分性 (semantic separability)**
- **REPA（arXiv:2410.06940）**：對齊 noisy hidden state 投影與 clean DINOv2 表徵，正則化係數 λ≈0.5 後飽和；SiT-XL/2 FID 1.42（w/ CFG + guidance interval），訓練加速 >17.5×（<400K steps 匹配 7M steps 表現）。
- **VA-VAE**：把語意先驗「烘焙」進 tokenizer 而非 diffusion 中間層。
- **MAETok**：以 mask autoencoding 學到「較少 GMM modes、較有判別性」的 latent；理論與實證都指向「latent 結構（而非 variational 約束）」才是關鍵；僅用 128 tokens 即達 SOTA、512² gFID 1.69。

**(b) 空間結構 (spatial structure)** — iREPA（arXiv:2512.10794，ICLR 2026）
- 在 27 個 vision encoder 上做大規模分析，發現驅動生成品質的是**空間結構（patch token 間的 pairwise cosine similarity）而非全域語意（ImageNet 準確率）**。
- 三行程式碼改動：spatial regularization layer + 用 conv 取代 MLP projection（MLP 會丟失空間資訊）。 [End2end-diffusion](https://end2end-diffusion.github.io/irepa/)
- 指標：LDS（近鄰是否比遠處更相似）、CDS（相似度隨距離衰減率）、SRSS（前景遮罩下 intra-foreground 一致性）、RMSC（空間多樣性，被 Zhong et al. 排除）。 [arxiv](https://arxiv.org/pdf/2606.03578)
- 加速隨 encoder 增大而增大：PE-B 22.2%、PE-L 38.8%、PE-G 39.6%。 [arxiv](https://arxiv.org/pdf/2512.10794)

**(c) 仿射等變性 (affine equivariance)** — EQ-VAE（arXiv:2502.09509）
- 診斷：既有 AE 對 scaling/rotation 缺乏等變性，導致 latent 複雜。
- 做法：fine-tune 使 decoder 對縮放/旋轉等變。
- 結果：DiT-XL/2 ×7、REPA ×4 加速，不損 reconstruction。
- **但 Zhong et al. 裁決此觀點與 gFID 相關性最低、兩個度量彼此不一致，故被剔除**——這是重要的反面證據。

**(d) 頻譜平滑度 (spectral smoothness)**
- Skorokhodov et al.（arXiv:2502.14831）：DCT 分析發現 bottleneck channel 越大，latent 高頻越過量；高頻干擾 diffusion 的 coarse-to-fine 過程。scale equivariance 對齊 latent 與 RGB 頻譜，僅需 ≤20K fine-tune steps，ImageNet-1K 256² FID −19%、Kinetics-700 影片 FVD ≥−44%。
- Prism Hypothesis（arXiv:2512.19693）：語意 encoder 偏低頻（抽象語意）、pixel encoder 保留高頻（細節）； [Semantic Scholar](https://www.semanticscholar.org/paper/The-Prism-Hypothesis:-Harmonizing-Semantic-and-via-Fan-Diao/e3e8cca6a793e617836ade10016d43218cce4ed8) 提出 UAE 用 frequency-band modulator 融合，DINOv2-L 版 PSNR 29.65 / SSIM 0.88 / rFID 0.19，reconstruction 勝過 RAE。 [HyperAI](https://hyper.ai/en/papers/2512.19693)
- Video VAE 頻譜偏置（Delving into Latent Spectral Biasing of Video VAEs，arXiv:2512.05394）為影片場景對應版本。

**(e) 流形連續性 (manifold continuity)** — iFID（arXiv:2603.05630）
- 動機：rFID 與 gFID 甚至負相關，因為 reconstruction 偏好「分離/斷裂」的 latent，而 diffusion 偏好連續 latent。 [Hugging Face](https://huggingface.co/papers/2603.05630)
- 做法：對每個 latent 取最近鄰、插值、解碼、與真實分布算 FID。 [ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260305630X/abstract)
- 理論：diffusion 樣本集中在連接鄰近訓練樣本的 ridge set 附近，故插值品質預測生成品質。 [arXiv](https://arxiv.org/html/2603.05630v2) 論文並細分：rFID 對應 diffusion 的 refinement phase、iFID 對應 navigation phase。
- 結果：跨多種 VAE 的 Pearson/Spearman ≈0.85，是首個與 gFID 強相關的指標。 [ADS](https://ui.adsabs.harvard.edu/abs/2026arXiv260305630X/abstract) VE（Variance Expansion Loss，arXiv:2603.21085）為同一流形連續性思路的訓練損失版本。

**(f) 分布均勻性 (distribution uniformity)** — VA-VAE 協定
- 用 t-SNE 投影 + density coefficient of variation / Gini / normalized entropy 評估 latent 是否被均勻利用。 [arxiv](https://arxiv.org/pdf/2606.03578) Zhong et al. 直接沿用，但發現其相關性偏弱且與頻譜/VIV 共線。

**(g) 速度歧義 (velocity ambiguity) — VIV（本主題最新貢獻）**
Zhong et al. 的核心方法論創新：
- Flow Matching 目標可分解為 **Reducible Error**（v* 與 vθ 的差）+ **Irreducible Variance**（v 與 Bayes 最優 v* 的差）。後者反映 trajectory crossing 造成的速度歧義。 [arxiv](https://arxiv.org/pdf/2606.03578)
- 把 class-conditional latent 建模為 anisotropic Gaussian（整體為 L-component GMM）。因 d≫M，採 **Kronecker Flip-Flop 協方差分解** Σ≈Σs⊗Σc（channel 與 spatial 可分離），大幅降低參數量、提升每個因子的有效樣本數。
- 解析形式：Iₖ(t)=Σλₖ,ᵢ/[(1−t)²+t²λₖ,ᵢ]；對 t∼U(0,1) 積分得 **Iₖ=(π/2)Σ√λₖ,ᵢ**，可再改寫為 Iₖ=(π/2)√(d(τₖ−d·Aₖ))，其中 τₖ=tr(Σₖ) 為總變異、Aₖ 為標準差譜的各向異性。 [arxiv](https://arxiv.org/pdf/2606.03578)
- **兩個 insight**：(1) **Intra-class compactness**：固定譜形，減小 τₖ 降低 Iₖ；(2) **Spectral anisotropy**：固定總變異，更各向異性（Aₖ 大）降低 Iₖ。∂Iₖ/∂τₖ>0、∂Iₖ/∂Aₖ<0。 [arxiv](https://arxiv.org/pdf/2606.03578)
- 對 T2I 等非 class-conditional 場景，target 不能化簡為單一 class Gaussian，改用 GMM + Monte Carlo 估計 I。

#### 3. 系統性比較研究（Zhong et al., arXiv:2606.03578）
- **實驗設計**：三個 tokenizer 家族——43 個 conv-f16d32、22 個 conv-f16d64、21 個 trans-f16d32（合計 86 個）；每個 tokenizer 訓練 16 epochs；四個下游 backbone SiT-B（400k steps）、SiT-XL（80k）、LightningDiT-B/XL（100k）。控制同家族 reconstruction 品質可比。
- **單一指標相關性排序**：VIV（Velocity Ambiguity）與 gFID 的 **Pearson=0.87**，為最穩定預測子；語意可分性（LNC）、空間結構（SRSS）緊隨其後「stand out」；流形連續性（iFID）、分布均勻性、仿射等變性相關性偏低，**仿射等變性最低且被剔除**（其兩個度量彼此不一致）。個別指標數值僅以散點圖呈現，原始數據於 GitHub（KlingAIResearch/diffusing-right-space）。
- **雙視角聯合迴歸**：**SRSS + LNC 可將 gFID 擬合到 R²=0.91**，說明在空間結構與語意可分性上位於 Pareto 前緣者生成品質較好。多數觀點近似正交，但頻譜平滑度、分布均勻性、VIV 三者呈現共線。
- **跨 backbone 泛化**：VIV 與空間結構最穩定；capacity 從 B→XL 時 SRSS 擬合更好、其他指標相關性下降或持平；LNC 在 SiT 上較好、SEC（Spectral Energy Concentration，本文提出的頻譜代理）在 LightningDiT 上較好。
- **跨家族泛化**：VIV、語意可分性、空間結構皆有效；iFID 在 conv-f16d64 家族相關性特別高（媲美 SRSS）但整體不穩定。
- **更好的 latent → 更直的 ODE 軌跡**：定義 Straightness（相鄰軌跡段 cosine 相似度均值）與 Efficiency（端點位移/累積路徑長），兩者皆與 gFID 高度相關（僅圖示、未給係數）；更好的 latent 讓 length ratio 更接近 1（更均衡的傳輸排程），且 irreducible variance 大的區域對應學到的步長較小。

#### 4. 評估指標的演進
- **gFID 的侷限**：只用 Inception 特徵空間，易被 Goodhart 化（部分 SOTA 模型 FID 已低於真實 validation set 卻仍可辨為假）。 [WisPaper](https://www.wispaper.ai/en/user-blog/frechet-loss-visual-generation-20260506/eng)
- **FDr⁶**（Representation Fréchet Loss，arXiv:2604.28190）：對六個表徵空間（Inception-v3、ConvNeXt-v2、DINOv2、MAE、SigLIP2、CLIP）各自以 validation FD 正規化後取算術平均；並發現 FD 可作為訓練目標（decouple 50k 估計 population 與 1024 batch）， [arXiv](https://arxiv.org/abs/2604.28190) 甚至可把多步 diffusion 免蒸餾轉一步。FDr-CLIP 另作 held-out 評估器。
- **FD_DINOv2 / FDr**：RAEv2 已把 FDr⁶ 與 gFID 並列為主要報告指標。
- **DiffusionBench**（arXiv:2606.24888）：NanoGen 統一框架同時支援 RAE/VAE/pixel-space/MeanFlow，橫跨 ImageNet 與 T2I。 [arXiv](https://arxiv.org/abs/2606.24888) 關鍵反面發現：21 個 latent diffusion 方法在 ImageNet 與 T2I 上的排名**無正相關（Pearson −0.377 至 −0.580）**，代表只在 ImageNet 上優化可能無法遷移到 T2I。 [Hugging Face](https://huggingface.co/papers/2606.24888)

#### 5. 理論
Wang & Pehlevan（arXiv:2503.03206，NeurIPS 2025 Spotlight）以 Gaussian-equivalence 原理解出線性/卷積 denoiser 的 gradient-flow 動態，得到普適的 **inverse-variance spectral law：某模態收斂時間 τ∝λ⁻¹**——高變異（粗結構）比低變異（細節）早數個數量級被學會（線性網路沿主成分、卷積網路沿 Fourier modes 收斂）。這直接支撐了 diffusability 研究的「latent 高頻能量少 = 較易學」直覺（Zhong et al. 在 Latent Smoothness 一節明確引用）。 [arxiv](https://arxiv.org/pdf/2606.03578)

#### 6. 綜合分析：正交還是共線？
- **可跨家族/backbone 泛化的結論**：VIV、語意可分性、空間結構是穩定預測子——這是目前最接近共識的部分。
- **各自 post-hoc 解釋、泛化性弱者**：仿射等變性（被裁決剔除）、分布均勻性、單獨的流形連續性（不穩定）。
- **正交 vs 共線**：多數觀點近似正交（故聯合迴歸能把 R² 提升到 0.91），但頻譜平滑度/分布均勻性/VIV 三者共線——暗示它們可能都是「velocity ambiguity/intra-class 結構」的不同投影。
- **未解問題**：VIV 的 Gaussian/GMM 假設在 RAE 這種極度各向異性、非高斯 latent 上是否成立；指標間的因果 vs 相關；T2I 場景（非 class-conditional）下這些指標是否仍預測有效（DiffusionBench 的負相關發現是警訊）。

**表 1：Diffusability 觀點與正則化方法比較**

| 方法/指標 | 年份/arXiv | 核心想法 | 正則化/度量手段 | 關鍵實驗結果 | 限制 |
|---|---|---|---|---|---|
| REPA | 2024.10 / 2410.06940 | 語意可分性（對齊 diffusion 中間層） | noisy hidden ↔ DINOv2 clean feat. | SiT-XL >17.5× 加速、FID 1.42 | 需選 target encoder；不改 latent 本身 |
| VA-VAE / LightningDiT | 2025.01 / 2501.01423 | 語意先驗烘焙進 tokenizer + 均勻性 | VF Loss（mcos+mdms）+ adaptive weight | FID 1.35、64ep 達 2.11（>21×） | VF Loss 過強會壓縮容量 |
| MAETok | 2025.02 / 2502.03444 | 少 GMM modes、判別性 latent | mask autoencoding（非 variational） | 512² gFID 1.69、76× 訓練、128 tokens | AE 無 variational 保證 |
| EQ-VAE | 2025.02 / 2502.09509 | 仿射等變性 | scale/rotation 等變 fine-tune | DiT-XL ×7、REPA ×4 | Zhong et al. 判定相關性最低、被剔除 |
| Improving Diffusability | 2025.02 / 2502.14831 | 頻譜平滑（抑高頻） | decoder scale equivariance | FID −19%、FVD ≥−44%、≤20K steps | 大 channel bottleneck 才明顯 |
| iFID | 2026.03 / 2603.05630 | 流形連續性 | 最近鄰插值後解碼算 FID | 跨 VAE Pearson/Spearman ≈0.85 | 不可直接最小化 |
| VIV（本文） | 2026.06 / 2606.03578 | 速度歧義（irreducible variance） | GMM + Kronecker flip-flop 解析式 | 與 gFID Pearson 0.87、最穩定 | Gaussian/可分離假設限制 |

---

### 主題二：RAE — Representation Autoencoder 取代 VAE

#### 1. 原始論文（Zheng, Ma, Tong, Xie，arXiv:2510.11690，ICLR 2026）
- **動機**：VAE backbone 過時（conv + 激進降採樣）、latent 維度過低（SD-VAE 把 256² 壓成 32²×4）限制資訊容量、純重建訓練導致表徵弱（ [arXiv](https://arxiv.org/abs/2510.11690) SD-VAE linear probing 僅約 8%）。 [github](https://rae-dit.github.io/)
- **方法**：凍結 DINOv2/SigLIP2/MAE encoder + 訓練 ViT decoder（ [arXiv](https://arxiv.org/abs/2510.11690) L1+LPIPS+GAN loss）；patch size=1，token length 256（與 VAE-DiT 相同，故**不增加 DiT 計算成本**——channels 在第一層投影到 DiT hidden dim）；SD-VAE encoder/decoder 分別比 RAE 貴約 6×/3× GFLOPs。
- **reconstruction**：MAE-B/16 版 rFID 0.16；decoder ViT-B 0.58→ViT-XL 0.49（皆勝 SD-VAE 0.62 且更省 FLOPs，ViT-B 已比 SD-VAE 省 14× GFLOPs）；encoder scaling（DINOv2-S/B/L）rFID 穩定。生成則 DINOv2 最佳（雖然 MAE reconstruction 更好）；linear probing DINOv2-B 84.5%、SigLIP-B 79.1%、MAE-B 68.0%，對比 SD-VAE ~8%。
- **高維 latent 導致 DiT 訓練困難的分析與解法**（直接訓練 DiT-S 完全失敗 gFID 215.76，DiT-XL 亦僅 23.08 遠差於 SD-VAE 的 7.13）：
  1. **Width≥token-dim**：DiT 寬度 d 必須 ≥ RAE token 維度 n，否則 loss 不收斂（DiT-S 對 DINOv2-B/768 完全失敗）。原因：diffusion 注入的 Gaussian noise 把 data manifold「擴散」成 full-rank，需要與完整資料維度匹配的容量。 [github](https://rae-dit.github.io/) 深度加倍幾乎無效。
  2. **Dimension-dependent noise schedule shift**：把 resolution-dependent shift 推廣到「effective data dimension = tokens × dim」；tₘ=αtₙ/[1+(α−1)tₙ]、α=√(m/n)，n=4096 為基準。無 shift gFID 23.08 → 有 shift 4.81。 [github](https://rae-dit.github.io/)
  3. **Noise-augmented decoding**：因 RAE decoder 只見過 clean latent（離散 delta 分布），生成 latent 會 OOD；訓練時加 Gaussian noise 平滑，gFID 4.81→4.28（但 rFID 0.49→0.57，是預期的取捨）。 [github](https://rae-dit.github.io/)
  4. **Wide DDT head（DiT^DH）**：輕量寬 DDT head 提升可擴展性。
- **結果**：DiT^DH 在 ImageNet 256² **1.51 FID（無 guidance）**、以及 256² 與 512² **皆 1.13（w/ AutoGuidance）**。宣稱「RAE 應成為 DiT 新預設」。

#### 2. 後續改進與延伸

**RAEv2 / Improved Baselines（arXiv:2605.18324）**
- **最後 k 層特徵求和**（Multi-Layer Sum, MLS）：x=Σ_{ℓ=L−K+1}^L z_ℓ，免 encoder fine-tune、免特殊資料即大幅改善 reconstruction。
- **RAE 與 REPA 非互斥而是互補**：大規模實證顯示同一表徵既可當 encoder 又可當中間層 distillation target。
- **CFG / self-guidance**：原始 RAE 的 CFG 有問題、需另訓一個弱模型做 AutoGuidance；RAEv2 發現 REPA head h_φ 只用早層特徵做 x-prediction，本身就是「較弱模型」，把 full model 也 re-parameterize 成 x-prediction 即可用來做內部 guidance，省掉第二個模型。
- **效率**：80 epoch 即達 SOTA gFID 與 FDr⁶（相對舊 baseline 的 800 epoch），>10× 收斂加速，且在 T2I、world model 設定皆一致改善。

**Scale-RAE / T2I（arXiv:2601.16208，rae-dit.github.io/scale-rae）**
- SigLIP-2 So400M encoder；相對 FLUX VAE，Qwen2.5-1.5B + 2.4B DiT 從頭訓練達 **GenEval 4.0×、DPG-Bench 4.6× 收斂加速**（apples-to-apples，只差 latent space 與 decoder）。
- decoder 訓練資料：web+synthetic+text-rendering；**文字渲染場景需 targeted data composition**，純 scale 不足。 [arXiv](https://arxiv.org/abs/2601.16208v1)
- 縮放簡化：dimension-dependent noise schedule 仍關鍵，但 wide diffusion head 與 noise-augmented decoding 在大規模下效益可忽略。 [Quantum Zeitgeist](https://quantumzeitgeist.com/fidelity-scaling-text-image-diffusion-transformers/)
- VAE baseline 在高品質資料上約 64 epoch 後災難性過擬合，RAE 因高維語意 latent 有隱式正則化而穩定至 256+ epoch；DiT 從 0.5B 到 9.8B、LLM 從 1.5B 到 7B 皆一致優於 VAE。
- **對多模態架構的意涵**：共享語意 latent 讓 LLM 可直接處理生成的 latent（無需 decode 回 pixel）， [Rae-dit](https://rae-dit.github.io/scale-rae/) 並消除 VAE-based unified model 的「two-tower」設計； [The Moonlight](https://www.themoonlight.io/en/review/scaling-text-to-image-diffusion-transformers-with-representation-autoencoders) 作者展示 latent-space test-time scaling 可行。

**蒸餾（Distilling Drifting Transformers，arXiv:2606.15553）**
- 診斷：RAE latent 因 DINO 特徵強烈 label-wise 聚類而**各向異性極高、ODE 軌跡曲率比 SD-VAE 大約兩個數量級**（isotropy 統計：participation ratio PR 0.2068→0.0630、spectral entropy SE 0.3781→0.2110；dispersion NN-d 4.3083→1.0664、S-MMD 0.00576→0.07913）。
- 傳統 trajectory-based distillation 假設 latent 近似 isotropic、軌跡平滑，故在 RAE 上不穩定。
- 解法：Drifting Models（Deng et al. 2026）直接用兩批樣本計算 drifting field 而非匹配 ODE 軌跡，避開高曲率問題； [arxiv](https://arxiv.org/pdf/2606.15553) 並理論證明 Drifting Model 在極度分散的 reconstruction-based VAE latent 上反而可能失敗——RAE 與 drifting 是互相契合的組合。

**影片對應**
- V-RAE（arXiv:2608.13556）：凍結 VFM + lightweight temporal pooling 去時間冗餘 + video decoder； [arXiv](https://arxiv.org/html/2608.13556) 提出 tFVD 量測時間 latent 平滑度； [X](https://x.com/shumpeiMaxwell/status/2088416836023177371) 以四種凍結 encoder 評估，重申「好 reconstruction 不保證好 generation」。
- VideoRAE（arXiv:2607.14088）：用 V-JEPA2/VideoMAEv2 多尺度特徵 + 1D self-attention projector 壓成緊湊 latent。 [arXiv](https://arxiv.org/html/2607.14088v1)
- PV-VAE（arXiv:2605.02134）、KVAE tokenizer 家族（arXiv:2608.05798）為相關影片 tokenizer 路線。

**其他表徵編碼器路線**
- One Layer Is Enough（arXiv:2512.07829）：只需 adapt 預訓練 encoder 的一層即可用於生成。
- Both Semantics and Reconstruction Matter：讓 representation encoder 適配 T2I 與編輯。
- DINO-SAE（DINO spherical autoencoder）、VAE-REPA、RPiAE（arXiv:2603.19206，representation-pivoted AE，同時強化生成與編輯）、Improving Reconstruction of RAE（arXiv:2602.08620）、VFM-VAE（arXiv:2510.18457）等變體。

#### 3. RAE 與 unified understanding/generation 的關係
RAE 的最大結構意涵是把「理解」（frozen encoder 的判別式表徵）與「生成」（在同一 latent 上跑 diffusion）統一到單一高維語意 latent。這消除了 unified 多模態模型常見的 two-tower（理解 encoder 與生成 VAE 各自為政）設計， [The Moonlight](https://www.themoonlight.io/en/review/scaling-text-to-image-diffusion-transformers-with-representation-autoencoders) 並讓 LLM 能直接在語意 token 上操作、甚至做 latent-space test-time scaling。

#### 4. 批判性分析：RAE 是否應成為新預設？
- **實際限制**：
  - decoder 訓練成本（尤其 T2I 需 targeted data，文字渲染是痛點）。
  - latent 維度高帶來的訓練需求（width≥token-dim、noise schedule shift）——雖然 token 數固定使 DiT 計算不增，但寬度需求推高參數。
  - **CFG 問題**：原始 RAE 需 AutoGuidance；RAEv2 才用 self-guidance 緩解。
  - **蒸餾困難**：各向異性/高曲率使傳統蒸餾失效，需 Drifting Models。
  - **對凍結 encoder 的依賴與偏誤**：直接繼承 DINOv2/SigLIP2 的偏誤；且 MAE reconstruction 最好卻 DINOv2 生成最好，顯示「哪個表徵最適合生成」仍是經驗選擇。
- **反面/未定證據**：DiffusionBench 指出 ImageNet 與 T2I 排名負相關，代表 RAE 在 ImageNet 上的巨大優勢未必等比例遷移；Prism Hypothesis 的 UAE 在 reconstruction（PSNR/SSIM/rFID）上勝過 RAE，暗示純語意 latent 在細節保真上有結構性上限。
- **判斷**：RAE 在 **ImageNet class-conditional 且已具備完整修正配方（noise shift + wide DiT + self-guidance）的設定**下確實應作為新預設；但在 T2I、需蒸餾/一步生成、或需極致 pixel 保真（文字、臉）的場景，宣稱「新預設」仍過早，需搭配額外工程。

**表 2：RAE 路線比較**

| 工作 | 年份/arXiv | 核心貢獻 | 手段 | 關鍵結果 | 限制/反證 |
|---|---|---|---|---|---|
| RAE（原始） | 2025.10 / 2510.11690 | 凍結 SEM encoder + ViT decoder | width≥dim、noise shift、DDT head | 256² 1.51（無G）/1.13（有G） | CFG 差需 AutoGuidance |
| RAEv2 | 2026.05 / 2605.18324 | 簡化 + 加速 | 末 k 層求和、self-guidance、RAE⊕REPA | 80ep 達 SOTA、>10× | 仍依賴凍結 encoder |
| Scale-RAE | 2026.01 / 2601.16208 | 擴展到 T2I | SigLIP-2 So400M | vs FLUX VAE 4.0×/4.6× | 文字需 targeted data |
| Drifting distill. | 2026.06 / 2606.15553 | 解決 RAE 蒸餾 | Drifting Models 取代軌跡匹配 | 曲率大 2 個數量級被繞開 | 需專用蒸餾範式 |
| V-RAE / VideoRAE | 2026.08 / 2608.13556、2607.14088 | 影片延伸 | temporal pooling / 1D projector | 提出 tFVD、改善 gFVD | 時間冗餘與壓縮更難 |

---

### 交叉分析

#### A. RAE 的成功可用哪些 diffusability 指標解釋？
- **語意可分性極高**：RAE 直接繼承 DINOv2 表徵，linear probing 84.5%（對比 SD-VAE ~8%）—— [github](https://rae-dit.github.io/) 完美命中 Zhong et al. 的頭號穩定預測子之一。
- **各向異性也極高**：Drifting 論文量到 RAE 曲率比 SD-VAE 大兩個數量級、PR/SE 更低。 [arxiv](https://arxiv.org/pdf/2606.15553) 這在 VIV 框架下是**雙面刃**：VIV 的 Insight 2 說「spectral anisotropy 大 → irreducible variance 小 → 較好 diffusability」， [arxiv](https://arxiv.org/pdf/2606.03578) 故 RAE 的高各向異性反而**有利於 diffusion 訓練收斂**；但同一各向異性/高曲率**不利於蒸餾**（trajectory-based 假設 isotropy）。這正是「語意可分性↑ 與 anisotropy↑ 的張力」的精確體現：對訓練有利、對蒸餾有害。
- **空間結構**：iREPA 已證明空間結構（而非全域語意）驅動生成——RAE 的 ViT patch token 保留了強空間結構，這也解釋為何它不需 REPA 的空間對齊仍收斂快（RAEv2 進一步證實兩者互補）。

#### B. 與 pixel-space 第三路線的對照
- **pixel-space 陣營**：
  - JiT（Li & He 2025，Back to Basics / arXiv:2511.13720 相關）：x-prediction 取代 velocity。
  - PixelDiT（arXiv:2511.20645）：單階段全 transformer，dual-level DiT 分離全域語意與 pixel 紋理，直接在 1K 解析度訓練；主打免 VAE artifact、編輯時背景一致性佳。
  - PixelGen（arXiv:2602.02493）：perceptual loss 讓 pixel diffusion 聚焦感知流形；ImageNet-256 無 CFG FID 5.11（80 epoch）、T2I GenEval 0.79。
  - There is No VAE / EPG（arXiv:2510.12586）：兩階段自監督預訓練 + end-to-end；ImageNet-256 FID 1.58、512 FID 2.35（75 NFE），僅用 DiT 約 30% compute；consistency model 直接在高解析度訓練達 FID 8.82。
- **核心爭論**：若 latent 的價值主要來自語意先驗（RAE）而非壓縮，pixel-space + 表徵預訓練是否就能取消 latent 層？
- **關鍵反面證據**：DiffusionBench 在統一 NanoGen 框架重訓 JiT/PixNerd/PixelGen [arXiv](https://arxiv.org/html/2606.24888v1) 時，**在 80 epoch 未觀察到加速收斂**（如同傳統 VAE），且 pixel-space 方法在 ImageNet FID 與 T2I 指標上都明顯落後 latent-space 方法（作者主分析甚至排除 pixel-space 以免人為抬高相關性）。iREPA 亦顯示其空間結構技巧能加速 pixel-space JiT——暗示「表徵/空間先驗」才是共通關鍵，而非 latent 與否。
- **判斷**：目前實證天平仍偏向「需要某種 encoder 承載語意/空間先驗」；pixel-space 尚未在統一框架下複現 RAE/latent 的收斂優勢。latent 層的必要性未被證偽，但其角色正從「壓縮」重新定義為「表徵基座」。

---

## Recommendations

**若你要選 tokenizer/latent space（研究或產品）：**
1. **ImageNet class-conditional、追求最快收斂**：直接採 RAE（DINOv2-B encoder + ViT decoder），並務必套用配方——width≥token-dim、dimension-dependent noise schedule shift、self-guidance（RAEv2 版）與末 k 層求和。基準：無 guidance 應接近 1.5 FID、80 epoch 內達 SOTA。
2. **T2I / 大規模**：用 Scale-RAE（SigLIP-2）但**投資 decoder 的 targeted data**（文字、臉），並以 DiffusionBench 同時評 ImageNet 與 T2I，不要只看 ImageNet FID。
3. **若無法負擔 RAE 的 decoder 訓練或高維成本**：退而用 VA-VAE / EQ-VAE / MAETok 對現有 VAE 做語意/等變/mask 正則化——EQ-VAE 只需 ≤20K fine-tune steps，CP 值最高。

**若你在做 diffusability 診斷或指標選擇：**
4. 優先量測 **VIV、LNC（語意可分性 proxy）、SRSS（空間結構）** 三者；用 SRSS+LNC 聯合迴歸預測 gFID（R²≈0.91）。**不要**倚賴仿射等變性或單獨的分布均勻性做決策。
5. 報告時以 **FDr⁶ + iFID** 補充 gFID；若做 T2I，DiffusionBench 是目前唯一橫跨兩任務的統一協定。

**若你要蒸餾/一步生成：**
6. RAE latent 上**避免 trajectory-based distillation**，改用 Drifting Models；或先評估 latent 的 PR/SE/曲率再決定。

**改變上述建議的門檻/benchmark：**
- 若 pixel-space 方法在統一框架（NanoGen 類）下開始展現與 latent 相當的收斂加速，且 T2I 指標追平 → 重新評估是否還需 latent 層。
- 若某 diffusability 指標在 T2I（非 class-conditional）上仍維持 >0.8 相關 → 可推廣為通用選型準則（目前僅在 ImageNet 驗證）。

---

## 開放問題與可能的研究方向
1. **VIV 假設的普適性**：Kronecker 可分離協方差與 anisotropic Gaussian/GMM 假設在極端非高斯、超高各向異性的 RAE latent 上是否成立？若不成立，需要非高斯版的 irreducible variance 估計。
2. **訓練–蒸餾張力的統一解**：能否設計一個 latent space（或 reparameterization）同時滿足「訓練時高各向異性（低 VIV）」與「蒸餾時低曲率（近 isotropic）」？Drifting Models 是一種繞道，但是否有 latent 側的根本解？
3. **T2I 遷移性**：DiffusionBench 顯示 ImageNet↔T2I 排名負相關，代表整條 diffusability/RAE 研究線的結論在真實 T2I 上可能失效。亟需在 freeform T2I 上重做相關性分析。
4. **指標的因果驗證**：目前 VIV/LNC/SRSS 與 gFID 的高相關仍是相關而非因果。需以介入式實驗（直接操控單一 latent 屬性、其他不變）建立因果。
5. **pixel-space 的表徵預訓練上限**：若把 RAE 的語意先驗以自監督預訓練形式注入 pixel-space（EPG 方向），能否在統一框架下複現 latent 的加速？這將直接回答「latent 層是否必要」。
6. **統一 tokenizer**：Prism Hypothesis/UAE 指出語意（低頻）與 pixel（高頻）可頻帶融合。能否設計一個同時最優 rFID、gFID、understanding linear-probe 的單一 tokenizer，讓生成/理解/編輯共用？
7. **評估的最終形態**：FDr⁶ 仍繼承 Fréchet 的 Gaussian 假設；當多個 SOTA 已「FID 低於真實 validation」，需要超越 moment-matching 的分布距離或與人評對齊的自動指標。

---

## Caveats
- **時間與可複現性**：本報告大量引用 2026 年的 arXiv 預印本（如 2606.03578、2605.18324、2606.24888、2606.15553），多數尚未經完整同儕審查，數字可能於改版變動。RAE 的「新預設」宣稱為作者立場，尚待獨立大規模複現。
- **86 個 tokenizer 的「86」**：論文正文未寫單一「86」字樣，該總數為三家族 43+22+21 之和；家族規模不均。個別指標（除 VIV 的 Pearson 0.87 與聯合 R²=0.91 外）的相關係數僅以散點圖呈現，精確數值需查其 GitHub 原始數據（KlingAIResearch/diffusing-right-space）。
- **VIV 的假設限制**：VIV 的解析形式建立在 anisotropic Gaussian / GMM 與 Kronecker 可分離協方差假設上，在高度非高斯、極端各向異性的 RAE latent 上是否精確成立，論文未直接驗證。
- **指標 = 相關而非因果**：diffusability 各指標與 gFID 的高相關不等於因果；多數「解釋」是各方法的 post-hoc 敘事，Zhong et al. 的貢獻正是揭露此點。
- **DiffusionBench 的負相關**：ImageNet 與 T2I 排名 Pearson −0.377 至 −0.580 [Hugging Face](https://huggingface.co/papers/2606.24888) 是強烈警訊——本報告中大量「ImageNet 上」的加速/FID 數字未必遷移到 T2I，讀者切勿外推。
- **部分 arXiv 編號與年份**：Scale-RAE 實際編號為 2601.16208（task 中以 project page URL 指涉）；Wang & Pehlevan 為 2503.03206，於 NeurIPS 2025 Spotlight 發表（task 標記 2026 為會議年份差異）。少數延伸工作（PV-VAE、KVAE、Both Semantics and Reconstruction Matter、Delving into Video VAE Spectral Biasing）僅由二手引用或摘要確認存在，未逐一取得原文核對全部數字。JiT 的 x-prediction 版本對應 Li & He「Back to Basics: Let denoising generative models denoise」（arXiv:2511.13720）。