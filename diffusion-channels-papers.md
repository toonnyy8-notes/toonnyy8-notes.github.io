# causal-diffusion & diffusion-model 論文整理

- 來源：Discord「研究手記」伺服器 › #causal-diffusion（19 則訊息）、#diffusion-model（89 則訊息，自 2025-02-26 頻道起點起）
- 擷取日期：2026-11-19
- 整理方式：讀取兩個頻道的全部訊息，依張貼時間排序；重複貼的連結已合併；標題缺漏或被截斷者，依 arXiv／論文頁面補上完整標題。

---

## 一、論文（依頻道）

1. causal-diffusion
   - [Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation](https://arxiv.org/abs/2605.15141)
   - [ABot-World-0: Infinite Interactive World Rollout on a Single Desktop GPU](https://arxiv.org/abs/2607.19191)
   - [Self Gradient Forcing: Native Long Video Extrapolation](https://arxiv.org/abs/2607.20368)
   - [AURORA-LM: Autoencoding Unified Representation for Continuous-Latent Diffusion Language Modeling](https://arxiv.org/abs/2608.02602)
   - [Streaming Autoregressive Video Generation via Diagonal Distillation](https://arxiv.org/abs/2603.09488)（頻道內貼的是 [GitHub repo](https://github.com/Sphere-AI-Lab/diagdistill)，論文連結取自 repo）
   - [XYZFlow: Scaling Multi dimensional Shortcut Flows for Efficient Generative Modeling](https://arxiv.org/pdf/2608.12276)
   - [Why Do Video Diffusion Models Violate Physics? Unveiling the Flaws in Attention Mechanisms](https://arxiv.org/abs/2609.23658)
   - [Looking Ahead: Modeling Future Evolution in Autoregressive Video Generation](https://arxiv.org/abs/2610.29989)

2. diffusion-model
   - [PFGM++: Unlocking the Potential of Physics-Inspired Generative Models](https://arxiv.org/abs/2302.04265)
   - [Denoising Diffusion Implicit Models](https://arxiv.org/pdf/2010.02502)
   - [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)（串內標註：AutoRegressive (LM Diffusion)）
   - [Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](https://arxiv.org/abs/2504.10612)
   - [Optimal Flow Matching: Learning Straight Trajectories in Just One Step](https://openreview.net/forum?id=kqmucDKVcU)
   - [Flow Matching with Semidiscrete Couplings](https://arxiv.org/abs/2509.25519)
   - [Unified Latents (UL): How to train your latents](https://arxiv.org/pdf/2602.17270)
   - [Elastic Token Compression for Pixel-Space Diffusion Transformers (RTI)](https://eduardzamfir.github.io/rti/)
   - [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://bfl.ai/research/self-flow)
   - [Beckmann Transport Models: From Autonomous Flows to One-Step Maps](https://arxiv.org/abs/2608.01692)
   - [Better Source, Better Flow: Learning Condition-Dependent Source Distribution for Flow Matching](https://arxiv.org/abs/2602.05951v2)
   - [On the Diffusibility of High-Dimensional Latents](https://cfeng16.github.io/on_the_diffusibility/)
   - [Diffusion Controller: Framework, Algorithms and Parameterization](https://arxiv.org/abs/2603.06981)
   - [PDMD: Projected Distribution Matching Distillation for Video Diffusion Models](https://arxiv.org/pdf/2609.35768)
   - [Looped Diffusion Transformer](https://arxiv.org/abs/2609.40305)
   - [LiFT: Loop Flow Transformers](https://www.alphaxiv.org/abs/2610.05538)
   - [DiffusionART: Efficient, Stable and Scalable Reinforcement of Diffusion Models](https://arxiv.org/abs/2610.12572)（從 ai-research-topic › RL 轉貼）
   - [Few-Step Generation via Data-Space Iteration](https://arxiv.org/abs/2610.12102)
   - [Are We Overthinking One-Step Generation?](https://arxiv.org/abs/2610.20331)
   - [Langevin Flow Maps: Efficient Molecular Dynamics and Transition Path Sampling](https://arxiv.org/abs/2610.05998)
   - [SCALLOP: Few-Step Boltzmann Generators via Scalable Likelihood Flow Maps](https://tonyauyeung.github.io/scallop/)
   - [Correction Training for Diffusion Models](https://sunovivid.github.io/correction-training/)
   - [Normalizing Flow Maps](https://the-bose-lab.github.io/normalizing-flow-maps/)
   - [Rethinking the Design Space of Vanilla Pixel Diffusion Transformers](https://soumik-kanad.github.io/mflow-diffusion/)
   - [Global Transport Couplings for Classifier-Free Guided Flows](https://arxiv.org/abs/2610.07555)
   - [LDM-is-AE: Latent Diffusion Model is an Auto-Encoder for End-to-End Image Generation](https://arxiv.org/abs/2609.37080)
   - [Strong Stochastic Flow Maps](https://arxiv.org/abs/2606.01086)
   - [Emergence of diffusion models from associative memory through memorization and generalization](https://www.nature.com/articles/s41467-026-78727-y)（Nature Communications）
   - [When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis](https://arxiv.org/abs/2509.24912)
   - [Understanding and Learning Prediction Targets in Diffusion Models](https://arxiv.org/abs/2601.21419)
   - [AnisoFlow: Learning Anisotropic Prediction Targets for Flow Models](https://hanshengchen.com/anisoflow/)
   - [eDMD: Endpoint Distribution Matching for Few-Step Diffusion Distillation](https://huggingface.co/papers/2611.17194)
   - [DWD: Decoding Without a Decoder](https://hanshengchen.com/dwd/)

---

## 二、未收錄內容

收錄標準與 ai-research-topic 整理相同：第一部分只收論文、論文 project page 與原創研究文章。以下為未收錄的部分。

- **causal-diffusion**
  - [meituan-longcat/LongCat-Video](https://github.com/meituan-longcat/LongCat-Video)（GitHub repo）
  - [Block-wise SMC for Zero-Shot Diffusion Source Separation](https://claude.ai/public/artifacts/85e9eaf3-8bfb-47d7-8097-4cb4a84ba807)（Claude artifact）
  - [Causal Attention & Diffusion Models: Sequence Generation Guide](https://claude.ai/public/artifacts/a97f9117-7f69-4d59-9083-cff2b36dc7b6)（Claude artifact）
  - 純文字討論：causal 的定義＝「前面影響後面」
- **diffusion-model**
  - [Diffusion学习笔记（二）——Score Matching、Langevin Equation 和 Diffusion的联系](https://zhuanlan.zhihu.com/p/618685170?utm_id=0)（知乎筆記）
  - [Diffusion学习笔记（八）——PFGM++](https://zhuanlan.zhihu.com/p/632080817?utm_id=0)（知乎筆記）
  - [X：@jrrhuang](https://x.com/jrrhuang/status/2110412722424983789)
  - [FORS 圖解（只用 Score 的高精度取樣）](https://claude.ai/artifact/XZ4unTbPR2qq9iTth4m61h)（Claude artifact）
  - [從零讀懂 FORS](https://claude.ai/artifact/MLUSL1LVqi6athavVnvUQt)（Claude artifact；FORS 論文本身頻道內沒有附連結）
  - [Flow Matching 配對入門](https://claude.ai/artifact/BCw5vBnNwhDLx54E1Ai4kd)（Claude artifact）
  - [Gemini 版本的教材](https://share.gemini.google/7EONuiRAZoju)（Gemini 分享，用來和 Claude 版本比較）
  - LDM-is-AE 導讀（2026-10-25 轉貼的導讀頁，無外部連結）
  - 純文字筆記：2026-02-13 轉貼「關鍵字是 Optimal-Transport-Based Flow Matching」
