# PDF 書庫 論文整理

- 來源：jh3168 提供的本機 PDF 書庫索引（README.md，2026-11-19 於 Discord 提供，之後更新為各 PDF 首頁的完整標題；266 份 PDF，不含 TTT/ebook 副本）
- 擷取日期：2026-11-20
- 整理方式：依原索引的主題分類；每篇以完整標題列出，連結優先用 arXiv（所有 arXiv ID 已以 export.arxiv.org 核對標題），非 arXiv 論文用會議、期刊或學校的官方頁面。同一份 PDF 在多個目錄重複收錄者只列一次，跨主題者在各主題各列一次。

---

## 一、論文（依主題）

1. 擴散模型：基礎與設計空間
   - [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
   - [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)
   - [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
   - [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
   - [Variational Diffusion Models](https://arxiv.org/abs/2107.00630)
   - [Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation](https://arxiv.org/abs/2303.00848)
   - [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
   - [Analyzing and Improving the Training Dynamics of Diffusion Models](https://arxiv.org/abs/2312.02696)（即 EDM2）
   - [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
   - [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720)
   - [Is Noise Conditioning Necessary for Denoising Generative Models?](https://arxiv.org/abs/2502.13129)
   - [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
   - [D-Flow: Differentiating through Flows for Controlled Generation](https://arxiv.org/abs/2402.14017)
   - [Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](https://arxiv.org/abs/2506.02221)
   - [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)
   - [A Continuous Time Framework for Discrete Denoising Models](https://arxiv.org/abs/2205.14987)
   - [Diffusion on the Probability Simplex](https://arxiv.org/abs/2309.02530)

2. 擴散模型：引導
   - [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
   - [Characteristic Guidance: Non-linear Correction for Diffusion Model at Large Guidance Scale](https://arxiv.org/abs/2312.07586)
   - [Guiding a Diffusion Model with a Bad Version of Itself](https://arxiv.org/abs/2406.02507)
   - [Self-Guidance: Boosting Flow and Diffusion Generation on Their Own](https://arxiv.org/abs/2412.05827)
   - [Upsample Guidance: Scale Up Diffusion Models without Training](https://arxiv.org/abs/2404.01709)

3. 擴散模型：取樣器與 ODE 求解
   - [DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics](https://arxiv.org/abs/2310.13268)
   - [UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2302.04867)
   - [Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](https://arxiv.org/abs/2312.00094)（即 AMED-Solver）
   - [High-accuracy sampling for diffusion models and log-concave distributions](https://arxiv.org/abs/2602.01338)

4. 擴散模型：蒸餾與少步生成
   - [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)
   - [Consistency Models](https://arxiv.org/abs/2303.01469)
   - [Improved Techniques for Training Consistency Models](https://arxiv.org/abs/2310.14189)
   - [Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models](https://arxiv.org/abs/2410.11081)（即 sCM）
   - [Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion](https://arxiv.org/abs/2310.02279)
   - [Bidirectional Consistency Models](https://arxiv.org/abs/2403.18035)
   - [One-step Diffusion with Distribution Matching Distillation](https://arxiv.org/abs/2311.18828)
   - [Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation](https://arxiv.org/abs/2404.04057)
   - [Universal Inverse Distillation for Matching Models with Real-Data Supervision (No GANs)](https://arxiv.org/abs/2509.22459)（即 RealUID）
   - [Mean Flows for One-step Generative Modeling](https://arxiv.org/abs/2505.13447)
   - [Improved Mean Flows: On the Challenges of Fastforward Generative Models](https://arxiv.org/abs/2512.02012)
   - [AlphaFlow: Understanding and Improving MeanFlow Models](https://arxiv.org/abs/2510.20771)
   - [Align Your Flow: Scaling Continuous-Time Flow Map Distillation](https://arxiv.org/abs/2506.14603)
   - [Terminal Velocity Matching](https://arxiv.org/abs/2511.19797)
   - [TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](https://arxiv.org/abs/2512.05150)
   - [Inductive Moment Matching](https://arxiv.org/abs/2503.07565)
   - [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770)
   - [Learning to Integrate Diffusion ODEs by Averaging the Derivatives](https://arxiv.org/abs/2505.14502)
   - [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)（Rectified Flow）
   - [Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow](https://arxiv.org/abs/2410.07303)
   - [COT-FM: Cluster-wise Optimal Transport Flow Matching](https://arxiv.org/abs/2603.13395)
   - [Simple ReFlow: Improved Techniques for Fast Flow Models](https://arxiv.org/abs/2410.07815)
   - [Straighten Viscous Rectified Flow via Noise Optimization](https://arxiv.org/abs/2507.10218)（即 VRFNO）

5. 離散擴散與擴散語言模型
   - [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)（D3PM）
   - [Autoregressive Diffusion Models](https://arxiv.org/abs/2110.02037)
   - [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)（SEDD）
   - [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
   - [Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data](https://arxiv.org/abs/2406.03736)
   - [The Diffusion Duality](https://arxiv.org/abs/2506.10892)
   - [The Diffusion Duality, Chapter II: Ψ-Samplers](https://arxiv.org/abs/2602.21185)
   - [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)（LLaDA）
   - [Dream 7B: Diffusion Large Language Models](https://arxiv.org/abs/2508.15487)
   - [Scaling Diffusion Language Models via Adaptation from Autoregressive Models](https://arxiv.org/abs/2410.17891)（DiffuLLaMA）
   - [Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)
   - [DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models](https://arxiv.org/abs/2210.08933)
   - [DiffusER: Discrete Diffusion via Edit-based Reconstruction](https://arxiv.org/abs/2210.16886)
   - [ImageBART: Bidirectional Context with Multinomial Diffusion for Autoregressive Image Synthesis](https://arxiv.org/abs/2108.08827)

6. 擴散模型：曝光偏差
   - [On Error Propagation of Diffusion Models](https://arxiv.org/abs/2308.05021)
   - [Alleviating Exposure Bias in Diffusion Models through Sampling with Shifted Time Steps](https://arxiv.org/abs/2305.15583)
   - [Anti-Exposure Bias in Diffusion Models](https://openreview.net/forum?id=MtDd7rWok1)（ICLR 2025）
   - [Input Perturbation Reduces Exposure Bias in Diffusion Models](https://arxiv.org/abs/2301.11706)（即 DDPM-IP）
   - [Elucidating the Exposure Bias in Diffusion Models](https://arxiv.org/abs/2308.15321)

7. 擴散模型：潛空間、反演與編輯
   - [Unifying Diffusion Models' Latent Space, with Applications to CycleDiffusion and Guidance](https://arxiv.org/abs/2210.05559)
   - [Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations](https://arxiv.org/abs/2410.10792)
   - [Dual Diffusion Implicit Bridges for Image-to-Image Translation](https://arxiv.org/abs/2203.08382)
   - [EDICT: Exact Diffusion Inversion via Coupled Transformations](https://arxiv.org/abs/2211.12446)
   - [BELM: Bidirectional Explicit Linear Multi-step Sampler for Exact Inversion in Diffusion Models](https://arxiv.org/abs/2410.07273)
   - [On Exact Inversion of DPM-Solvers](https://arxiv.org/abs/2311.18387)
   - [Direct Inversion: Boosting Diffusion-based Editing with 3 Lines of Code](https://arxiv.org/abs/2310.01506)（即 PnP Inversion，arXiv 現行標題為 Direct Inversion）
   - [Effective Real Image Editing with Accelerated Iterative Diffusion Inversion](https://arxiv.org/abs/2309.04907)
   - [FireFlow: Fast Inversion of Rectified Flow for Image Semantic Editing](https://arxiv.org/abs/2412.07517)
   - [Exact Diffusion Inversion via Bi-directional Integration Approximation](https://arxiv.org/abs/2307.10829)
   - [An Edit Friendly DDPM Noise Space: Inversion and Manipulations](https://arxiv.org/abs/2304.06140)
   - [Precise Diffusion Inversion: Towards Novel Samples and Few-Step Models](https://openreview.net/forum?id=2sMk2ShRdP)（NeurIPS 2025）
   - [Spectral Collapse in Diffusion Inversion](https://arxiv.org/abs/2602.13303)
   - [SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073)（相關）
   - [ChordEdit: One-Step Low-Energy Transport for Image Editing](https://arxiv.org/abs/2602.19083)（相關）

8. 擴散模型：最佳傳輸與耦合
   - [Optimal Flow Matching: Learning Straight Trajectories in Just One Step](https://arxiv.org/abs/2403.13117)
   - [Flow Matching with Semidiscrete Couplings](https://arxiv.org/abs/2509.25519)
   - [Immiscible Diffusion: Accelerating Diffusion Training with Noise Assignment](https://arxiv.org/abs/2406.12303)
   - [The Coupling Within: Flow Matching via Distilled Normalizing Flows](https://arxiv.org/abs/2603.09014)
   - [Explorative Modeling: Unlocking a Third Pretraining Axis and End-to-End Generation](https://arxiv.org/abs/2607.27372)

9. 擴散模型：逆問題與 Plug-and-Play
   - [Diffusion Models as Plug-and-Play Priors](https://arxiv.org/abs/2206.09012)
   - [Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687)（即 DPS）
   - [Principled Probabilistic Imaging using Diffusion Models as Plug-and-Play Priors](https://arxiv.org/abs/2405.18782)
   - [Provably Robust Score-Based Diffusion Posterior Sampling for Plug-and-Play Image Reconstruction](https://arxiv.org/abs/2403.17042)（即 DPnP）
   - [Plug-and-Play Split Gibbs Sampler: Embedding Deep Generative Priors in Bayesian Inference](https://arxiv.org/abs/2304.11134)（即 PnP-SGS；arXiv 標題為 Plug-and-Play split Gibbs sampler）
   - [Plug-and-Play Methods for Integrating Physical and Learned Models in Computational Imaging](https://arxiv.org/abs/2203.17061)（PDF 首頁為 Physics-Driven Machine Learning for Computational Imaging）
   - [ArrayDPS: Unsupervised Blind Speech Separation with a Diffusion Prior](https://arxiv.org/abs/2505.05657)
   - [UnDiff: Unsupervised Voice Restoration with Unconditional Diffusion Model](https://arxiv.org/abs/2306.00721)
   - [Score-Based Diffusion Models as Principled Priors for Inverse Imaging](https://arxiv.org/abs/2304.11751)
   - [Ambient Diffusion: Learning Clean Distributions from Corrupted Data](https://arxiv.org/abs/2305.19256)
   - [Measurement Score-Based Diffusion Model](https://arxiv.org/abs/2505.11853)
   - [Diffusion with Forward Models: Solving Stochastic Inverse Problems Without Direct Supervision](https://arxiv.org/abs/2306.11719)

10. 擴散模型 × 強化學習
   - [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
   - [GRPO-Guard: Mitigating Implicit Over-Optimization in Flow Matching via Regulated Clipping](https://arxiv.org/abs/2510.22319)
   - [Direct Discriminative Optimization: Your Likelihood-Based Visual Generative Model is Secretly a GAN Discriminator](https://arxiv.org/abs/2503.01103)
   - [Is Conditional Generative Modeling all you need for Decision-Making?](https://arxiv.org/abs/2211.15657)（即 Decision Diffuser）
   - [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)
   - [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning](https://arxiv.org/abs/2208.06193)（即 Diffusion-QL）
   - [IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies](https://arxiv.org/abs/2304.10573)
   - [Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning](https://arxiv.org/abs/2304.12824)（即 QGPO）
   - [Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling](https://arxiv.org/abs/2209.14548)（即 SfBC）
   - [Planning with Diffusion for Flexible Behavior Synthesis](https://arxiv.org/abs/2205.09991)

11. 擴散模型應用：影像、3D 與多模態
   - [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)（即 ControlNet）
   - [LION: Latent Point Diffusion Models for 3D Shape Generation](https://arxiv.org/abs/2210.06978)
   - [Diffusion Probabilistic Fields](https://arxiv.org/abs/2303.00165)
   - [Score-based Generative Modeling in Latent Space](https://arxiv.org/abs/2106.05931)（即 LSGM）
   - [Diffusion-GAN: Training GANs with Diffusion](https://arxiv.org/abs/2206.02262)
   - [Diffusion for World Modeling: Visual Details Matter in Atari](https://arxiv.org/abs/2405.12399)
   - [DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models](https://arxiv.org/abs/2501.18590)
   - [Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](https://arxiv.org/abs/2501.18804)
   - [Lance: Unified Multimodal Modeling by Multi-Task Synergy](https://arxiv.org/abs/2605.18678)
   - [Idempotent Generative Network](https://arxiv.org/abs/2311.01462)
   - [LEDiT: Your Length-Extrapolatable Diffusion Transformer without Positional Encoding](https://arxiv.org/abs/2503.04344)
   - [MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning](https://arxiv.org/abs/2307.16424)
   - [Meta-DM: Applications of Diffusion Models on Few-Shot Learning](https://arxiv.org/abs/2305.08092)
   - [ProtoDiff: Learning to Learn Prototypical Networks by Task-Guided Diffusion](https://arxiv.org/abs/2306.14770)

12. 語音：合成、轉換與聲碼器
   - [Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech](https://arxiv.org/abs/2105.06337)
   - [Diff-TTS: A Denoising Diffusion Model for Text-to-Speech](https://arxiv.org/abs/2104.01409)
   - [Guided-TTS: A Diffusion Model for Text-to-Speech via Classifier Guidance](https://arxiv.org/abs/2111.11755)
   - [Guided-TTS 2: A Diffusion Model for High-quality Adaptive Text-to-Speech with Untranscribed Data](https://arxiv.org/abs/2205.15370)
   - [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446)
   - [DiffSVC: A Diffusion Probabilistic Model for Singing Voice Conversion](https://arxiv.org/abs/2105.13871)
   - [SpecGrad: Diffusion Probabilistic Model based Neural Vocoder with Adaptive Noise Spectral Shaping](https://arxiv.org/abs/2203.16749)
   - [Diffusion-Based Voice Conversion with Fast Maximum Likelihood Sampling Scheme](https://arxiv.org/abs/2109.13821)
   - [UNIT-DDPM: UNpaired Image Translation with Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2104.05358)
   - [WavTTS: Towards High-Quality Zero-Shot TTS via Direct Raw Waveform Modeling](https://arxiv.org/abs/2606.03455)
   - [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://arxiv.org/abs/1905.05879)
   - [Global Rhythm Style Transfer Without Text Transcriptions](https://arxiv.org/abs/2106.08519)（即 AutoPST）
   - [Unsupervised Speech Decomposition via Triple Information Bottleneck](https://arxiv.org/abs/2004.11284)（即 SpeechSplit）
   - [VQMIVC: Vector Quantization and Mutual Information-Based Unsupervised Speech Representation Disentanglement for One-shot Voice Conversion](https://arxiv.org/abs/2106.10132)
   - [Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion](https://arxiv.org/abs/1906.00794)
   - [NVC-Net: End-to-End Adversarial Voice Conversion](https://arxiv.org/abs/2106.00992)
   - [HiFi-VC: High Quality ASR-Based Voice Conversion](https://arxiv.org/abs/2203.16937)
   - [FragmentVC: Any-to-Any Voice Conversion by End-to-End Extracting and Fusing Fine-Grained Voice Fragments With Attention](https://arxiv.org/abs/2010.14150)
   - [NoiseVC: Towards High Quality Zero-Shot Voice Conversion](https://arxiv.org/abs/2104.06074)
   - [CVC: Contrastive Learning for Non-parallel Voice Conversion](https://arxiv.org/abs/2011.00782)
   - [Any-to-Many Voice Conversion with Location-Relative Sequence-to-Sequence Modeling](https://arxiv.org/abs/2009.02725)
   - [Limited Data Emotional Voice Conversion Leveraging Text-to-Speech: Two-stage Sequence-to-Sequence Training](https://arxiv.org/abs/2103.16809)
   - [TTS Skins: Speaker Conversion via ASR](https://arxiv.org/abs/1904.08983)
   - [High Fidelity Speech Synthesis with Adversarial Networks](https://arxiv.org/abs/1909.11646)（即 GAN-TTS）
   - [DurIAN: Duration Informed Attention Network For Multimodal Synthesis](https://arxiv.org/abs/1909.01700)
   - [PortaSpeech: Portable and High-Quality Generative Text-to-Speech](https://arxiv.org/abs/2109.15166)
   - [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)
   - [Learning the Beauty in Songs: Neural Singing Voice Beautifier](https://arxiv.org/abs/2202.13277)（即 NeuralSVB）
   - [Online Phase Reconstruction via DNN-based Phase Differences Estimation](https://arxiv.org/abs/2211.08246)

13. GAN
   - [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
   - [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)
   - [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657)
   - [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
   - [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020)
   - [StarGAN v2: Diverse Image Synthesis for Multiple Domains](https://arxiv.org/abs/1912.01865)
   - [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)
   - [Energy-based Generative Adversarial Network](https://arxiv.org/abs/1609.03126)
   - [Idempotent Generative Network](https://arxiv.org/abs/2311.01462)
   - [CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training](https://arxiv.org/abs/1703.10155)
   - [A Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html)（JMLR 2012，MMD）

14. VAE
   - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
   - [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691)
   - [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl)（ICLR 2017）
   - [CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training](https://arxiv.org/abs/1703.10155)

15. Normalizing Flows
   - [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770)
   - [Normalizing Flows: An Introduction and Review of Current Methods](https://arxiv.org/abs/1908.09257)
   - [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)
   - [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)
   - [Improving Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/abs/1606.04934)
   - [Masked Autoregressive Flow for Density Estimation](https://arxiv.org/abs/1705.07057)

16. 能量模型
   - [How to Train Your Energy-Based Models](https://arxiv.org/abs/2101.03288)
   - [Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](https://arxiv.org/abs/2504.10612)
   - [Improving Adversarial Energy-Based Model via Diffusion Process](https://arxiv.org/abs/2403.01666)
   - [A Unified Contrastive Energy-based Model for Understanding the Generative Ability of Adversarial Training](https://arxiv.org/abs/2203.13455)

17. Neural ODE 與連續時間模型
   - [Generative Modeling with Neural Ordinary Differential Equations](https://uwspace.uwaterloo.ca/handle/10012/15354)（Tim Dockhorn 碩士論文，University of Waterloo）
   - [Liquid Time-constant Networks](https://arxiv.org/abs/2006.04439)
   - [Neural Operator: Learning Maps Between Function Spaces](https://arxiv.org/abs/2108.08481)

18. 自監督學習
   - [A Survey on Self-Supervised Representation Learning](https://arxiv.org/abs/2308.11455)
   - [A Survey on Self-supervised Learning: Algorithms, Applications, and Future Trends](https://arxiv.org/abs/2301.05712)
   - [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)（SimCLR）
   - [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)（MoCo）
   - [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)（MoCo v2）
   - [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057)（MoCo v3）
   - [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)（BYOL）
   - [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)（SimSiam）
   - [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)（DINO）
   - [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
   - [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)
   - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
   - [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
   - [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
   - [Self-Supervised MultiModal Versatile Networks](https://arxiv.org/abs/2006.16228)
   - [BYOL-Explore: Exploration by Bootstrapped Prediction](https://arxiv.org/abs/2206.08332)

19. 半監督學習
   - [A Survey on Deep Semi-supervised Learning](https://arxiv.org/abs/2103.00550)
   - [Semi-Supervised and Unsupervised Deep Visual Learning: A Survey](https://arxiv.org/abs/2208.11296)
   - [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252)
   - [Improved Noisy Student Training for Automatic Speech Recognition](https://arxiv.org/abs/2005.09629)

20. 元學習
   - [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)（MAML）
   - [Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML](https://arxiv.org/abs/1909.09157)（ANIL）
   - [Meta-Learning with Implicit Gradients](https://arxiv.org/abs/1909.04630)（iMAML）
   - [On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999)（Reptile）
   - [Meta-Learning with Adaptive Hyperparameters](https://arxiv.org/abs/2011.00209)（ALFA）
   - [Optimization as a Model for Few-Shot Learning](https://openreview.net/forum?id=rJY0-Kcll)（ICLR 2017）
   - [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474)
   - [MetaDiff: Meta-Learning with Conditional Diffusion for Few-Shot Learning](https://arxiv.org/abs/2307.16424)
   - [Meta-DM: Applications of Diffusion Models on Few-Shot Learning](https://arxiv.org/abs/2305.08092)
   - [ProtoDiff: Learning to Learn Prototypical Networks by Task-Guided Diffusion](https://arxiv.org/abs/2306.14770)

21. 測試時訓練與長期記憶
   - [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)（TTT）
   - [Test-time regression: a unifying framework for designing sequence models with associative memory](https://arxiv.org/abs/2501.12352)
   - [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
   - [Test-Time Training Done Right](https://arxiv.org/abs/2505.23884)
   - [In-Place Test-Time Training](https://arxiv.org/abs/2604.06169)
   - [End-to-End Test-Time Training for Long Context](https://arxiv.org/abs/2512.23675)（TTT-E2E）
   - [Modular TTT: Rethinking Test-Time Training as Composable Modules](https://arxiv.org/abs/2608.07110)
   - [TEMPO: Scaling Test-time Training for Large Reasoning Models](https://arxiv.org/abs/2604.19295)
   - [Learning to Discover at Test Time](https://arxiv.org/abs/2601.16175)（TTT-Discover）
   - [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/abs/2302.10866)
   - [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

22. 大型語言模型：推理與代理
   - [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
   - [The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity](https://arxiv.org/abs/2506.06941)
   - [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687)
   - [SkillOpt: Executive Strategy for Self-Evolving Agent Skills](https://arxiv.org/abs/2605.23904)

23. 大型語言模型：對齊、編輯與解碼時對齊
   - [Language Models Resist Alignment: Evidence From Data Compression](https://arxiv.org/abs/2406.06144)
   - [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/abs/2410.02355)
   - [Linear Alignment: A Closed-form Solution for Aligning Human Preferences without Tuning and Feedback](https://arxiv.org/abs/2401.11458)
   - [RAIN: Your Language Models Can Align Themselves without Finetuning](https://arxiv.org/abs/2309.07124)
   - [Transfer Q Star: Principled Decoding for LLM Alignment](https://arxiv.org/abs/2405.20495)

24. 大型語言模型：高效訓練、推論與長度外推
   - [Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful](https://arxiv.org/abs/2507.07101)
   - [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238)
   - [CLEX: Continuous Length Extrapolation for Large Language Models](https://arxiv.org/abs/2310.16450)
   - [Thus Spake Long-Context Large Language Model](https://arxiv.org/abs/2502.17129)
   - [With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation](https://arxiv.org/abs/2401.11504)

25. 大型語言模型：微調
   - [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
   - [Fine-Tuning Language Models with Just Forward Passes](https://arxiv.org/abs/2305.17333)（MeZO）

26. 非自回歸文字生成
   - [Insertion Transformer: Flexible Sequence Generation via Insertion Operations](https://arxiv.org/abs/1902.03249)
   - [Levenshtein Transformer](https://arxiv.org/abs/1905.11006)

27. 文字風格轉換
   - [Deep Learning for Text Style Transfer: A Survey](https://arxiv.org/abs/2011.00416)

28. 離線強化學習與決策模型
   - [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
   - [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)（IQL）
   - [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860)（TD3+BC）
   - [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
   - [Looped World Models](https://arxiv.org/abs/2606.18208)

29. 情感支持對話
   - [Towards Emotional Support Dialog Systems](https://arxiv.org/abs/2106.01144)
   - [ESCoT: Towards Interpretable Emotional Support Dialogue Systems](https://arxiv.org/abs/2406.10960)
   - [MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models](https://arxiv.org/abs/2308.09729)
   - [Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/abs/2308.09687)

30. 其他
   - [CLIP as RNN: Segment Countless Visual Concepts without Training Endeavor](https://arxiv.org/abs/2312.07661)（CVPR 2024）
   - [From data to functa: Your data point is a function and you can treat it like one](https://arxiv.org/abs/2201.12204)
   - [Spatial Functa: Scaling Functa to ImageNet Classification and Generation](https://arxiv.org/abs/2302.03130)
   - [Erasing Appearance Preservation in Optimization-based Smoothing](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/4677_ECCV_2020_paper.php)（ECCV 2020）
   - [Traits: Composable Units of Behaviour](https://doi.org/10.1007/978-3-540-45070-2_12)（ECOOP 2003）

## 二、未收錄內容

- What Is It Like to Be a Noise? An Entropy-based Gaussian Noise Regularization for Diffusion Models（`dpm/latent-inversion/`）：arXiv、OpenReview 皆查無，未找到官方頁面
- Diffusion-based RL 入門指南（`入門指南.pdf`、`diffusion_rl_beginner_guide.md.pdf`）：中文指南文件，非論文
- Diffusion-based plug-and-play prior in zero-shot speech separation with speaker consistency（`20250409.pdf`）：簡報，非論文
