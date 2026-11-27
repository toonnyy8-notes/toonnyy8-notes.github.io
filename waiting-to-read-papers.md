# 等待閱讀-論文 論文整理

- 來源：Discord「研究手記」伺服器 › #等待閱讀-論文（2025-01-06 頻道起點至 2026-11-19）
- 擷取日期：2026-11-19
- 整理方式：讀取匯出的完整聊天紀錄，依主題分類，同類內依張貼時間排序；重複貼的連結已合併；標題依 arXiv 或論文頁面補上完整標題。

---

## 一、論文（依主題）

1. 記憶、Test-Time Training 與持續學習
   - [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620)（另貼 [OpenReview](https://openreview.net/forum?id=eifW0W0xgt)）
   - [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)（實作：[lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch)）
   - [Test-Time Training Done Right](https://arxiv.org/abs/2505.23884)
   - [Self-Adapting Language Models](https://arxiv.org/abs/2506.10943)
   - [Introducing Nested Learning: A new ML paradigm for continual learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/)（Google Research）
   - [Self-Distillation Enables Continual Learning](https://arxiv.org/abs/2601.19897)
   - [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372)（頻道內貼的是 [Google 分享連結](https://share.google/MVDHEYQYg2FKjGFeH)）
   - [Rethinking Memory as Continuously Evolving Connectivity](https://arxiv.org/abs/2605.28773)
   - [MemTrain: Self-Supervised Context Memory Training](https://arxiv.org/abs/2606.03197)
   - [TEMPO: Scaling Test-time Training for Large Reasoning Models](https://arxiv.org/abs/2604.19295)
   - [Memory for Large Language Models](https://arxiv.org/abs/2607.25380)
   - [Memory Decoder at Scale: A Pretrained, Parametric Long-Term Memory](https://arxiv.org/abs/2607.27919)
   - [Memory Caching: RNNs with Growing Memory](https://arxiv.org/abs/2602.24281)
   - [Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models](https://arxiv.org/abs/2602.04509)（ICML 2026）
   - [Language Models Can Rewrite Their Own Weights](https://arxiv.org/abs/2611.21885)
   - [Learning Dynamics of Continual Learning: A Unified View of Data Attribution, Forgetting, and Plasticity Loss](https://arxiv.org/abs/2609.33620)（Oxford OATML／GDM；[專案頁](https://joshua-ren.github.io/learning-dynamics-cl/)；token 層級 learning dynamics 拆成 CH1／CH2 兩通道，統一解釋資料選擇、碰撞／侵蝕式遺忘與 readout 可塑性喪失；延續 Learning Dynamics of LLM Finetuning）
   - [For-Value: Efficient Forward-Only Data Valuation for finetuning LLMs and VLMs](https://arxiv.org/abs/2508.10180)（只用 forward 的資料估值，即上篇的 CH1）

2. 自蒸餾、強化學習與推理
   - [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
   - [Reasoning with Sampling: Your Base Model is Smarter Than You Think](https://arxiv.org/abs/2510.14901)
   - [PretrainZero: Reinforcement Active Pretraining](https://arxiv.org/abs/2512.03442)
   - [MaxRL: Maximum Likelihood Reinforcement Learning](https://zanette-labs.github.io/MaxRL/)
   - [Evolution Strategies at the Hyperscale](https://arxiv.org/abs/2511.16652)
   - [Self-Improving Training & Thinking](https://arxiv.org/abs/2601.21343)
   - [Reinforcement Learning via Self-Distillation](https://arxiv.org/abs/2601.20802)
   - [How to Steal Reasoning Without Reasoning Traces](https://arxiv.org/abs/2603.07267)
   - [AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning](https://arxiv.org/abs/2608.05987)
   - [Latent On-Policy Self-Distillation](https://arxiv.org/abs/2608.13040)
   - [Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization](https://arxiv.org/abs/2607.10169)（RIPO，ICML 2026）
   - [Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807)

3. Diffusion、Flow Matching 與生成模型
   - [Interpreting and Improving Diffusion Models from an Optimization Perspective](https://arxiv.org/abs/2306.04848)
   - [LanPaint: Training-Free Diffusion Inpainting with Asymptotically Exact and Fast Conditional Sampling](https://arxiv.org/abs/2502.03491)
   - [Token-Shuffle: Towards High-Resolution Image Generation with Autoregressive Models](https://arxiv.org/abs/2504.17789)
   - [Diffusion Model is Effectively Its Own Teacher](https://openaccess.thecvf.com/content/CVPR2025/papers/Ma_Diffusion_Model_is_Effectively_Its_Own_Teacher_CVPR_2025_paper.pdf)（CVPR 2025）
   - [Deep Researcher with Test-Time Diffusion](https://research.google/blog/deep-researcher-with-test-time-diffusion/)（Google Research）
   - [Diversity Has Always Been There in Your Visual Autoregressive Models](https://arxiv.org/abs/2511.17074)
   - [Why Diffusion Models Don't Memorize: The Role of Implicit...](https://openreview.net/forum?id=BSZqpqgqM0)（標題依頻道內預覽，原文截斷）
   - [Do Diffusion Models Learn to Generalize Basic Visual Skills?（Visual Generative Lab）](https://amishsethi.github.io/visual-generative-lab/)（NeurIPS 2026；以規則生成的合成資料測 diffusion 對大小、位置、旋轉、數量、顏色、形狀的外插與組合泛化；[arXiv 2611.22427](https://arxiv.org/abs/2611.22427)、[code](https://github.com/AmishSethi/visual-generative-lab)）
   - [Learning to Integrate Diffusion ODEs by Averaging the Derivatives](https://arxiv.org/abs/2505.14502)（Liu & Yue，NeurIPS 2025；secant loss 直接學 ODE 在一段步長上的積分（平均導數），屬 flow map / MeanFlow 系列的少步生成；[code](https://github.com/poppuppy/secant-expectation)）
   - [LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](https://arxiv.org/abs/2604.15311)
   - [AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation](https://arxiv.org/abs/2605.13724)
   - [Concurrent Image Understanding and Generation: Self-Correcting Coupled Markov Jump Processes](https://arxiv.org/abs/2607.13188)
   - [High-accuracy sampling for diffusion models and log-concave distributions](https://arxiv.org/abs/2602.01338)
   - [dRAE: Representation Autoencoder with Hyper-Spherical Codes](https://arxiv.org/abs/2607.22148)
   - [Hierarchical Diffusion Transformer for End-to-End Image Generation（HiT）](https://arxiv.org/abs/2611.22962)（pixel-space DiT 內以學習式漸進分組壓縮 token、再還原回 patch 位置；[project](https://nicholas0228.github.io/HiT-webpage/)）
   - [Scalable Beckmann Maps Improve One-step Generation（SBM）](https://arxiv.org/abs/2611.28286)（[project](https://tullebulle.github.io/projects/sbm/)；為 Beckmann Transport Models 加上 CFG 與可擴展訓練配方，ImageNet 256 一步 FID 1.61）

4. Diffusion Language Models（dLLM）
   - [DiffCoT: Diffusion-styled Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2601.03559)
   - [LLaDA2.0: Scaling Up Diffusion Language Models to 100B](https://arxiv.org/abs/2512.15745)
   - [One Token Is Enough: Improving Diffusion Language Models with a Sink Token](https://arxiv.org/abs/2601.19657)（dLLM 的 moving sink 現象；加一個只看自己、全域可見的額外 sink token 穩定注意力）

5. 可解釋性與學習理論
   - [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
   - [Why Language Models Hallucinate](https://cdn.openai.com/pdf/d04913be-3f6f-4d2b-b283-ff432ef4aaa5/why-language-models-hallucinate.pdf)（OpenAI）
   - [Maximal Brain Damage Without Data or Optimization: Disrupting Neural Networks via Sign-Bit Flips](https://arxiv.org/abs/2502.07408)
   - [The Remarkable Robustness of LLMs: Stages of Inference?](https://arxiv.org/abs/2406.19384)
   - [To Grok Grokking: Provable Grokking in Ridge Regression](https://arxiv.org/abs/2601.19791)
   - [(How) Do Language Models Track State?](https://arxiv.org/abs/2503.02854)
   - [Proving and Improving the Platonic Representation Hypothesis with JEPAs](https://arxiv.org/abs/2610.39302)
   - [Interpretable Weight-Space Programming](https://www.alphaxiv.org/abs/2611.interpretable-weight-space-programming)
   - [What drives the emergence of symbolic mechanisms in large language models?](https://arxiv.org/abs/2611.13270)（Li & Webb，NeurIPS 2026；符號機制的湧現主要來自資料多樣性，而非架構歸納偏置）
   - [Returning to the Neuron Basis for Interpretability, Part 1: Answering Three Objections](https://alignment.anthropic.com/2026/neuron-basis-theory/)（Anthropic；神經元輸出向量＋J-lens，回應多義性、跨層座標、事實分散三個反對理由）
   - [Returning to the Neuron Basis for Interpretability, Part 2: Using Neurons For Safety Applications](https://alignment.anthropic.com/2026/neuron-basis-applications/)（Anthropic；零樣本偵測、steering、微調稽核、後門 trigger 搜尋、unlearning 稽核）
   - [Logic and Conversation](https://www.sfu.ca/~jeffpell/Cogs300/GriceLogicConversation.pdf)（H. P. Grice，1975；合作原則、四準則與 conversational implicature；與 AI safety 中 CoT／輸出 monitor 只讀字面、無法捕捉言外之意的問題相關）

6. JEPA 與表徵學習
   - [LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures](https://arxiv.org/abs/2509.14252)
   - [Better Together: Leveraging Unpaired Multimodal Data for Stronger Unimodal Models](https://arxiv.org/abs/2510.08492)
   - [INTACT-JEPA](https://zju3dv.github.io/INTACT-JEPA/)
   - [SIGReg tutorial](https://rezabyt.github.io/blogposts/sigreg-tutorial.html)
   - [SADL: On the Demand-Supply Geometry of Representation Collapse](https://arxiv.org/abs/2611.21596)（JEPA 的表徵崩塌；實驗含 LeWorldModel 與 INTACT；[X 討論串](https://x.com/DavidSunok47/status/2123394363124523226)）
   - [bispectrum: Selective G-Bispectra Made Practical](https://arxiv.org/abs/2605.07270)（NeurIPS 2026；[專案頁](https://johmathe.github.io/bispectrum/)；選擇性 G-bispectrum 的對稱不變特徵 PyTorch 函式庫）

7. 資料篩選、合成資料與資料效率
   - [Achieving 10,000x training data reduction with high-fidelity labels](https://research.google/blog/achieving-10000x-training-data-reduction-with-high-fidelity-labels/)（Google Research）
   - [High-dimensional Analysis of Synthetic Data Selection](https://arxiv.org/abs/2510.08123)
   - [Why Less is More (Sometimes): A Theory of Data Curation](https://arxiv.org/abs/2511.03492)
   - [Less is Enough: Synthesizing Diverse Data in LLM Feature Space with Sparse Autoencoders](https://arxiv.org/abs/2602.10388)

8. VLM 與多模態
   - [Large (Vision) Language Models are Unsupervised In-Context Learners](https://arxiv.org/abs/2504.02349)
   - [Language Model as Visual Explainer](https://neurips.cc/virtual/2024/poster/96068)（NeurIPS 2024）
   - [TTRV: Test-Time Reinforcement Learning for Vision Language Models](https://arxiv.org/abs/2510.06783)
   - [Time Blindness: Why Video-Language Models Can't See What Humans Can?](https://arxiv.org/abs/2505.24867)
   - [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785)
   - [SAM Audio: Segment Anything in Audio](https://ai.meta.com/research/publications/sam-audio-segment-anything-in-audio/)（Meta）
   - [Beyond Text-Dominance: Understanding Modality Preference of Omni-modal Large Language Models](https://arxiv.org/abs/2604.16902)
   - [VISTA: A Visual Harness for Reasoning in an Interactive World](https://vista-research.github.io/)
   - [OpenReview fUvuEfZSEE](https://openreview.net/forum?id=fUvuEfZSEE&noteId=Ss61T4bJxw)（標題未知，串內討論歸為 VLM alignment）
   - [Rethinking Layer-Wise Information Allocation for Vision Foundation Model Adaptation](https://arxiv.org/abs/2607.21973)
   - [Editable Visual Design](https://arxiv.org/abs/2609.04034)
   - 2026-09-12 VLM 文獻清單：
     - [Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Answer Correctness in VLMs](https://arxiv.org/abs/2510.17771)
     - [Self-Correction is More than Refinement: A Learning Framework for Visual and Language Reasoning Tasks](https://arxiv.org/abs/2410.04055)
     - [VisCache: Visual KV Cache Pruning for Efficient Vision Large Language Model Inference](https://arxiv.org/abs/2608.24063)
     - [LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference](https://www.researchgate.net/publication/386197058_LOOK-M_Look-Once_Optimization_in_KV_Cache_for_Efficient_Multimodal_Long-Context_Inference)
     - [Make Each Token Count: Towards Improving Long-Context Performance with KV Cache Eviction](https://arxiv.org/abs/2605.09649)
     - [Don't Blind Your VLA: Aligning Visual Representations for OOD](https://openreview.net/pdf/72659aab554382b57e41f355d0fc9bf241a8347f.pdf)
     - [A Survey of Image-Grounded Chain-of-Thought Reasoning in Vision-Language Models](https://aclanthology.org/2026.acl-long.2087.pdf)
     - [An approach to Debugging Reasoning in Vision-Language Models (RVI)](https://openreview.net/forum?id=QOZ7VB1cJm)

9. 高效推論、架構與系統
   - [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
   - [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
   - [SwiftKV: Fast Prefill-Optimized Inference with Knowledge-Preserving Model Transformation](https://arxiv.org/abs/2410.03960)
   - [Deep Fourier-based Arbitrary-scale Super-resolution for Real-time Rendering](https://iamxym.github.io/DFASRR.github.io/)
   - [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)（Thinking Machines）
   - [The End of Manual Decoding: Towards Truly End-to-End Language Models](https://arxiv.org/abs/2510.26697)
   - [KV Cache Transform Coding for Compact Storage in LLM Inference](https://arxiv.org/abs/2511.01815)
   - [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569)
   - [TurboQuant: Redefining AI efficiency with extreme compression](https://share.google/HGnxpQwhssX4U9Kvf)（Google 分享連結）
   - [Quantizing Recurrent Language Models Is Easier Than You Think（LoReS）](https://phnazari.github.io/blog/2026/lores/)（線性遞迴 / SSM 模型的權重、激活與遞迴狀態 PTQ；[arXiv 2611.01864](https://arxiv.org/abs/2611.01864)、[code](https://github.com/camail-official/LoReS)）
   - [MegaTrain: Full Precision Training of 100B+ Parameter Large Language Models on a Single GPU](https://arxiv.org/abs/2604.05091)
   - [Reducing Doom Loops with Final Token Preference Optimization](https://www.liquid.ai/blog/antidoom)（Liquid AI）
   - [A Few Gaussians Is All You Need: SSOG-Attention That Steers Instead of Scores](https://pisoni.ai/posts/ssog/)
   - [Attention Sink Forges Native MoE in Attention Layers: Sink-Aware Training to Address Head Collapse](https://arxiv.org/abs/2602.01203)（ICML 2026；attention sink 讓 attention 層形成原生 MoE，head collapse 的成因，並以 head 負載平衡輔助損失訓練）
   - [DeepCrossAttention: Supercharging Transformer Residual Connections](https://arxiv.org/abs/2502.06785)（ICML 2025；[Google Research 部落格](https://research.google/blog/supercharging-transformer-residual-connections/)；以跨層動態路由與 depth-wise cross-attention 取代殘差連接的等權相加）
   - [Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals（LoopRes）](https://arxiv.org/abs/2611.31094)（NUS；[code](https://github.com/NUS-HPC-AI-Lab/LoopRes)；把預訓練 LLM 微調成 looped Transformer，以 Attention Residuals＋loop-aware key＋殘差窗口讓各 loop 直接讀取先前 loop 輸出）
   - [Give the Drafter a Second Chance to Guess: Parallel Reconsideration for Speculative Decoding](https://arxiv.org/abs/2611.05724)（Google；EAGLE drafter 在不確定位置平行多跑一次 forward 修正猜測，接受長度 +5.3–10.2%）

10. Agent、Context Engineering 與 LLM 應用
    - [A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications](https://arxiv.org/abs/2402.07927)
    - [Long Context vs. RAG for LLMs: An Evaluation and Revisits](https://arxiv.org/abs/2501.01880)
    - [Small Language Models are the Future of Agentic AI](https://arxiv.org/abs/2506.02153)
    - [One Token to Fool LLM-as-a-Judge](https://arxiv.org/abs/2507.08794)
    - [A Survey of AIOps in the Era of Large Language Models](https://arxiv.org/abs/2507.12472)
    - [LL3M: Large Language 3D Modelers](https://threedle.github.io/ll3m/)
    - [Robustness in Large Language Models: A Survey of Mitigation Strategies and Evaluation Metrics](https://arxiv.org/abs/2505.18658)
    - [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/abs/2508.21038)
    - [A brain-inspired agentic architecture to improve planning with LLMs](https://www.nature.com/articles/s41467-025-63804-5)（Nature Communications）
    - [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)（2025-10-11、2026-04-07 各貼一次）
    - [Agent Learning via Early Experience](https://www.themoonlight.io/zh/review/agent-learning-via-early-experience)（頻道內貼的是 Moonlight 導讀）
    - [Self-Harness: Harnesses That Improve Themselves](https://arxiv.org/abs/2606.09498)
    - [Shepherd: Enabling Programmable Meta-Agents via Reversible Agentic Execution Traces](https://arxiv.org/abs/2605.10913)
    - [From Code Foundation Models to Agents and Applications: A Comprehensive Survey and Practical Guide to Code Intelligence](https://arxiv.org/abs/2511.18538)

11. 教材
    - [Pen and Paper Exercises in Machine Learning](https://arxiv.org/abs/2206.13446)
    - [Optimal Transport for Machine Learners](https://arxiv.org/abs/2505.06589)
    - [MIT Diffusion 課程 2026](https://diffusion.csail.mit.edu/2026/index.html)

---

## 二、未收錄內容

收錄標準與 ai-research-topic 整理相同：第一部分只收論文、論文 project page 與原創研究文章。以下為未收錄的部分。

- **GitHub repo / 資源清單**
  - [VILA-Lab/Awesome-DLMs](https://github.com/VILA-Lab/Awesome-DLMs)
  - [piesauce/awesome-dLLM-resources](https://github.com/piesauce/awesome-dLLM-resources)
  - [NVIDIA-NeMo/labs-molt](https://github.com/NVIDIA-NeMo/labs-molt)
  - [nvidia/Nemotron-Labs-Diffusion-3B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B)（HuggingFace 模型）
- **標題未能確認的 OpenReview 連結**（OpenReview 擋自動存取，頻道內也沒有預覽）
  - [BDBdblmyzY](https://openreview.net/pdf?id=BDBdblmyzY)（2025-05-30）
  - [iiZy6xyVVE](https://openreview.net/forum?id=iiZy6xyVVE)（2026-07-16）
  - [GW3umRqsZZ](https://openreview.net/forum?id=GW3umRqsZZ&noteId=HiMKJ3Orgz)（2026-07-17）
- **新聞 / 報導**
  - [Why Your Next LLM Might Not Have A Tokenizer](https://towardsdatascience.com/why-your-next-llm-might-not-have-a-tokenizer/)（Towards Data Science）
  - [Google DeepMind Finds a Fundamental Bug in RAG](https://www.marktechpost.com/2025/09/04/google-deepmind-finds-a-fundamental-bug-in-rag-embedding-limits-break-retrieval-at-scale/)（MarkTechPost，對應 2508.21038）
  - [台積用頂大碩士「顧機台」很浪費？](https://www.cw.com.tw/article/5134290)（天下雜誌）
- **社群貼文**
  - [X：@ReeceShuttle](https://x.com/ReeceShuttle/status/1983224848735178843)（LoRA vs full fine-tuning，NeurIPS '25）
  - [X：@swann_bessa](https://x.com/swann_bessa/status/2120200714630594911)（k-means 升級）
  - [Threads：@theturingpost](https://www.threads.net/@theturingpost/post/DEzW2IsN6B4)（2025-01-15）
  - [Threads：@theturingpost](https://www.threads.com/@theturingpost/post/DMG36ODo0X-)、[@calvin.j.ku](https://www.threads.com/@calvin.j.ku/post/DMH-vGKsgtx)、[@theturingpost](https://www.threads.com/@theturingpost/post/DMIaNl-Ijr6)（2025-07-15）
  - [Threads：@oreo.ai.tech](https://www.threads.com/@oreo.ai.tech/post/DMPLFnDyhdk)（2025-07-19）
  - [Threads：@omarsar0](https://www.threads.com/@omarsar0/post/DNTLLWQNpg7)（2025-08-14）、[@omarsar0](https://www.threads.com/@omarsar0/post/DOt4a4CjC-U)（2025-09-18）
  - [Threads：@pupu123485074](https://www.threads.com/@pupu123485074/post/DNiNhydB2TR)（2025-08-19）
  - [Threads：@jwliao1209](https://www.threads.com/@jwliao1209/post/DQhkWaEk7l9)（2025-11-02）
  - [Threads 分享](https://www.threads.com/share/BARQ-XVFEB/)（2026-08-02）
- **AI 對話 / artifact 分享**
  - [ChatGPT 分享](https://chatgpt.com/share/685d0615-0efc-8001-8b34-81159cb5ae54)（2025-06-26）、[ChatGPT 分享](https://chatgpt.com/share/6a4dd66c-f99c-83ea-b6c2-502416dd6826)（2026-07-08）
  - [Claude artifact](https://claude.ai/public/artifacts/f8f34f33-ca98-4bb2-96fd-8501675e1450)（2026-07-09）
  - [Claude 分享](https://claude.ai/share/5ba71842-f27c-4fc7-994b-c8c3cee1cfff)、[Claude 分享](https://claude.ai/share/1c84df7f-b46c-4ad0-a535-3c2709f1cb70)（2026-07-18）
  - LLM_.html（2026-07-08 上傳的 Discord 附件）
- **其他**
  - [Google Scholar 引用頁](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=5sxXSfwAAAAJ&citation_for_view=5sxXSfwAAAAJ:LkGwnXOMwfcC)（2026-07-08）
  - [YouTube](https://www.youtube.com/watch?v=AGehkbQrBpY)（2026-11-13）
  - 頻道內部連結：[論文頻道](https://discord.com/channels/745220253990256651/1536135705288253522)（2026-08-12）、[轉發頻道](https://discord.com/channels/745220253990256651/1544834951520976926)（2026-11-06）
