# 層間權重共享 × 層級 LoRA 鬆綁：技術演進與批判性分析

## TL;DR
- **層間冗餘可以被壓縮掉，但目前壓縮的主要收益是「資源效率」（用更少參數與記憶體打平原有效能），而非「效能上界」的真正提升**——Bae et al. 的 Relaxed Recursive Transformer 把 Gemma 2B 壓成約 1B 參數，並在 60B token uptraining + 知識蒸餾下逼近原模型（few-shot 平均 58.4% vs 58.6%），這是「以少打平」的最佳證據，但沒有超越原本 2B 模型的上界。
- **唯一有較強證據支持「能力提升」的維度是「推理密集任務」**：Saunshi et al.（arXiv:2502.17416, ICLR 2025）在受控 iso-FLOP 比較下顯示 looped 模型在推理任務上逼近甚至偶爾超越同運算量的非 looped 模型（例如數學字題上 (12⊗2) looped 模型達 34.3%，超越 24 層 iso-FLOP 基線的 29.3%，「despite having 50% fewer parameters and worse perplexity」），但 perplexity 幾乎總是較差；Ouro（arXiv:2510.25741）與 Geiping et al. 的 recurrent-depth 模型把「深度即運算」擴展到 test-time compute。這些本質是「用同參數換更多迭代運算深度」，是把冗餘換成可迭代的運算，而非消滅冗餘後的免費午餐。
- **與 Lad et al. 四階段假說的交叉最關鍵的啟示是：U 形層敏感度（首尾脆弱、中間穩健）強烈暗示「共享程度與 LoRA rank 應隨深度非均勻分配」，但截至 2026 年 9 月，尚無論文直接以可解釋性為依據做深度非均勻的 rank 分配——這是明確且高價值的研究空白。**

## Key Findings

1. **技術樹主幹**：早期跨層共享（Universal Transformer 2018、ALBERT 2019、DEQ）→ 理論性 looped transformers（Giannou 2023、Fan 2024）→ 現代「轉換式」遞迴（Bae et al. RRT, 2024/ICLR 2025）→ 動態深度與 test-time compute（MoR 2025、Ouro 2025、Geiping 2025）。核心思想從「省參數」演進到「把深度當成一條獨立的 scaling 軸」。

2. **RRT 的核心貢獻與數字**：Recursive Transformer 以 Stepwise 初始化在 15B token uptraining 下，recursive Gemma 1B 達 few-shot 平均 51.7%，比從頭訓練的同尺寸 Gemma 1B（38.2%）高 13.5 個百分點（原文：「up to 13.5 absolute accuracy improvement (22% error reduction) on few-shot tasks compared to a non-recursive Gemma 1B model (pretrained from scratch)」），且勝過完整預訓練的 Pythia 1B（48.8%）。加上 LoRA 鬆綁（rank 512、SVD 初始化）與蒸餾、60B token 後達 58.4%，逼近完整 Gemma 2B（58.6%）。

3. **關鍵區分（資源效率 vs 能力提升）**：層共享/剪枝/合併/MoD 皆屬「資源效率」路線（證據強）；looped 模型的推理優勢屬「能力提升」但僅限特定任務、且以更多 test-time 運算換取（證據中等）；LoRA 鬆綁是把「純資源效率」往回退一步以換取更好的準確率（參數量回升，證據強）。

4. **LoRA 鬆綁的代價明確**：rank 從 0 → 512，Gemma 非嵌入參數由 0.99B 增至 1.60B（+0.61B，達 vanilla 的 80%），few-shot 從 46.7% 升至 55.5%。rank 越高越好但呈現遞減報酬，且會破壞「單一權重區塊常駐」的 serving 優勢與 continuous depth-wise batching 的效率。

5. **後續工作**：Mixture-of-Recursions（MoR, arXiv:2507.10524, NeurIPS 2025）加入 per-token 動態遞迴深度路由器； [alphaXiv](https://www.alphaxiv.org/abs/2507.10524v1) Ouro（LoopLM）證明「recurrence 提升的是知識操作能力而非知識儲存量」（原文：「this advantage stems not from increased knowledge capacity, but from superior knowledge manipulation capabilities」）；Geiping et al. 的 recurrent-depth 把 loop 數當成 test-time compute 旋鈕。

## Details

### A. 技術演進脈絡

**A.1 早期權重共享（2018–2021）**
- **Universal Transformer**（Dehghani et al., arXiv:1807.03819, ICLR 2019）：所有層共享同一組參數並沿深度遞迴更新， [arXiv](https://ar5iv.labs.arxiv.org/html/2310.07096) 配合 Adaptive Computation Time（ACT, Graves 2016）做 per-token 動態停止。原文報告 UT「achieve a 0.9 BLEU improvement over Transformers on the WMT14 En-De dataset」（並較 Weighted Transformer 高 0.5 BLEU），在 subject-verb agreement 上達約 99.2% 平均。但要達到相同表現需更寬的層、訓練時間約為 vanilla 的兩倍（Takase & Kiyono, arXiv:2104.06022 指出 UT 在 WMT En-De 需約兩倍訓練時間）。 [arxiv](https://arxiv.org/pdf/2104.06022)
- **ALBERT**（Lan et al., arXiv:1909.11942, ICLR 2020）：跨層全共享 + 分解式嵌入，ALBERT-large 相對 BERT-large 參數少 18×、訓練快約 1.7×。關鍵限制：ALBERT 是 encoder-only、以 MLM 目標訓練，共享在此設定下同時提供了正則化與梯度穩定。
- **Deep Equilibrium Models（DEQ）**（Bai et al. 2019）：把無限深度共享層視為求解不動點；ALBERT 論文觀察到其嵌入是振盪而非收斂到平衡點。 [arxiv](https://arxiv.org/pdf/1909.11942)
- **為何早期 encoder 可行、現代 decoder-only LLM 效果有限**：（i）現代 LLM 是 parameter-dominated 任務（語言建模的 perplexity 強烈依賴參數量），共享直接砍掉記憶容量；（ii）decoder-only 自回歸 + KV cache 使得「不同深度異質參數」的早退與批次更難整合；（iii）MoEUT（Csordás et al., NeurIPS 2024, arXiv:2405.16039）明言：在 MoEUT 之前「沒有任何工作成功做出在語言建模上有競爭力的共享層 Transformer」，因為要補回參數就得把層做得極寬，運算/記憶體成本高到不切實際。 [arXiv](https://arxiv.org/pdf/2405.16039)

**A.2 現代遞迴 / looped transformers（理論線）**
- **Looped Transformers as Programmable Computers**（Giannou et al., ICML 2023, PMLR 202:11398）：證明常數層（文獻歸納為 ≤13 層）的 looped transformer 可模擬 SUBLEQ、實作基本計算器、線性代數常式，達到圖靈完備。
- **Looped Transformers for Length Generalization**（Fan et al., arXiv:2409.15647, 2024）：decoder-only + input injection（每步注入原始輸入）， [arXiv](https://ar5iv.labs.arxiv.org/html/2409.15647) 在算術/copy 等任務展現超越訓練分布的長度外推。
- **Reasoning with Latent Thoughts: On the Power of Looped Transformers**（Saunshi, Dikkala, Li, Kumar, Reddi, arXiv:2502.17416, ICLR 2025, Google Research）：主張「許多推理問題需要大深度但不需大參數」。關鍵實驗設計與可信度——用 iso-param（同參數）與 iso-flop（同運算）雙基線做對照：k 層 looped L 次（k⊗L）在合成推理任務（加法、p-hop induction、i-GSM 數學）上接近 kL 層非 looped 模型； [Liner](https://liner.com/review/reasoning-with-latent-thoughts-on-the-power-of-looped-transformers) 在真實語言建模上，looped 模型 perplexity 較 iso-flop 差（只彌補 iso-param 與 iso-flop 間約 34–50% 的 perplexity gap），但在推理任務下游表現接近甚至偶爾超過 iso-flop 基線。論文明確結論：「Overall the looped models are worse on perplexity and closed book QA (memorization benchmarks), but the % Gap is much higher for task groups that require reasoning (open book QA, math word problems)」，並發現「accuracy on downstream tasks scale as logarithm of the effective depth」。具體例：數學字題上 (12⊗2) 達 34.3%，超越 24 層 iso-FLOP 基線的 29.3%（「despite having 50% fewer parameters and worse perplexity」）。此為「looped 在推理上有 inductive bias」的核心證據，設計嚴謹（同時控制參數與運算），可信度中高，但侷限在中小模型與特定任務組。

**A.3 Relaxed Recursive Transformers（Bae et al., arXiv:2410.20672, ICLR 2025；KAIST × Google DeepMind）**

方法四要素：
- **Looped layer tying（CYCLE 策略）**：Gemma 2B（18 層）只存前 9 層、迴圈兩次（ [arxiv](https://arxiv.org/html/2410.20672v3) B=2 blocks）。
- **層間初始化策略**：Stepwise（固定首尾、間隔取中間層）、Average（平均權重）、Lower（取前 K 層）。 [arxiv](https://arxiv.org/html/2410.20672v3) 非鬆綁遞迴模型 Stepwise 最佳（15B token、SlimPajama：PPL 12.85 / Avg 51.7，優於 Average 46.7、Lower 47.1、Random 37.6）。
- **層級 LoRA 鬆綁 + SVD 初始化**：對「原層權重 − 共享權重」的殘差矩陣做 truncated SVD 來初始化 LoRA， [arxiv](https://arxiv.org/html/2410.20672v3) 使 rank→0 時退化為純遞迴、rank→full 時退化為 vanilla。鬆綁模型改以 Average 初始化最佳（因平均後的殘差易被低秩逼近）。
- **Continuous Depth-wise Batching + early exiting**：利用「不同深度共享同一組權重」多開一個批次維度。 [arxiv](https://arxiv.org/html/2410.20672v3)

關鍵數字（皆 SlimPajama uptraining、7 項 few-shot 任務平均：LAMBADA/HellaSwag/PIQA/WinoGrande/ARC-e/ARC-c/OpenBookQA）：
- **rank 消融**（Gemma, Average+SVD, 15B）：rank 0/64/128/256/512 → 非嵌入參數 0.99/1.07/1.15/1.30/1.60B，few-shot Avg 46.7/52.1/52.5/54.2/55.5，PPL 15.15/12.83/12.52/12.10/11.83。高 rank 單調變好但遞減；Stepwise 初始化下則呈 U 形（中 rank 反而低於純遞迴）。LoRA 參數開銷：rank 64 +0.08B、128 +0.16B、256 +0.31B、512 +0.61B。
- **SVD vs Zero LoRA 初始化**：rank 512 下 SVD 55.5 vs Zero 46.7（最多提升約 6.5 點）。
- **uptraining vs from-scratch**：pretrained-init（Stepwise 51.7）vs random-init 遞迴從頭訓（37.6），差約 14.1 個百分點；Pythia 410M 對照（Table F.3）顯示多數增益來自利用預訓練權重（pretrained-init 非遞迴 Pile PPL 11.48 vs from-scratch 14.11），遞迴再加成（pretrained+recursive PPL 11.03 / Avg 41.54）。
- **蒸餾 headline**：relaxed Gemma（rank 512）+ 60B token + 知識蒸餾達 few-shot 58.4%，與 Gemma 2B 於 60B token uptraining 的 58.6% 打平（原文：「uptrained on 60 billion tokens, achieves performance on par with the full-size Gemma model」）。注意：原始 Gemma 2B 未 uptraining 為 61.7%，因 SlimPajama 分布偏移而下降，故論文以 58.6% 為對照。
- **吞吐量 2–3×（甚至近 4×）為理論值**：基於 oracle-exiting 模擬（假設中間預測與最終相符即可退出），非端到端實測；論文明言實用信心早退演算法「超出本文範圍」。

**A.4 後續工作與延伸（2025–2026）**
- **Mixture-of-Recursions（MoR）**（Bae, Kim, Bayat 等，arXiv:2507.10524, NeurIPS 2025）：統一「參數共享 + per-token 自適應深度 + recursion-wise KV caching」。135M–1.7B 規模下形成新 Pareto 前緣；>360M 時以約 1/3 唯一參數 match 或超越 vanilla； [alphaXiv](https://www.alphaxiv.org/abs/2507.10524v1) expert-choice routing（42.6%）優於 token-choice（40.0%）； [arXiv](https://arxiv.org/html/2507.10524v1) MoR-4 搭配 continuous depth-wise batching 達最高 2.06× 吞吐。 [alphaXiv](https://www.alphaxiv.org/abs/2507.10524v1) 135M 最小尺寸時 MoR 略遜 vanilla（遞迴瓶頸）。 [arXiv](https://arxiv.org/pdf/2507.10524)
- **Ouro / LoopLM**（Zhu et al., arXiv:2510.25741, 2025，33 位作者含 Bengio）：從頭預訓練 1.4B/2.6B LoopLM、7.7T token，配 entropy-regularized 自適應深度。 [arXiv](https://arxiv.org/abs/2510.25741) 摘要原文：「Ouro 1.4B and 2.6B models enjoy superior performance that match the results of up to 12B SOTA LLMs across a wide range of benchmarks」；Ouro-1.4B 在 GSM8K 達 78.92（vs 4B 基線 72.86）、MATH500 82.40（vs 59.60）、BBH 71.02（vs 70.95）。關鍵發現：recurrence「不增加原始知識儲存（looped 與非 looped 皆約每參數 2 bits），但大幅提升知識操作/多跳推理能力」，且「yields reasoning traces more aligned with final outputs than explicit CoT」——直接支持「能力提升侷限在推理/操作、而非容量」的論點。
- **Improving Recursive Transformers with Mixture of LoRAs（MoL）**（Nouriborji, Rohanian, Rohanian, arXiv:2512.12880, 2025）：在共享 FFN 內插入 token-conditional LoRA experts（ModernALBERT, 50M–120M）； [arXiv](https://arxiv.org/abs/2512.12880) GLUE 上 ModernALBERT 81.94 > RRT 80.95（同初始化/目標）；MoL 8-expert/top-2 為 77.24 > MoA 76.87； [The Moonlight](https://www.themoonlight.io/en/review/improving-recursive-transformers-with-mixture-of-loras) 並提出把多個 expert 合併成單一 adapter 以利部署。 [arXiv](https://arxiv.org/abs/2512.12880)
- **Skip a Layer or Loop it?（CoLa）**（Li, Li, Zhou, arXiv:2507.07996, 2025）：以 MCTS 在測試時對預訓練 LLM 逐層 skip/repeat/重排。 [GitHub](https://github.com/AkihikoWatanabe/paper_notes/issues/2610) >75% 原本答對的樣本可找到更短 CoLa；>60% 原本答錯的可找到答對的 CoLa—— [ADS](https://ui.adsabs.harvard.edu/abs/2025arXiv250707996L/abstract) 同時支持「冗餘可壓縮」與「重複層可提升」。
- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach**（Geiping et al., arXiv:2502.05171, NeurIPS 2025）：Huginn-3.5B、800B token，推理時最多 50 loops、等效約 50B 參數運算量，數學/程式碼受益最大。

**A.5 相鄰路線對照**
- **Mixture-of-Depths（MoD）**（Raposo et al., arXiv:2404.02258, 2024）：top-k routing 讓 token 動態決定是否參與某層運算，靜態計算圖、每次 forward FLOPs 較少，post-training 取樣最高快 50%。屬 per-token 運算分配，與遞迴的 per-token 深度互補。
- **深度剪枝**：The Unreasonable Ineffectiveness of the Deeper Layers（Gromov et al., arXiv:2403.17887, ICLR 2025）——移除相似度高的連續層區塊 + QLoRA heal（單卡 A100 即可），某些開源模型移除近半層仍幾乎不掉分，直接證明「深層冗餘」。
- **latent reasoning / recurrent depth**：Geiping et al.（上述）與 Ouro。
- **深度方向 MoE**：MoEUT（Csordás et al., NeurIPS 2024）以 MoE 補回共享層被砍掉的參數，首次讓共享層 Transformer 在 BLiMP/PIQA 上略勝標準 Transformer 且省運算記憶體。 [arXiv](https://arxiv.org/abs/2405.16039)

### B. 優劣分析（可比較維度）

| 方法 | 參數量 | 每 token FLOPs | KV cache | 訓練成本 | 推論吞吐 | 硬體友善度 | 定位 |
|---|---|---|---|---|---|---|---|
| Vanilla | 基準 | 基準 | 基準 | 基準 | 基準 | 高 | — |
| 純層共享（ALBERT/CYCLE） | ↓↓ | 不變（深度不變） | 不變 | 需 uptraining | 需靠早退才提升 | 中（早退異質性問題） | 資源效率 |
| RRT（+LoRA rank r） | ↓（隨 r 回升） | 略增 | 不變 | +15–60B token | continuous depth-wise batching 理論 2–3× | 中 | 資源效率 |
| MoR | ↓（~1/3 唯一參數） | ↓（per-token 深度 + 稀疏 KV） | ↓（recursion-wise） | 同 FLOPs 更省 | 實測至 2.06× | 中高 | 資源效率（Pareto） |
| Looped（Saunshi/Ouro） | ↓↓ | ↑（多 loop） | 依實作 | 需從頭或大量 | loop 越多越慢 | 中 | 能力提升（推理） |
| Recurrent-depth（Geiping） | 中 | ↑↑（test-time 可調） | 小 context 友善 | 大 | 隨 loop 線性變慢 | 中 | 能力提升（test-time compute） |
| MoD | 不變 | ↓ | 不變 | 同基準 | 快 | 高（靜態圖） | 資源效率 |
| 深度剪枝 | ↓↓ | ↓↓ | ↓ | 極省（QLoRA heal，單卡） | 快 | 高 | 資源效率 |
| MoEUT | ↑（MoE 補參數） | 稀疏活化 | 不變 | 中 | 中 | 中 | 資源效率 |

**資源效率 vs 能力提升的證據強度**：
- **資源效率（證據強）**：RRT、MoR、深度剪枝、MoD、MoEUT 皆有明確 iso-FLOP/iso-param 對照顯示「以更少參數或更少 FLOPs 打平」。RRT 的 58.4% vs 58.6%、MoR 的 Pareto 前緣、Gromov 移除半層仍不掉分，皆屬此類。
- **能力提升（證據中等、且有邊界條件）**：僅 looped/recurrent 在「推理密集任務」有 iso-FLOP 下逼近或偶爾超越的證據（Saunshi et al. 的 34.3% vs 29.3%；Ouro 的知識操作提升）。但需注意：（i）perplexity 幾乎總是較差；（ii）「提升」多半來自「用同參數換更多序列運算/loop 深度」，屬 test-time compute 換取，而非消滅冗餘後的免費午餐；（iii）證據集中於中小模型與合成/數學任務。

**looped 在推理上的優勢——查證**：Saunshi et al. 的雙基線設計嚴謹，結論可信但範圍有限；LoopFormer（arXiv:2602.11451）等後續一致確認「looped 在 perplexity 落後、在推理趨近 iso-FLOP」。 [arXiv](https://arxiv.org/html/2602.11451v1) Ouro 進一步用 controlled experiment 把「容量 vs 操作」拆開，是目前最強的機制性佐證。

**LoRA 鬆綁的代價**：參數量回升（rank 512 時 Gemma +0.61B，達 vanilla 的 80%）；破壞「單一權重區塊常駐」——每個 loop 需切換不同 LoRA，serving 上需管理 multi-LoRA（論文將此列為未來工作）；rank 敏感度高且與初始化耦合（Stepwise+SVD 在中 rank 出現 U 形退化）。

### C. 與可解釋性文獻的交叉分析

**C.1 Lad et al. 四階段與 U 形敏感度的啟示**
Lad et al.（The Remarkable Robustness of LLMs: Stages of Inference?, arXiv:2406.19384）：刪除/交換相鄰層後保留 72–95% top-1 準確率；層越多越穩健；** [Hugging Face](https://huggingface.co/papers/2406.19384) 首尾層脆弱、中間層穩健**（ [Emergent Mind](https://www.emergentmind.com/papers/2406.19384) U 形）；四階段為 detokenization → feature engineering → prediction ensembling → residual sharpening。 [arxiv](https://arxiv.org/pdf/2406.19384) 對權重共享的直接設計啟示：
- 中間「feature engineering」階段本質上就是「對殘差流做迭代精修」，與 looped/遞迴的「重複套用同一 block」高度同構——這解釋了為何中間層最適合被 tie/loop，也解釋了 Gromov et al. 為何能安全移除中段層。
- 首尾層脆弱 → 不應共享、應保留獨立參數（RRT 的 Stepwise 正是「固定首尾、只在中間取樣」， [arxiv](https://arxiv.org/html/2410.20672v3) 可視為對此假說的隱性呼應，雖然 Bae et al. 的動機引用的是 layer-skipping 文獻而非 Lad et al.）。

**C.2 深度非均勻的 rank / 共享分配——研究空白確認**
- 在通用 LoRA 微調領域，已有多篇證明「均勻 rank 次佳、rank 應隨深度非均勻」：AdaLoRA、LAARA（Fisher 資訊導向，arXiv:2607.19391）、RSLoRA（表徵敏感度探測，arXiv:2607.09757，明確觀察到「depth-dependent sensitivity transition」）、PARA、La-LoRA、ARD-LoRA 等。
- **但在「遞迴/共享 transformer 的層級 LoRA 鬆綁」脈絡下，尚無論文以可解釋性（四階段/U 形）為依據做深度非均勻 rank 分配。** RRT 本身用的是各 loop 均勻 rank；其 component-wise 消融（Table G.1）僅比較 Q/KV/Out/FFN 模組別（發現砍 FFN rank 傷害最大、砍 KV 最小），並未做「首尾大 rank、中間小 rank」的深度別實驗。**這是明確且高價值的研究空白。**

**C.3 對 looped/recurrent 模型的機械可解釋性分析**
- A Mechanistic Analysis of Looped Reasoning Language Models（arXiv:2604.11791, 2026）：分析 Huginn-0125、retrofitted Llama、Ouro，發現每個 cycle 層收斂到不同不動點、recurrent block 沿 latent 空間走一致的循環軌跡、attention head 行為隨迭代穩定；**經驗上 recurrent block 學到的「stages of inference」與 feedforward 模型相似，並沿深度每次迭代重複這些階段**—— [arXiv](https://arxiv.org/html/2604.11791v1) 這直接把 Lad et al. 的四階段結構延伸到 looped 模型，是本題最相關的交叉工作。（註：亦觀察到並非所有模型都達到嚴格不動點，例如 Ouro 未如 Huginn 般快速收斂。）
- LOTUS（arXiv:2606.31779）：把 post-loop latent 經 base LM head 投影可還原 gold reasoning steps，顯示 latent 空間可解釋、與 CoT 對齊。 [arXiv](https://arxiv.org/html/2606.31779v2)
- Pappone et al.（2025）分析 recurrent-depth 的「two-scale」latent 動態（快 intra-loop、慢 inter-token）。

**C.4 樂透票假說與「冗餘是否為必要鷹架」**
- Lottery Ticket Hypothesis（Frankle & Carbin, arXiv:1803.03635, ICLR 2019）：稠密網路含可獨立訓到全精度的稀疏子網（winning ticket）， [alphaXiv](https://www.alphaxiv.org/overview/1803.03635v5) 但從頭訓練該稀疏架構通常失敗——關鍵是初始化， [alphaXiv](https://www.alphaxiv.org/overview/1803.03635v5) 過參數化提供「找到好初始化子網」的搜尋空間。 [MIT DSpace](https://dspace.mit.edu/bitstream/handle/1721.1/129953/1803.03635.pdf?sequence=2&isAllowed=y) stabilized/late-rewinding 版本顯示需先讓網路建立穩定結構（約 0.1–7% 訓練步）再剪枝。
- 對本題的推論：這強烈支持 Bae et al.「先訓練大模型再轉換」優於「從頭訓遞迴」的策略選擇——RRT 的 pretrained-init vs random-init 差約 14.1 個百分點，恰是「冗餘作為訓練期鷹架」的實證。Ouro 從頭訓 LoopLM 成功，說明從頭訓可行但需 7.7T token 的巨量規模與精心穩定化（論文亦提到高 recursion 時訓練不穩）。** [Hugging Face](https://huggingface.co/papers/2510.25741) 綜合證據傾向：訓練期冗餘/過參數化是有用的鷹架，轉換式路線在中小規模的資料效率明顯更高。**（此為基於現有證據的推論，非單一論文直接定論。另注：關於過參數化為何有效，樂透票假說與 escape-dimension 觀點並存且互相質疑。）

### D. 潛在發展方向（附理論依據、既有證據、風險）

1. **深度非均勻的 rank 分配（可解釋性導向）**
   - 依據：Lad et al. U 形敏感度 + 通用 LoRA 的 depth-dependent 敏感度（RSLoRA）。
   - 已有證據：RRT 的 Stepwise（固定首尾）間接支持；component-wise 消融顯示模組別非均勻有效。
   - 假設：在遞迴模型中，對應「首/尾 loop」或「detokenization/residual sharpening 階段」的層給更大 rank（甚至不共享），中間 feature-engineering 階段高度共享 + 低 rank，可在同參數預算下優於均勻 rank。
   - 風險：Stepwise+SVD 已出現中 rank U 形退化，非均勻分配可能與 SVD 初始化交互產生不穩；需小心 rank–初始化耦合。

2. **per-token 動態遞迴深度**
   - 依據 + 證據：MoR（已實作，Pareto 前緣、2.06× 吞吐）、Ouro 的 entropy-regularized 深度、CoLa（>60% 錯題可靠重複層救回）。
   - 假設：把 per-token 深度路由與四階段對齊（簡單 token 少 loop、需多跳推理 token 多 loop）。
   - 風險：router 訓練不穩、token-choice 明顯遜於 expert-choice；小模型（135M）遞迴瓶頸使 MoR 落後 vanilla。

3. **遞迴 + MoE 組合**
   - 依據 + 證據：MoEUT（首個共享層在 LM 上勝標準 Transformer）、MoL（ModernALBERT 81.94 > RRT 80.95）、Tying the Loop（tied expert layers）。
   - 假設：MoE 補回共享砍掉的參數容量、遞迴提供深度，兩者解決彼此的短板（共享缺容量、MoE 缺深度 inductive bias）。
   - 風險：serving 複雜度疊加（multi-LoRA/multi-expert + 遞迴路由）；容量回升可能侵蝕資源效率初衷。

4. **遞迴 × test-time compute / latent reasoning**
   - 依據 + 證據：Geiping recurrent-depth（loop 數即 compute 旋鈕，Huginn 等效 50B）、Saunshi 深度-迭代等價、Ouro 知識操作提升。
   - 假設：把 loop 深度當成與 CoT token 並列的第三條 test-time scaling 軸；對數學/程式碼受益最大。
   - 風險：每多一 loop = 一次完整 forward，序列延遲線性上升；深 loop 訓練不穩（Ouro、mechanistic 分析皆提及）；path-independence/不動點是否總成立仍未定。

5. **可解釋性導向的共享策略設計（元方向）**
   - 依據：mechanistic 分析已證明 looped 模型重現四階段。
   - 假設：以「階段邊界」（attention→MLP 相變、prediction/suppression neuron 出現）為切點決定 block 邊界與共享粒度，而非機械地等分層。
   - 風險：階段邊界跨模型不完全一致、需逐模型探測；目前無端到端驗證，屬推測性最高的方向。

## Recommendations

**若目標是部署成本下降（資源效率）——證據充分，可直接採用：**
1. 首選「轉換式」路線而非從頭訓：以 Stepwise 初始化把現成 2× 大小模型轉成遞迴模型，先做 15B token uptraining 驗證，few-shot 掉分可控（RRT 顯示可打平）。
2. 若準確率不足，加入層級 LoRA（SVD 初始化、Average 底層初始化），rank 從 256 起試；**基準線**：rank 每翻倍帶來的 few-shot 增益 < 1 點時即停（RRT 中 256→512 僅 +1.3 點，已近遞減拐點）。
3. 若 serving 吞吐是瓶頸，優先評估 MoR（實測 2.06×）而非 RRT 的理論早退（2–3× 為 oracle 模擬，未實測）。

**若目標是推理能力提升——證據有邊界，需先驗證：**
4. 僅在推理/數學/程式碼為主的工作負載才考慮 looped/recurrent-depth；先在 iso-FLOP 下對照 vanilla，**門檻**：若 looped 在目標任務無法在 iso-FLOP 下追平非 looped（如 Saunshi 的開卷 QA / 數學字題模式），則不值得承受 perplexity 與延遲代價。
5. 資源允許從頭訓大模型者可參考 Ouro（需 T 級 token 與穩定化）；資源有限者留在轉換式路線。

**研究者——最高價值的空白：**
6. 直接做「深度非均勻 rank × 遞迴 LoRA」實驗，以 Lad et al. 四階段/U 形為先驗，對照均勻 rank baseline。這是目前文獻缺口且理論依據充分。
7. 用 mechanistic 工具（如 arXiv:2604.11791 的不動點/軌跡分析）檢驗你的共享設計是否破壞或保留四階段結構。

**會改變上述建議的觀測**：若未來出現「共享/遞迴模型在 iso-FLOP 下於一般（非推理）任務也穩定超越 vanilla」的強證據，則「能力提升」的判定應上修，遞迴路線的適用範圍應大幅擴張。

## Caveats
- **理論 vs 實測**：RRT 的 2–3×/近 4× 吞吐為 oracle-exiting 模擬值，非端到端實測；Continuous Depth-wise Batching 與信心早退的實用演算法作者列為未來工作。引用時務必標明「理論/假設性」。
- **對照基準的微妙處**：RRT 的 58.4% vs 58.6% 是與「SlimPajama uptraining 後的 Gemma 2B」相比；原始 Gemma 2B 為 61.7%，因分布偏移下降。故「打平」是相對於 uptraining 後基線，非原始上界——這強化了「資源效率而非上界提升」的結論。
- **perplexity vs 下游脫鉤**：looped 模型幾乎總是 perplexity 較差、下游推理較好，兩者不可互相推斷；比較時須指明用哪個指標。
- **規模外推風險**：多數遞迴/looped 的正面結果集中在 ≤3B 規模與合成/中小基準；是否外推到前沿規模仍未定（MoR 在 135M 甚至落後 vanilla）。
- **知識截止與新近性**：2026 年多篇（mechanistic 分析 arXiv:2604.11791、MoL arXiv:2512.12880、CoLa、LoopFormer、Ouro 等）為極新工作，部分尚未經完整同行評審，數字可能隨改版變動。
- **矛盾並陳**：關於「過參數化為何有效」，樂透票假說（好初始化子網）與 escape-dimension 觀點並存且互相質疑，本報告未選邊。
- **因果歸因**：RRT 的增益中「利用預訓練權重」與「遞迴結構本身」的貢獻，Table F.3 顯示前者為主、後者加成，但拆分僅在 Pythia 410M 小規模驗證。

---

*主要一手來源*：Bae et al. arXiv:2410.20672（RRT, ICLR 2025）；Bae et al. arXiv:2507.10524（MoR, NeurIPS 2025）；Saunshi et al. arXiv:2502.17416（ICLR 2025）；Zhu et al. arXiv:2510.25741（Ouro）；Geiping et al. arXiv:2502.05171（NeurIPS 2025）；Lad et al. arXiv:2406.19384；Gromov et al. arXiv:2403.17887（ICLR 2025）；Raposo et al. arXiv:2404.02258；Csordás et al. arXiv:2405.16039（MoEUT, NeurIPS 2024）；Dehghani et al. arXiv:1807.03819；Lan et al. arXiv:1909.11942；Giannou et al. PMLR 202（ICML 2023）；Fan et al. arXiv:2409.15647；Frankle & Carbin arXiv:1803.03635；Nouriborji et al. arXiv:2512.12880（MoL）；Li et al. arXiv:2507.07996（CoLa）；arXiv:2604.11791（looped 模型機械可解釋性分析）。二手/彙整來源（Emergent Mind、Medium、Substack 等）僅用於交叉定位，數字皆回溯至一手論文。