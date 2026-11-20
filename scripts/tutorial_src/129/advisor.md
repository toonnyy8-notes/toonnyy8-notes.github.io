# 顧問筆記：Score Centering Stabilizes Off-policy RL（arXiv 2609.20807v1, Marek & Ryabinin, Together AI）

給教學設計者，不是給讀者。凡標〔顧問推導〕者為我自己推的，論文沒有寫；其餘數字、式號、節號皆出自 paper.txt。
圖中曲線的數值論文沒有列表；我只能從文字抽取檔的「標籤上下排列」判斷名次，凡這類判斷都標〔讀圖〕，請對 PDF 複核後再用。

記號（§4 Notation）：固定 prefix y<t；v 為詞彙 token；p_v 為 trainer 的 next-token 機率、q_v 為 sampler 的；s_v = ∇θ log p_v（score，永遠是 trainer 的）；s̄ = Σ_v q_v s_v；R 為整條 rollout 的 reward 或 advantage；E_q、E_p 是對 prefix 之後的 rollout 取期望。

---

## 1. 論證的 9 個步驟

| # | 位置 | 假設 | 得到什麼 |
|---|---|---|---|
| 1 | §1, Eq.(1) | rollout 由 p_θ 自己取樣 | ∇θ E_p[R] = E_p[R ∇θ log p_θ(y)]。實作需要兩次 forward：sampler 產生 rollout、trainer 算梯度。兩者輸出不同即 TIM。 |
| 2 | §2.1 | — | TIM 來源：不同精度/不同 codebase 的 bug；浮點不可結合（註 2：fp4_e2m1 下 (0.5+0.5)+2=3，0.5+(0.5+2)=2）；自回歸取樣 vs 序列平行訓練導致 reduction 順序不同；staleness（非同步、continuous batching，一個 batch 甚至一條 rollout 的不同片段來自不同 checkpoint）。完全消除 TIM 代價是硬體利用率。 |
| 3 | §2.2, Eq.(2), Table 1 | q_θ ≠ p_θ，q 支撐集涵蓋 p | 序列層級 IS 精確修正：E_p[R∇log p] = E_q[(p/q) R ∇log p]。比值在稀有 token 上任意大 → 變異數大 → 實務全部 clip/mask → 引入偏差。Table 1 把 9 種方法排成格子（level × stat × low/high × outside × sign）。 |
| 4 | §3, Fig.2, 註 3、4 | 實驗觀察（Qwen3-1.7B, Countdown, 人為擾動 sampler 權重） | PG 在「advantage 為 +1/0 且完全 offline」時就是 SFT。所以 SFT 與 RL 只差兩件事：reward mode 與是否 online。Fig.2：offline 只有 +1/0 穩定；online 反過來，group-centered 最穩、+1/0 最不穩。作者**假設**（原文 "We hypothesize"）有兩個不同機制；本文只追 online 那個。 |
| 5 | §4 第一段 | on-policy，任意分佈 | 常數 reward 下期望梯度為 0：E_p[s_yt] = Σ_v p_v ∇log p_v = ∇Σ_v p_v = ∇1 = 0。 |
| 6 | §4, Eq.(3) | token 由 q 取樣、score 由 p 算；只用共變異數恆等式（註 5） | E_q[R s_yt] = E_q[R]·s̄ + Cov_q(R, s_yt)。前項 = drift，後項 = signal。drift 只透過 E_q[R] 依賴 reward；方向 s̄ 是「以 sampler 為 teacher 的 cross-entropy loss」的負梯度 → 在每個 prefix 把 trainer 蒸餾向 sampler，強度為該 prefix 的期望 reward。 |
| 7 | §4「Drift is an artifact…」「Group centering does not remove drift」 | sampler 是 trainer 的有偏副本，且每步（或定期）由 trainer 同步回去 | teacher 固定時蒸餾無害（收斂到 MLE）；teacher 跟著 student 動時，誤差在回饋迴路中累積而不收斂。這是**論述**，沒有定理、沒有收斂/發散速率。Group centering 讓同一 prompt 的 advantage 加總為 0，但 drift 發生在單一 prefix，E[advantage | prefix] ≠ 0，所以只縮小、不消除。 |
| 8 | §4, Eq.(4)(5) | s̄ 在給定 prefix 下是確定量，且可精確算出（對詞彙求和） | s̃_yt = s_yt − s̄ ⇒ E_q[s̃_yt]=0 ⇒ E_q[R s̃_yt] = Cov_q(R, s̃_yt) = Cov_q(R, s_yt)。與 on-policy 更新唯一差別：共變異數在 q 下量，不是 p 下。p=q 時 s̄=0，SC 是 no-op。 |
| 9 | §4.1, §A.1–A.3, Eq.(6)–(12) | 只記錄 sampler 的 top-k logprob（k=128）；尾部用 trainer 分佈按尾部質量比 ρ 重縮放 | 中心化項只需 head 的 k 個 token（Eq.8）；可寫成純量 loss（Eq.6/11/12）；可與任意 IS 權重函數 f(r) 合成（Eq.9/10，α = ρ f(1/ρ)）。 |
| 10 | §5.2–5.4, Fig.1/3/4/5, §A.4 | 刻意放大的 TIM（權重雜訊、只量化 sampler、每 64 步才同步）作為「溫和 TIM 長期訓練」的代理 | 權重雜訊最大時只有 SC 與 SC+TIS/MIS 穩定；量化下 SC（單獨或合成）最好；staleness 下 SC+TIS/MIS 最好且勝過單獨 SC；30B 上 SC 52%/30% vs TIS 51%/12%；k=32 與 k=128 與全詞彙 SC 相當。 |

---

## 2. 必須教對的 5 個推導

### D1. Drift/signal 分解（Eq.3）與「常數 reward 應無學習」

- on-policy：E_p[s_yt] = Σ_v p_v ∇θ log p_v = Σ_v ∇θ p_v = ∇θ(Σ_v p_v) = ∇θ 1 = 0。條件：p 是正規化分佈、可微。對「任意分佈」成立。
- off-policy：E_q[s_yt] = Σ_v q_v ∇θ log p_v = s̄，一般 ≠ 0（因為 Σ q_v ∇p_v / p_v 不再能合併成 ∇Σp_v）。
- 註 5：E[AB] = E[A]E[B] + Cov(A,B)，取 A=R、B=s_yt，期望在 q 下、條件於 prefix：
  E_q[R s_yt] = E_q[R]·E_q[s_yt] + Cov_q(R, s_yt) = E_q[R]·s̄ + Cov_q(R, s_yt)。
- 注意 Cov 是「純量 R 與向量 s」的共變異數（向量值）；R 是整條 rollout 的結果，與 y_t 相關是因為 y_t 影響後續。
- s̄ 的蒸餾解讀：CE(q,p) = −Σ_v q_v log p_v，把 q 當常數，−∇θ CE = Σ_v q_v ∇θ log p_v = s̄。論文說的是 cross-entropy；等價於 −∇θ KL(q‖p)（H(q) 與 trainer 參數無關）〔顧問推導〕。這裡 q 被當成資料（不對 q 微分），即使 q 實際上是 θ 的副本。

### D2. Score centering 消去 drift（Eq.4–5）

- s̃_yt = s_yt − s̄。E_q[s̃_yt] = s̄ − s̄ = 0。
- E_q[R s̃_yt] = E_q[R]·E_q[s̃_yt] + Cov_q(R, s̃_yt) = 0 + Cov_q(R, s̃_yt)。
- Cov_q(R, s̃_yt) = Cov_q(R, s_yt − s̄) = Cov_q(R, s_yt)，因為 s̄ 在給定 prefix 下是常數（不依賴被取樣的 token），常數與任何變數共變異數為 0。
- 條件：(a) 中心化的期望與取樣用的是**同一個**分佈 q；(b) s̄ 精確可算。top-k 近似下 (a) 只近似成立，見 D4。
- 對照 on-policy：E_p[R s_yt] = Cov_p(R, s_yt)（因 E_p[s]=0）。所以 SC 的期望更新 ≠ 真實 policy gradient；差在 Cov 的量測分佈（§6 Limitations 明說）。
- 對照 value baseline（§4「Relation to classical baselines」）：令 V = E_q[R | prefix]，E_q[(R−V) s_yt] = E_q[R]s̄ + Cov − V s̄ = Cov_q(R, s_yt)。期望相同，但需要精確 critic；SC 不需要。
- 〔顧問推導〕Eq.(5) 的說法「唯一差別是共變異數的取樣分佈」是在**固定 prefix** 下說的。對整條序列加總後，prefix 的造訪分佈也是 q 而非 p；SC 不修正這一點，per-token IS 也不修正。

### D3. 純量 loss（Eq.11 → Eq.6）與符號

- Eq.(11)：L = −R ( log p_yt − Σ_v sg[q_v] log p_v )。
- −∇θ L = R ( ∇log p_yt − Σ_v q_v ∇log p_v ) = R (s_yt − s̄) = R s̃_yt。sg 讓 q_v 不被微分。
- 被減掉的是 **R·s̄（用這條 rollout 自己的 R）**，不是 E[R]·s̄；其期望 E_q[R]·s̄ 正好等於 drift。
- 展開：L = −R log p_yt + R Σ_v q_v log p_v = −R log p_yt − R·CE(q,p)。R>0 時第二項在「最大化」CE(q,p)，即主動反蒸餾，抵銷 PG 隱含的蒸餾〔顧問推導：這個讀法〕。
- 符號慣例：Eq.(13) 的 J_base 是要最大化的目標；Eq.(6)(11)(12) 的 L 是要最小化的 loss。
- Eq.(13)：J_base = Σ_i A_i Σ_{t=1}^{L_i} log p_θ(y_{i,t} | y_{i,<t}) / Σ_i L_i，A_i = R_i − R̄_group，R_i ∈ {−1,+1}。除以 batch 總 token 數，不是每條各自平均。

### D4. Top-k 近似（Eq.7–8）

- q̂_v = q_v（v∈H）；q̂_v = ρ p_v（v∈T）；ρ = (1 − Σ_{v∈H} q_v) / (1 − Σ_{v∈H} p_v)。ρ 的選法使 Σ_v q̂_v = 1。
- E_q̂[s_yt] = Σ_H q_v s_v + ρ Σ_T p_v s_v
  = Σ_H q_v s_v + ρ ( E_p[s_yt] − Σ_H p_v s_v )　（E_p[s_yt]=0）
  = Σ_H (q_v − ρ p_v) s_v。
- 所以 Eq.(6)：L = −R ( log p_yt − Σ_{v∈H} sg[q_v − ρ p_v] log p_v )；ρ 也在 sg 裡面。
- 條件/誤差〔顧問推導〕：殘餘 drift = E_q[R]·(s̄ − E_q̂[s]) = E_q[R]·Σ_{v∈T} (q_v − ρ p_v) s_v。僅當 sampler 尾部與 trainer 尾部成比例（q_v = ρ p_v, ∀v∈T）時為 0；一般受尾部質量上界控制。論文只給經驗數字：k=128 時 head 平均覆蓋 >99.9% sampler 質量，唯 30B 的 INT8 W/A + INT4 KV 為平均 99.45%、最差 batch 95.8%（§A.4）。
- 記憶體算術（§A.1）：詞彙 152K × fp32 4 bytes ≈ 608KB/token；1024 條 × 32K token × 608KB ≈ 20TB。
- 成本：k=128 的 wall-clock 與基線差 1% 以內（0.6B 與 30B，自製 JAX sampler；vLLM/SGLang 的額外開銷**未測**）。

### D5. 與 IS 合成（Eq.9–10, 12）

- r_v = p_v/q_v，w_v = f(r_v)（TIS：f(r)=min(r,2)；MIS：r∈[0.5,5.0] 時為 r，否則 0）。
- 更新：R ( w_yt s_yt − E_q[w_yt s_yt] )，E_q[w_yt s_yt] = Σ_v q_v w_v s_v。常數 R 下期望為 0，對任何 f 成立（構造使然）。
- top-k：尾部 p_v/q̂_v = p_v/(ρ p_v) = 1/ρ 為常數，故
  E_q̂[ŵ s] = Σ_H q_v w_v s_v + ρ f(1/ρ) Σ_T p_v s_v = Σ_H (q_v w_v − α p_v) s_v，α = ρ f(1/ρ)。
- 檢查：f=1 ⇒ α=ρ（回到 Eq.8）；f(r)=r ⇒ α=1 且 q_v w_v = p_v ⇒ 中心化項為 0（精確 IS 本來就零期望）。
- Eq.(12)：L = −R ( sg[w_yt] log p_yt − Σ_{v∈H} sg[q_v w_v − α p_v] log p_v )。取樣 token 的權重與中心化係數都 detach。
- 〔顧問推導〕附錄程式中，被取樣 token 的權重用真實的 samp_token_logp（即使它落在 head 之外），而中心化項對尾部假設權重恆為 f(1/ρ)。故 top-k 合成版的「零期望」只在 q̂=q 時精確。
- 〔顧問推導〕「clip 重新引入 drift」只在 clip/mask 真的生效處成立：E_q[w s] = Σ_v q_v f(p_v/q_v) s_v；TIS 下若所有 p_v ≤ 2 q_v，則 w_v=r_v，Σ_v p_v s_v = 0，drift 為 0。

### 附：對教學有用的閉式特例（全部〔顧問推導〕，論文沒有）

- softmax、以 logit z 為參數時：s_v = e_v − p，故 s̄ = q − p；drift = E_q[R]·(q − p)；s̃_yt = e_yt − q。
  ⇒ 在 logit 座標下，SC 的期望更新 = q ⊙ (Q − R̄_q)（Q_v = E[R | 選 v]，R̄_q = Σ q_v Q_v），即「sampler 自己的 on-policy 更新」。只在 logit 空間成立；參數空間還乘 trainer 的 Jacobian。
- 註 4 的 1 − 1/G：A_i = R_i − (1/G)Σ_j R_j，on-policy 下 E[A_i s_i] = (1 − 1/G) E[R s]（用到 E[s]=0 與 rollout 獨立）。G=8 時為 7/8。
- 序列層級、off-policy：E_q[A_i S_i] = (1 − 1/G)·Cov_q(R, S)，S = Σ_t s_t。再用全共變異數定律：Cov_q(R, s_t) = E_prefix[Cov_q(R, s_t | prefix)] + Cov_prefix(E_q[R|prefix], s̄_t(prefix))。group centering 去掉了 E[R]·E[S]，但留下第二項（prefix 間的 drift）；SC 把第二項也去掉。這是論文「Group centering does not remove drift」那段文字的公式版。

---

## 3. 常見誤讀、內部不一致、未說明處

### 看起來對、其實不是論文所示

1. 「SC 讓梯度無偏」— 否。它只消去 drift；剩下的是 Cov_q 而非 Cov_p（Eq.5、§6 Limitations）。
2. 「SC 取代 IS」— 否。staleness=64 下，最好的是 TIS+SC / MIS+SC，且「outperforming vanilla score centering」（§5.3）。〔讀圖〕Fig.4 右圖 SC 單獨的標籤位在下段（TOPR 之下、TIS 之上），低於 PPO/DAPO/GSPO/DPPO。標題的 "off-policy" 不等於「對 staleness 單獨有效」。
3. 「SC 在量化下永遠勝過 IS」— 摘要是 "matches or outperforms"。30B FP8 W/A + FP8 KV 下連未修正的 PG 都穩定（58%）；FP4 KV 下 SC 52% vs TIS 51%；只有 INT8 W/A + INT4 KV 拉開（SC 30%、TIS 12%、其餘 <5%）。
4. 結論句「under severe quantization, it is the only method that trains stably」與 §5.4 自己說的 FP4 KV 下 TIS "stay stable" 不完全相容，也與 Fig.4 左圖 SC+TIS/MIS 同樣穩定不相容。教學時用 §5 的具體數字，不要用結論的概括句。
5. 「group centering 就夠了」— 否，只縮小 drift（§4）。
6. 「IS 一 clip 就有 drift」— 只在 clip/mask 生效處（見 D5 末）。這也符合溫和 TIM 下各方法重疊的觀察。
7. 「合成版在 30B 上驗證過」— 〔讀圖〕Fig.1 只有 10 種方法（SC, PG, naive IS, TIS, MIS, PPO, DAPO, GSPO, TOPR, DPPO），沒有 TIS+SC/MIS+SC。Table 2 的 47 個 run 恰等於 10 + (5+4+3+2+6) + (4+3+3+1+6)〔顧問推導〕，佐證此點。合成只在 0.6B Countdown 上測；staleness 也只在 0.6B、單一設定（64）測。
8. 「0.6B 到 30B」— 1.7B 只用於 Fig.2（reward mode 對照，沒有 SC）；SC 的結果只有 0.6B 與 30B 兩個模型、兩個任務。
9. 「Staleness = 64」— 是 inference engine **每 64 步才更新一次**，不是「最大 staleness 64」（§5.3 明講這是刻意加重）。
10. 「drift 是不穩定的原因已被證明」— 論證鏈是：Eq.3 的代數事實 + 回饋迴路的文字論述 + 「只消 drift 就穩定」的介入實驗。沒有發散定理。§3 的雙機制是假設。結論用了 "establishes"。
11. 「無超參數」— 精確 SC 無；實際用的 top-k 版有 k（=128）與 eps（=1e-6，§A.3 程式）。
12. 「所有圖都是同一種 accuracy」— Fig.2、3 是 held-out eval accuracy；Fig.1、4、5 是 training accuracy（取樣 rollout 的平均 reward，經 centered moving average 平滑）（§B.3）。Fig.5 的 weight-noise 面板與 Fig.3 設定相同但指標不同，數值不可直接對照。
13. 「量化兩邊都做」— 只量化 sampler（weights、activations、KV cache）；trainer 為 bf16。作者自己說實務上應兩邊同量化、且應盡量頻繁同步。
14. 「誤差帶都有意義」— 陰影是 ±1 standard error。30B：FP8 KV 面板全部單 seed；FP4 KV / INT4 KV 面板 SC 5/4 seeds、TIS 4/3、MIS 3/3、naive IS 2/1，其餘單 seed。單 seed 沒有 SE。額外 seed 是**事後只加給表現最好的方法**。
15. Eq.(2)「Either way, IS corrects for TIM」— 序列層級才精確；per-token 比值不修正 prefix 分佈〔顧問推導〕。
16. §5.2「SC、TIS、MIS 表現最好」是跨三個雜訊尺度的概括；同段說 TIS 在較大兩個尺度約 step 180、40 崩潰，DPPO 約 step 160、80、20 崩潰。

### 內部不一致 / 未說明

- reward 值：§3 與 Fig.2 用 +1/0、+1/−1、group-centered 三種；Eq.(13) 寫 R_i ∈ {−1,+1} 再 group-center。Fig.2 的 "group-centered" 是從 +1/0 還是 ±1 中心化未說明（差 2 倍 advantage，固定 lr 的 SGD 下不等價〔顧問推導〕）。
- 模型名：§1 寫 Qwen3-0.6B，§5.1/§B.1 寫 Qwen3-0.6B-Instruct；30B 為 Qwen3-30B-A3B-Base（MoE，未 replay router indices，trainer/sampler 可能路由不同）。
- Table 2 的 Fig.5「51 unique runs」我對不上〔顧問推導〕：照圖例 n 加總，0.6B 六個面板 53 條、30B 的 k=32 與 full 各 1×3 = 6 條，共 59；若 0.6B 的 k=128 也沿用 Fig.3/4 的 run，則為 41。論文只說 30B 的 k=128 是從 Fig.1 轉來。其餘列可對上：Fig.2 = 3×2×3 = 18；Fig.3 = 3×3×12 = 108；Fig.4 = 3×2×12 = 72；合計 296、6,180 H100-hours 加總正確。
- Fig.5 weight noise 0.01 的 k=32 為 n=2，其餘 0.6B 面板皆 n=3，未解釋。
- 未給：Fig.2 的權重擾動大小（"a small amount"）；Fig.2 offline 的資料如何產生；權重雜訊 N(0, 0.01²/0.02²/0.05²) 加在哪些參數、是否相對於權重尺度；INT8/FP8/FP4/INT4 的量化方案細節；平滑視窗長度；eval prompt 數；SGD 是否有 momentum/weight decay；DPPO 的 "binary total variation" 公式；Fig.1/3/4 曲線的數值表。
- lr 固定 10⁻²（SGD，依 Mukherjee et al. 2026），每 batch 一步；每 prompt 8 個 completion；0.6B：64 prompts（512 序列）、最大長度 512；30B：16 prompts（128 序列）、最大長度 1024。「省下最多 240GB 記憶體」是相對 AdamW。所有方法用原論文或 verl 預設值，**未調參**。
- §4.1 的短程式沒有 eps 保護（trainer 尾部質量為 0 時除以 0）；§A.3 的完整版有。`samp_logp` 是長度 k 的 top-k logprob，`train_logp` 是全詞彙。
- 基線不是原論文完整配方（Table 1 最後一欄）：TIS 省略 CISPO 的 dynamic sampling 與長度懲罰；MIS 省略 IcePop 的 PPO surrogate；DAPO 省略 dynamic sampling 與 overlong penalty；GSPO 改為 token 梯度加總並加 dual clipping at 3；PPO/DAPO dual clipping at 3。所以「SC 勝過 DAPO/GSPO」只能說成「勝過其修正規則」。
- Table 1 數值（教學若引用須照抄）：Naive IS 0…∞；TIS 0…2.0 clip both；MIS 0.5…5.0 drop both；PPO 0.8…1.2 drop ppo；DAPO 0.8…1.28 drop ppo；GSPO seq 0.9997…1.0004 drop ppo；TOPR seq 0…1.0 clip neg；DPPO token tv 0…0.2 drop ppo。

---

## 4. 哪些可以在瀏覽器即時示範

### 可以（閉式或小型玩具計算）

1. **Σ_v p_v ∇log p_v = 0 與 s̄ ≠ 0**：3–5 個 token 的 softmax，滑桿調 sampler 偏移，顯示 logit 空間 s̄ = q − p〔顧問推導〕。
2. **Eq.(3) 分解**：給每個 token 一個 Q_v，數值驗證 E_q[R s] = E_q[R] s̄ + Cov_q(R, s)；把 reward 全設成常數，signal 歸零而 drift 仍在。
3. **Eq.(4)(5)**：同一玩具上減去 s̄，顯示常數 reward 下更新為 0；非常數時等於 Cov_q，並與 Cov_p 並排顯示差異（示範「不是無偏」）。
4. **回饋迴路**〔顧問推導，論文無此模型，須標明是教學示意〕：logit bandit，q = softmax(z + δ)，δ 固定；常數 R=1；z ← z + η·R·(q − p)。小 δ 時 q − p ≈ p ⊙ (δ − Σ p_v δ_v)，z 沿 δ 持續漂移、熵下降；SC 下更新恆為 0。可再加「teacher 固定」對照（q 不隨 z 動 → 收斂到 p=q）來示範「重點是 teacher 是否跟著 student」。這不能宣稱重現 Fig.2/3。
5. **Top-k（Eq.7–8）**：20 個 token、k 可調，驗證 head-only 公式 = 在 q̂ 上全詞彙求和；顯示殘餘 Σ_T (q_v − ρ p_v) s_v 隨 k 變小。
6. **合成（Eq.10）**：f 切換 1 / r / min(r,2) / MIS，顯示 α = ρ f(1/ρ)，以及 f(r)=r 時中心化項為 0。
7. **IS 變異數與 clip 的偏差**：一個稀有 token 的 p/q 很大；顯示未 clip 的權重、clip 後 E_q[w s] ≠ 0〔顧問推導〕。
8. **fp4_e2m1 不可結合**（註 2）：可表示值 0, 0.5, 1, 1.5, 2, 3, 4, 6〔顧問推導〕；(0.5+0.5)+2=3，0.5+(0.5+2)=2.5→捨入為 2。
9. **記憶體算術**：152K × 4B = 608KB；× 1024 × 32K ≈ 20TB；對比 k=128。

### 不可以

- Fig.1/3/4/5 的訓練曲線：需要 0.6B–30B LLM、量化/自製 JAX sampler，合計約 6,180 H100-hours；論文沒有曲線數值表，只能引用文字中的數字（58%、52%、51%、30%、12%、<5%；DPPO 崩潰約 step 160/80/20；TIS 約 180/40；99.9%/99.45%/95.8%；1% wall-clock）。
- Fig.2 的 offline/online × reward mode 對照：依賴真實 LLM 的動態（負 reward 下 logprob 發散屬 Ren & Sutherland 的機制，本文未分析）；玩具模型只能示意 online +1/0 那一格。
- 「溫和 TIM 長期訓練也會崩」：論文自己也沒做，是以嚴重 TIM 當代理的推論。
- MoE 路由不一致、KV cache 量化的效應：無閉式。

---

## 5. 初學者先備知識（教學順序）

1. 機率分佈、期望、共變異數；E[AB] = E[A]E[B] + Cov(A,B)。
2. softmax 與 next-token 分佈；log-prob；LLM 自回歸取樣、prefix 的概念。
3. 梯度與 score ∇θ log p；log-derivative trick；為何 E_p[score] = 0。
4. Policy gradient / REINFORCE（Eq.1）；reward、advantage、baseline；group centering（GRPO 式，每 prompt G 條 rollout）。
5. SFT / cross-entropy / 蒸餾；「PG 在 +1/0 且 offline 時等於 SFT」。
6. On-policy vs off-policy；sampler（推論引擎）與 trainer 的分離；同步、staleness。
7. 浮點數精度與量化（bf16、fp8、int8、fp4；KV cache；不可結合性）→ TIM。
8. Importance sampling；比值 p/q；變異數爆炸；clip（TIS）與 mask（MIS）；偏差–變異數取捨。
9. stop-gradient 與「用純量 loss 表達想要的梯度」。
10. top-k logprob、尾部質量。
11. 讀圖素養：training vs eval accuracy、standard error、seed 數、moving average。
