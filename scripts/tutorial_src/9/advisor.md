# Advisor notes — Large Flow Language Models (Lafa), arXiv 2610.28559v1

給教學設計者,不是給讀者。凡標 〔顧問推導〕 者為我自己的推導或核算,不是論文原文;其餘數字、式號皆取自論文。
符號注意:論文的 `E` 同時是 vocabulary embedding 矩陣 E ∈ R^{|V|×d} 與期望值 E[·];教學時請改用不同字型。上標 i 是 block 內位置,不是次方。

---

## 1. 論證鏈(9 步)

**S1. Block-AR 分解(§2.1, Eq. 1)**
假設:B | M。內容:p(y|c) = ∏_{r=0}^{M/B−1} p(y_{rB+1:(r+1)B} | c, y_{≤rB})。這是恆等式(chain rule),沒有近似。作用:block 之間維持 causal,所以 prefix KV cache 可沿用;平行性只發生在 block 內。

**S2. 在 embedding 空間做 flow(§2.1 Embedded flows)**
x_1 := (E_{y_1},…,E_{y_B})^⊤ ∈ R^{B×d};x_0 ~ N(0, σ²I)。每個 token 的維度從 one-hot 的 |V| 降到 d。沒有「證明」任何事,是設計選擇。

**S3. Asynchronous(tokenwise time)interpolant(Eq. 2–3)**
假設:τ:[0,1]→[0,1]^B 每個分量可微且遞增,τ(0)=0, τ(1)=1。定義 I_t^i = (1−τ^i(t)) x_0^i + τ^i(t) x_1^i。存在 velocity field b_t(·;c) 使 ODE ẋ_t = b_t(x_t;c) 的邊際分佈為 p_t(·|c)(引用 [27],論文本身不證)。τ^i(t) > τ^j(t) (i<j) 時較前的位置先被還原。

**S4. Denoiser ⇒ velocity(Eq. 4–5, Prop. A.1)**
定義 D_t(x;c)^i := p(y_i = · | c, I_t = x)(每個位置的後驗邊際,條件是整個 block 的 noisy state)。證明:b_t(x;c)^i = τ̇^i(t)/(1−τ^i(t)) · (D_t(x;c)^i E − x^i),t ∈ [0,1)。所以只需學一個 tokenwise 分類器即可得到 flow。

**S5. 把預訓練 AR 模型改成 block denoiser(§2.2)**
四個改動:(a) adaLN 的 tokenwise time conditioning,prefix 位置恆為 time 1,block 位置 i 為 τ^i(t),identity 初始化;(b) block 內 bidirectional attention、prefix 仍 causal 且看不到 noisy block(Fig. 3)——加上 (a),prefix 的 K/V 與 (x,t) 無關,可跨 denoising step 與後續 block 重用;prefix 輸出仍是 AR 預測 f(c)_i;(c) E 與 prefix 共用、可訓練,但每個 E_v 的 L2 norm 固定在預訓練值,只學方向(避免 norm 膨脹讓固定 σ 的 corruption 實質變弱);(d) 一次 forward 平行訓練所有 block:clean 序列 + noisy 複本(Fig. 4, Alg. 1)。

**S6. 訓練目標(Eq. 6)**
L = E[ Σ_{i=1}^{N} −log f(c)^i_{c_{i+1}} + Σ_{i=1}^{B} −log D̂_t(I_t;c)^i_{y_i} + Σ_{i=1}^{B} ‖D̂_t(I_t;c)^i E − E_{y_i}‖² ],c_{N+1} := y_1,t ~ U[0,1],三項權重皆為 1。CE 監督 Eq. 4 的機率;MSE 監督決定 velocity 的 mean embedding(Eq. 5)。這是經驗性設計(Table 3 ablation),非理論結果。

**S7. Learned tokenwise time(§2.2, App. B.1, Eq. 22–23, Alg. 2)**
τ^i(t) := F_i^{-1}(t),F_i(τ) := ∫_0^τ ℓ^i(r)dr / ∫_0^1 ℓ^i(r)dr,ℓ^i(τ)>0 為位置 i 在 tokenwise time τ 的期望 denoising CE。結果 τ̇^i ∝ 1/ℓ^i(τ^i):loss 高處走得慢,uniform t 取樣與 uniform Euler grid 自動集中在難的 noise level。left-to-right 順序(Fig. 5)是學出來的「一般趨勢」,非強制。

**S8. Flow map distillation(§2.2, App. A.2, Eq. 7, 13–21, Prop. A.2–A.3)**
假設:prefix、E、τ 固定;τ^i ∈ C¹,τ̇^i(t)>0 (t<1);ODE 解唯一;denoiser 對時間連續可微;範圍 0 ≤ s ≤ t < 1,t=1 以極限理解。證明:(i) 存在 two-time denoiser δ_{s,t} 滿足 Eq. 15(等價 Eq. 7),且滿足 diagonal(δ_{s,s}=D̂_s)與 semigroup(Eq. 16)條件;(ii) 反過來,任何滿足這兩條件的 δ,經 Eq. 7 得到的就是真的 flow map;(iii) Prop. A.3:在 (s,u,t) 與 state 取樣 full support 下,KL 目標(Eq. 21)的 population minimum 為 0,且每個 minimizer 都還原 teacher 的 flow map。

**S9. 推論方式與實驗主張(§2.3, §3)**
- Flow decoding(Eq. 8, Alg. 3):K 步,t_k = k/K,加 self-conditioning 與 prefix guidance(Eq. 25–27)。
- Flow map decoding(Eq. 9):K=1 即 X̂_{0,1} 一步生成。
- Self-speculative decoding(Eq. 10, 28, Alg. 4):一步 flow draft,用同一個模型的 greedy AR 預測驗證,commit 最長一致 prefix;輸出等於該 checkpoint 的 greedy AR 輸出。
- 實驗主張:(a) 步數夠多時 flow decoding 在 B=4 接近(0.5B)或超過(7B)AR SFT(Fig. 6, Table 5);(b) flow map 改善 1–2 步(Table 5);(c) 聯合訓練使 single-token(AR)平均準確率高於 matched AR SFT(Table 1/6);(d) self-speculation 得到約 2.5–5 TPF(Table 2/7),7B 在單張 H100、batch size 1 下 2.02–2.54× AR SFT 的 TPS(Table 8, Fig. 1/8)。

---

## 2. 必須教對的推導

### D1. Velocity–denoiser 恆等式(Prop. A.1, Eq. 5, 11–12)
條件:線性 interpolant (2);t ∈ [0,1)(τ^i<1,否則除以零);b_t(x;c) := E[İ_t | c, I_t = x]。
1. İ_t^i = τ̇^i(t)(x_1^i − x_0^i)。 (11)
2. 由 (2) 解 noise:x_0^i = (I_t^i − τ^i x_1^i)/(1−τ^i) ⇒ x_1^i − x_0^i = (x_1^i − I_t^i)/(1−τ^i)。
3. 取條件期望,I_t = x 已知:b_t(x;c)^i = τ̇^i/(1−τ^i) · (E[x_1^i | c, I_t = x] − x^i)。
4. x_1^i = E_{y_i} 是 y_i 的確定函數 ⇒ E[x_1^i | c, I_t=x] = Σ_v p(y_i=v | c, I_t=x) E_v = D_t(x;c)^i E。 (12)
重點:條件是「整個 block 的 x」,所以位置 i 的 velocity 透過 D 依賴其他位置;但只需要每個位置的「邊際」後驗,不需要 joint。係數只用該位置自己的 τ^i。

### D2. Euler step (8) 其實是「τ 時間的 Euler」,且與 flow map (7) 同形 〔顧問推導〕
(8):x̂^i_{k+1} = x̂^i_k + (τ^i_{k+1} − τ^i_k)/(1 − τ^i_k) · (p_k^i E − x̂^i_k)
  = (1−τ^i_{k+1})/(1−τ^i_k) · x̂^i_k + (τ^i_{k+1}−τ^i_k)/(1−τ^i_k) · p_k^i E。
這正是 (7) 把 δ_{s,t} 換成 D̂_s(論文在 Eq. 9 前也這樣說)。它不是 base time 的 Euler(那會是 x + Δt·τ̇^i/(1−τ^i)(pE − x));它是「假設 denoiser 輸出在該步內不變」時 ODE 的精確解。後果:
- 最後一步 τ^i(t_K)=1 ⇒ x̂^i = p^i E,不會發散(雖然 (5) 在 t→1 有 1/(1−τ))。
- K=1 時與 τ 的形狀無關,直接得 (10):p = D̂_0(x̂_0;c),ŷ_i = argmax p^i。
- 產生的 token 是「最後一次 denoiser 預測的 argmax」(ŷ_i = argmax p^i_{K−1}),不是把 x̂_1 找最近 embedding。

### D3. Two-time denoiser 與 semigroup 權重 γ(Prop. A.2, Eq. 17–18, 16)
抑制 x, c;τ_r := τ^i(r)。
1. 由 ∂_r X = τ̇/(1−τ_r) · (D̂_r(X)E − X):
   d/dr [X_{s,r}/(1−τ_r)] = [(1−τ_r)∂_r X + τ̇ X]/(1−τ_r)² = w(r) D̂_r(X_{s,r})E,w(r) := τ̇(r)/(1−τ_r)²。 (17)
2. Z_{s,t} := ∫_s^t w dr = [1/(1−τ_r)]_s^t = (τ_t − τ_s)/((1−τ_t)(1−τ_s));δ_{s,t} := (1/Z_{s,t}) ∫_s^t w(r) D̂_r(X_{s,r}) dr —— 機率向量的加權平均,所以仍是機率向量。 (18)
3. 對 (17) 在 [s,t] 積分再乘 (1−τ_t):X_{s,t} = (1−τ_t)/(1−τ_s) · x + (τ_t−τ_s)/(1−τ_s) · δ_{s,t}E,即 (7)。
4. Semigroup:在 u 切開積分,r ≥ u 時 X_{s,r} = X_{u,r}(X_{s,u}) ⇒ Z_{s,t}δ_{s,t} = Z_{s,u}δ_{s,u} + Z_{u,t}δ_{u,t}(X_{s,u}),故 γ = Z_{s,u}/Z_{s,t} = (1−τ_t)(τ_u−τ_s)/((1−τ_u)(τ_t−τ_s))。 (16)
5. Diagonal:t→s 時 w/Z 集中在 r=s ⇒ δ_{s,t} → D̂_s。
條件與陷阱:(15) 只固定 mean embedding δE(因 |V| ≫ d,不同機率向量可有相同 mean embedding),δ 本身的唯一性來自 (18) 的建構;γ 是「每個位置各自的」γ^i;t→1 時 γ^i→0,即 δ_{s,1}(x) = δ_{u,1}(X_{s,u}(x)) 〔顧問推導〕;命題在 t<1 成立,t=1 僅「極限存在時」。
筆誤:證明中寫「which is (15)」,實際導出的是 (7)(兩者等價);App. A.2 後段稱「Theorem A.2」,實為 Proposition A.2。

### D4. Tokenwise time = loss 的 inverse CDF(Eq. 22)
F_i(τ) = ∫_0^τ ℓ^i / ∫_0^1 ℓ^i,τ^i(t) = F_i^{-1}(t)。對 F_i(τ^i(t)) = t 微分:F_i'(τ^i) τ̇^i = 1 ⇒ τ̇^i(t) = (∫_0^1 ℓ^i) / ℓ^i(τ^i(t)) 〔顧問推導:論文只寫 τ̇^i ∝ 1/ℓ^i〕。
等價說法:t ~ U[0,1] 時 τ^i(t) 的密度 ∝ ℓ^i(τ) 〔顧問推導〕。ℓ^i>0 保證 F_i 嚴格遞增、τ^i(0)=0、τ^i(1)=1。
實作(Alg. 2):L ∈ R^{B×M},M=200 個 uniform bin,初始化為 uniform;每 S=100 步做 4 次無梯度 forward;L_{i,m} ← βL_{i,m} + (1−β)L̄_{i,m},β=0.99(Eq. 23);沒被訪問的 bin 保留舊值;F_i 為正規化 row 的 piecewise-linear CDF。

### D5. Self-speculative 的接受規則與「輸出 = greedy AR」(Eq. 10, 28, App. B.3, Alg. 4)
- Draft:p^i = D̂_0(x̂_0;c)^i,ŷ_i = argmax p^i,無 prefix guidance。
- 第一個 token:與已有的 q_N = f(c)_N 比對(免費);不符則 ŷ_1 ← argmax(q_N),a = 1。
- 其餘:a = max{ j ∈ [B] : ŷ_{i+1} = argmax f(c⊕ŷ)_{N+i} ∀ 1 ≤ i < j }。 (28) j=1 為空條件,故 a ≥ 1。
- 為何等於 greedy AR:被接受的 ŷ_{i+1} 都等於「以 c⊕ŷ_{1:i} 為 prefix 的 greedy AR 預測」,而 ŷ_{1:i} 已歸納地等於 greedy AR 輸出(prefix 的 causal attention 使 f(c⊕ŷ)_{N+i} 只依賴 c⊕ŷ_{1:i})〔顧問補上歸納步驟;論文只陳述結論〕。
- Fused pass:輸入 = clean prefix c⊕ŷ(time 1)+ B 個各含 B 個 noise token 的 block z^(1..B)(time 0);z^(j) causal 地看 c⊕ŷ_{1:j}、block 內 bidirectional、block 間互不可見(Fig. 9)。驗證後取 z^(a) 的輸出當下一個 draft,並沿用 f(c⊕ŷ)_{1:N+a} 與 KV cache。修正情況 A={1},只需一個 noise block。
- 每次 fused forward commit 1 ≤ a ≤ B 個 token ⇒ 1 ≲ TPF ≤ B 〔顧問推導〕。TPF = Σ_j T_j / Σ_j F_j(Eq. 29),F_j = fused 次數 + 1 次初始 draft。
- 一次 fused forward 新處理 B + B² 個位置(B=16 時 272 個)〔顧問推導〕——這是 TPF 4.9 只換到 2.54× TPS 的原因之一(論文只說「forward-pass costs differ」)。

---

## 3. 常見誤讀、內部不一致、未說明之處

### 誤讀
1. **「flow 生成比 AR 快 2.54×」— 錯。** 2.54× 來自 self-speculative decoding,其輸出逐 token 等於同一 checkpoint 的 greedy AR;flow 只當 drafter,品質就是 AR 品質。純 flow decoding 要 AR 水準需 B=4、32–64 步(Table 5),即每個 token 約 8–16 次 denoiser forward,另加每步一次 prefix-free forward(guidance)——比 AR 慢 〔每 token forward 數為顧問推導〕。論文 §3.3 自己也說 flow decoding 需多步。
2. **一步 draft 不是「一步解 flow」。** t=0 時 I_0 = x_0 與 y 獨立,理想的 D_0(x_0;c)^i = p(y_i | c),即各位置獨立的邊際,noise 不帶資訊 〔顧問推導〕。這就是 multi-token prediction;論文稱之為「multi-token prediction effect」,並承認「One-step drafts can be inconsistent across positions」。能在一步內捕捉 joint 的是 distilled flow map X̂_{0,1},但 flow map 沒有用在 self-speculation。
3. **「2.54× 且品質相當」的範圍。** 是 7B、B=16、IFEval/GSM8K/HumanEval 三項平均(74.5 vs 73.9, Table 8)。六項平均下 7B B=16 是 58.0 < AR SFT 58.4(Table 2)。B=4:2.02×,76.5 vs 73.9。B=8:93.8 TPS(2.37× 〔顧問核算〕)。
4. **Lafa 不是 TPS 最高。** Dream k=4 為 113.7 TPS(準確率 37.7),高於 Lafa B=16 的 100.2;Lafa 的主張是 quality–speed frontier(Table 8)。
5. **Throughput 只在 batch size 1、單張 H100、bfloat16、32-token warm-up 下量測**(App. C.2),且只有 7B Lafa;沒有 Lafa 8B 的 TPS。大 batch 下 B+B² 個額外位置的成本論文未討論。
6. **「flow 訓練改善 AR」是平均值。** 7B B=4:HumanEval 78.7→76.8 下降、MMLU-Pro 41.6→41.8;8B:BBH 71.6→70.7、HumanEval 82.3→79.9 下降、MMLU-Pro 持平,平均 +6.3 幾乎全由 IFEval 38.1→62.3 與 GSM8K 62.9→77.8 貢獻(Table 2/6)。8B 由 Qwen3 8B *Base* 起訓(Instruct 版再訓練會退步),AR SFT 的 IFEval 只有 38.1。Table 1 的 Lafa 數字(31.6/60.9/63.1)即 B=4 checkpoint;B=8、B=16 的 AR 準確率要看 Table 2(0.5B:32.3、30.8;7B:59.3、58.0)。無 seed、無信賴區間;與 MARS/DeepSeek-MTP 的差距約 0.9–1.4 分。
7. **「8B」的範圍。** 8B 只有 B=4、只有 single-token 與 self-speculative 結果;flow/flow-map decoding、throughput、ablation 都沒有 8B。Ablation 全是 0.5B、B=4、32 步。
8. **Flow map「student 勝過 teacher」只在 7B、4 步**(69.4 vs 64.3)。0.5B:4 步 28.5 < 30.0,16 步 32.6 < 35.4;7B 無 16 步數據。1 步:0.5B 23.3 vs 20.8,7B 49.0 vs 43.7;2 步:25.2 vs 19.7,63.3 vs 46.2(Table 5)。一步 flow map 離 AR SFT(38.6 / 73.9)仍遠。
9. **步數–準確率非單調。** 0.5B B=4:1 步 20.8 → 2 步 19.7;64 步 38.5 → 128 步 37.1。論文用詞是「generally improves」。
10. **與 diffusion baseline 的比較不是 matched。** 只有標 † 的 baseline 同初始化/資料/預算;Dream、LLaDA、Efficient-DLM、Fast-dLLM v2(無 †)用釋出的 checkpoint。「≥18× fewer tokens」是對這些模型「reported budgets」(up to 500B)的比較,且 Lafa 是從預訓練 AR 模型起訓。Lafa 8B vs Efficient-DLM:63.1 vs 63.5(single-token)、63.1 vs 63.6(B=32)/61.6(B=4),TPF 皆 2.9——「similar」,不是勝過。
11. **Table 8 的 baseline 用各自 native decoder(多為 confidence threshold)**,不保證輸出等於 greedy AR;只有 Lafa 與 DeepSeek-MTP 是 greedy 驗證。
12. **left-to-right bias** 是學到的一般趨勢(「generally satisfy」),Fig. 5 只展示 7B;論文沒有給機制性解釋。
13. **Eq. 27 的 guided「denoiser」在 g>1 時不是機率向量**(可有負值)〔顧問推導〕;Alg. 3 實際在 embedding 空間做 x̃ = s′E + g(sE − s′E),且最終 token 取未 guided 的條件預測 s 的 argmax。

### 內部不一致
- **訓練量**:§3 寫 5–15 epochs(9–27B tokens),abstract 寫 <30B;Table 4 對三個尺度都寫「5 epochs / approximately 9B」;Table 6 說 † 模型訓練 five epochs。Table 4 caption 提到較大 block「continue training from smaller-block checkpoint」,故 15 epochs 可能是 B=4→8→16 累計 〔顧問推測,論文未明說〕。18× ≈ 500/27 〔顧問核算〕。
- **Table 3**:訓練區塊的完整設定為 38.0,推論區塊同一設定為 38.1;Table 5 對應格(0.5B, B=4, 32 步)為 38.1(三項平均 38.13 〔顧問核算〕)。
- **Table 2/7 的 7B B=8 平均 TPF** 寫 4.3;六個分項平均為 4.37 〔顧問核算〕(可能是分項已四捨五入)。Table 8 Lafa B=16 平均準確率寫 74.5,分項平均 74.57 〔顧問核算〕。
- **Noise scale**:訓練 σ=4.0,flow decoding 推論 σ=0.1(Table 4),正文定義 p_0 = N(0,σ²I) 為單一 σ,未解釋差異;self-speculative draft 用的 σ 未說明(Alg. 4 只寫「scale σ」)。
- **Fig. 10**:caption 說右圖 Lafa 的 B=4 是 8B、B∈{8,16} 是 7B;C.3 正文卻分別討論「At 7B」「At 8B」而不說明 block size。
- **Algorithm 2** 用 synced interpolant (1−t)x_0 + t x_1 估 ℓ^i,而 Eq. 22 的 ℓ^i 定義為「tokenwise time τ 下」的 loss、實際訓練用 asynchronous interpolant;兩者的條件分佈不同,論文只以「避免 starving low-probability regions」說明。
- **Self-conditioning 的時機**:§2.3 說每步重用前一步預測;第一步無前一步(Eq. 25 q_0 = D_0(x̂_0;c),Alg. 3 SC(x,∅)=x)。訓練時則是同一 t 的兩次 forward(Eq. 24)。

### 未說明
- Flow map distillation 的所有超參數:(s,u,t) 取樣分佈、訓練步數/token 數(是否計入 <30B)、學習率、三個 KL 的權重(Eq. 20 皆為 1)。student 推論時是否用 self-conditioning/guidance 未明說;§3.1 把兩者的整合列為 future work,暗示未使用,因此 Fig. 6 的同步數比較不是同 forward 數。
- Prefix-free 評估 D_t(x;∅) 如何訓練:沒有提 prefix dropout;訓練中空 prefix 似乎只出現在每個序列的第一個 block 〔顧問推測〕。
- 輸出 head 與 E 是否 tied;embedding norm 相對於 σ=4.0 的大小;Fig. 5、Fig. 6、Fig. 7、Fig. 10 的曲線數值(Fig. 5、Fig. 10 無對應表格)。
- Table 4:max sequence length「512–1024」未說明何時用哪個;adaLN 0.5B 為 per-layer,7B/8B 為 shared across layers,未解釋。
- 0.5B 的 MMLU-Pro 準確率 9.1–12.1,B=16 時 TPF 9.2 而準確率 9.1——該格的高 TPF 不宜當作「有效加速」的例子 〔顧問意見〕。

---

## 4. 哪些可在瀏覽器即時示範

### 可以(closed-form / toy)
1. **Async interpolant + 精確 denoiser + ODE(Eq. 2, 4, 5, 8)。** 小 vocab(3–5 個 token)、d=1 或 2、B=2,給定 joint prior p(y_1,y_2)。後驗有閉式 〔顧問推導〕:p(y | I_t = x) ∝ p(y) ∏_i N(x^i; τ^i E_{y_i}, (1−τ^i)²σ² I),對另一位置加總得 D_t(x)^i;代入 (5) 得精確 velocity,用 (8) 積分。可調:K、σ、synchronous vs asynchronous τ、每個位置的 τ 曲線。
2. **一步 draft 的不一致 vs 多步的一致。** 同一 toy:D_0 = 各位置邊際。例 〔顧問構造〕:p(A,B)=0.4, p(C,D)=0.3, p(E,D)=0.3 ⇒ 位置 1 argmax = A(0.4),位置 2 argmax = D(0.6),得到機率為 0 的「A D」;K 增大後樣本落在有支撐的組合。這是全文最值得示範的一點。
3. **Tokenwise time = inverse CDF(Eq. 22–23)。** 讓使用者畫/選 ℓ^i(τ),即時算 F_i 與 τ^i(t);顯示 uniform t-grid 映到 τ 的疏密;可用 M 個 bin 的 piecewise-linear 版本。注意:這不能重現 Fig. 5 的實際曲線。
4. **Flow map 與 semigroup(Eq. 7, 16, 18)。** 在 toy 上以數值積分算 δ_{s,t},驗證 X_{s,t} = X_{u,t}∘X_{s,u} 與 γ^i 公式;顯示 t→s 時 δ→D_s。也可示範「一步 X_{0,1} 是 noise→token 的確定映射,能保留 joint」。
5. **Self-speculative decoding 模擬(Eq. 10, 28, Alg. 4)。** toy AR 表(例如 bigram)+ 邊際 drafter;逐步顯示 draft、驗證、接受長度 a、fused pass 的 B 個候選 block 與 attention mask(Fig. 9);計算 TPF(Eq. 29);顯示輸出與 greedy AR 完全相同;顯示每次 forward 的位置數 B+B²。
6. **Attention mask(Fig. 3, 4b, 9b)。** 純規則,可互動生成。
7. **Prefix guidance(Eq. 27)。** toy 上有/無 prefix 的後驗,g 滑桿;可顯示 g>1 時「機率」出現負值。
8. **論文表格的重繪**(Table 2/5/7/8 的 accuracy–TPF、accuracy–TPS、accuracy–steps):靜態資料,需標明是論文數字。

### 不行
- 任何 benchmark 準確率、Table 3 ablation、「flow 訓練改善 AR」:需要 0.5B–8B 模型與評測。
- Fig. 5 的 learned τ 實際形狀、left-to-right 的湧現:需訓練好的模型的 loss 統計;toy 只能示範機制。
- Flow map distillation 的訓練過程(Eq. 20):需要訓練網路;toy 只能示範被 distill 的目標(精確 δ_{s,t})。
- TPS / 2.54×:取決於 H100、KV cache、batch size 1 的實測;瀏覽器只能做「位置數」成本模型,不能當成 wall-clock。
- norm-frozen embedding、adaLN identity init、self-conditioning 的效益:皆為大模型上的經驗結果。

---

## 5. 先備知識(教學順序)

1. AR 語言模型:next-token 分佈、greedy decoding、teacher forcing、一次 forward 一個 token。
2. Transformer 的 causal vs bidirectional attention mask;KV cache 為何要求 prefix 表示不隨後續輸入改變。
3. Token embedding 矩陣 E;「機率向量 × E = 期望 embedding」。
4. 高斯 noise、線性內插 (1−t)x_0 + t x_1;條件期望與 Bayes 後驗。
5. ODE 與 Euler 積分;velocity field 把 noise 分佈運到資料分佈(flow matching:b_t = E[İ_t | I_t])。
6. Denoiser 參數化:預測 clean token(分類)而非直接預測 velocity → D1。
7. Block-AR 分解(Eq. 1)與 block 內平行。
8. Tokenwise / asynchronous time;CDF 與 inverse CDF(time warping)→ D4。
9. Cross entropy、MSE、KL。
10. Self-conditioning;classifier-free guidance → prefix guidance。
11. Flow map X_{s,t}、semigroup 性質、distillation 與 stop-gradient → D3。
12. Multi-token prediction 的獨立性問題;speculative / blockwise parallel decoding(draft + verify)→ D5。
13. 指標:TPF(Eq. 29)vs TPS;macro-average;六項 vs 三項 benchmark 子集。
