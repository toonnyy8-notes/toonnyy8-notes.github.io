# Advisor notes — Flow Map Language Models (arXiv 2602.16813v3)

給教案設計者，不是給讀者。凡是論文沒有明寫、由我自己推出或自己算出的內容，一律標 〔顧問推導〕。
未標記的數字與式子皆直接取自論文（式號、表號、節號如括號所示）。符號：L 序列長度、|V| 詞表大小、
x0~N(0,I) 雜訊、x1 = one-hot 資料、t=0 是雜訊、t=1 是資料（注意與多數 diffusion 文獻方向相反）。

---

## 1. 論證鏈（9 步）

**步驟 1 — 問題設定：離散 diffusion 的 factorization error（§2，式 (2)，Fig. 3）**
- 假設：離散 diffusion 以 ancestral sampling 反轉 noising，需轉移機率 p_{t|s}(y_t|y_s)，其定義域是 V^L，輸出維度不可行。
- 作法：改用逐 token 乘積近似 p̃_{t|s}(y_t|y_s) = ∏_l p^l_{t|s}(y_t^l|y_s)（式 (2)）。
- 主張：條件獨立只在 t→s 的極限成立（引 [33]），步數少時忽略 token 間相關，產生 new-diego / san-york 這類無效組合。
- 這一步是「引用＋玩具圖」，不是本論文的證明。

**步驟 2 — 連續表示：one-hot + argmax（§3.1，式 (3)(4)）**
- f: y ↦ (onehot(y^1),…,onehot(y^L))ᵀ ∈ R^{L×|V|}；g: x ↦ (argmax(x^1),…,argmax(x^L))；g(f(y))=y。
- 不投影回 simplex，狀態活在無約束的歐氏空間。不需要正則化或另外訓練 embedding。

**步驟 3 — Flow matching（§3.2，式 (5)–(8)）**
- I_t = (1−t)x0 + t x1（式 (5)）；probability flow ẋ_t = b_t(x_t)（式 (6)）；b_t(x) = E[x1 − x0 | I_t = x]（式 (7)）。
- 可用 MSE 回歸學 b（式 (8)），但目標 x1−x0 繼承高斯雜訊的 full-rank 結構；當 |V| ≫ 網路內部維度 d 時 underfit（引 [40]）。
  實證：Velocity+MSE 的 Gen. PPL = 3801.36（Table 18）。

**步驟 4 — Denoiser 是 token 級後驗（式 (9)–(11)，Lemma 3.1，App. C.1.1）**
- D_t(x) := E[x1 | I_t = x]，b_t(x) = (D_t(x) − x)/(1−t)（式 (9)）。
- Lemma 3.1：D_t(x)^l = p^l_{1|t}(·|I_t = x) ∈ Δ^{|V|−1}。證明只用「x1^l 是 one-hot，所以條件期望＝機率向量」（式 (40)(41)）。
- 推論：可用 tokenwise softmax 輸出層。

**步驟 5 — Cross-entropy 訓練正確且有 W2 保證（Prop. 3.2，式 (12)(13)，App. C.1.2）**
- L_CE(D̂) = ∫₀¹ E[−Σ_l log D̂_t(I_t)^l · x1^l] dt（式 (12)）。
- 分解：L_CE = ∫E[H(x1|I_t)]dt + ∫E[Σ_l KL(p^l_{1|t} ∥ p̂^l_{1|t})]dt（式 (45)）⇒ 唯一極小點是真 denoiser。
- 若 excess risk ≤ ε，則 W2²(p̂_{1−ξ}, p_{1−ξ}) ≤ Cε（式 (13)）；需假設 C.1（b, b̂ Lipschitz）、C.2（二階動差有界）、C.3（輸出在 simplex 內部）。

**步驟 6 — 與離散 diffusion 的關鍵差別（§3.2 末段）**
- 兩者都學同一個逐 token 後驗 p^l_{1|t}。離散法拿它做 ancestral sampling（需要 joint，故有 factorization error）；
  連續法只拿它算 velocity，而 velocity 是條件期望（對 x1 線性），所以逐 token 邊際就已經給出「精確」的 b_t（式 (11)+(9)）。
- 這是全文的觀念樞紐，務必講清楚（見第 3 節誤讀 M1）。

**步驟 7 — Flow map 與 two-time denoiser（§3.3，式 (14)–(22)，Prop. 3.3，App. C.2–C.4）**
- Flow map：X_{s,t}(x_s) = x_t（式 (14)）；參數化 X_{s,t}(x) = x + (t−s)v_{s,t}(x)（式 (15)）；
  三個刻畫條件：X_{s,s}=x、lim_{s→t}∂_t X_{s,t} = b_t、X_{u,t}(X_{s,u}(x)) = X_{s,t}(x)（式 (16)）。
- 定義 δ_{s,t}(x) := x + (1−s)v_{s,t}(x)（式 (18)）。Prop. 3.3 四性質：
  (i) X_{s,t}(x) = (1−t)/(1−s)·x + (t−s)/(1−s)·δ_{s,t}(x)（式 (19)）；(ii) δ_{s,t}(x)^l ∈ Δ^{|V|−1}（式 (20)）；
  (iii) δ_{s,s} = D_s（式 (21)）；(iv) δ_{s,t}(x) = γ δ_{s,u}(x) + (1−γ) δ_{u,t}(X_{s,u}(x))，γ = (1−t)(u−s)/((1−u)(t−s)) ∈ [0,1]（式 (22)）。
- 意義：X_{s,t} 本身會離開 simplex，但 δ_{s,t} 永遠在 simplex 上 ⇒ flow map 也能用 softmax + KL/CE 學。

**步驟 8 — 蒸餾目標與「離散沒有 sample-level flow map」（Prop. 3.4 式 (23)；Prop. 3.5，App. C.1.3）**
- L_KL(δ) = E_{t,s,u}E[Σ_l KL(δ̄^l_{s,t} ∥ δ^l_{s,t}(I_s))] + E_t E[Σ_l KL(D_t^l(I_t) ∥ δ^l_{t,t}(I_t))]，
  δ̄_{s,t} := sg(γ δ_{s,u}(I_s) + (1−γ) δ_{u,t}(X_{s,u}(I_s)))（式 (23)）；(s,u,t) 的分布需在 {0≤s≤u≤t≤1} 有 full support。
- 實作上 D_t 換成凍結的預訓練 D̂_t（蒸餾）；換成 one-hot x1 則退化成式 (12) 的 CE（直接訓練 / self-distillation，式 (104)）。
- Prop. 3.5：S 有限，對任一 µ 存在 ν 使得 ν ≠ f_#µ 對所有確定性 f:S→S。證明：ν(y) = Σ_{x∈f⁻¹(y)} µ(x)（式 (70)），
  故 ν(y) ∈ {0} ∪ [µ_min, 1]；取 ν(y0) = µ_min/2 即不可達。

**步驟 9 — 工程選擇與實驗（§4，§5）**
- 時間重參數化：P_e(t) := (1/L)Σ_l P(g^l(x_t) ≠ g^l(x1))（式 (24)），P_e(0) = 1 − 1/|V|，P_e(1) = 0；
  τ(t) = (P_e(0) − P_e(t))/P_e(0) = 1 − |V|/(|V|−1)·P_e(t)（式 (25)）。訓練時 τ(t)~U[0,1]，取樣格點 t_n = t(n/N)。
- Off-diagonal 三元組：h~U[0,1]，τ(s)~U[0,1−h]，τ(t)=τ(s)+h，τ(u)=(τ(s)+τ(t))/2；以機率 p=1/32 直接取 (s,t)=(0,1)。
- 設定：LM1B（L=128，bert-base-uncased，|V|=30,522）、OWT（L=1024，gpt-2 tokenizer，|V|=50,257）；
  179M 參數、12 層 DiT + RoPE + AdaLN；FLM 1M steps、batch 512、Adam(β1=0.9, β2=0.999)、lr 3×10⁻⁴、2,500 warmup；FMLM 蒸餾 100k steps。
- 評估：每模型 1,024 個樣本，GPT-2 Large 算 Gen. PPL，並報 per-sample unigram entropy（資料：LM1B 4.31、OWT 5.44）。
- 主結果：
  - Table 10（1024 步）：FLM LM1B 96.91 / 4.29，OWT 62.23 / 5.33；Duo 98.14 / 4.31，77.69 / 5.55；MDLM 109.21 / 4.32，105.15 / 5.63；CANDI 120.99 / 4.35，143.13 / 5.71；RDLM 268.21 / 4.33（僅 LM1B）。
  - Table 13（FMLM，Gen. PPL / Ent.）：LM1B 1 步 119.34 / 4.16、2 步 110.19 / 4.21、4 步 98.76 / 4.21；OWT 1 步 168.30 / 5.17、2 步 133.29 / 5.25、4 步 111.31 / 5.26。
    一步 baseline：LM1B MDLM+SDTT 1429.48、MDLM+Di4C 1217.10、Duo+DCD 1224.52、Duo+Di4C 292.94（Ent. 3.79）；OWT 1260.86、1298.80、5743.29、370.51（Ent. 3.92）。
  - Table 15（OWT，50 token prefix → 50 token 續寫，一步）：FMLM 141.66 / 4.14；MDLM+SDTT 374.70 / 4.09。
  - Autoguidance（LM1B，1024 步）：96.91 → 51.62（η=50，entropy 4.00）；Duo 在 η≥10 崩（PPL > 1900），MDLM 在 η≥10 崩（PPL > 1750）。
  - Sudoku（Table 22，%）：FMLM unconditional 1/2/4/1024 步 = 9.38 / 57.03 / 79.69 / 94.92；held-out 條件式 39.84 / 85.15 / 92.19 / 98.83；Sudoku-Extreme 16.01 / 20.70 / 25.00 / 26.56。

---

## 2. 教師必須做對的推導（5 個）

### D1. Denoiser–velocity 關係與 Lemma 3.1（式 (9)(11)，Lemma C.8 式 (73)–(77)）
對 I_t = x 取條件期望：
- x = (1−t)E[x0|I_t=x] + t D_t(x) ⇒ E[x0|I_t=x] = (x − t D_t(x))/(1−t)
- b_t(x) = E[x1 − x0 | I_t=x] = D_t(x) − (x − t D_t(x))/(1−t) = (D_t(x) − x)/(1−t)
- 反寫：D_t(x) = x + (1−t) b_t(x)（式 (77)）⇒ denoiser 是「用 b_t 走一個大小 1−t 的 Euler 步」，不是 flow map 的終點。
- 一般內插 I_t = α_t x0 + β_t x1 時是式 (73)；主文只用 α_t=1−t、β_t=t。
- Lemma 3.1：D_t(x)^l = Σ_i e_i · p^l_{1|t}(x1^l = e_i | I_t = x)（式 (41)）。
條件：t<1（1/(1−t) 奇異）；後驗條件在「整個序列」的 I_t=x 上，不是只有第 l 個位置。

### D2. CE 分解與 W2 上界的常數（式 (42)–(65)）
- E_p[−log q] = H(p) + KL(p∥q)（式 (42)）⇒ 式 (45)。
- Pinsker：|p−q|² ≤ |p−q|₁² = 4TV² ≤ 2KL（式 (52)）⇒ ∫E|D̂_t(I_t) − D_t(I_t)|² dt ≤ 2Δ_D(D̂)（式 (50)）。
- b̂_t − b_t = (D̂_t − D_t)/(1−t)（式 (54)）；t ≤ 1−ξ 時 (1−t)⁻² ≤ ξ⁻² ⇒ ∫₀^{1−ξ}E|b̂_t − b_t|² dt ≤ 2ξ⁻²Δ_D（式 (53)，C_b(ξ) = 2ξ⁻²）。
- 同一個 x0 耦合兩條流，Gronwall + Cauchy–Schwarz：W2²(p^{b̂}_{1−ξ}, p_{1−ξ}) ≤ e^{2L_b̂(1−ξ)}(1−ξ)∫₀^{1−ξ}E|b̂_t − b_t|²dt（式 (64)）。
- 結論：W2(p^{b̂}_{1−ξ}, p_{1−ξ}) ≤ C_D(ξ)·√Δ_D，C_D(ξ) = √(2(1−ξ))·e^{L_b̂(1−ξ)}/ξ（式 (65)）。
  延伸到 t=1 要加 r_ξ ≤ ξ(√M_b + √M_b̂)（式 (58)）。
- 〔顧問推導〕式 (13) 的 C 即 C_D(ξ)² = 2(1−ξ)e^{2L_b̂(1−ξ)}/ξ²。ξ→0 時常數以 ξ⁻² 發散，且指數含 Lipschitz 常數，
  這是定性保證，不能拿來估實際誤差。式 (68) 上方的「(∫1² dt)」排版成「12 dt」，是 typo。

### D3. δ_{s,t} 在 simplex 上（Prop. 3.3 (i)(ii)，式 (78)–(83)）
- (i)：把 v_{s,t} = (δ_{s,t} − x)/(1−s) 代入式 (15)：X_{s,t} = x + (t−s)/(1−s)·(δ_{s,t} − x) = (1−t)/(1−s)·x + (t−s)/(1−s)·δ_{s,t}。
- (ii)：∂_τ X_{s,τ} = (D_τ(X_{s,τ}) − X_{s,τ})/(1−τ)（式 (79)）；乘積分因子 1/(1−τ)：∂_τ[X_{s,τ}/(1−τ)] = D_τ(X_{s,τ})/(1−τ)²（式 (80)）。
  由 s 積到 t：X_{s,t} = (1−t)/(1−s)·x + (1−t)∫_s^t D_τ(X_{s,τ})/(1−τ)² dτ（式 (81)）。
  與 (i) 比對：δ_{s,t}(x) = (1−s)(1−t)/(t−s) · ∫_s^t D_τ(X_{s,τ}(x))/(1−τ)² dτ（式 (82)）。
  權重非負，且 (1−s)(1−t)/(t−s) · [1/(1−t) − 1/(1−s)] = 1（式 (83)）⇒ δ 是沿軌跡的 denoiser 的凸組合 ⇒ 在 simplex 上。
- (iii)：δ_{s,s} = x + (1−s)b_s(x) = D_s(x)（式 (84)）。
- 條件：用的是「真」flow map 與「真」denoiser；學到的 δ̂ 在 simplex 上是因為 softmax 參數化，不是因為這個定理。
- 〔顧問推導〕t=1 時式 (19) 給 X_{s,1}(x) = δ_{s,1}(x)：跳到終點的 flow map 輸出就是 δ 本身。
  因 p1 支撐在 one-hot 上，真 δ_{0,1}(x0) 幾乎必然是 simplex 頂點（one-hot），所以一步生成是「對 softmax 輸出取 argmax」，不是從 softmax 抽樣。

### D4. δ 空間的 semigroup 條件與 γ（Prop. 3.3 (iv) / C.9 (iii)，式 (94)–(98)）
令 z := X_{s,u}(x) = (1−u)/(1−s)·x + (u−s)/(1−s)·δ_{s,u}(x)。
- X_{u,t}(z) = (1−t)/(1−u)·z + (t−u)/(1−u)·δ_{u,t}(z)
  = (1−t)/(1−s)·x + (1−t)(u−s)/((1−u)(1−s))·δ_{s,u}(x) + (t−u)/(1−u)·δ_{u,t}(z)（式 (95)）
- 與 X_{s,t}(x) = (1−t)/(1−s)·x + (t−s)/(1−s)·δ_{s,t}(x) 相等，消去 x 項，乘 (1−s)/(t−s)：
  δ_{s,t}(x) = (1−t)(u−s)/((1−u)(t−s))·δ_{s,u}(x) + (t−u)(1−s)/((1−u)(t−s))·δ_{u,t}(z)（式 (97)）
- 係數和：(1−t)(u−s) + (t−u)(1−s) = (t−s)(1−u)（式 (98)）⇒ 第二個係數 = 1−γ。
- u=(s+t)/2 時 γ = (1−t)/(2−s−t)，1−γ = (1−s)/(2−s−t)（Prop. C.9）。
- 注意：這個簡化式是「t 空間中點」。實際演算法取的是「τ 空間中點」τ(u)=(τ(s)+τ(t))/2，所以 γ 必須用換回 t 後的 (s,u,t) 代一般式算（Algorithm 7 正是如此寫）。
- 〔顧問推導〕t=1 時 γ=0，teacher 完全是 δ_{u,1}(X_{s,u}(I_s))；邊界對 (s,t)=(0,1) 即 δ̄ = δ_{u,1}(X_{0,u}(x0))。
- 〔顧問推導〕我用 L=2、|V|=2 的兩模態玩具（new-york / san-diego）數值驗證過：(s,u,t)=(0.1,0.5,0.9) 時 γ=0.1，
  式 (22) 兩邊都得到 [0.9557, 0.0443, 0.9557, 0.0443]，各位置和為 1。

### D5. Prop. 3.5 與 P_e(t) 的閉式
- Prop. 3.5 見步驟 8，三行即可，重點是 ν(y) 只能是 µ 值的子集和。
- 〔顧問推導〕P_e(t) 的閉式（論文未寫出，只在 App. E 說用 Gauss–Hermite 求積）：單一位置、真 token 為 k，
  I_t = (1−t)z + t e_k，z~N(0,I_{|V|})。解碼正確 ⇔ z_k + t/(1−t) > max_{j≠k} z_j，故
  P_e(t) = 1 − ∫ φ(z) Φ(z + t/(1−t))^{|V|−1} dz。
  檢查：t=0 時積分 = 1/|V|，符合 P_e(0) = 1 − 1/|V|；與資料分布無關，只依賴 |V|（位置間獨立，因為雜訊各維獨立）。
- 〔顧問推導〕我依上式算出的數值（200 點 Gauss–Hermite）：

  | |V| | P_e(0.5) | P_e(0.8) | τ=0.05 / 0.25 / 0.5 / 0.75 / 0.95 對應的 t |
  |---|---|---|---|
  | 10 | 0.6591 | 0.0168 | 0.203 / 0.487 / 0.619 / 0.703 / 0.777 |
  | 100 | 0.9175 | 0.0854 | 0.450 / 0.642 / 0.716 / 0.765 / 0.812 |
  | 1000 | 0.9834 | 0.2360 | 0.603 / 0.717 / 0.764 / 0.799 / 0.834 |
  | 30522 | 0.9987 | 0.5439 | 0.707 / 0.774 / 0.805 / 0.829 / 0.854 |
  | 50257 | 0.9991 | 0.5873 | 0.717 / 0.780 / 0.809 / 0.832 / 0.856 |

  即 |V|≈3–5 萬時，90% 的解碼進度發生在 t≈0.71–0.86 的窄帶，t=0.9 時 P_e 已幾乎為 0。
  論文文字說「concentrates acutely near t = 1」；依我的計算，窄帶在 t≈0.8 而非貼近 1。
  我無法從 paper.txt 讀出 Fig. 9 的實際曲線，設計者若要畫圖請以 PDF 的 Fig. 9 對照，文字敘述建議寫「集中在很窄的時間窗」而不要寫「貼近 t=1」。
- App. C.6（Prop. C.12）：假設每個位置的 x1^l 在 V 上均勻，由 Fano 不等式得 τ(t) ≤ σ(t) + O(1/log|V|)（式 (106)），σ 是 entropic time（式 (105)）。
  作者自己說這個假設「rather strong」。

---

## 3. 常見誤讀、內部不一致、未交代之處

### 誤讀
- **M1「連續法沒有做 factorization」**：錯。FLM/FMLM 的網路輸出同樣是逐 token 的 softmax，學的也是逐 token 邊際後驗（Lemma 3.1）。
  差別在「用途」：只用來算條件期望（velocity），不拿來獨立抽樣。token 間的相關性是由共同的初始雜訊 x0 經確定性映射帶出來的。
- **M2「一步生成是從 softmax 抽樣」**：錯。X_{0,1}(x0) = δ_{0,1}(x0)，再 argmax（Algorithm 8、式 (4)）。
- **M3「δ_{s,t} 是 x_t 處的 denoiser」或「δ 是終點預測」**：錯。它是沿 s→t 軌跡的 denoiser 加權平均（式 (82)），輸入是 x_s；只有 s=t 時等於 D_s。
- **M4「denoiser D_t(x) 就是一步 flow map」**：錯。App. C.2 明講 denoiser 是後驗平均，會把多模態平均掉；它只是一個 Euler 步。
- **M5「Prop. 3.5 證明離散 diffusion 不可能一步生成」**：過度解讀。它只說「對給定 µ 存在某個 ν 不可達」，沒有說真實資料分布不可達，也沒有量化近似誤差。
  〔顧問推導〕若 µ 是 V^L 上的均勻分布，µ_min = |V|^{−L}，可達的 ν 是所有機率為 |V|^{−L} 整數倍的分布，近似可以任意細；
  若 µ 是 masked diffusion 的全 mask 點質量，則 f_#µ 只能是點質量，才是真正的硬性障礙。實務上的障礙是步驟 1 的 factorization，不是這個命題。
- **M6「FLM 在所有步數都最好」**：錯。Fig. 11 圖說自己寫「performance degrades at low step counts」；FLM 的優勢陳述是在大步數（Table 10 是 1024 步）。
- **M7「FMLM 完整保留 teacher 品質」**：LM1B 上 4 步 98.76 接近 FLM 的 96.91；但 OWT 上 4 步 111.31、1 步 168.30，對比 FLM 1024 步的 62.23，差距很大。
  且 FMLM 的 entropy 低於資料（LM1B 一步 4.16 vs 4.31；OWT 一步 5.17 vs 5.44），PPL 低有一部分來自多樣性下降。
- **M8「CE 一定勝過 MSE」**：Table 18 支持（一步：δ-denoiser+softmax, CE 119.34 vs Average velocity, MSE 199.01），
  但 App. D 的兩階段 MSE 蒸餾在 LM1B 一步得到 102.49（第一階段）/ 104.37（第二階段），PPL 比 CE 的 119.34 更低（entropy 4.13 / 4.12 vs 4.16）。
  另外 Table 18 中「δ-denoiser, MSE」156.83 優於「δ-denoiser + softmax, MSE」181.72（但前者 entropy 3.94 < 4）。要寫成「在同一套單階段設定下 CE 最好」。
- **M9「Simplex diffusion 比較差」**：Table 18 中 Simplex [67] 的 Gen. PPL 是 85.07，低於 Euclidean 的 96.91；被判較差是因為 entropy 3.76（崩塌）。PPL 必須和 entropy 一起看。
- **M10「有 likelihood / validation perplexity 的比較」**：沒有。腳註 1 說本法量 validation perplexity 需額外訓練。全文只有 Gen. PPL（GPT-2 Large，1,024 樣本）與 entropy。
- **M11「速度有完整量測」**：只有 Fig. 1 的「8.3× speedup」與橫軸 Generation Time / Sample，正文沒有對應的表，也沒說 8.3× 是對誰、在什麼硬體。
  Sudoku 的 7.6× 是「一步＋rejection sampling 對 81 步自回歸」的估算。〔顧問推導〕81 × 0.0938 ≈ 7.6，隱含假設每次前向成本相同。
- **M12「time reparameterization 是學出來的」**：不是。τ(t) 由式 (25) 事先算好成 1,000 點查表 + cubic spline（App. E）；被比較的 CDCD [13] 才是線上估計。

### 內部不一致
- **一步 vs 8 步的說法四種版本**：摘要「exceeds the 8-step quality」；貢獻列表「nearing their 8-step quality」；Fig. 1 圖說「matches the 8-step」；
  §5.2「LM1B 相當於 baseline 的 8–16 步、OWT 相當於 4–8 步」。Table 13 只列 1/2/4 步，8 步以上的 baseline 數字只在 Fig. 12 圖上，沒有表。教案請用 §5.2 的保守說法。
- **Fig. 16**：圖例寫 Instability Region 為「Gen. PPL > 1k or Ent < 4.0」，圖說寫「entropy < 3.9」。而 FLM 最佳 guidance 點的 entropy 正好是 4.00。
- **Sudoku 一步 validity**：§5.5 寫 9%（Table 22 為 9.38），App. F 寫「achieves 5% validity and uniqueness」。以表為準。
- **Table 10 的 entropy 說法**：正文說 FLM 在 OWT 的 entropy「within ±0.1 of the data entropy, similar to the discrete baselines」；
  〔顧問推導〕5.44 − 5.33 = 0.11；baseline 偏差為 Duo 0.11、MDLM 0.19、CANDI 0.27（方向相反，偏高）。
- **Table 17 圖說**稱 FMLM/FLM 在「所有步數」reward 都較高；但 1024 步 Grammaticality：MDLM 0.255 > FLM 0.240。
- **計算成本**：結論說 one-hot 表示比 embedding diffusion 多約 30% 時間與記憶體；App. F 說比 MDLM 多 15–20% 記憶體但 wall-clock 更快（Table 32）。
  〔顧問推導〕由 Table 32 算記憶體比 FLM/MDLM = 1.16、1.18、1.24、1.16（有一格超過 20%）；App. F 說 Duo 比 FLM「up to 14% more memory」，但 78.0/63.4 = 1.23。時間 0.4717/0.3759 = 1.25 與文字的 25% 相符。
- **Self-BLEU**：正文說 FMLM「only slightly below those of real data」，但 Self-BLEU 越低越多樣，FMLM 是較高（較差）：LM1B 0.073 vs 0.047，OWT 0.121 vs 0.046（Table 25）。
  Table 26 標題寫 win rate (%)，數值卻是 0.39、0.42 這種比例。
- **取樣是否確定性**：Algorithm 8 是純確定性的 flow map 合成；App. E 卻說 FMLM 在兩個資料集都用 Kim et al. [91] 的「γ-sampling」並取 [92] 的最佳 γ（數值未給）；
  App. G 又說 FLM 與 FMLM 是 deterministic sampling。一步時兩者無差別，多步時不清楚。此處的 γ 與 semigroup 的 γ 同符號不同義。
- **式 (36)**（App. B 的 PMD loss）寫成 X̂_{s,t} − sg(X̂_{t,u}(X̂_{s,t}))，與式 (17)(39) 的下標順序不一致；式 (31) 的 (s,t,u) 順序也與式 (16) 不同。教學請統一用式 (16)(17)。
- **§3.2**說 Lemma 3.1 的後驗是「defined in (2)」，但式 (2) 定義的是 p^l_{t|s}，不是 p^l_{1|t}。
- **Table 18 的星號**：Time reparameterization 與 Continuous representation 兩組是 300k steps，不能和 1M steps 的 96.91 直接比（同為 decoding error rate：300k 是 106.98）。
  embedding 消融全部使用 [13] 的 learned time reparameterization，所以「One-hot encodings 130.42」與「Learned entropic time 130.42」是同一個模型。「Riemannian 268.21」就是 Table 10 的 RDLM。
- **Table 29/30** 混用有無 % 號，數值與 Table 22 的 unconditional 欄完全相同（App. F 解釋：所有 valid 樣本都互異且不在訓練集）。

### 未交代
- 網路如何同時吃兩個時間 (s,t)、δ̂ 是否由 D̂ 初始化：CE 蒸餾沒寫（只有 App. D 的 MSE 版本寫了初始化）。
- App. E：每個 batch 對半分成 diagonal（s=t）與 off-diagonal（s≠t），1/32 的邊界機率是「在 distillation 那一半之中」。主文沒提對半分。
- FMRG 的 λ、reward 梯度步數、Table 17 中 1024 步的未蒸餾 FLM 如何取得 look-ahead X_{t,1}：都沒寫。baseline 用的是 D-CBG [31] + 在加噪輸入上訓練的分類器，不是同一個 guidance 演算法。
- Autoguidance 的 weak model 是「推論時開 dropout 0.1」，沒有另訓模型。η=1 即無 guidance（式 (26)）。
- Table 28（prior 消融）寫「FMLM … 200k steps」，Gaussian 得 105.48 / 4.31，比主結果一步的 119.34 / 4.16 好；步數設定、是 FLM 還是 FMLM、幾步取樣都不清楚，不要拿來和主表比。
- Baseline 的蒸餾設定不對等：LM1B 上 SDTT/DCD 用 lr 6×10⁻⁵、batch 128、8 輪 × 10k steps；Di4C 取「32 步表現最好」的中間 checkpoint（LM1B 20k、OWT 50k）；OWT 用官方 checkpoint。
- Prop. 3.4 稱「unique critical point」，但 App. C.5 的證明只論述「兩項同時最小 ⇔ δ̂ = δ」，是兩三行的 sketch；且假設 teacher 精確。不要把它講成收斂性保證。
- Prop. 3.2 的 Lipschitz 假設要求 b̂_t 在 t∈[0,1) 上有一致的常數，但 b 含 1/(1−t)；作者用 ξ 截斷處理，常數隨之發散。
- 條件式生成需要另外 finetune 25k iterations（前 50 token 當 clean context），不是零成本。

---

## 4. 瀏覽器內可現場示範的內容

### 可以（閉式或小型數值）
1. **Fig. 3 的玩具：L=2、每個位置 |V|=2，資料只有 new-york 與 san-diego 各 50%。**
   - 離散一步：兩位置獨立各 50% ⇒ 四種組合各 25%，其中兩種無效（論文 Fig. 3 的數字）。
   - 連續：狀態 x ∈ R⁴，兩個模態 m_k。〔顧問推導〕後驗 p(k|I_t=x) ∝ exp(−|x − t m_k|²/(2(1−t)²))，D_t(x) = Σ_k p(k|x) m_k，b_t = (D_t − x)/(1−t)。
     用 RK2/Euler 積 ODE 就能畫軌跡、顯示確定性映射永不產生 new-diego。可順便展示 M1：每個位置的 denoiser 輸出是邊際，但兩個位置共用同一個 p(k|x)。
2. **δ_{s,t} 的 simplex 性質與 semigroup 恆等式**：在同一玩具上數值求 X_{s,t}，由式 (19) 反解 δ_{s,t} = ((1−s)X_{s,t} − (1−t)x)/(t−s)，
   讓使用者拖 s,u,t，看 X 離開 simplex、δ 留在 simplex、式 (22) 兩邊相等（我已驗證，見 D4）。
3. **P_e(t) 與 τ(t)（式 (24)(25)）**：用 D5 的一維積分（Gauss–Hermite 或梯形法）對 |V| ∈ {10, 100, 1k, 30k, 50k} 畫曲線與 τ 均勻取樣的刻度，重現 Fig. 9 的形狀。純閉式，毫秒級。
4. **γ 的幾何**：γ = (1−t)(u−s)/((1−u)(t−s))，中點時 γ = (1−t)/(2−s−t)；滑桿即可。t→1 時 γ→0。
5. **Prop. 3.5**：4 個狀態的 µ，列舉全部 4⁴=256 個 f，顯示可達的 ν 集合是有限點集，ν(y0)=µ_min/2 不在其中。
6. **Euler 少步失準 vs flow map**：在玩具上比較 N=1,2,4 步 Euler（Algorithm 6）與數值「真」flow map 的終點。這是示意，不是論文數字。
7. **Autoguidance 的兩種外插形式**：歐氏 b_weak + η(b − b_weak)（式 (26)）對比 logit 空間 p ∝ p_weak^{1−η} p^η；可用三維 simplex 示意高 η 時後者銳化。僅示意機制。
8. **Dir(1) 與高斯先驗的集中現象**（App. F）：Dir(1) 每維變異數 ∼1/|V|²，可抽樣顯示。

### 不可以
- 任何 Gen. PPL / entropy 數字（Tables 10、13、15、17、18、24）：需要 179M 參數模型、1M 訓練步與 GPT-2 Large 評分。只能引用表格。
- Sudoku validity（Table 22）：需要訓練 100k + 80k steps 的模型。
- Autoguidance / FMRG 的實際曲線（Fig. 16、Table 17）：需要訓練好的 LM 與分類器。
- Rank bottleneck（Velocity MSE = 3801.36）：需要 |V| ≫ d 的真實網路才會出現，玩具裡閉式 velocity 沒有這個問題。
- W2 上界（式 (13)）：常數含未知的 Lipschitz 常數，無法數值對照。
- 玩具中的任何數值都要明示是示意，不得與論文表格數字並列成「重現」。

---

## 5. 初學者先備知識（教學順序）

1. 語言模型在做什麼：token、詞表 |V|、序列長度 L、自回歸分解（式 (1)）與其序列化瓶頸。
2. One-hot 向量、機率 simplex Δ^{|V|−1}、softmax、argmax。
3. 條件期望與後驗；「one-hot 隨機變數的條件期望＝機率向量」（Lemma 3.1 的全部內容）。
4. Cross-entropy、熵、KL，以及 E_p[−log q] = H(p) + KL(p∥q)（式 (42)）。
5. 聯合分布 vs 邊際分布的乘積：為什麼逐 token 獨立抽樣會丟掉相關（Fig. 3）。這是離散 diffusion 的痛點。
6. 高斯雜訊與線性內插 I_t = (1−t)x0 + t x1；注意本文 t=0 是雜訊。
7. ODE、velocity field、Euler 法與步長誤差；probability flow 把雜訊分布搬到資料分布（式 (6)(7)）。
8. Flow matching：用回歸學條件期望（式 (8)(10)）；denoiser 與 velocity 互換（式 (9)）。
9. Flow map = ODE 的解算子；semigroup 性質 X_{u,t}∘X_{s,u} = X_{s,t}（式 (14)(16)）；average velocity（式 (15)）。
10. 蒸餾、teacher/student、stop-gradient；自舉式目標（用兩小步教一大步）。
11. 凸組合（為何平均後仍在 simplex）與積分因子法（只在講 D3 時需要，可放進階框）。
12. 評估指標：Gen. PPL（由 GPT-2 Large 打分）與 unigram entropy，為何兩者必須一起看（低 PPL + 低 entropy = 重複崩塌）。
13. （選讀）Pushforward 測度 f_#µ（Prop. 3.5）、Wasserstein-2、Pinsker、Gronwall（Prop. 3.2 的證明）、Fano 不等式（Prop. C.12）、guidance（式 (26)(27)）。
