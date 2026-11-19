# Advisor notes — arXiv 2609.20539v2 (Chen & Wang), "Parallelism, critical windows, and separations among diffusion language models"

給 tutorial designer,不是給讀者。全文(含 Appendix A/B/C)已讀完。凡是論文沒有寫、由我自己推出的內容都標 〔顧問推導〕;沒有標的數字與式子都對應論文的式號/節號。log 一律是自然對數(nats),除了 Section 8 的 ℓ 用 log₂。

論文是純理論:沒有訓練、沒有模型、沒有 benchmark 表。唯一的數值內容是 Figure 4(模擬)與 Appendix C。

---

## 1. 論證骨架(9 步)

**Step 0 — 計價單位(§1 Framework, §2.1–2.3)。**
一次 forward pass = 一次 score oracle query:給任意 (t, z),一次回傳所有座標 i 的 posterior marginal law((X₀)ᵢ | Xₜ = z)。三種腐蝕過程:
- Uniform:每個 token 以機率 1−e⁻ᵗ 重抽。Kᵁₜ(·|z) = αₜ·1[z=·] + βₜ,αₜ = e⁻ᵗ,βₜ = (1−αₜ)/S。Score = likelihood ratio sₜ(y)[i,a] = qₜ(y^{i←a})/qₜ(y)。誤差 ε_unif(t) = score-entropy(Bregman ψ(c,s) = s − c + c log(c/s),對 i、a 加總,Def 2.2)。
- Gaussian:one-hot 後跑 OU,Xₜ = e⁻ᵗX₀ + σₜG,σₜ = √(1−e⁻²ᵗ)。Tweedie(Lemma 2.4):E[X₀|Xₜ=x] = eᵗx + eᵗσₜ²sₜ(x)。誤差 ε_gauss(t) = σₜ²·(E‖sₜ−ŝₜ‖²)^{1/2}(Def 2.5,注意這是 L2 norm 不是平方)。
- Masked:每個 token 以機率 1−e⁻ᵗ 變 ⋆。Score = αₜ/(1−αₜ) · q_{i|I}(a|y_I)(Lemma 2.6),所以 oracle 其實只吃 partial assignment (I, x_I),與 t 無關。誤差以「level m = 已揭露座標數」定義,等於每個未揭露座標的平均 KL(Lemma 2.8, Eq. 1)。
證明上界時分析特定演算法;證明下界時排除所有用該 oracle 的演算法。

**Step 1 — DTC 與 product channel 的恆等式(§2.4, Lemma 2.10)。**
TC(X) = Σ H(Xᵢ) − H(X);DTC(X) = H(X) − Σᵢ H(Xᵢ|X₋ᵢ)。若 B 由 A 經獨立逐座標 channel 得到,則
DTC(A) − DTC(B) = DTC(A|B) + Σᵢ I(Bᵢ; A₋ᵢ | B₋ᵢ),因此 Σᵢ I(Bᵢ; A₋ᵢ | B₋ᵢ) ≤ DTC(A) − DTC(B)(Eq. 2)。推論:DTC(qₜ) 對 t 單調不增。
假設:只需要 channel 是 product。

**Step 2 — 單步誤差 = 條件 TC(§3.3 Lemma 3.8;§4.4 Lemma 4.9)。**
Sampler 每步用「product reverse kernel」(各座標獨立地從 posterior marginal 抽)。KL chain rule + data processing 給出
KL(q‖q̂) ≤ 初始化誤差 + 末端 rounding 誤差 + Σⱼ [ TC(X_{tⱼ} | X_{tⱼ₊₁}) + score 誤差項 ]。
也就是:平行更新所丟掉的東西恰好是條件相關性 TC(Xₛ|Xₜ)。

**Step 3 — Reverse DPI 把條件 TC 換成 DTC 落差(§3.1;§4.1–4.2)。**
- Uniform(Lemma 3.2):TC(Xₛ|Xₜ) ≤ cₛ(h)·(DTC(qₛ) − DTC(qₜ)),cₛ(h) = (e^{2h}−1)/(1−e⁻ˢ),h = t−s。關鍵是 Prop 3.3(Gibbs marginal 每個符號至少有質量 βₛ)+ Lemma 3.4(單 token reverse DPI)。
- Gaussian(Lemma 4.2,變數 u = e^{2t}−1,Zᵤ = X₀ + Bᵤ):TC(Zᵤ|Z_v) ≤ c_u(v)·(DTC(πᵤ) − DTC(π_v)),c_u(v) = (v/u)e^{3/u−3/v} − 1。這裡沒有 Prop 3.3 的類比,reverse DPI(Lemma 4.3)改用 de Bruijn 型恆等式 + Grönwall。

**Step 4 — 選步長使係數為常數 a,望遠鏡求和(Thm 3.1 / Thm 4.1 的證明)。**
Σⱼ c·(DTC(q_{tⱼ}) − DTC(q_{tⱼ₊₁})) ≤ a·DTC(q)。取 a ≍ ε/DTC 得誤差 ε,步數 ≈ (log 範圍)/log(1+a) = O((1 + DTC/ε)·log(edS/ε))。
結論:Thm 3.1:KL(q‖q̂) ≤ ε + O(ε_unif·log(dS/ε));Thm 4.1:KL(q_pre‖q̂) ≤ ε + O(ε_gauss²·log(dS/ε))。兩者都需要已知一個上界 DTC̄ ≥ DTC(q)。這讓 uniform/Gaussian 追平 masked 已知的結果 [CCL26, LZ25]。

**Step 5 — 困難分佈族(§5, Def 5.1)。**
q^C = Unif(C),C 是 {±1}^d 上均勻隨機的 M 元子集,M = ⌈e^{κd}⌉,κ ∈ [κ₋, κ₊] ⊂ (0, log 2)。Sampler 知道區間 [κ₋, κ₊],不知道 κ,也不知道 C。臨界時間由 Eq. (22) 定義:d·I_unif(t*_unif) = log M,d·I_gauss(t*_gauss) = log M,其中 I_unif(t) = log 2 − h₂(βₜ),I_gauss(t) = I(V; e⁻ᵗV + σₜG)。上界演算法查詢的時間都落在 I₀ = [τ₋/2, τ₊+1](Eq. 23)。

**Step 6 — Uniform/Gaussian 的 critical window 寬 w = A√(Λ/d)(Prop 6.4 / Prop 7.4)。**
Λ = log(Cd/(εδ))(Eq. 24, 54)。t ≥ t* + w:E_C TV(qₜ^C, 參考分佈) ≤ e^{−KΛ}(uniform 的參考分佈是 Unif(X);Gaussian 是 νₜ = law(e⁻ᵗU + σₜG),Eq. 56)。t ≤ t* − w:Bayes 最佳解碼錯誤率 E_C err ≤ e^{−KΛ}。

**Step 7 — 找窗 + 穿窗 + 一步解碼 ⇒ O(√(dΛ)/ε²)(Thm 6.1 / Thm 7.1)。**
(a) 找窗(Alg 1 / Alg 3):在格距 w/20、由高 noise 往低掃的時間格點上,每點用 R = 2⌈CΛ⌉+1 個「不依賴 C 的參考樣本」各查一次 oracle,算統計量 T̂ₜ 的中位數,首次低於 −(a+b)dw/2 就停。O(Λ/w) 次 query,輸出 t̂ ∈ [t*−3w, t*+2w](Prop 6.10 / 7.11)。
(b) 穿窗:從 t₊ = t̂+4w 以參考分佈起始,等距走到 t₋ = t̂−4w。Uniform 用 clipped score 的 product kernel,N = ⌈Cdw/ε²⌉(Prop 6.12 重用 Lemma 3.2 的望遠鏡,DTC ≤ d log 2);Gaussian 用 exponential integrator(Eq. 86)+ Girsanov,N = ⌈C(d∆²+d∆)/ε²⌉,∆ = 8w,誤差記在 posterior variance V(t) 的總增量 ≤ d 上(Lemma 7.12–7.14)。
(c) 窗下一步解碼:取估計 posterior mean 的符號(Eq. 51;Lemma 6.13 / 7.15)。
總計 O(Λ/w) + O(dw/ε²) + 1 = O(√(dΛ)/ε²);要求 sup_{t∈I₀} score 誤差 ≤ e^{−BΛ} = (εδ/(Cd))^B(Eq. 25, 55),保證是對 ≥ 1−δ 比例的 C 有 TV(q̂^C, q^C) ≤ ε。

**Step 8 — Masked 的 Ω(d/log d) 下界(§8, Thm 8.1)與 uniform 的 Ω(√(d/log d)) 下界(App. A, Thm A.1)。**
Prior:k ∼ Unif(K_d^M)(κ₋ ≤ k log 2/d ≤ κ₊ 的整數,共 Θ(d) 個),|C| = 2^k。建構一個「合法」的近似 oracle O_{k,C}(每個 level 誤差 ≤ η,Eq. 97),它在 m ≤ k−ℓ 回傳 uniform marginal、在不一致的 assignment 上也回傳 uniform,只有 |m−k| ≤ ℓ 的查詢才洩漏資訊,ℓ = ⌈log₂(Cd/(ηρ²))⌉(Eq. 96)。Coupling:Q 次查詢與「永遠回 uniform 的 O₀」出現差異的機率 ≤ (2ℓ+1)Q/|K_d^M| + 2^{1−ℓ}/η + Q·2^{−ℓ}(Eq. 100)。Q ≤ cρ²d/ℓ 時,以 ≥ 1−ρ 的機率 TV(q̂, q^C) ≥ 1−ρ。
Uniform 的版本同構:t*_unif(k) 的候選值取成彼此相距 ≥ 10w 的 Θ(1/w) = Θ(√(d/Λ)) 個(Eq. 108–109),oracle 只在 |t − t*| < 4w 說真話(Def A.4),Q ≤ cρ²√(d/Λ)(Eq. 111)。Gaussian 的 Ω̃(√d) 下界來自 [XP26],本文沒有重證。
**分離的機制(§1.2, Prop 8.6):** masked 的有資訊區間佔全 noise 範圍的 O(log d / d),uniform/Gaussian 佔 O(√(log d / d))。窗越窄,藏窗的位置越多,盲找越貴。

---

## 2. 必須講對的推導(5 個)

### D1. Lemma 2.10(DTC 落差恆等式)
條件:B 由 A 經 product channel 得到。
- H(A) − H(B) − H(A|B) = −H(B|A) = −Σᵢ H(Bᵢ|Aᵢ)(product channel)。
- H(Aᵢ|A₋ᵢ, B) = H(Aᵢ|A₋ᵢ, Bᵢ)。
- H(Aᵢ|A₋ᵢ) − H(Aᵢ|A₋ᵢ,Bᵢ) + H(Bᵢ|Aᵢ) = H(Bᵢ|A₋ᵢ)(因為 H(Bᵢ|Aᵢ) = H(Bᵢ|Aᵢ,A₋ᵢ))。
- 代入 Eq. (3):DTC(A) − DTC(B) − DTC(A|B) = Σᵢ [H(Bᵢ|B₋ᵢ) − H(Bᵢ|A₋ᵢ)],再用 H(Bᵢ|A₋ᵢ) = H(Bᵢ|A₋ᵢ,B₋ᵢ) 得 = Σᵢ I(Bᵢ; A₋ᵢ | B₋ᵢ)。
DTC(A|B) ≥ 0 給出 Eq. (2)。

### D2. Uniform reverse DPI → Lemma 3.2 → 步長排程(§3.1, §3.3)
**(a) Lemma 3.4。** 條件:p, p′ 在每個符號上 ≥ c/S。用恆等式 x log(x/y) − x + y = (x−y)²∫₀¹ (1−θ)/(y+θ(x−y)) dθ:
KL(p‖p′) = Σₐ (pₐ−p′ₐ)² ∫₀¹ (1−θ)/(p′ₐ+θ(pₐ−p′ₐ)) dθ(Eq. 4)。
加噪後差的平方乘上 e^{−2t},分母變成 e⁻ᵗ(·) + (1−e⁻ᵗ)/S ≤ (e⁻ᵗ + (1−e⁻ᵗ)/c)(·)。所以
KL(pKᵁₜ‖p′Kᵁₜ) ≥ KL(p‖p′)·e^{−2t}c/(1−e⁻ᵗ+e⁻ᵗc)。
**(b) Corollary 3.5。** 以 (Xₜ)₋ᵢ 為條件,有 Markov chain (Xₛ)_{i+1:d} − (Xₛ)ᵢ − (Xₜ)ᵢ。取 c = 1−e⁻ˢ(Prop 3.3:P[(Xₛ)ᵢ = a | (Xₜ)₋ᵢ] ≥ βₛ),noise 量取 h:
I((Xₛ)_{i+1:d}; (Xₜ)ᵢ | (Xₜ)₋ᵢ) ≥ C · I((Xₛ)_{i+1:d}; (Xₛ)ᵢ | (Xₜ)₋ᵢ),C = e^{−2h}(1−e⁻ˢ)/(1−e⁻ʰ+e⁻ʰ(1−e⁻ˢ))(Eq. 8)。
Chain rule(Eq. 9):I(…; (Xₛ)ᵢ | (Xₜ)₋ᵢ) = I(…; (Xₛ)ᵢ | Xₜ) + I(…; (Xₜ)ᵢ | (Xₜ)₋ᵢ)。整理:
I((Xₛ)_{i+1:d}; (Xₛ)ᵢ | Xₜ) ≤ (1/C − 1)·I((Xₛ)_{i+1:d}; (Xₜ)ᵢ | (Xₜ)₋ᵢ),
1/C − 1 = (1−e⁻ʰ)(1+e⁻ʰ(1−e⁻ˢ))/(e^{−2h}(1−e⁻ˢ)) ≤ (e^{2h}−1)/(1−e⁻ˢ)(Eq. 10)。
**(c) Lemma 3.2。** TC(Xₛ|Xₜ) = Σ_{i=1}^{d−1} I((Xₛ)ᵢ; (Xₛ)_{i+1:d} | Xₜ);右邊放大 (Xₛ)_{i+1:d} → (Xₛ)₋ᵢ 後用 Eq. (2)(A = Xₛ, B = Xₜ)。
**(d) 排程(Thm 3.1 證明)。** L_U = log(16edS/ε),δ_U = ε/(16dL_U),T_U = log(4d log(S)/ε),a_U = ε/(12 max{DTC̄, ε}) ≤ 1/12。令 uⱼ = e^{tⱼ} − 1,u₀ = e^{δ_U} − 1,uⱼ₊₁ = min((1+a_U)uⱼ, U),U = e^{T_U} − 1。則
e^{tⱼ₊₁−tⱼ} = (1+uⱼ₊₁)/(1+uⱼ) ≤ 1 + a_U(1−e^{−tⱼ}) =: 1 + xⱼ,
c_{tⱼ}(tⱼ₊₁−tⱼ) ≤ ((1+xⱼ)²−1)/(1−e^{−tⱼ}) = (2+xⱼ)a_U ≤ 3a_U。
誤差帳:離散化 ≤ 3a_U·DTC(q) ≤ ε/4;初始化 KL(q_{T_U}‖Unif) ≤ e^{−T_U}d log S ≤ ε/4(Lemma 3.10);末端 H(q_{δ_U}) − H(q) ≤ dδ_U log(eS/δ_U) ≤ ε/8(Lemma 3.11,要求 t ≤ 1;末端步是對最後 iterate 再加一次 forward kernel,Lemma 3.9 用到 kernel 對稱);score 項 3a_U N_U ε_unif。步數 N_U ≤ 1 + log(U/u₀)/log(1+a_U),log(U/u₀) ≲ L_U,1/log(1+a_U) ≤ 2/a_U。
直覺:t 小時 tⱼ₊₁ ≈ (1+a_U)tⱼ(幾何);t 大時 tⱼ₊₁ − tⱼ ≈ log(1+a_U)(等距)(Figure 6)。
**(e) Score 誤差處理(Lemma 3.7)。** 估計 score 必須先做 Bregman projection 到凸集 S_{t,b} = {v > 0 : v_b = 1, vₐ/Σv ≥ βₜ}(Prop 3.6),否則 p^gibbs/p^post 不一定是合法機率、reverse DPI 的假設也不成立。結果:E KL(R‖R̂) ≤ cₛ(h)·ε_unif(t)。作者在 AI usage 聲明中承認早期版本漏了這一步。

### D3. Gaussian reverse DPI(Lemma 4.3)與 F_G 排程(§4.2, §4.4)
設 K(u) = KL(f_{α,u}‖f_{β,u}),f_{α,u} = Σₐ αₐ N(eₐ, u·Id_S)。Lemma 4.6:
−K′(u) = ½FI ≥ 0;∂ᵤFI ≥ −(2/u + 3/u²)FI,即 K″(u) ≤ −(2/u + 3/u²)K′(u)。
(3 來自 Prop B.1:‖C(p) − C(p′)‖²_F ≤ 3‖p−p′‖²,C(p) = diag(p) − p^{⊗2};這一步用到 one-hot 結構。)
令 ξ = v/u > 1,對 −K′ 用 Grönwall(s ≥ u):
−K′(ξs) ≥ −K′(s)·exp(−∫ₛ^{ξs}(2/r + 3/r²)dr) = −K′(s)·ξ⁻²e^{3/(ξs)−3/s} ≥ −K′(s)·ξ⁻²e^{3/v−3/u}。
K(∞) = 0,所以 K(u) = −∫ᵤ^∞K′(s)ds ≤ e^{3/u−3/v}∫ᵤ^∞ ξ²(−K′(ξs))ds = (v/u)e^{3/u−3/v}K(v)。
即 KL(f_{α,v}‖f_{β,v}) ≥ (u/v)e^{3/v−3/u}·KL(f_{α,u}‖f_{β,u})。套 Cor 3.5 的同一段論證(C = (u/v)e^{3/v−3/u})得 c_u(v) = (v/u)e^{3/u−3/v} − 1。
排程:F_G(u) = log u − 3/u,因為 F_G(v) − F_G(u) = log(1 + c_u(v))。每步令 F_G 增加 log(1+a_G),a_G = ε/(4 max{DTC̄, ε}) ≤ 1/4,就有 c ≤ a_G。u₀ = 1/(8L_G),U = max{1, 4d/ε},L_G = log(64edS/ε)。步數 N_G ≤ 1 + (F_G(U) − F_G(u₀))/log(1+a_G),F_G(U) − F_G(u₀) ≤ log(8L_G U) + 24L_G ≲ L_G(Eq. 21)。
誤差帳:離散化 ≤ ε/4;初始化 KL(π_U‖N(0,U·Id)) ≤ d/(2U) ≤ ε/8(Lemma 4.10);rounding ≤ dSe^{−L_G} ≤ ε/4(Lemma 4.11,Fano + argmax decoder,p*(u) = ((S−1)/2)erfc(1/(2√u)));score 項 w_u(v)ε_gauss²,w_u(v) = (1+v)(e^{2(v−u)/(uv)} − 1) ≤ 8 log(1+c_u(v))(Lemma 4.8, Eq. 19,需 c_u(v) ≤ 1)。
注意:u 小時 3/u 項主導,步長 ≈ a_G u²/3,不是幾何(Figure 7 左邊的標註);u 大時才是 uⱼ₊₁ ≈ (1+a_G)uⱼ。

### D4. Uniform critical window(Lemma 6.2, 6.3 → Prop 6.4)
Binary overlap 形式(Lemma 2.3):2^d(Kᵁₜ)^{⊗d}(x|z) = (1+e⁻ᵗ)^d·tanh(t/2)^{d_H(x,z)}。有用的改寫:1+e⁻ᵗ = 2(1−βₜ),tanh(t/2) = βₜ/(1−βₜ)。
- 窗上(Lemma 6.2,二階矩 + 截斷半徑 r):E_C TV(qₜ^C, Unif) ≤ (1/(2√M))(1+e⁻ᵗ)^{d/2}tanh(t/2)^{(r+1)/2} + P[Bin(d,βₜ) ≤ r]。
- 窗下(Lemma 6.3,nearest-neighbor decoder + union bound):E_C err ≤ M·P[Bin(d,½) ≤ r] + P[Bin(d,βₜ) > r]。
取 r = ⌊d(βₜ + β_{t*})/2⌋,∆ = |βₜ − β_{t*}| ≍ w,並用 log M = d(log 2 − h₂(β_{t*})):
- Hoeffding 項:e^{−d∆²/2}(Eq. 28, 30)。
- 計數項:窗上 log[(1+e⁻ᵗ)^d tanh(t/2)^{r+1}/M] ≤ −d·KL(Ber(β_{t*})‖Ber(βₜ)) − (d∆/2)log((1−βₜ)/βₜ) ≤ −cd∆(Eq. 29);窗下 M·P[Bin(d,½) ≤ r] ≤ exp(−d[h₂(β_{t*}) − h₂((βₜ+β_{t*})/2)]) ≤ e^{−cd∆}(Eq. 31)。
- 合計 e^{−cd∆} + e^{−d∆²/2} ≤ e^{−KΛ},需要 d∆² ≳ Λ,即 w = A√(Λ/d)。
條件:d ≥ d_{K,A},log d ≤ Λ ≤ c_{K,A}d,t ∈ I₀(βₜ 遠離 0 與 ½)。期望是對 C 取的,不是對每個 C 成立。
〔顧問推導〕教學用的一句話:noisy 樣本離自己的 codeword 的 Hamming 距離是 Bin(d, βₜ);半徑 dβ 的球裡其他 codeword 的期望個數是 M·2^{−d}·e^{d·h₂(β)} = exp(d[κ − I_unif(t)]),在 I_unif(t) = κ 時由 ≫1 變 ≪1。窗寬 1/√d 來自 Bin(d,βₜ) 的 √d 漲落(Hoeffding 項 e^{−d∆²/2}),不是來自計數項(計數項是 e^{−cd∆},只需 ∆ ≳ Λ/d)。論文沒有用這句話陳述,但兩個項的來源在 Prop 6.4 的證明裡是明確的。
Gaussian 版本(Lemma 7.2, 7.3, Prop 7.4)結構相同:information density ıₜ(Y;Z) = λₜ⟨Z,Y⟩ − Σ log cosh(λₜZᵢ),λₜ = e⁻ᵗ/σₜ,是 d 個獨立項之和、均值 d·I_gauss(t),Gaussian concentration 給 P[|ıₜ − d·I_gauss| ≥ v] ≤ 2e^{−cv²/d}(Eq. 59);Eq. (61)–(62) 以 log M ∓ γ 為門檻,γ = c_I dw/2。√d 同樣來自這個和的漲落。

### D5. Masked 的窗與下界(Lemma 8.2, Prop 8.5, 8.6)
設 |C| = 2^k,固定 I(|I| = m),N(x_I) = 與 x_I 相符的 codeword 數,E_C N = 2^{k−m}。
- m ≤ k−ℓ(Eq. 104):N 是 hypergeometric,變異數 ≤ 2^{k−m},所以 E_C TV(law(X_{0,I}), Unif) ≤ ½·2^m·2^{−k}·√(2^{k−m}) = ½·2^{−(k−m)/2} ≤ 2^{−ℓ/2}。
- m ≥ k+ℓ(Eq. 105):唯一匹配 decoder,union bound,E_C err^M ≤ 2^{k−m} ≤ 2^{−ℓ}。
- O₀ 的誤差(Lemma 8.2):在 N > 0 的條件下,下一個座標為 +1 的比例 P 滿足 E[(P−½)²|N] ≤ 1/(4N),且 KL(Ber(p)‖Ber(½)) ≤ 4(p−½)²,所以 E_C[N·KL] ≤ 1;對 2^m 個 assignment 加總再除以 2^k:E_C ε_mask(O₀; q^C, m) ≤ 2^{m−k}。
- "good" C:O₀ 在所有 m ≤ k−ℓ 的誤差 ≤ η;P[not good] ≤ (1/η)Σ_{m≤k−ℓ}2^{m−k} ≤ 2^{1−ℓ}/η(Eq. 99)。
- Coupling(Prop 8.5):對 O₀ 的查詢序列只由內部隨機性決定,與 (k, C) 無關。三個壞事件:(i) 某次 |mⱼ − k| ≤ ℓ:機率 ≤ (2ℓ+1)Q/|K_d^M|;(ii) C 不 good:≤ 2^{1−ℓ}/η;(iii) 某次 m > k+ℓ 的查詢恰好與 C 一致:每次 ≤ 2^{k−m} ≤ 2^{−ℓ},共 ≤ Q·2^{−ℓ}。
- 收尾:TV(q̂, q^C) ≥ 1 − ν(C) − p_diff(Eq. 103),E_C ν(C) = 2^{k−d} ≤ e^{−(log 2−κ₊)d}。|K_d^M| ≥ c₁d,取 Q ≤ cρ²d/ℓ,三項與 ν(C) 合計 ≤ ρ²,Markov 得結論。ℓ = O(log d)(即使 η 是常數,ℓ 也含 log₂ d)⇒ Ω(d/log d)。
條件:這是對「存在一個誤差 ≤ η 的 oracle」的敘述,並且 k 對演算法未知。關鍵槓桿是 Lemma 2.8:誤差只約束 consistent assignment 上的回答,不一致的 assignment 上 oracle 可以任意回答。

---

## 3. 常見誤讀與論文的不一致/未明說之處

**誤讀(看 abstract/圖會以為對,但論文沒有證)**
1. 「Masked diffusion 一定要 Ω̃(d) 步」— 不是。Thm 8.1 是:存在一個每 level 誤差 ≤ η 的對抗性近似 oracle,使任何演算法失敗。論文沒有證明對 exact oracle、或對實際訓練出的網路也成立。對照:上界(Thm 6.1/7.1)對任何誤差 ≤ e^{−BΛ} 的 oracle 成立。所以分離的精確形式是「worst-case 近似 oracle 下的 query complexity」。
2. 「分離來自 masked 無法修改已 commit 的 token」— 論文明說不是(§1.2, §9)。來自 critical window 的寬度。下界論證完全沒用到 commitment。
3. 「窗窄 = 轉變快 = 應該更容易」— 方向相反。成本是「找窗」:候選位置數 ÷ 窗寬。Masked:Θ(d) 個 level ÷ (2ℓ+1) = Θ(d/log d)。Uniform/Gaussian:O(1) 的時間範圍 ÷ w = Θ(√(d/Λ))。
4. 「κ 已知也有這個分離」— 論文沒有主張。兩個下界都靠對 k 的 prior(演算法不知道 k)。若 κ 已知,下界的論證不成立;論文沒有討論這個情形。〔顧問推導〕此時上界演算法可省掉搜尋,但穿窗仍用 O(dw/ε²) 步。
5. 「Uniform/Gaussian 普遍比 masked 好」— 只對這一個隨機 codebook 分佈族。Result 1 是三者打平(都能做到 Õ(DTC/ε))。§9 明確說不應解讀為推崇某一 paradigm。
6. 「Θ̃(√d) 的 tight 是本文全部證的」— Uniform 的下界在 App. A;Gaussian 的下界引用 [XP26]。
7. 「DTC 界和 √d 界矛盾」— 不矛盾。此分佈族 DTC = Θ(d)(§1.2 直接陳述,沒有給證明),Thm 3.1/4.1 只給 Õ(d/ε);Thm 1.3 是利用分佈結構的專用演算法,超過 DTC 界。〔顧問推導〕DTC(q^C) = H − Σᵢ H(Xᵢ|X₋ᵢ),H = log M ≈ κd,而 H(Xᵢ|X₋ᵢ) ≠ 0 需要某 codeword 翻一位後仍在 C 中,機率 ≈ M/2^d 指數小,故 DTC ≈ κd。
8. Result 1 的 sampler 是「每步所有座標獨立重抽」的通用演算法;Result 2 的 sampler 是三階段專用演算法(找窗/穿窗/解碼)。兩者不是同一個演算法。
9. 三個「u」不同:Thm 3.1 證明裡 u = eᵗ − 1;§4 裡 u = e^{2t} − 1;§7/App. B.5 裡 u = λₜ = e⁻ᵗ/σₜ(訊號強度)。
10. 誤差度量不同,不能直接比:Thm 3.1/4.1 是 KL(q‖q̂),步數 ∝ 1/ε;Thm 6.1/7.1 是 TV,步數 ∝ 1/ε²,且是「對 ≥ 1−δ 比例的 C」;下界是 TV ≥ 1−ρ。ε_unif 對 d 個座標加總;ε_mask 是每個未揭露座標的平均、且取 max over m;ε_gauss 是 L2 norm(定理裡以 ε_gauss² 出現)。
11. Masked 的「noise level」是 m(已揭露個數),不是 t。Masked oracle 的輸入是 (I, x_I),對抗 oracle 依 m 分段。若用 t 參數化,m ∼ Bin(d, e⁻ᵗ) 本身就有 √d 漲落;論文在 §2.3 說明選 m 是因為它可由 Xₜ 觀測到、且是 sampler 實際控制的量。
12. Figure 4(d) 的 d⁻¹ 與 d^{−1/2} 是模擬中「recovery probability 由 0.2 到 0.8」的寬度(以每座標資訊量 nats 為橫軸)。理論的窗寬帶 log 因子:masked O(log d/d),uniform/Gaussian O(√(log d/d))。log 因子來自高機率要求,在 0.2→0.8 的定義下看不到。
13. Recovery probability(Figure 4)是 P[X₀ = y*|Xₜ] 的期望(posterior 質量),不是 sampler 的成功率,也不是 Step 7 的統計量 Tₜ。

**不一致 / 未明說**
- 所有常數 A, B, C, c₀, d₀, a, b, K, τ₋, τ₊, h₀ 都只說「depending only on [κ₋, κ₊]」,沒有數值。Alg 1–4 無法照字面實作:門檻 −(a+b)dw/2、R = 2⌈CΛ⌉+1、N 的常數都未指定。任何 demo 必須自選常數並標明是自選。
- 定理是漸近的(d ≥ d₀,log d ≤ Λ ≤ c₀d)。小 d 的 toy 不在定理適用範圍內。
- Thm 3.1/4.1 的 sampler 需要已知上界 DTC̄ ≥ DTC(q);informal 版(Thm 1.1/1.2)沒提。
- Thm 1.3 (informal) 寫「2^{Θ(d)} 個點」「score error 1/poly(d)」;正式版是 M = ⌈e^{κd}⌉,κ ∈ [κ₋,κ₊],誤差 e^{−BΛ},B 是未指定的大常數。下界另外限制 κ 在格點 {(k/d)log 2} 上,M = 2^k。
- Figure 4 caption 說「By Eq. (22), every process has its critical level at I = κ」,但 Eq. (22) 只定義 uniform 與 Gaussian;masked 的 I = (m/d)log 2 = κ ⇔ m = k 來自 §8,不是 Eq. (22)。
- Figure 4:κ = 0.2;(a)–(c) 圖例 d = 64, 256, 1024, 4096;(d) 橫軸 64…4096。樣本數、亂數種子、Gaussian saddlepoint/分箱的細節都沒給(App. C 說整個模擬由 coding agent 實作)。
- Alg 1 掃的格點覆蓋 [τ₋−3w, τ₊+3w],而誤差假設只在 I₀ = [τ₋/2, τ₊+1] 上;靠 w ≤ τ₋/16(uniform)/ τ₋/20(Gaussian)保證包含。沒觸發時回傳 τ₋。
- Prop 6.9/7.10 的低 noise 保證只在 [t*−3w, t*−2w];更低處 T̂ 的行為沒有保證,也不需要(由上往下掃)。
- 穿窗步數兩邊形式不同:uniform N = ⌈Cdw/ε²⌉(Alg 2);Gaussian N = ⌈C(d∆²+d∆)/ε²⌉(Alg 4)。代入後都是 O(√(dΛ)/ε²)(Gaussian 多一個 Λ/ε² 低階項)。
- 最終 TV 帳:uniform 7ε/16(失敗機率 over C ≤ 3δ/4);Gaussian 9ε/32(δ/2 + δ/4)。
- §3 的 Bregman projection 與 §6 的 clipping 是兩種不同的 score 修正;§4 用 Euclidean projection 到 ∆(Σ)^d(Eq. 17),§7 用 clip 到 [−1,1](Eq. 79)。
- PDF 轉文字後 Eq. (96) 的 ℓ 與 Alg 2 的 N 的 ceiling 符號遺失;依上下文應為 ⌈·⌉。

---

## 4. 可否在瀏覽器即時示範

**可以(閉式或小型計算)**
1. **三個資訊量曲線與臨界點。** I_unif(t) = log 2 − h₂((1−e⁻ᵗ)/2) 是閉式;I_gauss(t) = λ² − E_W log cosh(λ² + λW)(Lemma 7.2 證明內,λ = e⁻ᵗ/σₜ)要一維數值積分;masked I = (m/d)log 2。滑桿 κ,解 I = κ。〔顧問推導〕κ = 0.2 時:β* ≈ 0.1948,t*_unif ≈ 0.4937(e^{−t*} ≈ 0.6103);t*_gauss ≈ 0.5521(λ ≈ 0.7042);masked m* = κd/log 2 ≈ 0.2885d(d = 64 時 ≈ 18.5)。
2. **Masked 的 recovery 曲線(Figure 4a)是閉式。** App. C:recovery = E[1/N],N − 1 ∼ Poi(M·2^{−m})。〔顧問推導〕E[1/(1+Poi(λ))] = (1−e^{−λ})/λ,λ = exp(d(κ − I))。0.8 ↔ λ ≈ 0.4642,0.2 ↔ λ ≈ 4.965,寬度 = ln(4.965/0.4642)/d ≈ 2.370/d nats(d = 64:0.0370;d = 4096:5.79×10⁻⁴),與 Figure 4(d) 的 d⁻¹ 線一致。以 m 計則寬度 ≈ 2.370/log 2 ≈ 3.42 個座標,與 d 無關(App. C 有說 m 的寬度與 d 無關)。
3. **Uniform 的 recovery 曲線(Figure 4b)Monte Carlo。** App. C 的公式:P[X₀ = y*|Xₜ] = 1/(1 + Σⱼ Nⱼ·tanh(t/2)^{j − d_H}),d_H ∼ Bin(d, βₜ),Nⱼ ∼ Poi(M·C(d,j)·2^{−d}) 獨立。每個樣本 d 個 Poisson + 1 個 binomial。瀏覽器可做,但必須在 log 空間算(M·C(d,j)·2^{−d} 與 tanh 的冪次都會上溢/下溢),大均值 Poisson 用常態近似。能重現 d^{−1/2} 的寬度。
4. **顯式小 codebook 的精確模擬。** 〔顧問推導〕不需要枚舉 2^d,只要存 M = e^{κd} 個隨機字串:κ = 0.2 時 d = 32 → M ≈ 602;d = 48 → M ≈ 14,765;d = 64 → M ≈ 362,217(瀏覽器的上限附近)。可以精確算 posterior、三種腐蝕下的 recovery probability、以及 uniform 的統計量 Tₜ(x) = ⟨x, mₜ(x)⟩ − de⁻ᵗ。小 d 時轉變是平滑的,寬度的冪律需要 d 跨幾個數量級才看得清楚,所以冪律圖應該用第 2、3 項。
5. **找窗統計量的形狀。** 對與 C 無關的均勻 x:窗上 Tₜ ≥ −a₀dw,窗下 Tₜ ≤ de^{−t*} − de⁻ᵗ + O(Λ)(Prop 6.5 的證明)。〔顧問推導〕所以 Tₜ(x)/d 大致沿 min(0, e^{−t*} − e⁻ᵗ) 走,在 t* 有一個折點;這是啟發式的形狀,不是論文的定理。可用第 4 項的顯式 codebook 畫出來。
6. **DTC 望遠鏡與排程。** 排程是閉式:uniform uⱼ₊₁ = (1+a_U)uⱼ、tⱼ = log(1+uⱼ);Gaussian 解 F_G(uⱼ₊₁) − F_G(uⱼ) = log(1+a_G)。可以畫格點、算 N_U、N_G 隨 DTC̄/ε 的變化。若要顯示 DTC(qₜ) 曲線本身,需對小例子(例如 d ≤ 10 的二元分佈,枚舉 2^d)精確計算 entropy;可行。也可數值驗證 Lemma 3.2 的不等式。
7. **下界的計數。** Eq. (100) 三項對 (d, Q, ℓ, η) 的閉式計算器;以及「候選位置數 ÷ 窗寬」的比較:masked |K_d^M|/(2ℓ+1) 對 uniform |K_d^U| ≈ c√(d/Λ)。可做一個「盲找窗」的遊戲:隱藏 k,每次查詢只有落在窗內才有回應。這忠實於 Prop 8.5/A.6 的 coupling 論證。
8. **Lemma 8.2 / Prop 8.6 的數值驗證。** 小 d 顯式 codebook 下,計算 O₀ 在各 level 的誤差,對照 2^{m−k};計算 TV(law(X_{0,I}), Unif) 對照 ½·2^{−(k−m)/2}。

**不行或不建議**
- **Figure 4(c) Gaussian 的大 d 曲線**:App. C 用 saddlepoint 近似 Σ bᵢaᵢ 的密度再分箱做 Poisson process,細節未給,無法保證與圖一致。替代:第 4 項的顯式 codebook(d ≤ ~48)直接算 Z = Σ_y K^G(Xₜ|y)/K^G(Xₜ|y*)。
- **照字面跑 Alg 1–4 並宣稱驗證定理**:常數未指定且定理要求 d ≥ d₀。只能做「自選常數的示意版」。
- **實際 query 數對 d 的 √d vs d 縮放曲線**:需要大 d 的 score oracle(M 指數大),只能用 Poissonization 之類的代理;論文自己也沒有做這個實驗。
- **Thm 3.1/4.1 在真實語言分佈上的步數**:需要 DTC 與 score 網路;論文沒有任何這類實驗。
- **對抗 oracle 對真實 masked 模型的影響**:論文沒有主張,無法示範。

---

## 5. 先備知識(教學順序)

1. 離散機率基礎:joint / marginal / conditional、Bayes 規則、posterior。
2. Entropy H、條件 entropy、KL divergence、mutual information 與 chain rule;TV distance 與 Pinsker(KL → TV);data processing inequality(加噪不會增加 KL/MI)。
3. Total correlation 與 dual total correlation 的定義與直覺(TC:與獨立乘積的距離;DTC:總 entropy 中不能由「給定其餘座標後的局部 entropy」解釋的部分)。兩個小例子:完全複製的 d 個 bit(TC = (d−1)log 2,DTC = log 2)與 parity。〔顧問推導〕例子的數值。
4. Diffusion 的框架:forward 腐蝕過程、reverse 去噪;一次 forward pass 給出所有座標的 posterior marginal;「平行更新 = 用 marginal 的乘積取代 joint」所以丟掉條件相關性。
5. 三種腐蝕過程的單座標 kernel(uniform / masked / Gaussian-OU 與 one-hot embedding);Tweedie 公式;score 作為 likelihood ratio(離散)或 ∇log 密度(連續)。
6. Query complexity 與 oracle 模型:上界 = 一個演算法;下界 = 對所有演算法;「近似 oracle」與對抗性 oracle 的意思。
7. 二項分佈的集中(Hoeffding / Chernoff)、binary entropy h₂ 與 C(d,r) ≈ e^{d·h₂(r/d)}、Hamming ball 體積、union bound、Markov/Chebyshev。
8. Random code 的直覺(Shannon random coding / channel capacity:rate κ 對 mutual information I;all-or-nothing 現象)。這是理解「I = κ 是臨界點」的最短路徑。
9. 望遠鏡求和與幾何步長排程。
10. Coupling 論證與「與 dummy oracle 無法區分」的下界技巧;Poisson 近似(只在 Figure 4 用到)。
11. 只在 Gaussian 部分需要、可以當黑盒子:Fisher information 與 de Bruijn 恆等式、Grönwall、Girsanov、Itô、Fano。對初學者建議只陳述結論(Lemma 4.3 的形式與 F_G 排程),不展開。
