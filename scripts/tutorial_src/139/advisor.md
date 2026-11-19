# 顧問筆記：Value Forcing（arXiv 2611.02160v1）

對象：教學設計者。不是給讀者看的。
標記規則：沒有標記的內容都能在論文指定位置找到；〔顧問推導〕＝論文沒寫、我自己推的（已自行驗算，但論文未背書）；〔顧問判讀〕＝我對論文未說明處的推測。

符號：t 是 RL 時間步，τ∈[0,1] 是 flow 的 noise level（τ=0 純雜訊、τ=1 乾淨回報）。注意：IQL 原文用 τ 當 expectile，本文改用 µ（A.2 有註明 "Vτ and Qτ in Kostrikov et al."）。教學時不要混用。

---

## 1. 論證骨架（9 步）

**步驟 1 — 問題設定（Sec. 1, 3）**
假設：無限期 MDP，reward 為確定性且有界 r∈[rmin,rmax]，γ∈[0,1)；離線資料 D={(s,a,r,s′)}。
主張（動機，非證明）：每筆 transition 只給一個 r，監督訊號稀少；分佈式 RL 用整個回報分佈當 TD target 訊號較多。
分佈式 Bellman 方程 Eq. 1：Z^π(s,a) =d r(s,a) + γ Z^π(S′,A′)，T^π 在 max-Wasserstein 下是 γ-contraction（引 Bellemare 2017，本文不重證）。

**步驟 2 — Flow matching 與 Lemma 1（Sec. 3, Eq. 2–4）**
ODE（Eq. 2）：dϕ/dτ = u(ϕ|τ)，ϕ(x0|0)=x0。標準 CFM loss（Eq. 3）：E‖v(x^τ|τ) − (x1−x0)‖²，x^τ=(1−τ)x0+τx1。
Lemma 1（非正式，引 Liu et al. 2022）：若隨機過程 X(τ) 在給定隨機變數集合 𝒳 後是 τ 的確定函數，則 u(x|τ) ≜ E[dX(τ)/dτ | X(τ)=x] 產生該過程的邊際密度路徑 p(x^τ|τ)；且可用 Eq. 4 的回歸 E‖v(X(τ)|τ) − dX(τ)/dτ‖² 擬合 u。
這是全文唯一的「工具定理」，Sec. 4.2 與 A.2 的所有 loss 都是「定義一個過程 → 對 τ 微分 → 套 Lemma 1」。

**步驟 3 — Flow Bellman 方程（Sec. 4.1, Eq. 5）**
定義每個 τ 的 noisy return 隨機變數 Z^π(s,a,τ)，τ=1 時 =d Z^π(s,a)。提出算子 T^π_τ：
Z^π(s,a,τ) =d τ r(s,a) + γ Z^π(S′,A′,τ) + (1−τ)√(1−γ²) N，N∼N(0,1) 且獨立。
論文只做兩端檢查：τ=0 得 Z(s,a,0) =d γZ(S′,A′,0)+√(1−γ²)N，兩邊都是 N(0,1)（γ²+(1−γ²)=1）；τ=1 回到 Eq. 1。
論文沒有說這個式子怎麼「推」出來，也沒有明寫 Z(s,a,τ) 的閉式定義——見 §2 推導 A。

**步驟 4 — 每個 τ 有唯一不動點（Remark 1 / Prop. 1, Eq. 6, App. A.1）**
假設：固定策略 π、固定 τ、p∈[1,∞)。
證明：W_p^sup(T_τ Z1, T_τ Z2) ≤ γ W_p^sup(Z1,Z2)。再引 Banach 不動點定理得唯一不動點。
注意：contraction 係數是 γ，與 τ 無關；這是對「固定 π 的 policy evaluation 算子」的敘述。

**步驟 5 — 拆成 V 與 Q（Sec. 4.2, Eq. 7–9）**
Z_V(s,τ) =d Z_Q(s,A,τ)（A∼π）；Z_Q(s,a,τ) =d τr + γZ_V(S′,τ) + (1−τ)√(1−γ²)N。
用兩條 flow ϕ_V、ϕ_Q 把獨立高斯 N_V、N_Q 傳輸成這兩個分佈（Eq. 8, 9）。等式都只是「分佈相等」，不是逐路徑相等。

**步驟 6 — 由 Lemma 1 得兩個 loss（Eq. 10, 11）**
V：過程 X_V(s,τ)=ϕ_Q(N_Q|s,A,τ)，導數 = u_Q(ϕ_Q(...)|s,A,τ) → Eq. 10。
Q：過程 X_Q(s,a,τ)=τr+γϕ_V(N_V|S′,τ)+(1−τ)√(1−γ²)N，導數 = r+γu_V(ϕ_V(...)|S′,τ) − √(1−γ²)N → Eq. 11。
假設：能取得真實速度 u_V、u_Q（實際上沒有）。

**步驟 7 — 實作 loss（Eq. 12, 13）**
用學到的 v_V、v_Q 互為 target（類比 Fitted Q-Evaluation），交替更新；reward 乘 (1−γ)。
L(v_V)=E[(v_V(z^τ|s,τ) − v_Q(z^τ|s,a,τ))²]，a∼π，z^τ∼p_Q(z^τ|s,a,τ)。
L(v_Q)=E[(v_Q(τ(1−γ)r+γz^τ+(1−τ)√(1−γ²)ε | s,a,τ) − ((1−γ)r+γv_V(z^τ|s′,τ) − √(1−γ²)ε))²]，z^τ∼p_V(z^τ|s′,τ)，ε∼N(0,1)。
這一步沒有任何收斂證明（Conclusion 自己承認）。

**步驟 8 — 兩個演算法（Sec. 5, Eq. 14–16, Alg. 1）**
VF（explicit）：Q(s,a)≜E_{z0}[∫₀¹ v_Q dτ]；策略 = 從 flow BC 策略 π_β 抽 K 個動作取 argmax Q（Eq. 14）。
IVF（implicit）：V loss 改用資料集動作，加權重 w_µ(s,a)=µ·1(Q≥V)+(1−µ)·1(Q<V)（Eq. 15），µ∈[0.5,1)，得 Eq. 16。權重由「純量」Q、V 決定，再乘在速度殘差平方上。
App. A.2 的理論：在「精確最佳化 + 收斂到自洽不動點」的假設下，(1) V 估計是 Q 估計的 expectile；(2) µ→1 時收斂到資料支撐內的最佳值（借用 IQL 的 Theorem 3）。

**步驟 9 — 實驗（Sec. 6, App. B–D）**
(a) 單一狀態 policy evaluation 的 1-Wasserstein（Fig. 2, 6, 7）；(b) 20 個 OGBench + 6 個 OCBench 任務、不同資料量的成功率（Fig. 3, Table 2–5）；(c) 消融（Fig. 4, 5, 8）。
實驗是「相關性證據」：沒有任何一個消融單獨拿掉高斯修正項（見 §3）。

---

## 2. 必須講對的推導

### 推導 A：Eq. 5 從哪裡來、不動點長什麼樣〔顧問推導〕
論文只給 Eq. 5 並檢查兩端。以下是補上的中間步驟。

(i) 由線性內插反推修正項。令 Z(τ)=(1−τ)Z⁰+τZ¹，Z¹=r+γZ¹′（Eq. 1），並要求起點雜訊滿足 Z⁰ =d γZ⁰′+√(1−γ²)N（三者皆 N(0,1)、N 獨立，變異數 γ²+(1−γ²)=1）。代入：
Z(τ) = (1−τ)(γZ⁰′+√(1−γ²)N) + τ(r+γZ¹′)
　　 = τr + γ[(1−τ)Z⁰′+τZ¹′] + (1−τ)√(1−γ²)N
　　 = τr + γZ′(τ) + (1−τ)√(1−γ²)N。即 Eq. 5。

(ii) 不動點閉式。把 Eq. 5 沿軌跡展開（每步用一個新的獨立 N_t）：
Z(s,a,τ) =d τ·Σ_t γ^t r_t + (1−τ)√(1−γ²)·Σ_t γ^t N_t。
Σ_t γ^t N_t 是高斯，變異數 Σγ^{2t}=1/(1−γ²)，所以第二項 =d (1−τ)N̄，N̄∼N(0,1) 且與軌跡獨立。
結論：Z^π(s,a,τ) =d τ·Z^π(s,a) + (1−τ)·N̄。這正是「獨立耦合、線性內插」的標準 flow-matching 邊際路徑。
條件：每步的 N 與 (S′,A′) 及未來全部獨立；起始雜訊為標準高斯；γ<1。
教學用法：這一句話就是全文核心——「VF 要求每個 τ 的邊際分佈，都是『乾淨回報縮 τ 倍 + (1−τ) 倍獨立高斯』，並且這個族在 Bellman backup 下自洽」。

(iii) 變異數檢查。Var[不動點]=τ²Var(Z)+(1−τ)²。右邊（忽略 s′ 混合）：γ²(τ²Var(Z′)+(1−τ)²)+(1−τ)²(1−γ²)=γ²τ²Var(Z′)+(1−τ)²。一致。

(iv) 不加修正會怎樣。若每個 τ 都用乾淨方程 Z_τ=r+γZ′_τ（D.2 對 Value Flows DCFM 的描述：輸入是 r+γz^τ），這就是原本的 T^π，其唯一不動點在「每個 τ」都是 Z^π 本身——與「τ=0 必須是 N(0,1)」矛盾。與正確的內插邊際相比，誤差在所有 τ<1 都存在（平均差 (1−τ)r、雜訊變異數 γ²(1−τ)² 對 (1−τ)²），τ→1 才消失。論文只強調「τ=0 不正確」。

(v) 數量級。γ=0.999 時 √(1−γ²)=√0.001999≈0.0447；γ=0.9999 時 ≈0.0141。單步修正很小，但有效視界約 1/(1−γ) 步，累積後才是整個單位高斯。不要告訴讀者「修正項很大」。

### 推導 B：contraction 證明（App. A.1）
1. 對下一步 (S′,A′) 用分位數耦合：C_i = F⁻¹_{Z_i}(U|S′,A′,τ)，同一個 U∼Unif[0,1]，U 與 (S′,A′) 獨立。P(F⁻¹(U)≤y)=P(U≤F(y))=F(y)，所以 C_i =d Z_i。
2. 一維 Wasserstein 的分位數公式（Eq. 17）：W_p(Z1,Z2)=(∫₀¹|F⁻¹_{Z1}(u)−F⁻¹_{Z2}(u)|^p du)^{1/p}=(E_U|C1−C2|^p)^{1/p}。
3. T_i ≜ τr+γC_i+(1−τ)√(1−γ²)N，兩者共用同一組 (S′,A′,U,N)。T_i =d T_τ Z_i，所以 (T1,T2) 是一個合法耦合。
4. W_p(T_τZ1,T_τZ2) ≤ (E|T1−T2|^p)^{1/p}（W_p 是對所有耦合取 inf）
　 = γ(E_{S′,A′,U}|C1−C2|^p)^{1/p}（τr 與高斯項相消）
　 = γ(E_{S′,A′}[W_p(Z1(S′,A′,τ),Z2(S′,A′,τ))^p])^{1/p}
　 ≤ γ sup_{s′,a′} W_p(...) = γW_p^sup(Z1,Z2)。對 (s,a) 取 sup 得 Eq. 6。
條件與提醒：回報是一維（才有分位數公式）；固定 π 與 τ；關鍵是「相同的 τr 與相同的 N 會相消」，所以修正項不影響 contraction 係數；Banach 定理需要度量空間完備，論文未驗證 (Z, W_p^sup) 的完備性。

### 推導 C：Q-loss（Eq. 11 → 13）
X_Q(s,a,τ)=τr+γϕ_V(N_V|S′,τ)+(1−τ)√(1−γ²)N，𝒳_Q={N_V,N,S′}。
dX_Q/dτ = r + γ·u_V(ϕ_V(N_V|S′,τ)|S′,τ) − √(1−γ²)N（用了 dϕ_V/dτ=u_V）。
套 Lemma 1：回歸的「輸入點」是 X_Q 本身 = τr+γz^τ+(1−τ)√(1−γ²)ε；「目標」是導數 = r+γu_V(z^τ|s′,τ)−√(1−γ²)ε；z^τ=ϕ_V(z⁰|s′,τ)。
最常出錯三處：(1) 輸入點是 **+**(1−τ)√(1−γ²)ε，目標是 **−**√(1−γ²)ε，同一個 ε；(2) 輸入點是 τr（乘 τ），目標是 r（不乘 τ）；(3) u_V 在「未變換的 z^τ、下一狀態 s′」求值，v_Q 在「變換後的點、當前 (s,a)」求值。
實作版 Eq. 13 把 r 換成 (1−γ)r，其餘不變。

### 推導 D：V-loss（Eq. 10 → 12）與其最小解
X_V(s,τ)=ϕ_Q(N_Q|s,A,τ)，𝒳_V={N_Q,A}，dX_V/dτ=u_Q(ϕ_Q(N_Q|s,A,τ)|s,A,τ)。套 Lemma 1 得 Eq. 10：在同一個 z^τ 上讓 v_V(z^τ|s,τ) 回歸 u_Q(z^τ|s,a,τ)，沒有 reward、沒有 ε。
最小解〔顧問推導，標準邊際速度公式〕：u_V(z|s,τ)=E[u_Q(z|s,A,τ) | X_V=z] = Σ_a π(a|s)p_Q(z|s,a,τ)u_Q(z|s,a,τ) / Σ_a π(a|s)p_Q(z|s,a,τ)。是「以 z 為條件的後驗加權平均」，不是對 π 的單純平均。A.2 把它寫成 E_{a∼π, z^τ∼p_Q}[v_Q(z^τ|s,a,τ)]，記號鬆散（左邊是 z^τ 的函數，右邊卻對 z^τ 取期望）。
取樣 z^τ∼p_Q(z^τ|s,a,τ) 需要把 Q-flow 從 0 積分到 τ：VF 不是 simulation-free（標準 CFM 是）。

### 推導 E：IVF 的 expectile 性質（App. A.2）
假設：類 Fitted-Q 迭代、每輪精確最小化、k→∞ 收斂到自洽不動點（Eq. 22, 23；這是假設不是結論）；分析時 reward 不乘 (1−γ)。
1. 隱式策略（Eq. 24）：π^µ_{k−1}(a|s)=π_β(a|s)w^µ_{k−1}(s,a)/E_{π_β}[w^µ_{k−1}(s,a)]。因 w∈[1−µ,µ]、µ∈[0.5,1)，密度非負。權重只依賴 (s,a)，所以 Eq. 16 = (只依賴 s 的常數)×(以 π^µ 抽動作的未加權 V-loss)。
2. 對兩個 loss 各套一次 Lemma 1（同推導 C、D），不動點滿足 ϕ^∞_V(N_V|s,τ) =d ϕ^∞_Q(N_Q|s,A,τ)（A∼π^µ_∞）與 Eq. 9 形式的 Q 方程，即策略 π^µ_∞ 的 flow Bellman 方程（Eq. 7）。
3. 取期望：V^∞(s)=E[ϕ^∞_V(z⁰|s,1)]=E_{a∼π^µ_∞}[Q^∞(s,a)]；Q^∞(s,a)=r+γE_{s′}[V^∞(s′)]。
　（這裡用了 E[∫₀¹v dτ]=E[ϕ(z⁰|1)]，隱含 E[z⁰]=0；論文沒說。）
4. 〔顧問推導〕為什麼這就是 expectile：V=E_β[wQ]/E_β[w] ⟺ E_β[w_µ(s,a)(Q(s,a)−V(s))]=0，而這正是 expectile 回歸 min_V E_β[|µ−1(Q−V<0)|(Q−V)²] 的一階條件。論文此處只寫「套用 Hansen-Estruch et al. (2023) 的 Theorem 4.1 與 Eq. 9」。
5. µ→1：引 IQL Theorem 3，lim V^∞_µ(s)=max_{a:π_β(a|s)>0}Q*_β(s,a)，lim Q^∞_µ(s,a)=r+γE[V*_β(s′)]。
為什麼不能直接對速度做 expectile（D.5）：µ→1 時逐點 expectile 給 max_a v̄_Q(z^τ|s,a,τ)，而 E_{z⁰}[∫max_a v̄_Q dτ] ≠ max_a E_{z⁰}[∫v̄_Q dτ]=max_a Q̄(s,a)。

### 附帶：一步估計為何夠用〔顧問推導〕
對獨立耦合線性內插，真實速度在 τ=0 滿足 u(z⁰|0)=E[Z¹−Z⁰|Z⁰=z⁰]=E[Z¹]−z⁰，故 z⁰+u(z⁰|s,a,0)=E[Z¹]=Q(s,a)，對任何 z⁰ 都成立。這解釋 D.2 的訓練用估計 Q(s,a)≈z⁰+v_Q(z⁰|s,a,0)（論文稱 Tweedie's formula，未推導），也讓 Fig. 8a「1 步也差不多」不那麼意外。僅在速度學得準時成立。

---

## 3. 容易讀錯之處、論文內部不一致、未指明處

### 3.1 看起來對、但論文沒有證明的說法
1. 「高斯修正項是效能提升的原因」——沒有消融。Sec. 6.3 的消融是：去掉 V-velocity、one-step actor、三種 expectile 加權、reward scaling；B.2 是 flow 步數與 M。沒有任何實驗拿掉 (1−τ)√(1−γ²)N 或只在部分 τ 監督。與 Value Flows 的比較同時混了 BCFM loss、confidence weight、return clipping 等差異。論文自己的措辭是 "offering an explanation"、"suggest"。
2. 「先前方法會 collapse 成單點」——只有 4 個狀態（每個 domain 一個）、專門收集的 1000+1000 條軌跡上的觀察（Fig. 2, 7）。沒有理論說它們必然 collapse。
3. 「VF/IVF 全面優於先前方法」——看表：
   - humanoidmaze-giant：VF 在 10%/25%/50%/100% 的 domain 平均是 0/0/0/1（Table 2–5），所有 explicit 方法都接近 0；只有 implicit 方法能做（IVF 40/50/49/55）。
   - 100% OGBench：IVF 65±5，Implicit C51 66±3（Table 5）。puzzle-4x4 100%：IVF 82±11，Implicit C51 100±0、Implicit IQN 95±4。
   - 10% puzzle-4x4：VF 96±4，C51 與 QC 都是 100±0。
   - OGBench 總平均（explicit 最佳 baseline → VF）：10% 43(QC)→48；25% 48(IQN)→54；50% 51(C51)→53；100% 52(C51,QC)→57。（implicit 最佳 baseline → IVF）：10% 27→43；25% 46→61；50% 58→66；100% 66→65。
   - OCBench：25% VF 67±2 對 Value Flows 60±3；IVF 58±2 對 Implicit QC 57±3。100% VF 92±1 對 C51 91±2；IVF 91±1 對 Implicit C51 86±2。
   增益最明確的是 IVF 在縮減資料上；VF 的領先多在 2–6 個百分點。
4. 「limited data」——OGBench 的 10% 是 100M 的 10%，即約 10M transitions；不是小資料。OCBench 完整資料為 0.77M–5.81M transitions。
5. 子集不是均勻抽樣：以每條軌跡的平均 reward 為分數、softmax 不放回抽樣（D.1）。溫度未說明。子集偏向高 reward 軌跡。
6. 「VF 接近 Oracle」只對 cube-triple（115 vs 92）與 puzzle-4x4（22 vs 20）成立。cube-quadruple 是 179±6 對 Oracle 57±6，且與 PCBF 184±9 幾乎相同。humanoidmaze-giant VF 13±2 反而低於 Oracle 25±3——Oracle 不是下界，只是用同批軌跡的實際回報做標準 CFM。
7. 「richer learning signals」是動機敘述，沒有被量測。
8. 理論保證的範圍：contraction 是固定 π、固定 τ、理想算子；對神經網路交替訓練、對 VF 的 rejection-sampling 策略改進、對 IVF 的自洽不動點是否存在/唯一，都沒有證明。A.2 說 flow Bellman 方程「有唯一不動點」，但 π^µ_∞ 本身依賴解，所以那只是「給定 π 後唯一」。
9. Fig. 2 的距離數值是幾百的量級，但訓練時 reward 乘了 (1−γ)（scaled 回報應落在 [rmin,rmax]，例如 [−3,0]）。圖軸（如 −800…−400）顯然是未縮放單位；論文沒說明換算。〔顧問判讀〕
10. Eq. 5 的等號是「分佈相等」。ϕ_Q 與 ϕ_V 用獨立雜訊，並非同一條路徑上的等式。把它畫成「同一顆粒子的軌跡滿足 Bellman」是 PCBF 的做法，不是 VF。
11. Table 1 把 Value Flows 的「intermediate noise levels 的 Bellman constraint」打 ✓、「initial 高斯」打 ✗†；正文與 Fig. 1 稱其為 "incorrect supervision at initial noise"。依推導 A(iv)，乾淨方程在所有 τ<1 都與內插邊際不一致，不只 τ=0。〔顧問推導〕教學時說「在雜訊端錯得最多，越靠近 τ=1 越小」較準確。
12. 腳註 1 說換成 Unif 起始雜訊會有「不同的修正項」，沒有給式子。〔顧問推導〕要有獨立加性修正，需存在 X 使 Z⁰ =d γZ⁰′+X；對均勻分佈，特徵函數比 γ·sin(t)/sin(γt) 在 1/γ 非整數時無界，不是合法特徵函數，所以一般 γ 下不存在這種修正。不要把腳註當成已證事實展開。
13. 「Value Forcing」這個名字：論文引了 Diffusion Forcing (Chen et al., 2024) 但沒有解釋命名。不要自己編典故。

### 3.2 內部不一致 / 筆誤
1. A.2 的 Q-loss 最小解式子（p.18）：寫成 γv_V^{k−1}(τr+γz^τ+(1−τ)√(1−γ²)ε | s′,τ) 且 z^τ∼p_V^{k−1}(z^τ|s,a,τ)。與 Eq. 11/21 對照，v_V 的引數應為 z^τ，取樣條件應為 s′；且應是以「變換後的點＝查詢點」為條件的期望。以 Eq. 11、21 為準。
2. Sec. 5 定義 Q(s,a)≜E_{z⁰}[∫₀¹v_Q dτ]（沒有 z⁰ 項），D.2 的實作是 z⁰+v_Q(z⁰|s,a,0)。期望值相同（E[z⁰]=0），單樣本不同。
3. Sec. 5 說 Q、V 由「M 個回報樣本平均」得到；D.3 說 VF 訓練時 M=1、一步積分，評估時 M=16、10 步、候選動作共用起始雜訊；IVF 固定 M=16。Table 6 沒有 M 這一列；Table 6 的「Return-flow integration steps 10」與 VF 訓練時一步不同；「BC flow integration steps 10 (VF = 4 during training)」。
4. OCBench 敘述：Fig. 3 圖說「VF 在 25% 勝過 explicit、IVF 在完整資料領先 implicit」；Sec. 6.2 正文說 VF 在 25% 與 100% 都最高（100% 是 92±1 對 91±2）。
5. Sec. 6.2「VF barely performs worse as the dataset size decreases」：總平均是 57→53→54→48。
6. D.2 對 Value Flows DCFM 的寫法：目標是 v̄_Q(z^τ|s′,a′,τ)，沒有 γ 係數；而本文自己的 Eq. 11 目標有 γu_V。若逐路徑 z_τ=r+γz′_τ 成立，速度應差一個 γ〔顧問推導〕。我無法從本文判斷這是原方法如此還是轉述筆誤；不要在教學中細講 DCFM 公式。
7. 消融與主表數字不能直接對：Fig. 5 的 VF「約 86%」是 4 個任務、最後三次評估（300K/400K/500K）平均；Table 5 同 4 個任務的最終評估是 98/78/98/100（平均 93.5〔顧問計算〕）。IVF 76% 對 Table 5 的 90/58/85/75（平均 77）。協定不同，不算矛盾，但不要混引。
8. 「one-step actor reduces success rates by 44%」：沒說是相對下降還是百分點，Fig. 4 無數值標註。只能照原句引用。

### 3.3 未指明的超參數與協定
- softmax 抽樣溫度（D.1）。
- 取樣 z^τ∼p_Q / p_V 到任意 τ 時用幾步 Euler、如何對齊 τ（Alg. 1 第 4、6、8 行）。
- Alg. 1：Q 有 target 網路 ψ̄（EMA，rate 0.005）；V 沒有 target 網路，Q-loss 的 z^τ 與目標都用線上 v_V^θ。twin 兩個成員如何合併（min／平均）未說明。
- IVF「一個網路同時當 V 與 Q velocity（仿 classifier-free guidance）」：動作如何遮蔽未說明。
- action chunk H=5 下，γ 是每步還是每 chunk、chunk 內 reward 如何累加，未說明。
- baseline 是否也用 (1−γ) reward scaling，未說明（C51/DIVL 的 support 是 [rmin,rmax]/(1−γ)，暗示未縮放）。〔顧問判讀〕
- Fig. 2/6 的 ± 是什麼（推測為 4 seeds 的標準差）未明寫。
- Fig. 6 的 IVF 如何做 SARSA 式 policy evaluation（µ 取多少）未說明；其數值與 VF 幾乎相同（116/174/18/13 對 115/179/22/13）。
- OGBench 每任務每次評估只有 10 個 episode × 4 seeds，所以單任務數字是粗粒度（3、5、8、10…）、標準差常達 20–40；只有 domain／總平均較可信。OCBench 是 500 個 episode。
- 超參數在每個 domain 的 task 2、4、完整資料上調，然後套用到全部 5 個任務與所有資料量；消融也用同樣的 task 2、4。VF、FlowCritic、C51、QC、Implicit QC 未調參；Floq 的 κ、vel-depth 沿用「最相近 domain」。
- 11 個 baseline = 9 個既有方法（Floq、FlowCritic、Value Flows、PCBF、C51、IQN、QC、DIVL、Implicit QC）+ 2 個作者自建（Implicit C51、Implicit IQN）。Table 1 另列 FlowIQN、EVOR，但它們不在實驗裡。DIVL 的 policy extraction 被換成 rejection sampling，atom 數由 201 改為 2001。

### 3.4 可直接引用的數字（已核對位置）
Fig. 2，mean 1-Wasserstein（Oracle / VF / FlowCritic / Value Flows / PCBF）：
- cube-triple：92±13 / 115±16 / 212±14 / 295±16 / 204±15
- cube-quadruple：57±6 / 179±6 / 203±9 / 289±11 / 184±9
- puzzle-4x4：20±5 / 22±5 / 125±6 / 134±7 / 118±6
- humanoidmaze-giant：25±3 / 13±2 / 127±3 / 462±3 / 122±3
相對 PCBF 下降 81%（puzzle）、89%（humanoidmaze）為正文數字；1−22/118、1−13/122 吻合。
Fig. 6（Oracle / IVF / Implicit C51 / Implicit IQN / DIVL）：
- cube-triple：92±13 / 116±16 / 299±15 / 251±13 / 287±13
- cube-quadruple：57±6 / 174±6 / 497±10 / 233±7 / 290±7
- puzzle-4x4：20±5 / 18±4 / 279±4 / 141±6 / 128±6
- humanoidmaze-giant：25±3 / 13±2 / 205±3 / 88±2 / 255±3
（以上由 PDF 文字層的欄位位置對應，長條順序依圖的 x 軸標籤；若要上圖請再對 PDF 目視一次。）
OGBench 總平均（Table 2–5；Floq / FlowCritic / Value Flows / PCBF / C51 / IQN / QC / VF ‖ DIVL / Impl. C51 / Impl. IQN / Impl. QC / IVF）：
- 10%：13 / 34 / 36 / 29 / 39 / 39 / 43 / 48 ‖ 14 / 26 / 27 / 25 / 43
- 25%：19 / 34 / 47 / 26 / 47 / 48 / 44 / 54 ‖ 15 / 39 / 46 / 38 / 61
- 50%：19 / 32 / 46 / 23 / 51 / 50 / 45 / 53 ‖ 15 / 58 / 57 / 51 / 66
- 100%：16 / 31 / 48 / 23 / 52 / 48 / 52 / 57 ‖ 18 / 66 / 61 / 62 / 65
OCBench 總平均（Value Flows / C51 / QC / VF ‖ Impl. C51 / Impl. QC / IVF）：25%：60 / 58 / 45 / 67 ‖ 51 / 57 / 58；100%：85 / 91 / 62 / 92 ‖ 86 / 80 / 91。
Reward scaling（Fig. 5）：VF 約 86%→11%，IVF 76%→11%。
設定：Adam、lr 3e-4、500K steps、batch 1024、γ=0.999（OGBench）/0.9999（OCBench）、H=5、K=16、4 seeds；policy evaluation 實驗訓練 100K steps、抽 5000 個回報樣本對 1000 條評估回報。IVF 的 µ：cube-triple 0.9、cube-quadruple 0.8、puzzle-4x4 0.9、humanoidmaze-giant 0.8、OCBench 0.9。[rmin,rmax]：[−3,0]、[−4,0]、[−16,0]、[−1,0]、[−1,0]。

---

## 4. 哪些可以在瀏覽器即時示範

### 可以（閉式或小型粒子模擬）
1. **τ=0 的變異數檢查**：滑桿 γ，顯示 γN′+√(1−γ²)N 的變異數恆為 1；對照不加修正的 r+γN′（平均 r、變異數 γ²）。直接對應 Sec. 4.1。
2. **不動點族**〔顧問推導 A(ii)〕：玩具 MDP（2–3 個狀態、確定性 reward、隨機轉移），用粒子反覆套 T_τ，顯示收斂到 τZ+(1−τ)N；滑桿 τ。對照「乾淨算子」：每個 τ 都收斂到 Z^π 本身，τ=0 不是高斯。示範用 γ≈0.8–0.9；γ=0.999 收斂太慢看不出來。須標明這是玩具與顧問推導，不是論文圖。
3. **Contraction**：兩組粒子、同一 τ，每次迭代後計算一維 W₁（排序後配對），顯示比值 ≤ γ。對應 Eq. 6。
4. **Q-loss 的輸入點／目標**（Eq. 11）：畫一條 V-flow 路徑上的 z^τ，顯示變換後的輸入點 τr+γz^τ+(1−τ)√(1−γ²)ε 與目標 r+γu_V−√(1−γ²)ε；拉 ε 看正負號。
5. **閉式速度場**：回報為高斯 N(m,σ²) 時 u(z|τ)=m+[(τσ²−(1−τ))/(τ²σ²+(1−τ)²)]·(z−τm)〔顧問推導〕；雙點分佈（Intro 的 ±1 對 0 例子）或高斯混合可用後驗權重寫出閉式，JS 以 10 步 Euler 積分畫出類似 Fig. 7 的路徑（但那是真實速度，不是學到的）。
6. **一步估計**：驗證 z⁰+u(z⁰|0)=E[Z] 與 z⁰ 無關〔顧問推導〕。
7. **Expectile 與隱式策略**：離散動作、給定 Q 值與 π_β，解 Σπ_β w_µ(Q−V)=0 得 V；滑桿 µ 從 0.5 到接近 1，V 從平均移向 max。再加一個「逐點對速度取 max」對「先積分再取 max」的對照，重現 D.5 的不等式（兩個動作、高斯 Q 分佈即可）。
8. **PCBF 的代數恆等式**：(1−τ)z⁰+τ(r+γz′)=τr+γz̃^τ+(1−τ)(1−γ)z⁰（D.2）。對照 VF：PCBF 的修正用同一個 z⁰（係數 1−γ），VF 用獨立 N（係數 √(1−γ²)）；兩者在 τ=0 都保持單位高斯。
9. **尺度直觀**：未縮放回報範圍 [rmin,rmax]/(1−γ)（γ=0.999、cube-triple 為 [−3000,0]）對單位高斯；乘 (1−γ) 後回到 [−3,0]。只能示範尺度，不能示範成功率差異。
10. **表格數字的靜態圖**：Table 2–5、Fig. 2/6 的數字可直接畫長條圖（用 §3.4）。

### 不行
- Fig. 2/6/7 的學到分佈、baseline 的 collapse：需要在 OGBench 模擬器收集的軌跡上訓練 4 層 1024 寬 MLP 共 100K steps。玩具訓練未必出現 collapse，論文也沒有 collapse 的理論；不要用玩具「演」出 collapse 然後說是論文結果。
- Fig. 3、Table 2–5 的成功率：100M transitions、500K gradient steps、機器人模擬器。
- Fig. 4/5/8 的消融：同上。尤其 reward scaling 的 86%→11%，論文沒有機制解釋，瀏覽器裡只能講尺度。
- 「資料少時優勢更大」：需要完整離線 RL 流程。
- A.2 的收斂假設（自洽不動點存在）：論文未證，玩具裡收斂不代表一般成立。

---

## 5. 先備知識（教學順序）

1. 隨機變數、「分佈相等」(=d) 與「逐點相等」的差別；獨立性。
2. 高斯的線性組合：aN₁+bN₂∼N(0,a²+b²)（獨立時）；幾何級數 Σγ^{2t}=1/(1−γ²)。
3. MDP、策略、折扣回報 Σγ^t r_t、有效視界 1/(1−γ)。
4. V、Q、Bellman 方程；TD bootstrapping（用自己的估計當目標）；SARSA 與 policy evaluation 的差別。
5. 分佈式 RL：回報是隨機變數 Z，Q=E[Z]；分佈式 Bellman 方程 Eq. 1；C51／分位數法各一句話。
6. 一維 Wasserstein 距離：CDF、分位數函數、耦合、排序配對。
7. Contraction 與不動點（Banach）：反覆套用會收斂到唯一解。
8. ODE、速度場、Euler 積分。
9. Flow matching：線性內插 x^τ=(1−τ)x⁰+τx¹、條件速度 x¹−x⁰、平方誤差回歸的最小解是條件期望（→ 邊際速度）；一維 flow 路徑不交叉、保分位數。
10. Lemma 1：把任意「以 τ 為參數的隨機過程」變成一個回歸問題。
11. 離線 RL：OOD 動作問題、行為複製（flow BC 策略）、從 K 個候選取 argmax 的 rejection sampling。
12. Expectile 回歸與 IQL：不對稱平方損失、µ→1 逼近 max、只用資料內動作。
13. 實作零件：target 網路／EMA、twin critics、action chunking、reward scaling。
