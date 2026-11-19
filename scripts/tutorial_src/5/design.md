# 設計文件：〈Parallelism, critical windows, and separations among diffusion language models〉互動教學

- 論文：Sitan Chen、Liye Wang，arXiv 2609.20539（v2）。資料庫編號 No. 5。純理論論文：沒有訓練、沒有 benchmark，唯一的數值內容是 Fig. 4 與 App. C 的模擬。
- 原文：`tut/5/src/paper.txt`（PDF 同目錄）。本文件中「Eq. n」「Sec. n」「Thm/Lemma/Prop/Def n」「Fig. n」「App. X」「Alg. n」皆指原文編號。理解備忘：`tut/5/advisor.md`。
- 產出：`tut/5/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/dllm-parallelism-critical-windows.html`（slug `dllm-parallelism-critical-windows`），站內連結以上線位置撰寫。
- 驗算腳本（與頁面同一算法，用 `python3 -I` 執行）：`tut/5/verify/w23_exact.py`（C2、C3）、`w4_window.py`（C4；加 `--mc` 另跑 App. C 的 Monte Carlo 交叉檢查，約 45 秒）、`w5_count.py`（C5）。
- 研究筆記：`tut/5/notes/`（hub `dllm-parallelism-lineage-hub` + 6 篇 topic + `timeline.json`，13 筆），上線後為 `tutorials/<slug>.html`。
- 慣例：log 一律為自然對數（nats），只有 Sec. 8 的 $\ell$ 用 $\log_2$。本設計自行推導、論文沒有寫出的結論標「〔本頁推導〕」；為了示範而自選、論文沒有給數值的常數標「〔自選〕」；本頁不含「用論文數字校準」的推定值（論文沒有可校準的實驗數字），因此沒有「推定」標籤。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

diffusion language model（dLLM）宣稱的優點是平行生成：長度 $d$ 的序列可以用少於 $d$ 次 forward pass 生成。本文把「一次 forward pass」形式化為一次 **score oracle query**：給定任意 noise level $t$ 與帶噪序列 $z$，一次回傳所有座標 $i$（masked 僅未揭露座標）的 posterior marginal $\mathrm{law}((X_0)_i\mid X_t=z)$（Sec. 1 Framework），並比較 masked、uniform、Gaussian 三種腐蝕過程所需的 query 數。結果有兩部分。**Result 1**（Thm 3.1、4.1）：對任意分佈 $q$，uniform 與 Gaussian diffusion 用 $O\big((1+\overline{\mathrm{DTC}}/\varepsilon)\log(edS/\varepsilon)\big)$ 次 query 就能達到 $\mathrm{KL}(q\|\hat q)\le\varepsilon$（$\varepsilon>0$ 是使用者指定的目標精度，$\hat q$ 是取樣器的輸出分佈；另加 score 誤差項），其中 DTC 是 dual total correlation；這追平了先前只對 masked diffusion 成立的結果 [CCL26, LZ25]。關鍵是 reverse data-processing inequality：每一步平行更新丟掉的條件相關 $\mathrm{TC}(X_s\mid X_t)$ 不超過步長係數乘上 $\mathrm{DTC}(q_s)-\mathrm{DTC}(q_t)$，這些差可以telescoping 相加（Fig. 2）。**Result 2**：對隨機碼本分佈（$\{\pm1\}^d$ 上 $M=\lceil e^{\kappa d}\rceil$ 個隨機點的均勻分佈），uniform 與 Gaussian 用 $O(\sqrt{d\Lambda}/\varepsilon^2)$ 次 query，即可對至少 $1-\delta$ 比例的碼本 $C$ 達到 $\mathrm{TV}(\hat q^C,q^C)\le\varepsilon$（Thm 6.1、7.1）。這裡 $\varepsilon$ 是 TV 的目標精度；$\delta\in(0,1/4)$ 是失敗機率，即保證不成立的碼本所佔比例（Alg. 1 的輸入寫作 "failure probability δ"）；$\Lambda=\log(Cd/(\varepsilon\delta))$ 是對數因子（Eq. 24，這裡的 $C$ 是只依賴 $[\kappa_-,\kappa_+]$ 的常數，不是碼本），且 $\tilde\Omega(\sqrt d)$ 是必要的（App. A；Gaussian 引用 [XP26]）；masked diffusion 則存在每個 level 誤差 $\le\eta$ 的近似 oracle，使任何演算法需要 $\Omega(d/\log d)$ 次 query（Thm 8.1）。分離的原因是 **critical window** 的寬度：masked 的有資訊區間只佔 noise 範圍的 $O(\log d/d)$（Prop 8.6，需 $\eta=d^{-\Theta(1)}$ 使 $\ell=O(\log d)$），uniform／Gaussian 的窗寬為 $w=A\sqrt{\Lambda/d}$（Eq. 26；$A>0$ 是 Prop 6.4 證明中選定、只依賴 $[\kappa_-,\kappa_+]$ 的常數，論文沒有給數值），在 $\varepsilon$ 為常數、$\delta=d^{-\Theta(1)}$ 時是 $O(\sqrt{\log d/d})$（Prop 6.4、7.4）；窗越窄，窗可能的位置越多，不知道窗位置時的搜尋越貴（Fig. 3、Fig. 4）。作者主張分離來自窗寬這個機制，而不是常見說法中的「masked diffusion 無法修改已確定的 token」（abstract、Sec. 1.2、Sec. 9）。

背景（研究筆記）：不利用資料結構時，uniform diffusion 以 uniformization 可在 $O(d)$ 次 query 內達到零離散化誤差 [Chen & Ying 2024]；本文的目標是 $o(d)$。autoregressive 模型可視為腐蝕過程為「由右至左 erasure」的特例（Sec. 1）。

### 1.2 研究動機（Sec. 1）

1. 平行解碼一定有統計代價：用 posterior marginal 的乘積近似 joint posterior 會產生離散化誤差，代價大小取決於推論排程與腐蝕過程。
2. 實務文獻常說 masked diffusion 較難少步生成，因為已解開的 token 不能修改；uniform 與 Gaussian 可以在取樣過程中反覆修正（Sec. 1 引 [SSP+25, SDG+25, SLY+26]）。這些說法是啟發式的。
3. 理論上只知道 masked diffusion 在適當排程下步數可隨內在複雜度（TC、DTC）縮放 [LC26, CCL26, LZ25]；uniform 與 Gaussian 沒有對應結果。作者的問題：「腐蝕過程的選擇如何影響 dLLM 的少步生成能力？」（Sec. 1）

### 1.3 貢獻

1. Uniform diffusion 的 DTC 縮放（Thm 3.1）與 Gaussian diffusion 的 DTC 縮放（Thm 4.1），證明路線是「單步條件 TC ≤ 係數 × DTC 下降量」再以 telescoping 相加，不需要 masked 情形中誤差的精確刻畫（Sec. 1.1）。
2. 隨機碼本分佈上 uniform 與 Gaussian 的 $\tilde O(\sqrt d)$ 上界（Thm 6.1、7.1），演算法為「找窗 → 在窗內取樣 → 一步解碼」，第一次用 critical window 推出新的 query 上界（Sec. 1.2）。
3. 同一分佈族上 masked diffusion 的 $\tilde\Omega(d)$ 下界（Thm 8.1）與 uniform 的 $\tilde\Omega(\sqrt d)$ 下界（Thm A.1），合起來是三種範式之間第一個可證明的平行度分離。
4. 機制解釋：論文主張分離來自 critical window 寬度（Prop 6.4、7.4、8.6），而不是常見說法中的 commitment（abstract、Sec. 9）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供；各符號仍在第一次使用處定義）

| 符號 | 意義 | 出處 |
| --- | --- | --- |
| $d$、$S$、$\Sigma=[S]$、$\mathcal X=\Sigma^d$ | 序列長度、字母表大小、字母表、序列空間 | Sec. 2.1 |
| $q$、$X_0\sim q$、$X_t$、$q_t$ | 資料分佈、乾淨樣本、noise level $t$ 的帶噪樣本、其分佈 | Sec. 2.1 |
| $\alpha_t=e^{-t}$、$\beta_t=(1-\alpha_t)/S$ | uniform kernel 的保留機率與每個符號的均勻質量 | Sec. 2.1 |
| $K^U_t(\cdot\mid z)=\alpha_t\mathbb 1[z=\cdot]+\beta_t$ | uniform 單座標 kernel（每個 token 以機率 $1-e^{-t}$ 重抽） | Sec. 2.1 |
| $K^M_t(\cdot\mid z)=\alpha_t\mathbb 1[z=\cdot]+(1-\alpha_t)\mathbb 1[\star=\cdot]$ | masked 單座標 kernel，$\star$ 為遮罩符號 | Sec. 2.3 |
| $X_t=e^{-t}X_0+\sigma_tG$、$\sigma_t=\sqrt{1-e^{-2t}}$ | Gaussian（OU）過程，$X_0$ 為 one-hot 嵌入 | Sec. 2.2 |
| $s_t$ | score：uniform/masked 為 likelihood ratio $q_t(y^{i\leftarrow a})/q_t(y)$，Gaussian 為 $\nabla\log q_t$ | Sec. 2.1–2.3 |
| $\varepsilon$、$\hat q$ | 使用者指定的目標精度、取樣器的輸出分佈；Thm 3.1/4.1 中 $\varepsilon$ 是 $\mathrm{KL}(q\|\hat q)$ 的上限，Thm 6.1/7.1 中是 $\mathrm{TV}(\hat q^C,q^C)$ 的上限 | Thm 3.1、4.1、6.1、7.1 |
| $\delta$ | 失敗機率：Thm 6.1/7.1 的保證對至少 $1-\delta$ 比例的碼本 $C$ 成立，$\delta\in(0,1/4)$。與 Thm 3.1 證明的末端時間 $\delta_U$ 是不同的量 | Thm 6.1、Alg. 1 輸入 |
| $\varepsilon_{\rm unif},\varepsilon_{\rm gauss},\varepsilon_{\rm mask}$ | 三種 score 誤差：score entropy（對 $i,a$ 加總）、$L^2$ norm、每個未揭露座標的平均 KL | Def 2.2、2.5、2.7、Lemma 2.8 |
| $H$、$\mathrm{KL}$、$I(\cdot;\cdot)$ | entropy、KL divergence、mutual information（nats） | 標準 |
| $\mathrm{TC}(X)=\sum_iH(X_i)-H(X)$ | total correlation（與獨立乘積的 KL） | Def 2.9 |
| $\mathrm{DTC}(X)=H(X)-\sum_iH(X_i\mid X_{-i})$ | dual total correlation | Def 2.9 |
| $t$、$s=t-h$、$h$ | 單步的起點（噪聲較多）、終點（噪聲較少）、步長；取樣器由 $X_t$ 退到 $X_s$ | Lemma 3.2、Eq. 12 |
| $c_s(h)=\frac{e^{2h}-1}{1-e^{-s}}$ | uniform 的 reverse DPI 係數 | Lemma 3.2 |
| $a_U$、$u=e^t-1$、$N_U$、$\overline{\mathrm{DTC}}$ | 步長參數、uniform 排程變數、步數、已知的 DTC 上界 | Thm 3.1 證明 |
| $C$、$M=\lceil e^{\kappa d}\rceil$、$q^C=\mathrm{Unif}(C)$、$\kappa\in[\kappa_-,\kappa_+]\subset(0,\log2)$ | 碼本、碼本大小、隨機碼本分佈、碼率 | Sec. 5、Def 5.1 |
| $I_{\rm unif}(t)=\log2-h_2(\beta_t)$、$I_{\rm gauss}(t)$、$h_2$ | 單一 bit 經腐蝕後保留的資訊量、binary entropy | Sec. 5 |
| $t^\star_{\rm unif},t^\star_{\rm gauss}$ | 臨界時間：$d\,I(t^\star)=\log M$ | Eq. 22 |
| $\Lambda=\log(Cd/(\varepsilon\delta))$、$w=A\sqrt{\Lambda/d}$、$A$ | 對數因子（$C$ 為常數，不是碼本）、critical window 寬度、只依賴 $[\kappa_-,\kappa_+]$ 的常數（無數值） | Eq. 24、26 |
| $T_t$、$\hat T_t$、$\hat t$、$t_\pm$、$N$ | 找窗統計量、其估計、估計的臨界時間、窗內取樣區間端點、窗內取樣步數 | Sec. 6.2–6.3 |
| $k$、$\mathcal K^M_d$、$\ell=\lceil\log_2(Cd/(\eta\rho^2))\rceil$、$m$ | $|C|=2^k$、$k$ 的候選集合、worst-case oracle 的窗半寬、已揭露座標數 | Sec. 8、Eq. 96 |
| $\mathcal O_0$、$\mathcal O_{k,C}$、$Q$、$p_{\rm diff}$ | 永遠回傳均勻 marginal 的 oracle、worst-case oracle、query 數、兩次執行出現差異的機率 | Sec. 8 |

注意（頁面以 callout 寫出）：論文中有三個不同的 $u$（Thm 3.1 證明 $u=e^t-1$；Sec. 4 $u=e^{2t}-1$；Sec. 7 的 $\lambda_t=e^{-t}/\sigma_t$ 是訊號強度）。頁面只使用第一個，並以 $u$ 標明「uniform 排程變數」。

#### 步驟 0：計價單位是 score oracle query（Sec. 1 Framework、Sec. 2.1–2.3）

- 一次 forward pass = 一次 query：輸入任意 $(t,z)$，回傳所有座標的 posterior marginal（masked 僅未揭露座標 $i\notin I$）。三者的 score 都等價於 posterior marginal：uniform 由 Lemma 2.1（$p^{\rm post}$、$p^{\rm gibbs}$ 兩個仿射映射）；Gaussian 由 Tweedie（Lemma 2.4）$\mathbb E[X_0\mid X_t=x]=e^tx+e^t\sigma_t^2s_t(x)$；masked 由 Lemma 2.6 $s_t(y)[i,a]=\frac{\alpha_t}{1-\alpha_t}q_{i\mid I}(a\mid y_I)$，$i\notin I$。
- masked 的係數 $\frac{\alpha_t}{1-\alpha_t}$ 依賴 $t$，但 posterior marginal $q_{i\mid I}$ 不依賴 $t$；對估計的 score，論文**假設**估計的 $\hat q_{i\mid I}$ 也不依賴 $t$（Sec. 2.3）。在此假設下 oracle 只吃 partial assignment $(I,x_I)$；自然的 noise level 是已揭露座標數 $m$，因為它可由 $X_t$ 觀測、也是取樣器實際控制的量（Sec. 2.3 末）。
- masked 的誤差只約束 **consistent** partial assignment 上的回答；在不一致的 assignment 上 oracle 可以任意回答（Sec. 2.3、Lemma 2.8）。Sec. 8 的下界依賴這一點。
- 上界分析特定演算法；下界排除所有使用該 oracle 的演算法（Sec. 1）。

#### 步驟 1：平行更新丟掉的是條件 TC（Sec. 3.3 Lemma 3.8）

- 取樣器在格點 $t_0<\dots<t_M$ 上由高往低，每步從噪聲較多的 $t$ 退到噪聲較少的 $s=t-h$（$h$ 為步長），用 **product reverse kernel**：各座標獨立地從 posterior marginal $P[(X_s)_i=\cdot\mid X_t=y]$ 抽（Eq. 12），最後一步對 $X_{t_0}$ 再套一次 forward kernel（Sec. 3.3 步驟 3）。
- KL chain rule + data processing（Lemma 3.8）：$\mathrm{KL}(q\|\hat q)\le\mathrm{KL}(q_{t_M}\|\mathrm{Unif})+H(q_{t_0})-H(q)+\sum_j c_{t_j}(t_{j+1}-t_j)\big[\mathrm{DTC}(q_{t_j})-\mathrm{DTC}(q_{t_{j+1}})+\varepsilon_{\rm unif}(t_{j+1})\big]$。其中離散化項在證明中**恰好等於** $E^{\rm disc}_j=\mathrm{TC}(X_{t_j}\mid X_{t_{j+1}})$（用 marginal 乘積取代 joint 的 KL 就是條件 TC）。末端項等於 $H(q_{t_0})-H(q)$（Lemma 3.9，用到 kernel 對稱）。頁面分兩句寫：(i) 證明中的實際分解：路徑 KL = 初始化 + $\sum_j\mathrm{TC}(X_{t_j}\mid X_{t_{j+1}})$ + $\sum_j E^{\rm sc}_j$ + 末端，其中 $E^{\rm sc}_j\le c_{t_j}(t_{j+1}-t_j)\,\varepsilon_{\rm unif}(t_{j+1})$（Lemma 3.7）；(ii) Lemma 3.8 的敘述是上界，TC 已被換成 $c\cdot$DTC 落差（Lemma 3.2）。
- 〔本頁推導〕小例子（$d$ 個 bit）：全部相同（複製）時 $\mathrm{TC}=(d-1)\log2$、$\mathrm{DTC}=\log2$；偶同位（parity，均勻分佈在偶數個 1 的字串上）時 $\mathrm{TC}=\log2$、$\mathrm{DTC}=(d-1)\log2$；獨立時兩者皆 0。依據：Def 2.9 直接代入（複製：$H=\log2$、$H(X_i\mid X_{-i})=0$；parity：$H=(d-1)\log2$、$H(X_i)=\log2$、$H(X_i\mid X_{-i})=0$）。`w23_exact.py` 對 $d=6$ 驗證。

#### 步驟 2：DTC 的落差恆等式（Sec. 2.4 Lemma 2.10）

- 若 $B$ 由 $A$ 經獨立逐座標 channel 得到：$\mathrm{DTC}(A)-\mathrm{DTC}(B)=\mathrm{DTC}(A\mid B)+\sum_iI(B_i;A_{-i}\mid B_{-i})$，因此 $\sum_iI(B_i;A_{-i}\mid B_{-i})\le\mathrm{DTC}(A)-\mathrm{DTC}(B)$（Eq. 2）。推論：$\mathrm{DTC}(q_t)$ 對 $t$ 單調不增（Thm 3.1 證明引用）。
- 證明要點（頁面可收合）：$H(A)-H(B)-H(A\mid B)=-\sum_iH(B_i\mid A_i)$；$H(A_i\mid A_{-i},B)=H(A_i\mid A_{-i},B_i)$；$H(B_i\mid A_{-i})=H(B_i\mid A_{-i},B_{-i})$（Eq. 3 之後）。

#### 步驟 3：reverse DPI 把條件 TC 換成 DTC 落差（Sec. 3.1）

- Prop 3.3：$P[(X_s)_i=a\mid(X_t)_{-i}=y_{-i}]\ge\beta_s$（加噪後每個符號至少有 $\beta_s$ 的質量）。
- Lemma 3.4（單 token reverse DPI）：若 $p,p'$ 每個符號質量 $\ge c/S$，則 $\mathrm{KL}(pK^U_t\|p'K^U_t)\ge\mathrm{KL}(p\|p')\cdot\frac{e^{-2t}c}{1-e^{-t}+e^{-t}c}$。證明用恆等式 $\mathrm{KL}(p\|p')=\sum_a(p_a-p'_a)^2\int_0^1\frac{1-\theta}{p'_a+\theta(p_a-p'_a)}d\theta$（Eq. 4）：加噪後分子乘 $e^{-2t}$，分母最多放大 $(e^{-t}+(1-e^{-t})/c)$ 倍。
- Cor 3.5（Fig. 5）：以 $(X_t)_{-i}$ 為條件有 Markov 鏈 $(X_s)_{i+1:d}-(X_s)_i-(X_t)_i$，取 $c=1-e^{-s}$、noise 量 $h$；用 chain rule（Eq. 9）整理得 $I((X_s)_{i+1:d};(X_s)_i\mid X_t)\le(1/C-1)\,I((X_s)_{i+1:d};(X_t)_i\mid(X_t)_{-i})$，$1/C-1\le\frac{e^{2h}-1}{1-e^{-s}}$（Eq. 10）。
- Lemma 3.2：$\mathrm{TC}(X_s\mid X_t)\le c_s(h)\big(\mathrm{DTC}(q_s)-\mathrm{DTC}(q_t)\big)$，$c_s(h)=\frac{e^{2h}-1}{1-e^{-s}}$。由 $\mathrm{TC}(X_s\mid X_t)=\sum_{i=1}^{d-1}I((X_s)_i;(X_s)_{i+1:d}\mid X_t)$、Cor 3.5、把 $(X_s)_{i+1:d}$ 放大成 $(X_s)_{-i}$，再用 Eq. 2（$A=X_s$、$B=X_t$）。
- 直覺（頁面字面敘述）：$c_s(h)$ 在 $h\ll s$ 時約為 $2h/(1-e^{-s})$，步長越小係數越小；$s$ 很小時分母趨近 0，所以低 noise 端必須用很小的步長。
- `w23_exact.py` 顯示這個上界在 $d=6$ 的例子中是成立但寬鬆的（真值／上界 = 0.018–0.19，見第 5 節 C2 表）。頁面要寫出「上界寬鬆」。

#### 步驟 4：選步長讓係數固定，telescoping 相加（Thm 3.1 證明、Fig. 6）

- 先定義（頁面 S4 開頭一句）：$\varepsilon>0$ 是目標精度，要求 $\mathrm{KL}(q\|\hat q)\le\varepsilon$，$\hat q$ 是取樣器的輸出分佈。下面的 $\delta_U$ 是末端時間（取樣停在 $t_0=\delta_U$），不是 Result 2 的失敗機率 $\delta$。
- 參數：$L_U=\log(16edS/\varepsilon)$、$\delta_U=\varepsilon/(16dL_U)$、$T_U=\log(4d\log S/\varepsilon)$、$a_U=\varepsilon/(12\max\{\overline{\mathrm{DTC}},\varepsilon\})\le1/12$；$u_0=e^{\delta_U}-1$、$u_{j+1}=\min((1+a_U)u_j,U)$、$U=e^{T_U}-1$、$t_j=\log(1+u_j)$。
- 由 $e^{t_{j+1}-t_j}\le1+a_U(1-e^{-t_j})=:1+x_j$ 得 $c_{t_j}(t_{j+1}-t_j)\le\frac{(1+x_j)^2-1}{1-e^{-t_j}}=(2+x_j)a_U\le3a_U$（$x_j\le1$ 即可）。
- 誤差帳：離散化 $\le3a_U\,\mathrm{DTC}(q)\le\varepsilon/4$；初始化 $\mathrm{KL}(q_{T_U}\|\mathrm{Unif})\le e^{-T_U}d\log S\le\varepsilon/4$（Lemma 3.10）；末端 $H(q_{\delta_U})-H(q)\le d\delta_U\log(eS/\delta_U)\le\varepsilon/8$（Lemma 3.11，需 $t\le1$）；score 項 $3a_UN_U\varepsilon_{\rm unif}$。步數 $N_U\le1+\log(U/u_0)/\log(1+a_U)$，$\log(U/u_0)\lesssim L_U$、$1/\log(1+a_U)\le2/a_U$。
- 排程形狀（Fig. 6）：$t$ 小時 $t_{j+1}\approx(1+a_U)t_j$（幾何），$t$ 大時 $t_{j+1}-t_j\approx\log(1+a_U)$（等距）。
- Score 誤差（Sec. 3.2）：估計的 score 必須先做 Bregman projection 到凸集 $S_{t,b}=\{v>0:v_b=1,\ v_a/\sum v\ge\beta_t\}$（Prop 3.6、Eq. 11），否則換算出的 $p^{\rm gibbs}$、$p^{\rm post}$ 不一定是合法機率、Lemma 3.4 的假設也不成立；結果 $\mathbb E\,\mathrm{KL}(R\|\hat R)\le c_s(h)\varepsilon_{\rm unif}(t)$（Lemma 3.7）。作者在 AI usage 聲明中說早期版本漏了這一步。
- 結論 Thm 3.1：$\mathrm{KL}(q\|\hat q)\le\varepsilon+O(\varepsilon_{\rm unif}\log(dS/\varepsilon))$，query 數 $O\big((1+\overline{\mathrm{DTC}}/\varepsilon)\log(edS/\varepsilon)\big)$。需要已知 $\overline{\mathrm{DTC}}\ge\mathrm{DTC}(q)$（informal 的 Thm 1.1 沒寫）。

#### 步驟 5：Gaussian 版本只改 reverse DPI 與時鐘（Sec. 4，頁面只陳述結論）

- 換變數 $u=e^{2t}-1$、$Z_u=X_0+B_u$（Eq. 14）。沒有 Prop 3.3 的類比（Gaussian 沒有「每個符號至少有質量」的性質），改用 Lemma 4.3：$\mathrm{KL}(f_{\alpha,v}\|f_{\beta,v})\ge\mathrm{KL}(f_{\alpha,u}\|f_{\beta,u})\cdot\frac uve^{3/v-3/u}$（$f_{\alpha,u}$ 為 one-hot 點加 $N(0,u\,I)$ 的混合），證明用 de Bruijn 型恆等式與 Grönwall（Lemma 4.6）。
- 係數 $c_u(v)=\frac vue^{3/u-3/v}-1$（Lemma 4.2）；時鐘 $F_G(u)=\log u-3/u$，因為 $F_G(v)-F_G(u)=\log(1+c_u(v))$；每步令 $F_G$ 增加 $\log(1+a_G)$，$a_G=\varepsilon/(4\max\{\overline{\mathrm{DTC}},\varepsilon\})$（Thm 4.1 證明、Fig. 7）。$u$ 小時步長約 $a_Gu^2/3$，$u$ 大時才是幾何。
- Thm 4.1：$\mathrm{KL}(q_{\rm pre}\|\hat q)\le\varepsilon+O(\varepsilon_{\rm gauss}^2\log(dS/\varepsilon))$，query 數同 Thm 3.1 的形式。

#### 步驟 6：困難分佈族：隨機碼本（Sec. 5、Def 5.1）

- $\Sigma=\{\pm1\}$、$\beta_t=(1-e^{-t})/2$；$C$ 是 $\{\pm1\}^d$ 上均勻隨機的 $M$ 元子集，$M=\lceil e^{\kappa d}\rceil$，$q^C=\mathrm{Unif}(C)$。取樣器知道 $[\kappa_-,\kappa_+]$，不知道 $\kappa$，也不知道 $C$（Sec. 5）。下界額外允許 $\kappa$ 限制在格點 $\{(k/d)\log2\}$，此時 $M=2^k$。
- 臨界時間（Eq. 22）：$d\,I_{\rm unif}(t^\star_{\rm unif})=\log M$、$d\,I_{\rm gauss}(t^\star_{\rm gauss})=\log M$，即「每座標資訊量 = 碼率 $\kappa$」。masked 的對應條件是 $m=k$（Sec. 8；Fig. 4 圖說寫「by Eq. (22)」，但 Eq. 22 只定義 uniform 與 Gaussian）。上界演算法的 query 時間都落在 $I_0=[\tau_-/2,\tau_++1]$（Eq. 23）。
- 〔本頁推導〕$\kappa=0.2$ 時 $\beta^\star=0.1948$、$t^\star_{\rm unif}=0.4937$、$t^\star_{\rm gauss}=0.5521$、masked $m^\star/d=\kappa/\log2=0.2885$（`w4_window.py`）。
- 此分佈族 $\mathrm{DTC}=\Theta(d)$（Sec. 1.2 直接陳述，沒有證明），所以 Thm 3.1/4.1 只給 $\tilde O(d/\varepsilon)$；Result 2 的演算法利用分佈結構，超過 DTC 界。〔本頁推導〕直覺：$H=\log M\approx\kappa d$，而 $H(X_i\mid X_{-i})\neq0$ 需要某個碼字翻一位後仍在 $C$ 中，機率約 $M/2^d$，指數小，故 $\mathrm{DTC}\approx\kappa d$。

#### 步驟 7：critical window 的寬度（Prop 6.4、7.4、8.6；Fig. 3、4；App. C）

- 先定義（頁面 S6 開頭，符號在此第一次出現於正文）：Result 2 的目標是對至少 $1-\delta$ 比例的碼本 $C$ 有 $\mathrm{TV}(\hat q^C,q^C)\le\varepsilon$；$\varepsilon$ 是 TV 精度，$\delta\in(0,1/4)$ 是失敗機率（Thm 6.1）。對數因子 $\Lambda=\log(Cd/(\varepsilon\delta))$（Eq. 24；$C$ 是只依賴 $[\kappa_-,\kappa_+]$ 的常數，不是碼本）；$A>0$ 是 Prop 6.4 證明中選定、只依賴 $[\kappa_-,\kappa_+]$ 的常數，論文沒有給數值。
- Uniform（Prop 6.4）：$w=A\sqrt{\Lambda/d}$（Eq. 26），$t\ge t^\star_{\rm unif}+w$ 時 $\mathbb E_C\,\mathrm{TV}(q^C_t,\mathrm{Unif})\le e^{-K\Lambda}$；$t\le t^\star_{\rm unif}-w$ 時 Bayes 最佳解碼錯誤率 $\mathbb E_C\,\mathrm{err}^U_C(t)\le e^{-K\Lambda}$。證明工具：Lemma 2.3 $2^d(K^U_t)^{\otimes d}(x\mid z)=(1+e^{-t})^d\tanh(t/2)^{d_H(x,z)}$；Lemma 6.2（二階矩 + 截斷半徑 $r$）、Lemma 6.3（最近鄰解碼 + union bound）；取 $r=\lfloor d(\beta_t+\beta_{t^\star})/2\rfloor$，得到 Hoeffding 項 $e^{-d\Delta^2/2}$（Eq. 28、30）與計數項 $e^{-cd\Delta}$（Eq. 29、31），$\Delta=|\beta_t-\beta_{t^\star}|\asymp w$；需要 $d\Delta^2\gtrsim\Lambda$，所以 $w\propto\sqrt{\Lambda/d}$。條件：$d\ge d_{K,A}$、$\log d\le\Lambda\le c_{K,A}d$，期望是對 $C$ 取的。
- Gaussian（Prop 7.4）結構相同：information density $\imath_t(Y;Z)=\lambda_t\langle Z,Y\rangle-\sum_i\log\cosh(\lambda_tZ_i)$ 是 $d$ 個獨立項之和、均值 $d\,I_{\rm gauss}(t)$，$P[|\imath_t-dI_{\rm gauss}|\ge v]\le2e^{-cv^2/d}$（Eq. 59）；參考分佈為 $\nu_t=\mathrm{law}(e^{-t}U+\sigma_tG)$（Eq. 56）。
- Masked（Prop 8.6）：$|C|=2^k$；$m\le k-\ell$ 時 $\mathbb E_C\,\mathrm{TV}(\mathrm{law}(X_{0,I}),\mathrm{Unif})\le2^{-\ell/2}$（Eq. 104，hypergeometric 變異數）；$m\ge k+\ell$ 時 $\mathbb E_C\,\mathrm{err}^M_C(m)\le2^{-\ell}$（Eq. 105）。窗在 $m$ 上寬 $2\ell$，$\ell=O(\log d)$，佔全範圍 $O(\log d/d)$。
- Fig. 4（$\kappa=0.2$；App. C）：recovery probability $=\mathbb E\,P[X_0=y^\star\mid X_t]$（posterior 質量，不是取樣器成功率），以 Poissonization 模擬：除了植入的 $y^\star$，其餘碼字是強度 $M2^{-d}$ 的 Poisson 點過程。橫軸為每座標資訊量 $I$：masked $(m/d)\log2$、uniform $I_{\rm unif}(t)$、Gaussian $I_{\rm gauss}(t)$。窗寬定義為 recovery 由 0.2 到 0.8 的距離；(d) 顯示 masked $\propto d^{-1}$、uniform/Gaussian $\propto d^{-1/2}$。
- 〔本頁推導〕masked 閉式：$N-1\sim\mathrm{Poi}(\lambda)$、$\lambda=M2^{-m}=e^{d(\kappa-I)}$，$\mathbb E[1/N]=(1-e^{-\lambda})/\lambda$。0.8 對應 $\lambda=0.4642$、0.2 對應 $\lambda=4.9651$，窗寬 $=\ln(4.9651/0.4642)/d=2.3698/d$ nats，以 $m$ 計為 $3.419$ 個座標，與 $d$ 無關（App. C 有說 $m$ 的寬度與 $d$ 無關）。
- 〔本頁推導〕uniform 的 $\sqrt d$ 從哪裡來：App. C 的式子 $P[X_0=y^\star\mid X_t]=1/(1+Z)$、$Z=\sum_jN_j\tanh(t/2)^{j-d_H}$，$d_H=d_H(X_t,y^\star)\sim\mathrm{Bin}(d,\beta_t)$、$N_j\sim\mathrm{Poi}(M\binom dj2^{-d})$。由二項式定理 $\mathbb E[Z\mid d_H]=M2^{-d}(1+\tanh\tfrac t2)^d\tanh(\tfrac t2)^{-d_H}$；代入 $1+\tanh\tfrac t2=1/(1-\beta_t)$、$\tanh\tfrac t2=\beta_t/(1-\beta_t)$，在 $d_H=d\beta_t$ 時 $\log\mathbb E[Z\mid d_H]=d(\kappa-I_{\rm unif}(t))$。$d_H$ 有 $\sqrt{d\beta_t(1-\beta_t)}$ 的漲落，使條件均值 $\log\mathbb E[Z\mid d_H]$（對 Poisson 計數取平均後的量，不是 $Z$ 本身）漲落 $\sqrt{d\beta_t(1-\beta_t)}\log\frac{1-\beta_t}{\beta_t}$（$d=256,t=0.6$ 時 8.25）。若把轉變近似為「$\mathbb E[Z\mid d_H]\approx1$ 的位置」，轉變在 $I$ 上寬 $\propto1/\sqrt d$。這一步是**啟發式**，論文沒有證明；論文只給充分窗寬 $w=A\sqrt{\Lambda/d}$（Prop 6.4），$1/\sqrt d$ 的實際縮放依據是本頁求積（擬合斜率 $-0.478$）與 Fig. 4(d)。masked 的 $m$ 是觀測到的固定值，沒有這個漲落，只剩 Poisson 計數本身的 $O(1)$ 轉變。這與論文 Prop 6.4 證明中「Hoeffding 項需要 $d\Delta^2\gtrsim\Lambda$、計數項只需 $d\Delta\gtrsim\Lambda$」一致；頁面把這一段標為〔本頁推導〕、寫明「啟發式」，並附「與 Prop 6.4 的 Hoeffding 項同源」。

#### 步驟 8：上界演算法：找窗、在窗內取樣、一步解碼（Sec. 6.2–6.3、Alg. 1–2；Gaussian：Sec. 7、Alg. 3–4）

- 找窗統計量：$T_t(x)=\langle x,m_t(x)\rangle-de^{-t}=\frac{1-e^{-2t}}{2e^{-t}}\sum_i(1-s_t(x)[i,-x_i])$，$m_t(x)=\mathbb E[X_0\mid X_t=x]$（Sec. 6.2）。查詢點 $x\sim\mathrm{Unif}(\mathcal X)$（不依賴 $C$）。Prop 6.5：窗上（$t\ge t^\star+2w$）$T_t<-a_0dw$ 的機率 $\le e^{-K\Lambda}$；窗下（$t\le t^\star-2w$）$T_t>-b_0dw$ 的機率 $\le e^{-K\Lambda}$，$b_0>4a_0$。
- 實際可算的版本：score 先 clip 到 $[\ell_t,\ell_t^{-1}]$，$\ell_t=\beta_t/(1-\beta_t)$（Eq. 32–33；這裡用 clipping，不是 Sec. 3 的 Bregman projection），得 $\hat T_t$（Eq. 34），一次 query 一個值。Prop 6.9：門檻變成 $a=2a_0<b=b_0-a_0$。
- Alg. 1：在格距 $w/20$、覆蓋 $[\tau_--3w,\tau_++3w]$、由高往低的時間格點上，每點抽 $R=2\lceil C\Lambda\rceil+1$ 個均勻樣本各查一次，若 $\hat T_t$ 的中位數 $<-\frac{a+b}2dw$ 就回傳 $t$；都沒有觸發就回傳 $\tau_-$。Prop 6.10：$O(\Lambda/w)$ 次 query；對至少 $1-\delta/4$ 比例的 $C$，$P[t^\star-3w\le\hat t\le t^\star+2w\mid C]\ge1-\varepsilon/8$（Eq. 46）。
- Alg. 2：$t_+=\hat t+4w$、$t_-=\hat t-4w$、$N=\lceil Cdw/\varepsilon^2\rceil$、$h=8w/N$；從 $\mathrm{Unif}(\mathcal X)$ 起始，每步每座標以機率 $p_h(\tilde s)$ 翻轉，$p_h(r)=\frac{\beta_h}{\alpha_h}((1-\beta_h)r-\beta_h)$（Eq. 47–48）；最後一次 query 取估計 posterior mean 的符號 $\hat z_t(y)=\mathrm{sgn}(\hat m_t(y))$（Eq. 51，Lemma 6.13）。窗內取樣誤差由 Prop 6.12 重用 Lemma 3.2 的telescoping，$\mathrm{DTC}\le d\log2$。
- 總 query：$O(\Lambda/w)+N+1=O(\sqrt{d\Lambda}/\varepsilon^2)$。Thm 6.1 條件：$d\ge d_0$、$\varepsilon,\delta\in(0,1/4)$、$\Lambda\le c_0d$、$\sup_{t\in I_0}\varepsilon_{\rm unif}(t)\le e^{-B\Lambda}=(\varepsilon\delta/(Cd))^B$（Eq. 25）；保證對至少 $1-\delta$ 比例的 $C$ 有 $\mathrm{TV}(\hat q^C,q^C)\le\varepsilon$（帳：$7\varepsilon/16$，失敗機率 $3\delta/4$）。$\varepsilon$ 常數、$\delta=d^{-\Theta(1)}$ 時為 $O(\sqrt{d\log d})$。
- Gaussian（Thm 7.1）：查詢點取自 $\nu_t$；統計量 $T_t(x)=\langle z,m_t(x)-m^0_t(x)\rangle$、$z=x/\sigma_t$、$m^0_t(x)_i=\tanh(e^{-t}x_i/\sigma_t^2)$（Eq. 65–66）；窗內取樣用 exponential integrator，$N=\lceil C(d\Delta^2+d\Delta)/\varepsilon^2\rceil$、$\Delta=8w$；證明中的計數為 $O((\sqrt{d\Lambda}+\Lambda)/\varepsilon^2+1)$，Thm 7.1 敘述為 $O(\sqrt{d\Lambda}/\varepsilon^2)$；因為 $\Lambda\le c_0d$ 時 $\Lambda\le\sqrt{c_0d\Lambda}$，兩者同階，頁面寫定理敘述版；帳 $9\varepsilon/32$。
- 所有常數 $A,B,C,a,b,\dots$ 只說「只依賴 $[\kappa_-,\kappa_+]$」，沒有數值；Alg. 1–4 無法照字面執行。頁面不執行 Alg. 1–4，只以逐行對照表呈現。

#### 步驟 9：下界：worst-case oracle 與 coupling（Sec. 8、App. A）

- Prior：$k\sim\mathrm{Unif}(\mathcal K^M_d)$，$\mathcal K^M_d=\{k\in\mathbb Z:\kappa_-\le k\log2/d\le\kappa_+\}$，$|\mathcal K^M_d|\ge c_1d$；給定 $k$，$C$ 是均勻隨機的 $2^k$ 元子集。
- Lemma 8.2：永遠回傳均勻 marginal 的 $\mathcal O_0$ 滿足 $\mathbb E_C\,\varepsilon_{\rm mask}(\mathcal O_0;q^C,m)\le2^{m-k}$。這是對 $C$ 取期望的敘述；對 $m\le k-\ell$ 的各 level 用 Markov 不等式，得到「good」$C$（這些 level 上 $\mathcal O_0$ 的誤差都 $\le\eta$）的機率 $\ge1-2^{1-\ell}/\eta$（Eq. 98–99）。
- Def 8.3：對 good $C$，$\mathcal O_{k,C}$ 在 $m\le k-\ell$ 回傳均勻、在 $m>k-\ell$ 的 consistent assignment 回傳真實 marginal、在 inconsistent assignment 回傳均勻。Prop 8.4：每個 level 誤差 $\le\eta$（Eq. 97）——這依賴 Lemma 2.8：誤差只看 consistent assignment。
- Prop 8.5（coupling）：對 $\mathcal O_0$ 的查詢序列只由演算法內部隨機性決定。與 $\mathcal O_{k,C}$ 出現差異的機率 $\mathbb E\,p_{\rm diff}\le\frac{(2\ell+1)Q}{|\mathcal K^M_d|}+\frac{2^{1-\ell}}\eta+Q2^{-\ell}$（Eq. 100）：三項分別是「某次 query 的 $m$ 落在 $k\pm\ell$」「$C$ 不是 good」「某次 $m>k+\ell$ 的 query 恰好與 $C$ 一致（每次 $\le2^{k-m}\le2^{-\ell}$）」。
- Thm 8.1：$\ell=\lceil\log_2(Cd/(\eta\rho^2))\rceil$（Eq. 96），$Q\le c\rho^2d/\ell$ 時以機率 $\ge1-\rho$（對 $k,C$）有 $\mathrm{TV}(\hat q,q^C)\ge1-\rho$；$\eta=d^{-\Theta(1)}$、$\rho$ 常數時需要 $\Omega(d/\log d)$ 次 query。收尾用 Eq. 103 與 $\mathbb E_C\nu(C)=2^{k-d}$。
- Uniform（Thm A.1）：$t^\star_{\rm unif}(k)$ 的候選值取成彼此相距 $\ge10w$（Eq. 108），$|\mathcal K^U_d|\ge\frac{c_U}A\sqrt{d/\Lambda}$（Eq. 109）；oracle 只在 $|t-t^\star|<4w$ 回傳真實 score（Def A.4）；$Q\le c\rho^2\sqrt{d/\Lambda}$（Eq. 111）；coupling 界 $Q(1/|\mathcal K^U_d|+e^{-4\Lambda})$（Eq. 127）。Gaussian 的 $\tilde\Omega(\sqrt d)$ 下界引用 [XP26]，本文沒有重證；依研究筆記，[XP26] 的論證顯示在此分佈上連把 $\kappa$ 估計到足夠精度都需要這麼多 query。另一個不同的下界：[CCL26] 對 masked 以 MDS code 構造給出 $\Omega(d)$（Sec. 1.3），那是另一個分佈族，與本文的分離無關；頁面以一句話區分。
- 機制（Sec. 1.2）：成本 ≈ 候選位置數 ÷ 窗寬。masked：$\Theta(d)$ 個 level ÷ $(2\ell+1)$；uniform/Gaussian：$O(1)$ 的時間範圍 ÷ $w$。

#### 步驟 10：範圍與限制（Sec. 9 與全文）

- 下界是「存在一個每 level 誤差 $\le\eta$ 的worst-case 近似 oracle」，沒有證明對 exact oracle 或訓練出的網路成立；上界則對任何誤差 $\le e^{-B\Lambda}$ 的 oracle 成立。
- 只對這一個隨機碼本分佈族；Result 1 是三者打平。Sec. 9 明說不應解讀為推崇某一範式；masked 的 any-order 生成等優點未被分析；Gaussian 只考慮最基本的 one-hot 嵌入。
- 兩個下界都依賴 prior（演算法不知道 $k$）；論文沒有討論 $\kappa$ 已知的情形。
- 誤差度量不同：Thm 3.1/4.1 為 KL、步數 $\propto1/\varepsilon$；Thm 6.1/7.1 為 TV、$\propto1/\varepsilon^2$、且是「對 $\ge1-\delta$ 比例的 $C$」；下界為 $\mathrm{TV}\ge1-\rho$。
- 定理是漸近的（$d\ge d_0$），常數未給；Fig. 4 的寬度不含 log 因子，因為 0.2→0.8 的定義不要求高機率。

### 1.5 實作細節與常見錯誤（頁面 S9 的「常見誤讀」與各節 callout）

1. 把 Thm 8.1 讀成「masked diffusion 一定要 $\tilde\Omega(d)$ 步」：它是 worst-case 近似 oracle 下的 query complexity。
2. 把分離歸因於 commitment：論文主張分離來自窗寬（abstract、Sec. 1.2、Sec. 9）；論文沒有證明 commitment 在一般情形下無關，頁面不做這個更強的宣稱。
3. 以為窗窄比較容易：成本來自找窗，窄窗代表窗可能的位置更多。
4. 把 Result 1 與 Result 2 的取樣器混為一談：前者是「每步所有座標獨立重抽」的通用演算法；後者是三階段的專用演算法。
5. Uniform 取樣器使用估計 score 時漏掉 Bregman projection（Sec. 3.2）；Sec. 6 用 clipping（Eq. 33）、Sec. 4 用 Euclidean projection 到 $\Delta(\Sigma)^d$（Eq. 17）、Sec. 7 clip 到 $[-1,1]$（Eq. 79）。
6. 把 recovery probability 當成取樣器成功率，或當成找窗統計量 $T_t$。
7. 混用三個 $u$（見符號表下方）。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details>`（summary「檢查理解」）放該章檢查題與參考答案；**參考答案的內容必須同時出現在該章正文的文字中**（不只在元件讀數、表格、圖或答案框）。第 3 節大綱中以粗體標出的「LO 句」是建置時必須寫進該章正文的句子。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明一次 score oracle query 給什麼 | 本文把一次 forward pass 形式化成什麼？masked 的 oracle 輸入與 $t$ 有關嗎？ | 對任意 $(t,z)$ 一次回傳所有座標（masked 僅未揭露座標 $i\notin I$）的 posterior marginal $\mathrm{law}((X_0)_i\mid X_t=z)$（Sec. 1）；masked 的 score 等於 $\frac{\alpha_t}{1-\alpha_t}q_{i\mid I}(a\mid y_I)$，係數依賴 $t$，但 posterior marginal $q_{i\mid I}$ 不依賴 $t$（Lemma 2.6）；論文假設估計的 marginal 也不依賴 $t$（Sec. 2.3），因此 oracle 只依賴 partial assignment $(I,x_I)$ | S1b |
| LO2 | 寫出三種腐蝕過程 | uniform、masked、Gaussian 的 forward 過程各是什麼？ | uniform：每個 token 以機率 $1-e^{-t}$ 重抽，$K^U_t=\alpha_t\mathbb 1[z=\cdot]+\beta_t$、$\beta_t=(1-e^{-t})/S$；masked：以機率 $1-e^{-t}$ 變成 $\star$；Gaussian：one-hot 後 $X_t=e^{-t}X_0+\sigma_tG$、$\sigma_t=\sqrt{1-e^{-2t}}$（Sec. 2.1–2.3） | S1a |
| LO3 | 計算 TC 與 DTC | $d$ 個完全相同的 bit 與 $d$ 個 bit 的偶同位分佈，TC、DTC 各是多少？ | 複製：TC $=(d-1)\log2$、DTC $=\log2$；同位：TC $=\log2$、DTC $=(d-1)\log2$（〔本頁推導〕，由 Def 2.9） | S2 |
| LO4 | 說明平行更新的單步誤差 | 用 posterior marginal 的乘積取代 joint 時，一步的 KL 誤差等於什麼？總誤差由哪些項組成？ | 等於條件 total correlation $\mathrm{TC}(X_s\mid X_t)$。實際分解：路徑 KL = 初始化 $\mathrm{KL}(q_{t_M}\|\mathrm{Unif})$ + 各步 TC + 各步 score 誤差（每步 $\le c\,\varepsilon_{\rm unif}$，Lemma 3.7）+ 末端 $H(q_{t_0})-H(q)$（Lemma 3.9）。Lemma 3.8 的敘述是它的上界：各步 TC 換成 $c_{t_j}(t_{j+1}-t_j)\cdot$DTC 落差 | S2 |
| LO5 | 寫出 reverse DPI 的結論 | Lemma 3.2 是什麼？靠哪兩個事實證明？ | $\mathrm{TC}(X_s\mid X_t)\le\frac{e^{2h}-1}{1-e^{-s}}(\mathrm{DTC}(q_s)-\mathrm{DTC}(q_t))$；Prop 3.3（加噪後每個符號質量 $\ge\beta_s$）與 Lemma 3.4（單 token reverse DPI），再用 Lemma 2.10 的 Eq. 2 | S3 |
| LO6 | 說明排程如何讓誤差與 DTC 成正比 | Thm 3.1 的排程怎麼選？為什麼總離散化誤差 $\le3a_U\mathrm{DTC}(q)$？query 數是多少？ | $u=e^t-1$ 每步乘 $1+a_U$，使每步係數 $c\le3a_U$；$\mathrm{DTC}(q_t)$ 對 $t$ 不增，各步 DTC 落差telescoping 相加 $\le\mathrm{DTC}(q)$；取 $a_U=\varepsilon/(12\max\{\overline{\mathrm{DTC}},\varepsilon\})$，query 數 $O((1+\overline{\mathrm{DTC}}/\varepsilon)\log(edS/\varepsilon))$，需已知 $\overline{\mathrm{DTC}}$ | S4 |
| LO7 | 描述困難分佈與臨界點 | 隨機碼本分佈怎麼定義？取樣器知道什麼？臨界時間由什麼式子定義？ | $q^C=\mathrm{Unif}(C)$，$C$ 是 $\{\pm1\}^d$ 上均勻隨機的 $M=\lceil e^{\kappa d}\rceil$ 元子集；取樣器知道 $[\kappa_-,\kappa_+]$，不知道 $\kappa$ 與 $C$；$d\,I(t^\star)=\log M$（Eq. 22），masked 為 $m=k$ | S5 |
| LO8 | 比較三者的窗寬 | masked 與 uniform／Gaussian 的 critical window 各佔 noise 範圍多少？Fig. 4 的模擬顯示什麼？ | masked $O(\log d/d)$（Prop 8.6，$\eta=d^{-\Theta(1)}$ 時 $\ell=O(\log d)$）；uniform／Gaussian $w=A\sqrt{\Lambda/d}$，$\Lambda=\log(Cd/(\varepsilon\delta))$，在 $\varepsilon$ 常數、$\delta=d^{-\Theta(1)}$ 時為 $O(\sqrt{\log d/d})$（Prop 6.4、7.4，需 $\log d\le\Lambda\le cd$）；Fig. 4：recovery 由 0.2 到 0.8 的寬度，masked $\propto1/d$、uniform／Gaussian $\propto1/\sqrt d$ | S6 |
| LO9 | 解釋 $\sqrt d$ 的來源 | 為什麼 uniform 的窗寬是 $1/\sqrt d$ 而 masked 是 $1/d$？ | 論文給的是窗寬上界 $w\propto\sqrt{\Lambda/d}$（Prop 6.4，來自 Hoeffding 項 $e^{-d\Delta^2/2}$ 需要 $d\Delta^2\gtrsim\Lambda$）；Fig. 4(d) 與本頁求積顯示實際寬度約 $\propto1/\sqrt d$。啟發式解釋〔本頁推導〕：uniform 的帶噪樣本與原碼字的 Hamming 距離 $\sim\mathrm{Bin}(d,\beta_t)$，有 $\sqrt d$ 的漲落，使競爭碼字總權重的條件均值 $\mathbb E[Z\mid d_H]$ 在對數尺度上漲落 $\Theta(\sqrt d)$；masked 的已揭露數 $m$ 是固定觀測值，轉變只來自 $\mathrm{Poi}(2^{k-m})$，以 $m$ 計寬 $O(1)$ 個座標 | S6 |
| LO10 | 說出上界演算法的三階段 | Thm 6.1 的演算法怎麼做？各用多少 query？需要什麼 score 精度？ | 找窗：由高往低掃格距 $w/20$ 的時間格點，每點用 $R=2\lceil C\Lambda\rceil+1$ 個均勻樣本算 $\hat T_t$ 的中位數，低於 $-\frac{a+b}2dw$ 即停，$O(\Lambda/w)$ 次；窗內取樣：從 $\hat t+4w$ 以均勻分佈起始走到 $\hat t-4w$，$N=\lceil Cdw/\varepsilon^2\rceil$ 步；一步解碼：取估計 posterior mean 的符號，1 次；合計 $O(\sqrt{d\Lambda}/\varepsilon^2)$；需 $\sup_{I_0}\varepsilon_{\rm unif}\le e^{-B\Lambda}$ | S7 |
| LO11 | 說明 masked 下界的機制 | worst-case oracle 怎麼構造？為什麼它是合法的？為什麼需要 $\Omega(d/\log d)$ 次 query？ | 對 good $C$（Def 8.3）：$m\le k-\ell$ 一律回傳均勻（不論 consistent 與否）；$m>k-\ell$ 時 consistent assignment 回傳真實 marginal、inconsistent 回傳均勻。合法是因為誤差只在 consistent assignment 上計算（Lemma 2.8），且 $\mathbb E_C\,\varepsilon_{\rm mask}(\mathcal O_0;q^C,m)\le2^{m-k}$（Lemma 8.2），由 Markov 不等式得 good $C$ 的機率 $\ge1-2^{1-\ell}/\eta$（Eq. 98–99）。演算法對 $\mathcal O_0$ 的查詢序列與 $k,C$ 無關；與 $\mathcal O_{k,C}$ 出現差異只可能因為：某次 query 的 $m$ 落在 $\lvert m-k\rvert\le\ell$（機率 $\le(2\ell+1)Q/|\mathcal K^M_d|$）、$C$ 不是 good、或某次 $m>k+\ell$ 的 query 恰好與 $C$ 一致（$\le Q2^{-\ell}$）（Eq. 100）。$|\mathcal K^M_d|=\Theta(d)$、$\ell=O(\log d)$，所以 $Q\le c\rho^2d/\ell$ 時以機率 $\ge1-\rho$ 有 $\mathrm{TV}\ge1-\rho$ | S8 |
| LO12 | 界定結論範圍 | 這個分離能推論出哪些結論、不能推論出哪些結論？ | 可以：在隨機碼本分佈上，存在每 level 誤差 $\le\eta=d^{-\Theta(1)}$ 的 worst-case oracle，使 masked 對常數比例的 $(k,C)$ 需要 $\tilde\Omega(d)$ 次 query；uniform 在精度足夠（$\le e^{-B\Lambda}$）的 oracle 下 $\tilde O(\sqrt d)$ 次即可（Gaussian 同）。兩者相比 masked 至少多 $\tilde\Omega(\sqrt d)$ 倍，但兩邊的 oracle 類別與誤差要求不同；論文主張原因是窗寬。不可以：masked 對 exact oracle 或訓練出的網路一定慢、分離來自 commitment、uniform／Gaussian 普遍較好（Result 1 三者打平；Sec. 9） | S9 |

---

## 3. 頁面大綱

單欄閱讀（文字欄 680–720px），元件放在加寬圖版（最大 1080px）。章節編號以 JetBrains Mono 顯示（`01 · …`）。本文件的章節代號（S1a、S1b、S2…S11）在修訂後保持不變，顯示編號依下表（S1 拆成兩章後 S2 起顯示編號 +1）。標題與用語一律字面敘述，不用比喻。視覺記號：一條水平的「noise 軸」（左低右高），在 Hero、C1、C4、C5 重複出現，窗以 `--accent-soft` 色帶標示。章節標題用陳述句。每章依序：先定義符號與概念 → 元件 → 檢查題。

| # | id | 標題 | 內容（依序；粗體為正文必寫的 LO 句） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈dLLM 的平行度：DTC、critical window 與三種腐蝕過程的分離〉 | 副標一句：「這篇論文用 forward pass 次數比較 masked、uniform、Gaussian 三種 diffusion language model，證明 uniform 與 Gaussian 的步數可隨 DTC 縮放，並在一個分佈族上證明三者的平行度有差距。」三張事實卡（見下）；noise 軸示意（三條軸各畫一個窗：masked 窄、uniform／Gaussian 寬）；連結：arXiv、`../index.html#paper-5`、`dllm-parallelism-lineage-hub.html` | 1 螢幕 |
| S1a | `paradigms` | 01 · 三種腐蝕過程把序列變成噪聲 | 先定義 $d,\Sigma,S,X_0,X_t,q_t$；三個 kernel（**LO2 句**）；一句「autoregressive 可視為由右至左 erasure 的特例」（Sec. 1）；C1。延伸：`dllm-three-corruption-paradigms.html` | 300–350 字 + C1 |
| S1b | `oracle` | 02 · 一次 forward pass 回傳所有位置的 posterior marginal | 定義 posterior marginal 與 score oracle、query complexity（**LO1 句**，含 masked 只給未揭露座標、只依賴 $(I,x_I)$ 的假設、$m$ 是 masked 的 noise level）；上界/下界的意思；基準一句：不利用結構時 uniform 用 $O(d)$ 次 query 可做到零離散化誤差（Chen & Ying，筆記），本文目標是 $o(d)$；可收合符號表。延伸：`dllm-discrete-sampling-theory.html` | 250–300 字 |
| S2 | `product` | 03 · 平行更新丟掉的是條件相關 | 前置只複習 $H$、KL、條件熵各一句（$I$、chain rule、TV、Pinsker 在首次使用處一句帶過）；定義 TC、DTC，各附 $d=3$ 算式（TC：複製 $3\log2-\log2=2\log2$；DTC：同位 $2\log2-3\cdot0=2\log2$），再給一般 $d$ 的複製／同位結論（**LO3 句**）；條件版本；product reverse kernel（Eq. 12），此處定義「從 $t$ 退到 $s=t-h$」；Lemma 3.8 的實際分解與上界分開寫，「離散化項 = $\mathrm{TC}(X_s\mid X_t)$」（**LO4 句**）；Lemma 2.10 與 DTC 單調（證明可收合）。延伸：`dllm-masked-schedule-dtc-theory.html` | 550–650 字 |
| S3 | `rdpi` | 04 · reverse DPI 把單步誤差換成 DTC 的下降量 | DPI 是什麼（加噪不增加 KL）→ reverse DPI 的意思（少量加噪不會讓 KL 縮太多）；Prop 3.3、Lemma 3.4（Eq. 4 可收合）、Cor 3.5（附 Fig. 5 風格的靜態示意：兩列 $d$ 格，灰＝條件、白＝邊際化、弧線連接兩個引數）、Lemma 3.2（**LO5 句**）；C2；一句「上界成立但寬鬆」 | 450–550 字 + C2 |
| S4 | `telescope` | 05 · 幾何步長讓各步的 DTC 落差成為逐項抵消的和（telescoping sum） | 先定義目標精度 $\varepsilon$（$\mathrm{KL}(q\|\hat q)\le\varepsilon$）與輸出分佈 $\hat q$；排程與 $c\le3a_U$ 推導（可收合）、誤差帳、Thm 3.1（**LO6 句**）；Bregman projection callout；C3；Gaussian 正文一句：「Gaussian 用同一論證，reverse DPI 的常數與排程變數不同，結論同形式（Thm 4.1）」；Lemma 4.3、$c_u(v)$、$F_G$ 放 `<details>` | 400–500 字 + C3 |
| S5 | `codebook` | 06 · 困難分佈是隨機碼本上的均勻分佈 | Def 5.1 與取樣器的已知資訊；$I_{\rm unif}$、$I_{\rm gauss}$ 的定義（一個 bit 經腐蝕後保留的資訊）；Eq. 22（**LO7 句**）；$\kappa=0.2$ 的臨界值（本頁推導）；DTC $=\Theta(d)$ 與「Result 2 超過 DTC 界」；random coding 直覺一句：$I<\kappa$ 時帶噪樣本附近有很多碼字，$I>\kappa$ 時只剩原碼字。延伸：`dllm-score-oracle-lower-bounds.html` | 350–450 字 |
| S6 | `window` | 07 · critical window 在 masked 寬 1/d，在 uniform 與 Gaussian 寬 1/√d | 先定義 Result 2 的 $\varepsilon$（TV 精度）、$\delta$（失敗機率）、$\Lambda$、$A$（步驟 7 第一條）；recovery probability 定義與 App. C 的 Poissonization；Prop 6.4、7.4、8.6（**LO8 句**，含成立條件）；masked 閉式與 uniform 的 $\mathbb E[Z\mid d_H]$ 啟發式推導（**LO9 句**，標〔本頁推導〕與「啟發式」，先寫論文的窗寬上界再寫 Fig. 4(d)／求積的實際縮放）；C4；一句說明 Fig. 4(c) Gaussian 曲線本頁不重算的原因。延伸：`dllm-critical-windows-speciation.html` | 600–700 字 + C4 |
| S7 | `upper` | 08 · 上界演算法先找窗、再在窗內取樣、最後一步解碼 | $T_t$ 的定義與直覺；Prop 6.5、6.9；Alg. 1、Alg. 2 逐行對照表（沿用參考頁的演算法格線）；query 計數與 Thm 6.1 條件（**LO10 句**）；Gaussian 差異一段；常數未指定的 callout | 500–600 字 + 2 個演算法表 |
| S8 | `lower` | 09 · worst-case oracle 使 masked 取樣器必須搜尋窗的位置 | prior、Lemma 8.2（$\mathbb E_C$ 敘述＋Markov 得 good $C$）、Def 8.3（單側：$m\le k-\ell$ 全均勻，$m>k-\ell$ 的 consistent 為真實）、Lemma 2.8 的作用、Eq. 100 三項各自的事件、Thm 8.1（**LO11 句**）；Thm A.1 與 [XP26]；一句區分 [CCL26] 的 MDS code $\Omega(d)$ 下界（不同分佈族）；C5；漸近性 callout（自選常數下 $d\le2^{13}$ 時 $Q_{\max}<1$，保證無意義；$d\le256$ 時 $n_M\le n_U$，因為 log 因子與自選常數在小 $d$ 占優）。延伸：`dllm-parallelism-separation-result.html` | 500–600 字 + C5 |
| S9 | `scope` | 10 · 結論的適用範圍與常見誤讀 | 步驟 10 全部 + 1.5 的 7 條常見誤讀（**LO12 句**）；誤差度量對照小表（定理／度量／步數對 $\varepsilon$／oracle 誤差要求／保證對象） | 350–450 字 + 表 |
| S10 | `history` | 11 · 研究脈絡 | 時間線（6.4）+ 一段文字：連續 diffusion（DDPM）→ 離散 diffusion 三條路線（D3PM、CDCD、SEDD、MDLM、LLaDA）→ 取樣理論（CTMC、uniformization 的 $O(d)$ 基準）→ masked 的 DTC 排程（CCL、LZ）→ critical window → Xun & Price 下界 → 本文 | 時間線 + 140 字 |
| S11 | `next` | 12 · 延伸閱讀 | 筆記依 hub 的閱讀順序；站內 `eps-prediction-ddpm-score.html`、`continuous-flow-lm-overview.html`（Gaussian 路線，筆記 `dllm-three-corruption-paradigms` 已連結）；arXiv | 連結清單 |
| — | — | 頁尾 | 引用、arXiv；`#pm-bar` | — |

Hero 事實卡（不用未定義術語）：
1. 「模型一次 forward pass 對所有位置同時給出『這個位置原本是什麼』的機率。平行取樣把這些機率當成互相獨立來抽，每一步的誤差等於被忽略的位置間相關性。」
2. 「uniform 與 Gaussian diffusion 的步數可以只隨分佈的內在複雜度（dual total correlation）成長，與 masked diffusion 先前的結果相同。」
3. 「在一個隨機碼本分佈上，模型夠準時 uniform 與 Gaussian 約 √d 次 forward pass 就夠；masked diffusion 則存在誤差很小的模型，使任何取樣方法至少需要約 d 次（相差 log 因子）。論文把差距歸因於『資訊突然出現的 noise 範圍』在 masked 中窄得多。」

桌機（≥1024px）左側 sticky 章節目錄，<1024px 改為頂部橫列（沿用參考頁）。

---

## 4. 互動元件規格

### 4.0 共用

- 二元字母表 $\{0,1\}$（C2、C3，與 $\{\pm1\}$ 等價）或 $\{\pm1\}$（C1、C4）。log 為自然對數，讀數單位 nats。
- PRNG：`mulberry32(seed)`；`gauss()` 用 Box–Muller，每次取兩個均勻數 $u_1,u_2$，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$（只用 cos 分支）。
- 所有 live region（`aria-live="polite"`）只在 `change` 事件或輸入停止 500ms 後更新；重算超過 50ms 的元件用 `setTimeout` 分段，期間讀數框顯示「計算中…」，完成後一次重畫；新輸入到來時以遞增的 `jobId` 作廢舊工作。
- 曲線一律以線型＋直接文字標籤區分：uniform 實線（`--c-unif`）、masked 虛線（`--c-mask`）、Gaussian 點線（`--ink-1`）；標記形狀 uniform ●、masked ■。實心／外框長條一律在長條旁加文字標籤（不只靠填色區分）。
- 滑桿：數值標籤在 `input` 事件即時更新（不經 live region）；重算只在 `change` 觸發。
- 動態：全域 `@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}`；Hero noise 軸、滑桿重畫、「計算中…」都不用過場動畫以外的動態；C5 掃描在 reduced motion 時一次完成。
- SVG 文字：SVG 內一律用 Unicode 字元與 `<tspan>`（上標以 `baseline-shift="super"`、`font-size` 0.75em），例如「I = κ」「∝ 1/√d」「t⋆」；需要 KaTeX 的標籤只放在圖外的 HTML 圖例，每次動態更新該容器後呼叫 `renderMathInElement(container)`。
- 窄螢幕（390px）：所有 `<table>`（S9 對照表、C3 讀數表、驗收用寬度表）與演算法逐行格線外包 `<div style="overflow-x:auto">`；SVG 一律 `viewBox`＋`width:100%`，並設最小高度（主圖 220px、小圖 160px）；C2 左右兩圖在 <600px 改上下堆疊；時間線 rail 沿用參考頁的窄螢幕樣式。
- 暗色：每個新元件於建置時做暗色截圖；圖形元素對背景對比 ≥3:1（以 selfTest 計算 `--c-unif`、`--c-mask`、`--ink-1`、`--accent` 對 `--surface` 的對比，見 6.5）。

### C1 · 三種腐蝕過程（S1a）

- **教的概念**：同一個 $t$ 下三種過程的樣子；uniform 與 masked 動到同一批位置的機率都是 $1-e^{-t}$，但 uniform 重抽的位置只有 $(S-1)/S$ 真的改變（本元件 $S=2$，即一半；整體改變比例 $(S-1)\beta_t$，binary 時 $=\beta_t$）。$I(t)$ 不在 C1 出現（到 S5 才定義，讀數放 C4 小圖）。
- **輸入**：滑桿 $t\in[0,3]$，步長 0.01，預設 0.5；按鈕「重新抽樣」（seed +1，初始 seed = 11）。
- **計算**：$d=32$。依序對 $i=0..31$ 抽 $x_i$（$<0.5\to-1$，否則 $+1$）、$u_i$、$r_i$（同規則）、$g_i$（gauss）。觸及集合 $\{i:u_i<1-e^{-t}\}$。uniform：觸及位置取 $r_i$；masked：觸及位置顯示 $\star$；Gaussian：$e^{-t}x_i+\sigma_tg_i$（$\sigma_t=\sqrt{-\mathrm{expm1}(-2t)}$，$t=0$ 時為 0）。
- **顯示**：四列 32 格。第 1 列 $X_0$（＋／−字元＋底色深淺）；第 2 列 uniform（改變的格加粗外框，未改變但被重抽的格加細虛框）；第 3 列 masked（$\star$ 格為 `--surface-2` 底與 ⋆ 字元）；第 4 列 Gaussian（數值條：高度 ∝ 值，範圍 $[-3,3]$，超出截斷並加 ▲▼ 標記；格下印一位小數）。<600px 時改為 2 行 × 16 格。讀數：觸及比例（實際／期望 $1-e^{-t}$）、uniform 實際改變比例（期望 $\beta_t=(1-e^{-t})/2$）。
- **觀察**：$t=0$ 三列都等於 $X_0$；$t$ 增大時 masked 與 uniform 動到同一批格子；$t=0.5$ 時期望觸及比例 0.3935、期望改變比例 0.1967。
- **無障礙**：格子 `aria-hidden="true"`；每列容器一句摘要 `aria-label`（例「uniform 列：32 格中 11 格被重抽，6 格改變」）；圖下附可收合的文字表（`<details>`「以文字顯示 32 個位置」，四欄 $X_0$／uniform／masked／Gaussian）。

### C2 · 單步：條件 TC 與 reverse DPI 上界（S3）

- **教的概念**：Lemma 3.2 的兩邊可以在小例子上精確算出；上界成立且寬鬆；$s$ 小或 $h$ 大時係數 $c_s(h)$ 爆增。
- **輸入**：分佈 chip（`codebook` 碼本〔預設〕、`copy` 複製、`parity` 同位、`indep` 獨立）；滑桿 $s$（對數，$[0.01,3]$，預設 0.5）；滑桿 $h$（對數，$[0.01,1.5]$，預設 0.2）。
- **分佈**（$d=6$，$N=64$ 個狀態，狀態以整數 $x\in[0,63]$ 的 bit 表示）：copy：$q(0)=q(63)=\tfrac12$；parity：偶數個 1 的 32 個字串均勻；codebook：$\{5,10,19,28,38,45,51,60\}$ 均勻（〔自選〕的小碼本，示意 S5 的結構）；indep：每 bit 為 1 的機率 0.8、獨立。
- **計算**（與 `w23_exact.py` 同算法）：
  - $(K_t)^{\otimes6}(y\mid x)=w_t[d_H(x,y)]$，$w_t[k]=(a+b)^{6-k}b^k$，$a=e^{-t}$、$b=(1-a)/2$。預先建 `HAM`（Uint8Array 4096）。
  - $q_s=qK_s$；joint $J(x,y)=q_s(x)w_h[d_H(x,y)]$；$q_t=\sum_xJ(x,\cdot)$。
  - $\mathrm{TC}(X_s\mid X_t)=\sum_i\big[H((X_s)_i,X_t)-H(X_t)\big]-\big[H(J)-H(q_t)\big]$。
  - $\mathrm{DTC}(p)=H(p)-\sum_i\big[H(p)-H(p_{-i})\big]$，$p_{-i}(x)=p(x)+p(x\oplus2^i)$（只對 bit $i$ 為 0 的 $x$）。
  - $c_s(h)=\mathrm{expm1}(2h)/(-\mathrm{expm1}(-s))$；上界 $=c_s(h)(\mathrm{DTC}(q_s)-\mathrm{DTC}(q_t))$。
  - $H(p)=-\sum p\log p$，$p\le0$ 的項略過。
- **數值保護**（相對門檻，$H_{\max}=6\log2$）：TC、DTC、DTC 落差若 $|x|<10^{-9}H_{\max}$ 視為 0；比值只在上界 $>10^{-9}$ 時顯示，否則顯示「—（兩者皆接近 0）」；indep 預期 TC = DTC = 0。
- **顯示**：左圖 $\mathrm{DTC}(q_t)$ 對 $t$（橫軸對數 $[0.01,6]$，80 點；縱軸 $[0,\mathrm{DTC}(q)\cdot1.05]$，indep 時固定 $[0,0.1]$），在 $s$ 與 $t=s+h$ 畫垂直線並把 DTC 落差標成雙箭頭「DTC 落差」。右圖兩條水平長條（同一線性尺度，以上界為滿格）：「TC(X_s | X_t)（精確）」實心、「上界 c_s(h)·DTC 落差」外框；長條旁印名稱與數值。<600px 時左右兩圖改上下堆疊。讀數表：$c_s(h)$、$\mathrm{DTC}(q_s)$、$\mathrm{DTC}(q_t)$、落差、TC、上界、TC／上界。
- **觀察**：預設（codebook，$s=0.5,h=0.2$）TC = 0.024176 ≤ 上界 0.147974（比值 0.163）；$s=0.05,h=0.5$ 時 $c=35.23$、上界 44.16 遠大於 TC 0.798；indep 兩者皆 0（獨立時平行更新沒有誤差）。
- **無障礙**：滑桿 `aria-valuetext`（「s = 0.50」）；左圖 `aria-label` 描述目前 $s,t$ 與落差。

### C3 · 整個排程：telescoping 相加與實際誤差（S4）

- **教的概念**：Thm 3.1 的排程；每步係數 $\le3a$；各步 TC 的和、DTC telescoping 上界、$3a\,\mathrm{DTC}(q)$ 三者的大小關係；以及取樣器實際的 $\mathrm{KL}(q\|\hat q)$ 比這些都小。
- **輸入**：分佈 chip（同 C2，共用狀態但各自選取；預設 codebook）；滑桿 $a$（對數，$[0.02,1]$，預設 0.1）。端點固定 $\delta_U=0.001$、$T=6$〔自選；論文的 $\delta_U,T_U$ 依 $\varepsilon,d,S$ 而定，頁面寫明；$\delta_U$ 是末端時間，不是失敗機率 $\delta$〕。
- **計算**（與 `w23_exact.py` `run_schedule` 同算法）：
  - $u_0=\mathrm{expm1}(\delta_U)$、$U=\mathrm{expm1}(T)$、$u_{j+1}=\min((1+a)u_j,U)$ 直到 $=U$；$t_j=\mathrm{log1p}(u_j)$，$N=$ 步數。
  - 每步 $j=N-1..0$（$s=t_j$、$t=t_{j+1}$）：C2 的 `tcCond` 得 TC、$q_s$、$q_t$；$c_j=c_s(h)$；累加 $\sum\mathrm{TC}$、$\sum c_j(\mathrm{DTC}(q_s)-\mathrm{DTC}(q_t))$；posterior marginal $m(y,i)=P((X_s)_i=1\mid X_t=y)=\sum_{x:\,x_i=1}J(x,y)/q_t(y)$；取樣器分佈 $p\leftarrow pR$，$R(y,x)=\prod_im(y,i)^{x_i}(1-m(y,i))^{1-x_i}$；$p$ 初值為均勻。
  - 結尾 $\hat q=pK_{t_0}$；init $=\sum q_T\log(64q_T)$（論文界 $e^{-T}\cdot6\log2$）；end $=H(qK_{t_0})-H(q)$（論文界 $6\delta_U\log(2e/\delta_U)$）；path $=$ init $+\sum\mathrm{TC}+$ end；$\mathrm{KL}(q\|\hat q)=\sum_{q>0}q\log(q/\hat q)$。
  - 成本：每步約 6 萬次運算，$a=0.02$ 時 $N=652$；每 50 步 `setTimeout` 一次，重算只在 `change` 時觸發。
- **數值保護**：$q_t(y)$ 對所有 $y$ 都 $>0$（$t>0$ 時 kernel 全正），但仍以 `qt[y] > 1e-300` 保護除法；$m$ 夾在 $[0,1]$。
- **顯示**：上圖橫軸 $t$（對數 $[10^{-3},6]$）：$\mathrm{DTC}(q_t)$ 曲線（重用 C2 的曲線函式），軸下畫格點刻度 $t_j$（$N>200$ 時只畫每第 $\lceil N/200\rceil$ 個並註明）。下圖同一橫軸：每步的 TC（實心柱）與 $c_j\cdot$DTC 落差（外框柱），圖內直接文字圖例「實心＝TC（實際）」「外框＝c·DTC 落差（上界）」，縱軸對數 $[10^{-8},1]$，低於 $10^{-8}$ 不畫。讀數表（兩欄：數值、意義）：$N$；$\max_jc_j$（並印「≤ 3a = …」）；$\sum\mathrm{TC}$；$\sum c_j\Delta\mathrm{DTC}$；$3a\,\mathrm{DTC}(q)$；init 與其界；end 與其界；path；實際 KL。一行文字：「KL(q‖q̂) ≤ 路徑 KL = init + ΣTC + end ≤ init + Σc·ΔDTC + end」。
- **觀察**：$a$ 每減為 1/3，$\sum\mathrm{TC}$ 與上界約減為 1/3，$N$ 約乘 3；末端項不隨 $a$ 改變；實際 KL 遠小於上界（codebook，$a=0.1$：KL 0.0059、$\sum$TC 0.0381、上界 0.348、$3a\mathrm{DTC}$ 0.520）。
- **無障礙**：完成時宣告一次「N = 136 步，實際 KL = 0.0059」。

### C4 · critical window：recovery probability 與窗寬縮放（S6）

- **教的概念**：Fig. 4(a)(b)(d) 的重算；masked 窄窗 $\propto1/d$、uniform 寬窗 $\propto1/\sqrt d$；三條資訊量曲線與臨界點。
- **輸入**：滑桿 $\kappa\in[0.05,0.6]$，步長 0.01，預設 0.2；滑桿 $d\in\{64,128,256,512,1024,2048,4096\}$，預設 256。
- **共用函式**：
  - `lf[n]` = log n!（Float64Array 到 4096，累加 log）。$h_2(p)=-p\log p-(1-p)\,\mathrm{log1p}(-p)$。$I_{\rm unif}(t)=\log2-h_2(\beta_t)$。
  - $I_{\rm gauss}(t)=u^2-\mathbb E_W\log\cosh(u^2+uW)$、$u=e^{-t}/\sigma_t$（Lemma 7.2 證明）：Simpson，$w\in[-10,10]$，401 點；$\log\cosh x=|x|+\mathrm{log1p}(e^{-2|x|})-\log2$。$t<10^{-4}$ 不計算（橫軸下限 0.01）。
  - $t$ 由 $I$ 反解（uniform）：二分 $\beta\in[10^{-15},0.5-10^{-15}]$ 解 $\log2-h_2(\beta)=I$（80 次），$t=-\mathrm{log1p}(-2\beta)$。臨界點：$t^\star_{\rm unif}$ 用同一反解；$t^\star_{\rm gauss}$ 在 $[10^{-4},10]$ 二分 80 次。
  - masked：$f(\lambda)=-\mathrm{expm1}(-\lambda)/\lambda$（$\lambda<10^{-12}$ 時為 1），$\lambda=e^{d(\kappa-I)}$（指數 $>700$ 時 $f=0$）。$\ln(\lambda_{0.2}/\lambda_{0.8})=2.3698$ 在載入時以二分求出，masked 寬度 $=2.3698/d$。
  - uniform（〔本頁推導〕的確定性求積，與 App. C 的 Poisson 模型完全相同，無 Monte Carlo 誤差）：$\log\lambda_j=\kappa d+\mathrm{lf}[d]-\mathrm{lf}[j]-\mathrm{lf}[d-j]-d\log2$（以 $\log M=\kappa d$，忽略上取整）。由 $1/(1+Z)=\int_0^\infty e^{-s(1+Z)}ds$ 與 Poisson 過程的 Laplace 泛函：$P[\text{recover}\mid d_H]=\int_0^\infty e^{-s}\exp(-G(s,d_H))\,ds$，$G=\sum_j\lambda_j(1-e^{-s\,\tanh(t/2)^{j-d_H}})$；recovery $=\sum_{d_H}\mathrm{Bin}(d,\beta_t)(d_H)\cdot P[\cdot\mid d_H]$。
  - 求 $G$：$L=\log\tanh(t/2)<0$、$v=\log s$、$x_j=v+(j-d_H)L$（隨 $j$ 遞減）。$x_j>\log40$ 的項取 $\lambda_j$（前綴和 `logP0`），$x_j<\log10^{-8}$ 的項取 $\lambda_je^{x_j}$（後綴和 `logS1[j]=log Σ_{j'≥j} λ_{j'} e^{j'L}`），中間約 $22/|L|$ 項逐項算 $\lambda_j\cdot(-\mathrm{expm1}(-e^{x_j}))$。所有項以 log 值相加；任一 log 值 $>700$ 時 $G=\infty$（積分項為 0）。索引：$j_a=\mathrm{clamp}(\lceil(v-d_HL-\log40)/(-L)\rceil,0,d+1)$、$j_b=\mathrm{clamp}(\lfloor(v-d_HL-\log10^{-8})/(-L)\rfloor+1,j_a,d+1)$。
  - $s$ 積分：$v\in[-36,4]$，步長 0.25，Simpson 161 點，被積函數 $e^{v-e^v-G}$；除以同一網格上 $\int e^{v-e^v}dv$ 以消去求積偏差，結果夾在 $[0,1]$。
  - $d_H$ 求和：範圍 $[\max(0,\lfloor d\beta-7\sigma-2\rfloor),\min(d,\lceil d\beta+7\sigma+2\rceil)]$，$\sigma=\sqrt{d\beta(1-\beta)}$。$P[\cdot\mid d_H]$ 對 $d_H$ 遞減（$G$ 隨 $d_H$ 增加），以二分找第一個 $<1-10^{-10}$ 與第一個 $<10^{-10}$ 的位置 $b_1,b_2$：$\sum_{<b_1}\mathrm{pmf}+\sum_{b_1}^{b_2-1}\mathrm{pmf}\cdot P$；$P$ 以 Map 快取（每個 $t$ 約 40 次積分）。
  - 窗寬：recovery 對 $I$ 遞增；在 $I\in[I_{\rm lo},I_{\rm hi}]=[\max(10^{-4},\kappa-0.2),\min(\log2-10^{-4},\kappa+0.2)]$ 二分 26 次找 0.8 與 0.2 的位置。先檢查端點：若 $\mathrm{rec}(I_{\rm lo})>0.2$ 或 $\mathrm{rec}(I_{\rm hi})<0.8$，回傳 `NaN`，讀數顯示「窗超出範圍」；寬度圖不畫該點，斜率擬合略過 `NaN`，有效點 $<3$ 時不顯示斜率。
- **計算排程**：主圖曲線（目前 $d$，$I$ 網格 $[\kappa-0.1,\kappa+0.1]\cap[0.005,\log2-10^{-6}]$，81 點）於 $\kappa$ 或 $d$ 的 `change` 時重算；寬度圖（7 個 $d$）只在 $\kappa$ 的 `change` 時重算並以 $\kappa$ 為 key 快取。分段：每個 $I$ 點或每次二分一個 `setTimeout`。預設 $\kappa=0.2$ 的寬度在載入後背景計算。
- **顯示**：
  - 主圖「recovery probability」：橫軸 $I$（nats），縱軸 $[0,1]$；masked 曲線＋（$d\le256$ 時）整數 $m$ 的 ■ 點；uniform 曲線；垂直虛線 $I=\kappa$ 標「I = κ」；水平細線 0.2、0.8；兩條曲線各以水平雙箭頭標出 0.2→0.8 的寬度與數值。
  - 寬度圖：log–log，橫軸 $d$（64–4096），縱軸 $[10^{-4},0.3]$；masked ■、uniform ●；兩條參考線斜率 $-1$、$-1/2$，標「∝ 1/d」「∝ 1/√d」；讀數印出 uniform 擬合斜率（最小平方法於 log–log）。
  - 小圖：$I_{\rm unif}(t)$、$I_{\rm gauss}(t)$ 對 $t\in[0.01,2]$，與水平線 $I=\kappa$ 的交點標 $t^\star$；文字讀數 $\beta^\star$、$t^\star_{\rm unif}$、$t^\star_{\rm gauss}$、masked $m^\star=\kappa d/\log2$；另印 $t=0.5$ 的參考值 $I_{\rm unif}=0.1973$、$I_{\rm gauss}=0.2272$（原 C1 讀數移到這裡）。
  - 圖下兩行：「masked 曲線為 App. C 模型的閉式；uniform 曲線為同一模型的確定性求積（本頁推導），與 App. C 的 Monte Carlo 一致（見驗收）。」「Gaussian 的 recovery 曲線（Fig. 4c）在論文中用 saddlepoint 近似與分箱計算，細節未給，本頁不重算。」
- **觀察**：$d$ 增大時 masked 曲線變陡的速度比 uniform 快；寬度圖兩組點分別貼著斜率 $-1$ 與 $-1/2$（$\kappa=0.2$ 擬合斜率 $-0.478$）；在連續理想化下（$\log M=\kappa d$ 忽略 $\lceil\cdot\rceil$、$m$ 取實數），masked 在 $I=\kappa$ 的 recovery 恆為 $1-e^{-1}=0.632$；頁面寫明這個前提（整數 $m$ 的 ■ 點不一定落在 $I=\kappa$）。
- **無障礙**：主圖 `aria-label` 寫目前 $d$、兩個寬度；寬度表以 `<table>` 提供同一組數字（可收合）。

### C5 · 不知道窗位置時的搜尋：下界的計數（S8）

- **教的概念**：Eq. 100 第一項的機制：查詢若沒落在窗內，回答與 $\mathcal O_0$ 相同，不帶任何資訊；需要的 query 數 ≈ 候選位置數 ÷ 窗寬；masked 與 uniform 的比值隨 $\sqrt d$ 成長（log 因子在小 $d$ 時占優）。
- **常數**（全部〔自選〕，頁面寫明）：Eq. 96 的 $C=1$；$\kappa_-=0.1$、$\kappa_+=0.5$；$\rho=0.2$；uniform 的候選窗數取 $n_U=\lceil\sqrt{d/\Lambda}\rceil$、$\Lambda=\log(d/(\eta\rho^2))$（Eq. 107、109 的階，$C_\Lambda=A=c_U=1$）。
- **輸入**：滑桿 $d=2^p$，$p\in\{10,\dots,20\}$，預設 $p=16$（小 $d$ 的反轉改用 S8 callout 說明，不放進滑桿範圍）；分段按鈕 $\eta\in\{0.09,10^{-3},10^{-6}\}$，預設 $10^{-3}$；查詢段號滑桿（每條帶各一個，整數 $1..n$，`step=1`，`aria-valuetext`「第 k/n 段」）與「查詢」按鈕；「等距掃描」「重新抽窗位置」。
- **計算**：$|\mathcal K^M_d|=\lfloor\kappa_+d/\log2\rfloor-\lceil\kappa_-d/\log2\rceil+1$；$\ell=\lceil\log_2(d/(\eta\rho^2))\rceil$；masked 窗數 $n_M=\lceil|\mathcal K^M_d|/(2\ell+1)\rceil$。Eq. 100 三項於目前的已用查詢數 $Q$：$(2\ell+1)Q/|\mathcal K^M_d|$、$2^{1-\ell}/\eta$、$Q2^{-\ell}$；以及第一項 $\le\rho^2/4$ 所允許的 $Q_{\max}=\rho^2|\mathcal K^M_d|/(4(2\ell+1))$。
- **遊戲**：兩條水平帶（masked、uniform），各分成 $n$ 個等寬段；各自以 `mulberry32`（seed 每次「重新抽窗位置」+1，初始 5）均勻選一段作為窗的位置。查詢段號 $k$：未命中的段塗 `--surface-2` 並標「回答 = 均勻（與 O₀ 相同）」；命中時該段塗 `--accent` 並標「回答 = 真實 marginal」。「等距掃描」：由左至右固定 32 個 tick（每 tick 250ms，整條帶 ≤8 秒），每 tick 查 $\lceil n/32\rceil$ 段，直到命中（reduced-motion 時立即完成）；$d=2^{20}$ 時 masked 每 tick 267 段。讀數：已用查詢數、$n$、$|\mathcal K^M_d|$、$\ell$、Eq. 100 三項、$Q_{\max}$；$Q_{\max}<1$ 時讀數旁標「保證無意義（自選常數下 $Q_{\max}<1$）」。$n>400$ 時段不畫分隔線，只畫已查詢段。
- **觀察**：預設 $d=65536,\eta=10^{-3}$：$n_M=601$、$n_U=56$（比值 10.7），$Q_{\max}=6.0$；$d=2^{20}$：8523 對 210（比值 40.6）；$p=10$ 時 12 對 8，$Q_{\max}=0.116$ 標「保證無意義」。S8 callout 另寫：$d\le256$ 時 $n_M\le n_U$（log 因子與自選常數造成），下界是漸近敘述。
- **無障礙**：查詢以滑桿＋按鈕完成（不需點擊帶狀圖）；每次查詢結果以 live region 宣告一次。

---

## 5. 驗收數值表

全部由 `verify/` 腳本產生（同算法）。容許誤差：C1 期望比例（$1-e^{-t}$、$\beta_t$）$\pm10^{-4}$；C2、C3 絕對 $10^{-6}$（KL 與和）；C4 recovery $\pm5\times10^{-4}$、寬度相對 1%、臨界點 $\pm10^{-4}$、$I_{\rm unif}$／$I_{\rm gauss}$ 各列 $\pm10^{-5}$；C5 整數完全相同。

**分佈的 TC／DTC（$d=6$，nats）**：copy TC 3.4657、DTC 0.6931；parity TC 0.6931、DTC 3.4657；codebook $H$ 2.0794、TC 2.0479、DTC 1.7329；indep 0、0。

**C2（單步）**

| 分佈 | $s$ | $h$ | $c_s(h)$ | TC | DTC 落差 | 上界 | 比值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| codebook | 0.5 | 0.2 | 1.2500 | 0.024176 | 0.118382 | 0.147974 | 0.1634 |
| codebook | 0.05 | 0.5 | 35.2319 | 0.798092 | 1.253516 | 44.163802 | 0.0181 |
| codebook | 1.0 | 0.5 | 2.7183 | 0.007402 | 0.017124 | 0.046549 | 0.1590 |
| copy | 0.5 | 0.2 | 1.2500 | 0.033732 | 0.142931 | 0.178660 | 0.1888 |
| parity | 0.1 | 0.05 | 1.1052 | 0.018617 | 0.370788 | 0.409784 | 0.0454 |
| parity | 0.5 | 0.2 | 1.2500 | 0.000796 | 0.005637 | 0.007046 | 0.1129 |
| indep | 任意 | 任意 | — | 0 | 0 | 0 | — |

**C3（排程，$\delta_U=0.001$、$T=6$）**

| 分佈 | $a$ | $N$ | $\max c_j$ | ΣTC | Σc·ΔDTC | 3a·DTC | end | path | KL(q‖q̂) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| codebook | 1.0 | 19 | 2.9924 | 0.277032 | 3.707104 | 5.198604 | 0.023278 | 0.300310 | 0.012022 |
| codebook | 0.3 | 50 | 0.6897 | 0.105059 | 1.061592 | 1.559581 | 0.023278 | 0.128337 | 0.006900 |
| codebook | 0.1 | 136 | 0.2100 | 0.038090 | 0.348041 | 0.519860 | 0.023278 | 0.061368 | 0.005906 |
| codebook | 0.03 | 437 | 0.0609 | 0.011799 | 0.103751 | 0.155958 | 0.023278 | 0.035077 | 0.005613 |
| copy | 0.1 | 136 | 0.2100 | 0.030080 | 0.141373 | 0.207944 | 0.025791 | 0.055870 | 0.006134 |
| parity | 0.1 | 136 | 0.2100 | 0.025194 | 0.674868 | 1.039721 | 0.020371 | 0.045565 | 0.007476 |
| indep | 0.1 | 136 | 0.2100 | 0 | 0 | 0 | 0.002492 | 0.002499 | 0.000007 |

init：codebook $1.92\times10^{-7}$、copy $2.83\times10^{-10}$、parity $\approx0$、indep $6.64\times10^{-6}$，皆 $\le e^{-6}\cdot6\log2=1.03\times10^{-2}$。

**C4（資訊量與臨界點）**

| $\kappa$ | $\beta^\star$ | $t^\star_{\rm unif}$ | $t^\star_{\rm gauss}$ | $m^\star/d$ |
| --- | --- | --- | --- | --- |
| 0.1 | 0.2802 | 0.8219 | 0.8534 | 0.1443 |
| 0.2 | 0.1948 | 0.4937 | 0.5521 | 0.2885 |
| 0.3 | 0.1336 | 0.3108 | 0.3904 | 0.4328 |
| 0.5 | 0.0482 | 0.1013 | 0.2028 | 0.7213 |

$t=0.25/0.5/1/2$：$I_{\rm unif}$ = 0.34538／0.19730／0.06928／0.00919；$I_{\rm gauss}$ = 0.43845／0.22716／0.07268／0.00924。

**C4（uniform recovery，$\kappa=0.2$）**：$d=64$：$I=0.15/0.18/0.20/0.22/0.25$ → 0.2545／0.4407／0.5660／0.6797／0.8151；$d=256$：0.0591／0.2966／0.5357／0.7525／0.9374；$d=1024$：0.0006／0.1203／0.5183／0.8856／0.9980。$d=256$：$I=0.10$ → $8.12\times10^{-5}$，$I=0.30$ → 0.99810。

**C4（窗寬，$\kappa=0.2$）**

| $d$ | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniform | 0.105902 | 0.076140 | 0.054900 | 0.039519 | 0.028355 | 0.020278 | 0.014461 |
| masked | 0.037029 | 0.018514 | 0.009257 | 0.004629 | 0.002314 | 0.001157 | 0.000579 |

uniform 擬合斜率 $-0.478$；$\text{寬度}\times\sqrt d$ 由 0.847 增到 0.926。$d=256$：$\kappa=0.1$ 寬 0.040774、$\kappa=0.4$ 寬 0.065392。masked：$\lambda_{0.8}=0.4642$、$\lambda_{0.2}=4.9651$、$\ln$ 比 2.3698、以 $m$ 計 3.4190。

**Monte Carlo 交叉檢查**（`w4_window.py --mc`，App. C 方法，20000 樣本，只作為求積正確性的證據，不放頁面 selfTest）：$d=64$，$I=0.17/0.19/0.20/0.21/0.23$：求積 0.3769／0.5043／0.5660／0.6248／0.7299，MC 0.3788／0.5069／0.5654／0.6278／0.7322（SE ≈ 0.003）；$d=256$：求積 0.1943／0.4138／0.5357／0.6514／0.8340，MC 0.1953／0.4116／0.5345／0.6510／0.8319。

**C5（$\eta=10^{-3}$、$\rho=0.2$）**

| $d$ | 64 | 256 | 1024 | 4096 | 16384 | 65536 | 262144 | 1048576 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\lvert\mathcal K^M_d\rvert$ | 37 | 148 | 591 | 2364 | 9455 | 37820 | 151277 | 605110 |
| $\ell$ | 21 | 23 | 25 | 27 | 29 | 31 | 33 | 35 |
| $n_M$ | 1 | 4 | 12 | 43 | 161 | 601 | 2258 | 8523 |
| $\Lambda$ | 14.286 | 15.672 | 17.058 | 18.444 | 19.831 | 21.217 | 22.603 | 23.990 |
| $n_U$ | 3 | 5 | 8 | 15 | 29 | 56 | 108 | 210 |
| $Q_{\max}$ | 0.009 | 0.031 | 0.116 | 0.430 | 1.603 | 6.003 | 22.579 | 85.227 |

（$d=64,256$ 只用於 S8 callout；滑桿範圍為 $2^{10}$–$2^{20}$。`w5_count.py` 已延伸到 $p=20$。）

Eq. 100 三項（$d=1024$）：$Q=1$：0.08629、$5.96\times10^{-5}$、$2.98\times10^{-8}$；$Q=12$：1.036、$5.96\times10^{-5}$、$3.576\times10^{-7}$。$\eta=10^{-6}$、$d=65536$：$\ell=41$、$n_M=456$、$n_U=49$。

---

## 6. 實作計畫

### 6.1 檔案與外殼

- 單一檔案 `tut/5/build/index.html`。**完整沿用** `tutorials/prediction-targets-k-diff.html` 的外殼：CSS tokens（含深色模式與 `--ink-3` 深色 #9C978C）、字型載入、appbar 與主題切換（`pm-theme`）、`#pm-bar`、sticky 章節目錄／窄螢幕 `#tocbar`、`.plate`/`.plate-h`/`.plate-lead`/`.plate-f`、callout、`<details>` 檢查題、演算法逐行格線、時間線 rail（`#rail`、`#tl-chips`、`#tl-prev`/`#tl-next`）、`?test` 的 `selfTest()` 與 `#selftest` 輸出、KaTeX（auto-render、`.katex-display{overflow-x:auto}`、動態 DOM 呼叫 `renderMathInElement`）、SVG 輔助（`el`、`scaleLin`、`scaleLog`、`linePath`、`axis`）、「本頁推導」pill 標籤樣式。以下只列新增項目。
- 連結：論文 `../index.html#paper-5`；arXiv `https://arxiv.org/abs/2609.20539`；筆記 `<slug>.html`。

### 6.2 新增 tokens

- `--c-unif: var(--c-par)`、`--c-mask: var(--c-perp)`（沿用參考頁的兩個資料色，只改名稱語意；暗色對比由 6.5 自檢確認）；Gaussian 用 `--ink-1` 點線。
- 新增「〔自選〕」pill：同「本頁推導」樣式，文字「自選常數」。

### 6.3 程式結構（一個 IIFE；每個元件一個 init 函式，只碰自己的容器）

- 共用：`mulberry32`、`gauss`、`schedule(job)`（setTimeout 分段執行器，帶 `jobId` 作廢）、`fmt(x, digits)`、`debounceLive(el, text)`（500ms）。
- 列舉模組（C2、C3）：`ENUM = {D:6, N:64, POP:Uint8Array, HAM:Uint8Array(4096)}`、`PRESETS`、`kernelW(t)`、`push(q,t)`、`H(p)`、`dtc(p)`、`tcCond(q,s,h)`（回傳 `{tc, qs, qt, J}`）、`cS(s,h)`、`gridT(a)`、`runSchedule(q,a,onDone)`。
- 資訊量模組（C1、C4）：`lf`、`h2`、`Iunif`、`Igauss`（預先算 Simpson 權重）、`tOfIunif`、`tStarUnif`、`tStarGauss`、`maskedRec`、`MASK_LNRATIO`、`makeUnifRec(d,kappa)` → `{rec(t), recI(I), width()}`（4.4 的演算法，內部快取 `setupT` 的前後綴和）。
- 元件：`initC1`、`initC2`、`initC3`、`initC4`（`widthCache = new Map()` 以 κ 為 key）、`initC5`。
- 靜態圖：S0 的三條 noise 軸、S3 的 Fig. 5 風格示意（純 SVG）。

### 6.4 研究脈絡（`notes/timeline.json`）

- 建置時原樣貼進 `<script type="application/json" id="timeline-data">`（13 筆，欄位 `year,title,authors_short,arxiv,role,source`），不手改。依 `year` 排序。
- `TL_GROUP`（以 arXiv 為 key，未列者歸「其他」）：

| 類別 | arXiv | 卡片連到的筆記 |
| --- | --- | --- |
| 基礎 | 2006.11239 | `eps-prediction-ddpm-score.html` |
| 三種腐蝕 | 2107.03006、2211.15089、2310.16834、2406.07524、2502.09992 | `dllm-three-corruption-paradigms.html` |
| 取樣理論 | 2205.14987、2402.08095 | `dllm-discrete-sampling-theory.html` |
| masked DTC | 2511.04647、2510.25544 | `dllm-masked-schedule-dtc-theory.html` |
| critical window | 2403.01633 | `dllm-critical-windows-speciation.html` |
| query 下界 | 2604.10857 | `dllm-score-oracle-lower-bounds.html` |
| 本文 | 2609.20539 | `dllm-parallelism-separation-result.html` |

- 本文卡片 `--accent` 外框；JSON 解析失敗時整節顯示「時間線資料尚未提供」。
- S11 延伸閱讀順序：`dllm-parallelism-lineage-hub`、`dllm-three-corruption-paradigms`、`dllm-discrete-sampling-theory`、`dllm-masked-schedule-dtc-theory`、`dllm-critical-windows-speciation`、`dllm-score-oracle-lower-bounds`、`dllm-parallelism-separation-result`（標題取各 frontmatter 的 `title`）；另列站內既有 `eps-prediction-ddpm-score.html`、`continuous-flow-lm-overview.html`（建置時確認兩者存在於 `tutorials/`，不存在就不列）。

### 6.5 selfTest（`?test`）

以 `console.assert` 比對，最後一行 `selfTest: N passed, M failed`：
- `dtc`、TC：copy DTC = log 2、parity DTC = 5 log 2、codebook DTC 1.7329（±1e-4）。
- `tcCond(codebook, 0.5, 0.2).tc` = 0.024176、上界 0.147974（±1e-6）。
- `runSchedule(codebook, 0.1)`：N = 136、ΣTC 0.038090、KL 0.005906（±1e-6）。
- `tStarUnif(0.2)` 0.4937、`tStarGauss(0.2)` 0.5521（±1e-4）；`Igauss(0.5)` 0.22716（±1e-5）。
- `MASK_LNRATIO` 2.3698（±1e-4）；`makeUnifRec(256,0.2).recI(0.2)` 0.5357（±5e-4）；`makeUnifRec(64,0.2).width()` 0.105902（相對 1%）。
- C5：$d=4096,\eta=10^{-3}$ → $|\mathcal K|=2364$、$\ell=27$、$n_M=43$、$n_U=15$；$d=2^{20}$ → $n_M=8523$、$n_U=210$。
- C4：`width()` 在端點包不住時回傳 `NaN`（以人為縮小的區間呼叫內部函式測試）。
- 對比：`--c-unif`、`--c-mask`、`--ink-1`、`--accent` 對 `--surface` 的 WCAG 對比 ≥3（淺色與 `data-theme="dark"` 各算一次，以 `getComputedStyle` 取值）。
- `document.documentElement.scrollWidth <= innerWidth`。

### 6.6 建置後自我檢查

截圖 390／1440 與深色模式各一張，深色模式需涵蓋 C1–C5 每個元件（C1 的＋／−底色、Gaussian ▲▼、C5 未命中段、Gaussian 點線）；390px 截圖確認表格與演算法格線在容器內水平捲動、無整頁水平捲動；`?test` 全數通過；執行 `extract_page_text.py` 確認 12 個 LO 的參考答案都出現在正文。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 定理常數未給 | $A,B,C,a,b,K,c_0,d_0,\tau_\pm$ 都只說依賴 $[\kappa_-,\kappa_+]$；Alg. 1–4 無法照字面執行 | 頁面不執行 Alg. 1–4；C5 的常數全標〔自選〕並寫出保證在小 $d$ 無意義 |
| R2 | uniform 求積是本頁推導 | 論文用 Monte Carlo（App. C）；本頁改用 Laplace 泛函的確定性求積 | 標「本頁推導」；第 5 節列出與 MC 的一致性（差 ≤ 0.003，在 SE 內） |
| R3 | Fig. 4(c) Gaussian 曲線 | saddlepoint + 分箱細節未給 | 不重算；C4 只畫 $I_{\rm gauss}$ 與 $t^\star_{\rm gauss}$，並寫明原因 |
| R4 | Fig. 4 數值不可逐點比對 | 論文只有圖，樣本數、種子未給 | 頁面只比較縮放（斜率）與定性形狀，不宣稱重現某個數值 |
| R5 | C2/C3 的小例子 | $d=6$ 與自選碼本不在任何定理的漸近範圍；端點 $\delta_U,T$ 為自選 | 頁面寫「這是驗證不等式的小例子，不是論文的實驗」；$a$ 可到 1（論文 $a_U\le1/12$），但 $c\le3a$ 的推導只需 $x_j\le1$，頁面註明 |
| R6 | 誤讀風險 | 讀者可能把分離讀成「masked 一定慢」或「因為 commitment」 | S9 專節 + Hero 第 3 卡措辭用「存在…模型」；LO12 |
| R7 | C4 計算時間 | $d=4096$ 的寬度計算（約 360 次積分評估）在低階手機可能超過 1 秒 | 分段執行、顯示「計算中…」、以 κ 快取；只在 `change` 觸發 |
| R8 | 數值溢位 | $\lambda_j$ 可達 $e^{\kappa d}$（$d=4096$ 時 $e^{819}$） | 全部在 log 空間累加，$>700$ 截為無窮（4.4） |
| R9 | 時間線年份 | timeline.json 中 CCL 標 2025，論文引用寫 [CCL26] | 照 JSON 原樣，不修改（AGENTS.md「保留事實」） |
| R10 | DTC $=\Theta(d)$ 無證明 | Sec. 1.2 直接陳述 | 頁面寫「論文陳述」，直覺說明標本頁推導 |
| R11 | 名稱衝突 | 論文 $\ell_t$（Sec. 6）與 $\ell$（Sec. 8）、三個 $u$ | 頁面只用 Sec. 8 的 $\ell$；clip 範圍寫成 $[\beta_t/(1-\beta_t),(1-\beta_t)/\beta_t]$；$u$ 見符號表 callout |
| R12 | 筆記措辭與本頁不同 | `dllm-parallelism-separation-result` 寫 uniform/Gaussian「Θ̃(√d) 次即足夠」（對 uniform 單獨而言上下界合起來確為 Θ̃(√d)，但不能讀成 masked 與 uniform 的比值是 Θ̃(√d)）；`dllm-score-oracle-lower-bounds` 寫「TV 至少 0.99」（來自 Sec. 1 的 informal 敘述，line 311；Thm 8.1 為 $1-\rho$） | 筆記不由本頁修改；頁面一律用 LO12 的措辭與 Thm 8.1 的 $1-\rho$ |
| R13 | uniform $1/\sqrt d$ 的解釋是啟發式 | 「轉變 ≈ $\mathbb E[Z\mid d_H]\approx1$」論文未證；Prop 6.4 只給充分窗寬 | 頁面標〔本頁推導〕＋「啟發式」，縮放的證據寫成求積斜率 $-0.478$ 與 Fig. 4(d) |
| R14 | S1 拆章後顯示編號改變 | 章節代號 S1a/S1b/S2…與顯示編號 01–12 不再一一同號 | 第 3 節大綱明列對照；LO 表的「章節」欄用代號 |

---

## 8. 修訂紀錄

- **Round 1**（合併兩份審查，`reviews/round1.md` 26 項）：
  - 忠實度：LO11 改為 Def 8.3 的單側規則並補 Eq. 100 三項的事件、Lemma 8.2 的 $\mathbb E_C$＋Markov；LO12／Hero 卡 3 改為「至少 $\tilde\Omega(\sqrt d)$ 倍、oracle 類別不同」；LO1／步驟 0 補「masked 僅未揭露座標」與 $t$ 無關性的假設；LO8／1.1 補窗寬比例的成立條件；LO4 分開實際分解與 Lemma 3.8 上界；LO9／步驟 7 改為「論文給窗寬上界、$1/\sqrt d$ 來自求積與 Fig. 4(d)」並標啟發式；commitment 一律寫成「論文主張」；C1 $(S-1)/S$、C4 0.632 的理想化前提、步驟 8 Gaussian 計數兩種寫法並列。
  - 教學與建置：S1 拆為 S1a／S1b；S2 前置縮減並加 $d=3$ 算式、定義 $t\to s=t-h$；S4 Gaussian 移入 `<details>`；比喻用語全部改為字面用語；C1 移除 $I$ 讀數；C2 改相對門檻；C4 窗寬端點檢查與 NaN；C5 範圍 $2^{10}$–$2^{20}$、整數段號滑桿、32 tick 掃描、$Q_{\max}<1$ 標示，驗收表延伸到 $2^{20}$；4.0 增列窄螢幕、SVG 文字、reduced motion、暗色對比規格；Gaussian 線色改 `--ink-1`；6.5 增 C5、C4 NaN 與對比自檢。
  - 研究筆記：S1a 加 autoregressive 特例、S1b 加 Chen & Ying 的 $O(d)$ 基準、S8 加 [XP26] 的 $\kappa$ 估計說法與 [CCL26] MDS 下界的區分、S11 加 `continuous-flow-lm-overview`；筆記與本頁措辭差異記為 R12。
- **建置前修正**（readiness 報告的 non-blocking 註記，建置時採用）：(1) 第 5 節補 C1 期望比例 $\pm10^{-4}$、C4 $I$ 列 $\pm10^{-5}$ 的容許誤差，selfTest 一併檢查；(2) 本文件的 `tut/5/` 即 `scripts/tutorial_src/5/`；(3) 頁面把 C3／S4 的步長參數一律寫成 $a_U$，S7 的門檻寫成「Alg. 1 的門檻常數 $a=2a_0$、$b=b_0-a_0$（與 $a_U$ 無關）」；(4) C5 與 S8 寫「Eq. 96 的常數 $C$ 取 1（這個 $C$ 不是碼本）」；(5) Hero 卡 3 改為「約 √d 次（相差 log 因子）」；(6) Fig. 4(c) 不重算（R3，維持）；(7) 自選常數的理由在 plate-f 各一句：C5 的 $\eta\in\{0.09,10^{-3},10^{-6}\}$ 分別代表「誤差大、使 good $C$ 的保證失效」「$d^{-1}$ 量級」「$d^{-2}$ 量級」，對應 Thm 8.1 的 $\eta=d^{-\Theta(1)}$；$\rho=0.2$ 使結論為 TV ≥ 0.8；C4 的 $\kappa\in[0.05,0.6]$ 落在 $(0,\log2)$ 內且使 $I=\kappa\pm0.1$ 的網格不碰到端點太多。
- **Readiness 修正**（Goldfish NO-GO 三項）：$\varepsilon$（目標精度；Result 1 為 KL、Result 2 為 TV）、$\hat q$ 與 $\delta$（失敗機率）在 1.1 第一次使用處定義，並加入符號表；$\Lambda$（Eq. 24）與 $A$ 在 1.1 第一次使用處定義，步驟 4、7 與 S4、S6 大綱各加「先定義」；C3 的自選端點改記為 $\delta_U$，與失敗機率 $\delta$ 區分；步驟 3 的比值範圍改為 0.018–0.19，與第 5 節 C2 表（0.0181–0.1888）一致。
- **Elephant 修正**（no-context 讀者只讀正文時 LO6、LO7、LO11 為 partial）：S4 正文新增一段「從 $\varepsilon$ 推出 query 數」：$a_U=\varepsilon/(12\max\{\overline{\mathrm{DTC}},\varepsilon\})$ 使 $3a_U\mathrm{DTC}\le\varepsilon/4$、預算合計 $\le\varepsilon$、每步一次 query、$N_U\le1+24\max\{\cdot\}/\varepsilon\cdot O(L_U)=O((1+\overline{\mathrm{DTC}}/\varepsilon)\log(edS/\varepsilon))$（LO6）。S5 正文說明 masked 以已揭露座標數 $m$ 表示腐蝕、總資訊 $m\log2=\log M=k\log2$，臨界點 $m=k$，並給一致碼字數約 $1+2^{k-m}$（LO7）。S8 正文補 Lemma 8.2 的直覺、good $C$ 的定義（Eq. 98）、Markov＋union bound 的等比級數並把 Eq. 99 列為顯示式；Eq. 100 的三項改為編號清單，逐項說明事件（$|m-k|\le\ell$、$C$ 不是 good、$m>k+\ell$ 的 query 碰巧與 $C$ 一致）與機率來源；結論段補 Thm 8.1 證明的三步（coupling 得 $\mathrm{TV}(\hat q,\nu)\le p_{\rm diff}$、Eq. 103、選常數使 $\mathbb E[\nu(C)+p_{\rm diff}]\le\rho^2$ 後用 Markov），明寫「$Q\le c\rho^2d/\ell$ 時以機率 $\ge1-\rho$ 有 $\mathrm{TV}\ge1-\rho$」（LO11）。
