# 設計文件：〈When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis〉互動教學

- 論文：Xiang Li、Zebang Shen、Ya-Ping Hsieh、Niao He（ETH Zurich），arXiv 2509.24912v2（2026-03-16）。資料庫編號 No. 237。
- 原文：`tut/237/src/paper.txt`（PDF 同目錄）。「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Thm」「Lemma」「Assumption」皆指論文原文編號。`tut/237/advisor.md` 是顧問備忘（本文件已逐條比對原文）。
- 產出：`tut/237/build/index.html`（單一 HTML）。上線位置 `tutorials/scores-learn-geometry.html`（slug `scores-learn-geometry`），站內相對連結以上線位置為準。
- 驗算腳本（`tut/237/verify/`）：`core.js` 是頁面數值核心，**建置時逐字複製進頁面**；`check_c1.js`（C1）、`check_c2.js`（C2）、`check_c3.js`（C3）、`check_c4.js`（C4）用 `core.js` 產生第 5 節的驗收表（輸出存為 `c*_out.txt`）；`check_c2_perf.js` 量測 C2 橢圓的視窗化求積與完整求積的差異與耗時（`c2_perf_out.txt`）；`ref.py` 以 mpmath 與直接求積獨立驗證 `core.js` 的 Bessel 函式與封閉式（`ref_out.txt`）。
- 研究筆記：`tut/237/notes/`（總覽 `score-geometry-lineage-hub` + 8 篇主題筆記 + `timeline.json`），上線後為 `tutorials/<slug>.html`。
- 慣例：本設計自行推導、論文沒有寫出的結論標「〔本頁推導〕」；用論文數字校準、論文沒寫明的值標「推定」。頁面用同樣的標籤（6.5）。本頁只講 No. 237，不提其他教學頁。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

分數式方法（擴散模型、貝氏逆問題）通常被解讀為在噪音 σ → 0 時學到資料分佈。本文在流形假設下（資料落在 d 維空間中的 n 維緊緻流形 M 上）分析小 σ 時的 log 密度，得到 Thm 3.1：$\log p_\sigma(x)=-d_M(x)/\sigma^2+\log p_{data}(\Phi^{-1}(P_M(x)))-\tfrac{d-n}{2}\log(2\pi\sigma^2)+H(x)+o(1)$（Eq. 6）。$\sigma^{-2}$ 階只含到流形的距離 $d_M$，資料密度 $p_{data}$ 只出現在 Θ(1) 階，作者稱為**速率分離**（rate separation）。三個結果：(i) Thm 4.1：生成分佈的分數誤差 $o(\sigma^{-2})$ 就足以集中到 M，還原 $p_{data}$ 需要 $o(1)$，Θ(1) 的誤差可以讓極限變成 M 上任意分佈；(ii) Thm 5.1–5.2：把 Langevin 的分數乘上 $\sigma^\alpha$（**Tempered Score Langevin**，Eq. 8），在誤差 $o(\sigma^\beta)$、$\beta>-2$、$\max\{-\beta,0\}<\alpha<2$ 下，平穩分佈收斂到 M 上對黎曼體積的均勻分佈；(iii) Thm 6.1：逆問題中把先驗分數換成 TS 版（乘上 $\sigma^\alpha$，Eq. 10）、似然項 $v$ 有界，在與 Thm 5.2 相同的假設下（$p_{data}\in C^2$、$\max\{-\beta,0\}<\alpha<2$、Assumption B.2），平穩分佈收斂到均勻先驗下的後驗，分數誤差容許 $o(\sigma^\beta)$、$\beta>-2$。實驗（作者稱 preliminary）：橢圓上的玩具模擬（Fig. 2）與 Stable Diffusion 1.5 的 Predictor–Corrector 採樣（Table 1–4）。

### 1.2 研究動機（Sec. 1）

1. 低溫（σ → 0）的分數最有價值，也最難估計（作者引 Karras et al. 2022、Song et al. 2021 等）。
2. 既有收斂理論把幾何與密度合在同一個分數誤差裡（Sec. 1.1，Chen et al. 2023 等），看不出尺度分離；既有「擴散模型偵測流形」的研究只處理內在維度（Stanczuk et al.）、線性流形（Ventura et al.）或給出不含 σ 階數的條件（Pidstrigach 2022）。
3. 分數漸近研究（Lu et al. 2023 等）把 $p_{data}$ 的資訊併入有界餘項，沒有分離出來。

### 1.3 貢獻

1. Thm 3.1 / Thm B.2：小 σ 展開（Eq. 6、7、12），VE 與 VP 都成立。
2. Thm 4.1：既有框架下，集中到流形與還原 $p_{data}$ 的誤差門檻差 $\sigma^{-2}$。
3. TS Langevin（Eq. 8，「one-line modification」）與 Thm 5.1（梯度場）、Thm 5.2（非梯度場，用 WKB 與強極大值原理）。
4. Thm 6.1：逆問題中均勻先驗（最大熵先驗）的穩健性；CFG 的對應。
5. App. D：單位圓例子上 TS 混合不比標準 Langevin 慢（Remark 5.1）；實驗（Sec. 7、App. C）。

### 1.4 從作者的角度逐步拆解

#### 符號表（取自 repo 擁有者的 `user_terms_note.md`；頁面 S2 末尾以可收合表格呈現）

| 符號 | 意義 |
| --- | --- |
| $d$ | 外部空間（ambient space）維度，$x\in\mathbb R^d$ |
| $M$、$n$ | 資料流形與它的維度，$n<d$ |
| $\sigma$ | 高斯噪音標準差；論文只關心 σ → 0 |
| $\mu_{data}$、$p_{data}(u)$ | 資料分佈；它在座標 $u$ 上的密度（對 $U$ 上 Lebesgue 測度，不是對體積測度） |
| $\mu_\sigma$、$p_\sigma(x)$ | 加噪後的分佈與它在 $\mathbb R^d$ 上的密度 |
| $s^*(x,\sigma)$ | 真實分數 $\nabla\log p_\sigma(x)$ |
| $s(x,\sigma)$ | 模型學到的分數（score oracle） |
| $\Phi$、$u$、$g(u)$ | 座標圖、局部座標、黎曼度量 $g=J^\top J$ |
| $P_M(x)$、$d_M(x)$ | $x$ 在 M 上的最近點；$\tfrac12\mathrm{dist}^2(x,M)$（Eq. 2，有 ½） |
| $T_M(\epsilon)$ | 距離 M 不到 ε 的管狀鄰域 |
| $\alpha$、$\beta$ | TS Langevin 的縮放指數；分數誤差的階數 $o(\sigma^\beta)$ |
| $\pi_\sigma$、$f_\sigma$、$E_\sigma$ | Sec. 4：實際生成分佈、$-\log\pi_\sigma$、$\|s_{\pi_\sigma}-s^*\|_{L^\infty(K)}$ |
| $\theta$ | Gibbs 形式 $\exp(-f_\theta/\theta)$ 的溫度；Sec. 4 取 $\sigma^2$，Sec. 5 取 $\sigma^{2-\alpha}$ |

#### 步驟 1：問題設定（Sec. 2，Eq. 1–3）

- Eq. 1：$\mu_\sigma:=\mathrm{law}(X+\sigma Z)$（VE）或 $\mathrm{law}(\sqrt{1-\sigma^2}X+\sigma Z)$（VP），$X\sim\mu_{data}$、$Z\sim\mathcal N(0,I)$。
- Assumption 2.1：$\mu_{data}$ 支撐在緊緻、無邊界、$C^4$ 的嵌入子流形 $M\subset\mathbb R^d$，$\dim M=n$。Assumption 2.2：$p_{data}$ 在 $U$ 上 $C^1$ 且嚴格正。$p_{data}$ 是母體密度，不是有限樣本經驗分佈（Sec. 2.1 末句）。
- 座標：$\int_Mf\,dM=\int_Uf(\Phi(u))\sqrt{\det g(u)}\,du$，$\sqrt{\det g}$ 是體積修正因子（Sec. 2）。$d_M(x):=\tfrac12\mathrm{dist}^2(x,M)$（Eq. 2）；$T_M(\epsilon)$ 內最近點 $P_M(x)$ 唯一。
- Eq. 3：$p_\sigma(x)=\int_M(2\pi\sigma^2)^{-d/2}\exp\!\big(-\|x-\gamma(\sigma)\Phi(u)\|^2/(2\sigma^2)\big)p_{data}(u)\,du$，VE $\gamma=1$，VP $\gamma=\sqrt{1-\sigma^2}$。
- Langevin（Sec. 2.3）：$dX_t=s(X_t)dt+\sqrt2dW_t$；$s=-\nabla f$ 時平穩分佈 $\propto e^{-f}$。頁面 S1 同時用一句話定義平穩 Fokker–Planck 方程：SDE $dX=b\,dt+\sqrt{2\theta}dW$ 的密度 $\pi$ 不隨時間改變的條件是 $0=\mathrm{div}(-b\pi+\theta\nabla\pi)$；$b=-\nabla f$ 時 $\pi\propto e^{-f/\theta}$ 滿足它（S7 才用到展開）。並說明「溫度」θ：$e^{-f/\theta}$ 的 θ 越小越集中在 $f$ 的極小點。$s$ 不是梯度場時動力學不可逆，平穩分佈一般沒有封閉式，文獻用 WKB ansatz $\exp(-V/\epsilon)c_\epsilon$（Eq. 5）。
- 逆問題（Sec. 2.2）：$y=A(x)+\xi$，似然 $\propto e^{-v(x;y)}$（高斯時 $v=\|A(x)-y\|^2/(2s^2)$）；後驗 Langevin $dX=-\nabla_xv\,dt+\hat s\,dt+\sqrt2dW$（Eq. 4）。既有保證要求 $\hat s$ 誤差 $o(1)$（Laumont et al. 2022）或精確（Pesme et al. 2025）。
- 下一步的理由：要比較「幾何」與「密度」需要多少分數精度，先得知道兩者在 $\log p_\sigma$ 中各在哪一階。

#### 步驟 2：小 σ 展開（Thm 3.1 = Thm B.2，Eq. 6、7、12；Corollary B.1）

- 工具（Corollary B.1，Łapiński 2019）：$\int g\,e^{-F/\theta}dx=e^{-F(x^*)/\theta}\frac{(2\pi\theta)^{m/2}}{\sqrt{|\nabla^2F(x^*)|}}(g(x^*)+h(\theta))$，$h=O(\sqrt\theta)$，在常數有界的函數類上一致。
- 對 Eq. 3 的 $u$ 積分取 $F(u)=\tfrac12\|x-\Phi(u)\|^2$、$\theta=\sigma^2$、$m=n$：極小點 $\Phi^{-1}(P_M(x))$，Hessian 為 $\hat H_{ij}=\langle\partial^2\Phi/\partial u_i\partial u_j,\Phi(u)-x\rangle+\langle\partial\Phi/\partial u_i,\partial\Phi/\partial u_j\rangle$。得 Eq. 12：$p_\sigma(x)=\exp(-\|x-P_M(x)\|^2/(2\sigma^2))\,(2\pi\sigma^2)^{(n-d)/2}|\hat H|^{-1/2}(p_{data}+h(\sigma^2))$，$h=O(\sigma)$。取 log 即 Eq. 6（VE：$H=-\log\sqrt{|\hat H|}$；VP 另加 $-\tfrac12\langle P_M(x),x-P_M(x)\rangle$）。
- Eq. 7：$\lim_{\sigma\to0}\sigma^2\log p_\sigma(x)=-d_M(x)$，在 $T_M(\epsilon)$ 上一致。
- 作者的解讀（Sec. 3）：$\sigma^{-2}$ 係數只含 $d_M$、不含 $p_{data}$；$d_M$ 的任何誤差都會被 $\sigma^{-2}$ 放大，所以必須先學對幾何，才可能學到密度。
- 在 M 上 $\hat H=g(u)$，所以 $|\hat H|^{1/2}=\sqrt{\det g}=dM/du$；這一點在步驟 4 讓體積因子相消。
- 〔本頁推導〕單位圓（$n=1,d=2$）、$x=\rho(\cos\varphi,\sin\varphi)$：$\hat H=\rho$；von Mises $p_{data}(t)=e^{\kappa\cos(t-\theta_0)}/(2\pi I_0(\kappa))$ 時有封閉式 $p_\sigma(x)=(2\pi\sigma^2)^{-1}e^{-(\rho^2+1)/(2\sigma^2)}I_0(R)/I_0(\kappa)$，$R=\|x/\sigma^2+\kappa e_0\|$，$e_0=(\cos\theta_0,\sin\theta_0)$。用 $I_0(z)\approx e^z/\sqrt{2\pi z}$ 可逐項得到 Eq. 6 的四項（$H=-\tfrac12\log\rho$）。C1 以此封閉式計算；`ref.py` 以直接求積驗證到 $10^{-7}$。數值上 Eq. 6 的殘差在此例按 $\sigma^2$ 衰減（ρ = 1.1、φ = 90°：σ = 0.1 為 0.0735，σ = 0.01 為 0.000739），比論文保證的 o(1) 快。
- 下一步的理由：展開描述的是真密度；要談「誤差多大還可以」，需要一個把任意 $\exp(-f_\theta/\theta)$ 的極限算出來的通用工具。

#### 步驟 3：通用極限框架（App. B.1，Eq. 11，Assumption B.1，Lemma B.1–B.2，Thm B.1）

- 密度 $\propto\exp(-f_\theta/\theta)$，$f_\theta=f_0+\theta f_1+\hat f$（Eq. 11），$f_0$ 的極小集是 M，$\hat f=o(\theta)$ 一致。Assumption B.1 另要求法向 Hessian $\partial^2f_0/\partial r^2$ 一致正定（第 5 條）。
- Lemma B.1：用管狀座標 $\Phi(u,r)=\Phi(u)+N(u)r$（$N$ 的行為單位法向量）把 $r$ 積掉。Lemma B.2：只要 $f_\theta=f_0+o(1)$，質量就集中到 M（不需要 $f_1$）。
- Thm B.1：若管外質量 → 0，則 $\pi(u)\propto\exp(-f_1(u,0))\,|\partial^2f_0/\partial r^2(u,0)|^{-1/2}\,dM(u)/du$（弱收斂；證明沿用 Hwang 1980）。
- 這個式子讀法：$f_0$ 決定集中在哪裡，$f_1$ 決定 M 上的形狀；比 $\theta$ 小的項全部消失。後面三個定理都是挑 $\theta$、算 $f_0,f_1$。

#### 步驟 4：既有框架的速率分離（Thm 4.1，Assumption 4.1，Lemma B.3–B.4）

- 設定：$\pi_\sigma=e^{-f_\sigma}$ 是演算法實際產生的分佈（訓練與離散化誤差都算在內），$f_\sigma\in C^1(K)$，質量集中在緊緻集 $K\supset T_M(\epsilon)$，K「一致可求長路徑連通」（Assumption 4.1）。誤差 $E_\sigma=\|s_{\pi_\sigma}-s^*\|_{L^\infty(K)}$。
- Thm 4.1：(1) $E_\sigma=o(\sigma^{-2})$ ⇒ 集中到 M；(2) 對 M 上任意 $C^1$ 密度 $\hat\pi$，可構造 $E_\sigma=\Omega(1)$ 而 $\pi_\sigma\to\hat\pi$；(3) $E_\sigma=o(1)$ ⇒ $\pi_\sigma\to p_{data}$。
- 證明三步：(a) 沿 K 內長度有界的路徑積分，把梯度誤差 $o(\sigma^\eta)$ 變成 log 密度誤差 $o(\sigma^\eta)$（$\eta\ge-2$；常數由歸一化固定）；(b) Lemma B.3（$p_\sigma$ 集中在管內）、Lemma B.4（$e^{-f_\sigma}$ 管外質量 → 0）；(c) 套步驟 3，$\theta=\sigma^2$。(1)：$\sigma^2f_\sigma=d_M+o(1)$，用 Lemma B.2。(3)：$f_0=d_M$，$f_1=-\log p_{data}+\log\sqrt{|\hat H|}$，在 r = 0 時 $\sqrt{|\hat H|}=dM/du$、$\partial^2f_0/\partial r^2=I$，所以 $\pi(u)\propto p_{data}(u)\,(dM/du)^{-1}\,(dM/du)=p_{data}(u)$。(2)：把 $\log p_{data}$ 換成 $\log\hat\pi$，與 $-\log p_\sigma$ 只差 Θ(1)（構造中的誤差是有界的 Θ(1) 量）。
- 作者的解讀：擴散模型可能在還原 $p_{data}$ 之前就已產生在流形上的樣本，這可能是它成功的原因之一（Sec. 4 末）。
- 下一步的理由：只做到集中時，M 上的分佈可以是任意的（結論 (2)）。作者要找一個在同樣寬鬆的誤差下仍有確定極限的目標：均勻分佈。

#### 步驟 5：TS Langevin，梯度場情形（Sec. 5，Eq. 8–9，Thm 5.1）

- Eq. 8：$dX_t=\sigma^\alpha s(X_t,\sigma)dt+\sqrt2dW_t$；α = 0 就是標準 Langevin／PC 的 corrector。
- 若 $s=-\nabla f_\sigma$：平穩分佈 $\tilde\pi_\sigma\propto\exp(-\sigma^\alpha f_\sigma)=\exp(-[\sigma^2f_\sigma]/\sigma^{2-\alpha})$，即 $\theta=\sigma^{2-\alpha}$。
- Thm 5.1：假設 Assumption 2.1、2.2 與 Assumption 4.1（**把 $\pi_\sigma$ 換成 TS 的輸出 $\tilde\pi_\sigma$**：質量集中在緊緻、一致可求長路徑連通的 $K$ 上，$f_\sigma\in C^1(K)$），誤差 $\|s-s^*\|_{L^\infty(K)}=o(\sigma^\beta)$、$\beta>-2$（Eq. 9），$\max\{-\beta,0\}<\alpha<2$ ⇒ $\tilde\pi_\sigma$ 弱收斂到 M 上對內蘊體積的均勻分佈，$\tilde\pi(u)\propto dM/du=\sqrt{\det g(u)}$。
- 為什麼（Thm 5.1 的證明只寫「其餘項相對 $\sigma^{2-\alpha}$ 漸近地小」；下列逐項條件是〔本頁推導〕，與 Thm 5.2 證明中「because α > max{−β, 0}」一句（App. B.5）一致）：$\sigma^2f_\sigma=d_M+O(\sigma^2)$（$p_{data}$ 與曲率項）$+o(\sigma^{2+\beta})$（誤差）。兩者都是 $o(\theta)$ 的條件：$\sigma^2=o(\sigma^{2-\alpha})\Leftrightarrow\alpha>0$；$\sigma^{2+\beta}\lesssim\sigma^{2-\alpha}\Leftrightarrow\alpha\ge-\beta$（論文取嚴格 $\alpha>-\beta$）；$\theta\to0\Leftrightarrow\alpha<2$。於是 $f_0=d_M$、$f_1\equiv0$，代入 Thm B.1 得 $\propto dM/du$。管外質量另行估計（App. B.5，用到 $2-\alpha>0$ 與 $\alpha+\beta>0$）。
- 〔本頁推導〕精確分數時 $\tilde\pi_\sigma\propto p_\sigma^{\sigma^\alpha}$：法向高斯寬度由 σ 變成 $\sigma^{1-\alpha/2}$（仍趨近 0），切向密度變成 $p_{data}^{\sigma^\alpha}\to1$。對 von Mises，殘留的最大／最小比在 σ → 0 時 $\approx e^{2\kappa\sigma^\alpha}$（這是只看切向因子的漸近式；κ = 4、α = 1、σ = 0.01 為 1.083，與 C2 的精確值 1.083 相同；σ 較大時有差距，例如 σ = 0.2 時漸近式 4.953、C2 精確值 4.783，σ = 0.05 時 1.492 對 1.491）。等價於溫度 $\sigma^{-\alpha}$ 的退火。
- 「均勻」是對嵌入後的黎曼體積，不是對座標 $u$。〔本頁推導〕橢圓 $\Phi(t)=(a\cos t,b\sin t)$：$\sqrt{\det g}=\sqrt{a^2\sin^2t+b^2\cos^2t}$；$(a,b)=(1,2)$ 周長 9.6884，t 上的密度最大 0.2064（t = 0、π）、最小 0.1032（t = ±π/2），比值 2。換一個嵌入（例如像素空間與潛空間）均勻分佈就不同。

#### 步驟 6：分數的首項與 Eq. 20（App. B.5）

- 證明 Thm 5.2 需要比 Eq. 6 更強的分數展開：$\sup_{T_M(\epsilon)}\|\nabla\log p_\sigma(x)+\tfrac1{\sigma^2}\nabla\tfrac12\|x-P_M(x)\|^2\|=O(1)$（Eq. 20），其中 $\nabla\tfrac12\|x-P_M(x)\|^2=x-P_M(x)$。
- 作法：$\sigma^2\nabla\log p_\sigma+(x-P_M(x))$ 是後驗平均 $\mathbb E[\Phi(u)-P_M(x)\mid x]$；$O(\sqrt\theta)$ 的 Laplace 只給 $O(\sigma)$，不夠；改用 $O(\theta)$ 版本（Majerski 2015, Thm 2.4，需要 $F\in C^4$、$g\in C^2$，因此 Thm 5.2 多假設 $p_{data}\in C^2$），並在分子加 1 使前因子在極小點不為 0，得 $(1+O(\sigma^2))/(1+O(\sigma^2))-1=O(\sigma^2)$。
- 結果：$b_\theta:=\sigma^2s=-\nabla d_M+o(\sigma^{2+\beta})+O(\sigma^2)=-\nabla d_M+o(\sigma^{2-\alpha})$。
- 「法向 ~σ⁻²、切向 ~1」只在**離 M 固定距離**的點成立〔本頁推導〕：在 M 上法向分量本身是 O(1)（ρ = 1 時 −0.500），在離 M 約 σ 的典型加噪點約為 1/σ。論文的速率分離是 Eq. 6、7 中 log 密度的階數；Eq. 20 只給 O(1) 的餘項。頁面以 ρ = 1.1（距離 0.1）當示例並寫明此限定。
- 〔本頁推導〕單位圓 C1 的數值：ρ = 1.1、φ = 90°，$s_{rad}+(\rho-1)/\sigma^2$ 在 σ = 0.1、0.01 為 −0.521、−0.455，趨近 $-1/(2\rho)=-0.4545$（曲率項 $-\tfrac12\log\rho$ 的徑向導數），確實是 O(1)。

#### 步驟 7：非梯度場的 WKB 分析（App. B.4，Eq. 13–19，Assumption B.2，Thm B.3，Lemma B.5–B.6）

- SDE：$dx=b_\theta dt+\sqrt{2\theta}dW$，$b_\theta=-\nabla f_0-\theta\nabla f_1+\hat b$，$\hat b=o(\theta)$；等價寫法 $dx=(b_\theta/\theta)dt+\sqrt2dW$（Eq. 13），平穩分佈相同。
- Assumption B.2（**假設，不是證明**）：平穩分佈管內質量 → 1，管內有局部 WKB 形式 $\pi_\theta\propto e^{-V/\theta}c_\theta$，$c_\theta\to c_0$（$C^2$），$c_0>0$，V 的解唯一。
- 平穩 Fokker–Planck $0=\mathrm{div}(-b_\theta\pi_\theta+\theta\nabla\pi_\theta)$，代入得 Eq. 14，按 θ 的冪次逐階比對：
  - $\theta^{-1}$：$\langle\nabla f_0,\nabla V\rangle=\|\nabla V\|^2$（Hamilton–Jacobi），取 $V=f_0$。
  - $\theta^0$：$\langle\nabla f_1,\nabla f_0\rangle c_0+\langle\nabla c_0,\nabla f_0\rangle=0$（輸運方程）；令 $\tilde c_0=e^{f_1}c_0$ 得 $\langle\nabla\tilde c_0,\nabla f_0\rangle=0$（Eq. 15）：$\tilde c_0$ 沿 $f_0$ 的梯度流（法向）不變。在 M 上 $\nabla f_0=0$，所以此式完全不限制 $\tilde c_0$ 在 M 上的切向變化。這是本文的難點：既有文獻假設唯一穩定點，流線匯到一點，$c_0$ 自動是常數；這裡流線匯到整個流形（Sec. 5）。
  - $\theta^1$（只在 M 上求值，$\nabla f_0=0$ 讓含 $\hat c$ 的高階項消失）：$\mathrm{Tr}(\nabla^2\tilde c_0)-\langle\nabla\tilde c_0,\nabla f_1\rangle=0$。轉到 $(u,r)$ 座標（Eq. 16）得 Lemma B.5（Eq. 17）；Lemma B.6 由 Eq. 15 的一、二次微分（Eq. 18–19）在 r = 0 求得 $\partial\tilde c_0/\partial r=0$、$\mathrm{Tr}(\partial^2\tilde c_0/\partial r^2)=\langle h(u),\partial\tilde c_0/\partial u\rangle$。
  - 代回後得到 $u$ 上二階部分為 Laplace–Beltrami $\Delta_M$、只有一階項、**沒有零階項**的方程；緊緻無邊界流形上由強極大值原理（Gilbarg–Trudinger Thm 3.5）$\tilde c_0(\cdot,0)$ 為常數；再由 $\theta^0$ 階推到管內同一常數。
- Thm B.3：$V=f_0$，$c_0=C\exp(-f_1)$。於是 $\pi_\theta\propto\exp(-(f_0+\theta f_1+o(\theta))/\theta)$，回到步驟 3。
- 用語提醒（照錄 `user_terms_note.md`）：正文第 5 節寫「c₀ 滿足拋物型（parabolic）PDE」，但附錄推出的是不含時間的二階方程，引用的 Gilbarg–Trudinger Thm 3.5 也是橢圓型的結果。理解證明時以附錄為準。
- 〔本頁推導〕兩個未交代處：強極大值原理只給「每個連通分支上為常數」，Assumption 2.1 沒寫連通（Lemma B.2 才加上 connected）；$\theta^{-1}$ 階也有 $V\equiv0$ 等其他解，論文以 Assumption B.2 的「V 唯一」排除。

#### 步驟 8：Thm 5.2（非梯度場）

- 假設 Assumption 2.1、2.2、Eq. 9，另加 $p_{data}\in C^2$；對 $\max\{-\beta,0\}<\alpha<2$，假設 SDE 有唯一平穩分佈且具局部 WKB 形式（Assumption B.2，$\theta=\sigma^{2-\alpha}$）。結論同 Thm 5.1。
- 證明：把 Eq. 8 寫成 $dX=\sigma^2s/\sigma^{2-\alpha}\,dt+\sqrt2dW$，$b_\theta=\sigma^2s$；由步驟 6，$f_0=d_M$、$f_1\equiv0$；Thm B.3 + Thm B.1 → 均勻。
- 作者的口號「one-line modification」：Langevin／corrector 更新中的分數乘上 $\sigma^\alpha$。

#### 步驟 9：逆問題與 CFG（Sec. 6，Eq. 10，Thm 6.1，App. B.6，App. C.2）

- Thm 6.1：同 Thm 5.2 的假設，$v$ 在 $\mathbb R^d$ 有界、在 $T_M(\epsilon)$ 上 $C^1$；$dx=-\nabla v\,dt-\sigma^\alpha\nabla f_\sigma dt+\sqrt2dW$（Eq. 10）的平穩分佈 → M 上密度 $\propto\exp(-v(\Phi(u)))\,dM/du$。證明：同 Thm 5.2，取 $f_1=v$（乘上 $\theta=\sigma^{2-\alpha}$ 後 $b_\theta=-\theta\nabla v-\nabla d_M+o(\theta)$）。似然項沒有被 temper，被 temper 的只有先驗分數。
- 這是**均勻先驗下的後驗**，不是 $p_{data}$ 先驗下的後驗；後者的 $o(1)$／精確條件是引用的既有充分條件，不是本文證明的下界。
- CFG（Sec. 6、App. C.2）：$s_t(x,y)=-\tfrac1{\sigma_t}\big(\epsilon(x,t,\varnothing)+w(\epsilon(x,t,y)-\epsilon(x,t,\varnothing))\big)$，$\sigma_t=\sqrt{1-\bar\alpha_t}$；TS 版 $\tilde s_t=-\tfrac1{\sigma_t}\big(\sigma_t^\alpha\epsilon(x,t,\varnothing)+w(\epsilon(x,t,y)-\epsilon(x,t,\varnothing))\big)$，只縮放無條件部分。引導項扮演 $-\nabla v$；作者稱此對應「can also be applied」，屬啟發式（$v$ 實為兩個學到的分數之差且隨 $\sigma_t$ 變）。
- 〔本頁推導〕Thm 6.1 要求 $v$ 在 $\mathbb R^d$ 有界，但 Sec. 2.2 的高斯似然 $v=\|A(x)-y\|^2/(2s^2)$ 在 $\mathbb R^d$ 無界；頁面 C2b 用有界的 $v=\lambda(1-\cos t)$。

#### 步驟 10：混合速度（Remark 5.1，App. D）

- 只在單位圓、梯度場、連續時間的例子上：$-\log p_\theta=d(x)/\sigma^2+\sigma^\beta\phi(x)$，$\phi=(|x_1|-1)^2$，$-2<\beta<0$。標準 Langevin 有兩個模態 $(\pm1,0)$，Poincaré 常數可為 $O(\exp(-\sigma^\beta))$（原文記號）；TS（$\alpha>-\beta$）由 Holley–Stroock 擾動與平滑位能 $V_c$（$c=\sigma^{3-\alpha}$）、Gong et al. 2024 的 log-PL 結果，得到與 σ 無關的下界。論文的 Poincaré 常數定義是 $\mathrm{Var}\le(1/C_{PI})\int\|\nabla f\|^2$，$C_{PI}$ 越大越快。
- 一般收斂率留待未來（Remark 5.1）。頁面只以文字介紹，不做元件（見 R6）。

#### 步驟 11：實驗（Sec. 7，App. C，Fig. 2–4，Table 1–4）

- Fig. 1 是示意圖（一維圓嵌在 ℝ²，說明「既有範式」與「新範式」），不是實驗結果。
- 橢圓模擬（Sec. 7.1、App. C.1）：$M=\{(x/a)^2+(y/b)^2=1\}$，$(a,b)=(1,1)$ 圓、$(1,2)$ 橢圓；$p_{data}$ 為角度參數上的 von Mises；Sec. 7.1 寫「evaluate the learned score function with σ = 10⁻² and perform Langevin dynamics until convergence」「Training hyperparameters are tuned to minimize the test loss」。Fig. 2：標準 Langevin（L）偏離 $p_{data}$，TS-1 接近內蘊均勻分佈；四格的縱軸 (a)(b)(d) 到 0.40、(c) 到 0.8，角度軸 −π 到 π。C.1 設定（頁面 S10 設定表 A，逐項照錄）：

  | 項目 | App. C.1 原文 |
  | --- | --- |
  | 網路 | 4 個 transformer block，hidden dimension 128 |
  | 訓練目標 | $\hat s(x,\sigma):=\sigma^2s(x,\sigma)$；損失 $\tfrac12\mathbb E_{u\sim p_{data}}\mathbb E_{x\sim\mathcal N(\Phi(u),\sigma^2I)}\big[\tfrac1{\sigma^2}\|\hat s(x,\sigma)+x-\Phi(u)\|^2\big]$ |
  | 資料 | von Mises，κ = 1 |
  | 噪音 | σ ∈ [0.01, 50] 抽樣（分佈未寫） |
  | 最佳化 | AdamW，weight decay 1×10⁻⁴，global gradient clipping 1.0 |
  | 學習率 | 初始 3×10⁻³，cosine 衰減「over 4×10⁻⁴ steps」到初值的 1%，之後「constant learning rate of 4×10⁻⁴」（兩處互相矛盾，見常見錯誤 7） |
  | batch | 1024 |
  | 取樣 | $dx=\hat s(x,\sigma_{min})dt+\sqrt{2\sigma_{min}^2}dW$，$\sigma_{min}=0.01$（與 $dx=s\,dt+\sqrt2dW$ 平穩分佈相同）；TS 擴散係數 $\sqrt{2\sigma_{min}^{2-\alpha}}$ |
  | 離散化 | Euler–Maruyama，步長 0.1，10,000 步，10,000 條 |
- 受控實驗（App. C.3，Fig. 4）：單位圓、κ = 4、θ₀ = π，真實分數加上 $e(x)=-\nabla\big(\tfrac12\|x-(1,0)\|^4\big)$（O(1)）；**標準反向擴散**（不是固定 σ 的 Langevin）偏離 $p_{data}$，TS-1 接近均勻。
- Stable Diffusion 1.5（Sec. 7.2、App. C.2；頁面 S10 設定表 B，逐項照錄）：

  | 項目 | App. C.2 原文 |
  | --- | --- |
  | 模型與採樣器 | 預訓練 SD 1.5（預設設定），DDPM 採樣器的 predictor–corrector 方案 |
  | CFG 分數 | $s_t=-\tfrac1{\sigma_t}(\epsilon(x,t,\varnothing)+w(\epsilon(x,t,y)-\epsilon(x,t,\varnothing)))$，$\sigma_t=\sqrt{1-\bar\alpha_t}$ |
  | TS 修改 | $\tilde s_t=-\tfrac1{\sigma_t}(\sigma_t^\alpha\epsilon(x,t,\varnothing)+w(\epsilon(x,t,y)-\epsilon(x,t,\varnothing)))$ |
  | corrector | 每個 DDPM predictor 步後 $n_{corr}$ 次：$x_{k+1}=x_k+\delta_i\tilde s_{t_i}(x_k,y)+\sqrt{2\delta_i}\xi_k$，$\delta_i$ 依 Song et al. 2021 Algorithm 5 |
  | 投影步 | 反向過程結束後 $n_{corr}$ 次 $dx_\tau=\nabla\log p_{t_0}(x_\tau)d\tau$（無引導、無噪音），PC 與 TS 相同 |
  | 超參數 | w = 7.5，30 inference steps |
  | Table 1 調參 | corrector 步數 {5, 10, 15, 20, 30}、α ∈ {0.1, 0.5, 1.0, 1.5} 格點搜尋；PC 在同樣步數上調 |
  | 評估 | 每設定 512 張，縮到 256×256 後計算 CLIP 分數 |

  指標：P-sim（生成圖與提示詞的平均 CLIP 分數，越高越好，品質）、I-sim（同提示詞所有圖兩兩平均 CLIP 分數，越低越多樣）。提示詞：“Creative furniture”、“An innovative car design”、“A creative architecture”。
- Table 1（P-sim↑ / I-sim↓）：Furniture DDPM 29.56/80.78、PC 29.40/81.24、TS 30.20/80.76；Car DDPM 26.23/87.30、PC 26.30/87.20、TS 26.62/87.14；Architecture DDPM 27.36/81.53、PC 27.13/81.03、TS 27.32/80.76。
- Table 2（TS 用 α = 1、不再調參；每格 P-sim↑ / I-sim↓；欄為 corrector 步數）：

  | 提示詞 | 方法 | 5 | 10 | 15 | 20 | 30 |
  | --- | --- | --- | --- | --- | --- | --- |
  | Furniture | PC | 29.40 / 81.34 | 29.30 / 81.24 | 29.32 / 81.64 | 28.98 / 81.72 | 28.67 / 82.33 |
  | Furniture | TS | 29.54 / 81.11 | 29.58 / 80.95 | 29.68 / 81.34 | 29.52 / 81.15 | 29.43 / 81.87 |
  | Car | PC | 26.20 / 87.20 | 26.30 / 87.57 | 26.24 / 87.98 | 26.26 / 88.06 | 26.17 / 87.94 |
  | Car | TS | 26.23 / 87.14 | 26.37 / 87.42 | 26.32 / 87.88 | 26.28 / 88.07 | 26.20 / 87.87 |
  | Architecture | PC | 27.13 / 81.83 | 27.13 / 81.81 | 26.92 / 81.64 | 26.87 / 81.60 | 26.60 / 81.03 |
  | Architecture | TS | 27.23 / 81.58 | 27.27 / 81.57 | 27.14 / 81.54 | 27.06 / 80.97 | 26.84 / 80.76 |

  30 組比較中 TS 只有一處不如 PC（Car、20 步的 I-sim：PC 88.06、TS 88.07），頁面以粗體與文字註明。
- Table 3（10 步）與 Table 4（20 步）（C.4；每格 P-sim / I-sim；α = 0 即 PC 基線）：

  | 步數 | 提示詞 | α = 0 | 0.1 | 0.5 | 1.0 | 1.5 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 10 | Architecture | 27.13 / 81.81 | 27.12 / 81.73 | 27.14 / 81.67 | 27.27 / 81.57 | 27.32 / 81.52 |
  | 10 | Furniture | 29.30 / 81.24 | 29.32 / 81.37 | 29.33 / 81.06 | 29.58 / 80.95 | 30.16 / 80.76 |
  | 10 | Car | 26.30 / 87.57 | 26.30 / 87.58 | 26.31 / 87.44 | 26.37 / 87.42 | 26.50 / 87.34 |
  | 20 | Architecture | 26.87 / 81.60 | 26.85 / 81.56 | 26.97 / 81.49 | 27.06 / 80.97 | 27.10 / 81.13 |
  | 20 | Furniture | 28.98 / 81.72 | 28.99 / 81.65 | 29.07 / 81.40 | 29.52 / 81.15 | 30.20 / 81.39 |
  | 20 | Car | 26.26 / 88.06 | 26.26 / 88.09 | 26.25 / 87.95 | 26.28 / 88.07 | 26.62 / 87.70 |

  作者稱 α ≥ 0.5 一致改善、α ∈ [1.0, 1.5] 特別穩定，α = 1.5 可再提升；例外見 R4。
- 〔本頁推導，逐格比對 Table 2–4〕Table 1 每格是分別調參的最佳值，同一列的兩個數可能來自不同設定：Furniture PC 的 29.40 是 5 步的 P-sim、81.24 是 10 步的 I-sim；Furniture TS 的 30.20 與 Table 4 α = 1.5、20 步相同，80.76 與 Table 3 α = 1.5、10 步相同。Table 1 中 DDPM 在 Architecture 的 P-sim（27.36）高於 TS（27.32）；PC 有 3 格不如 DDPM。所有差距約 0.01–1.2，無誤差棒、每類 1 個 prompt。

#### 步驟 12：實作細節與常見錯誤（頁面 S10 逐條列出）

1. 把「誤差 $o(\sigma^{-2})$ 就夠」讀成「誤差多大都行」：精確條件是 $o(\sigma^\beta)$、$\beta>-2$ 嚴格，且 $\max\{-\beta,0\}<\alpha<2$；誤差越接近 $\sigma^{-2}$，α 必須越接近 2，$\theta=\sigma^{2-\alpha}$ 趨近 0 越慢。摘要與結論的「Θ(σ⁻²) stronger」「tolerates O(σ⁻²) larger errors」是口語，定理層級是 $o(\sigma^{-2})$ 對 $o(1)$。
2. 只縮放 drift、忘了時鐘：C.1 的取樣寫法 drift 是 $\hat s=\sigma^2s$；標準 Langevin 擴散係數 $\sqrt{2\sigma^2}$，TS 改為 $\sqrt{2\sigma^{2-\alpha}}$（drift 不變）。兩種寫法與 Eq. 8 只差時間重標，平穩分佈相同。
3. 在 CFG 中縮放整個分數：TS 只乘無條件分數 $\sigma_t^\alpha\epsilon(x,t,\varnothing)$，引導增量 $w(\epsilon(x,t,y)-\epsilon(x,t,\varnothing))$ 不變（C.2）。
4. 把「均勻分佈」當成座標上平坦：它對黎曼體積均勻，座標密度 $\propto\sqrt{\det g}$；依賴嵌入。
5. 把 Thm 5.2 當無條件結論：它假設平穩分佈存在、唯一且具 WKB 形式（Assumption B.2）；只有梯度場（Thm 5.1）不需要 ansatz。
6. 把有限步的 corrector 當成定理設定：定理是固定小 σ、連續時間、跑到平穩；PC 每層只跑 5–30 步、σ_t 隨層變（論文自稱 proof of concept；離散化誤差未分析，Limitations d）。
7. 論文未交代或不一致的設定：C.1 寫學習率「cosine 衰減 over 4×10⁻⁴ steps」、降到初值的 1% 後「以常數 4×10⁻⁴ 繼續」（步數與數值互相矛盾，疑為筆誤）；總訓練步數、σ 的抽樣分佈、Langevin 初始分佈未寫；C.1 用 κ = 1、C.3 用 κ = 4、θ₀ = π，兩個實驗不要混用；C.2 投影步的步長與 corrector 的 signal-to-noise 參數未寫。

#### 步驟 13：限制（Sec. 8）

(a) 未追蹤取樣軌跡上的誤差累積，只假設知道最終生成分佈的誤差；(b) $L^\infty$ 誤差假設可望放寬為 $L^2$（實際訓練控制 $L^2$）；(c) 未轉成樣本複雜度；(d) 均勻取樣分析為連續時間，未量化離散化誤差；(e) 實驗是初步的。另外（Sec. 2.1 末句）：$p_{data}$ 取母體密度，不是有限樣本的經驗分佈；經驗分佈支撐在有限點集上、不滿足 Assumption 2.2 的嚴格正，所以定理不涵蓋記憶化（memorization）的情形〔本頁推導：後半句的推論〕。

---

## 2. 學習目標（驗收測試）

每個參考答案都必須寫在指定章節的**內文段落**中（不能只出現在元件、表格、圖、答案框或可收合的推導區塊）。每節結尾以 `<details>`（summary「檢查理解」）列該節檢查題與參考答案，原文照搬。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明加噪密度與 Langevin 平穩分佈 | VE 與 VP 的加噪方式差在哪？漂移 $s=-\nabla f$ 時 Langevin 的平穩分佈是什麼？ | VE：$X+\sigma Z$（$\gamma=1$）；VP：$\sqrt{1-\sigma^2}X+\sigma Z$（$\gamma=\sqrt{1-\sigma^2}$）（Eq. 1、3）；平穩分佈 $\propto\exp(-f)$（Sec. 2.3） | S1 |
| LO2 | 寫出流形相關的定義 | $d_M(x)$、$P_M(x)$、$T_M(\epsilon)$ 是什麼？$p_{data}$ 是對哪個測度的密度？ | $d_M=\tfrac12\mathrm{dist}^2(x,M)$（Eq. 2，有 ½）；$P_M$ 為 $T_M(\epsilon)$ 內唯一最近點；$T_M(\epsilon)=\{x:\mathrm{dist}(x,M)<\epsilon\}$；$p_{data}$ 是對 $U$ 上 Lebesgue 測度的密度（Assumption 2.2） | S2 |
| LO3 | 區分「對體積均勻」與「對座標均勻」 | 流形上的均勻分佈在座標 $u$ 上的密度是什麼？橢圓 $(a,b)=(1,2)$ 時 t 上密度的最大與最小值？ | $\propto\sqrt{\det g(u)}=dM/du$（Thm 5.1）；橢圓為 $\sqrt{a^2\sin^2t+b^2\cos^2t}/9.6884$，最大 0.2064（t = 0、π）、最小 0.1032（t = ±π/2），比值 2〔本頁推導〕 | S2 |
| LO4 | 讀懂 Eq. 6 的四項 | Eq. 6 有哪四項？各是什麼階？$\lim\sigma^2\log p_\sigma$ 是什麼？ | $-d_M/\sigma^2$（Θ(σ⁻²)，只含幾何）、$\log p_{data}(\Phi^{-1}(P_M(x)))$（Θ(1)）、$-\tfrac{d-n}2\log(2\pi\sigma^2)$（與 x 無關）、$H(x)$（曲率，與 σ 無關），再加一致的 o(1)；$\lim\sigma^2\log p_\sigma=-d_M$（Eq. 7） | S3 |
| LO5 | 比較分數的法向與切向大小 | 在離 M 固定距離的點，小 σ 時分數的法向分量近似什麼、誤差多大？單位圓 κ = 4、θ₀ = π、ρ = 1.1、φ = 90°、σ = 0.01 時兩分量各是多少？ | 法向 $\approx-(x-P_M(x))/\sigma^2$，誤差 O(1)（Eq. 20）；切向 O(1)、攜帶 $p_{data}$；例：法向（沿 $x/\|x\|$）−1000.455，切向（沿逆時針單位切向量 $(-\sin\varphi,\cos\varphi)$，指向密度中心 θ₀ = π）+3.636，比值約 275。在 M 上或離 M 約 σ 的點，法向分量分別是 O(1) 與約 1/σ〔本頁推導〕 | S3 |
| LO6 | 說出 Thm 4.1 的三個門檻 | 集中到 M、還原 $p_{data}$ 各需要多小的 $E_\sigma$？為什麼 Θ(1) 的誤差足以得到任意分佈？ | 集中：$E_\sigma=o(\sigma^{-2})$；還原：$E_\sigma=o(1)$；可構造 $E_\sigma=\Omega(1)$（構造中為有界 Θ(1)）使 $\pi_\sigma\to$ 任意 $C^1$ 的 $\hat\pi$，因為 $p_{data}$ 只出現在 Eq. 6 的 Θ(1) 項；$E_\sigma$ 是最終生成分佈 $\pi_\sigma$ 的 $L^\infty(K)$ 分數誤差，不是網路的訓練誤差（Thm 4.1） | S4 |
| LO7 | 使用 Thm B.1 的極限公式 | Thm B.1 的極限密度是什麼？套到 Thm 4.1(3) 時體積因子為什麼消掉？ | $\pi(u)\propto\exp(-f_1(u,0))|\partial^2f_0/\partial r^2(u,0)|^{-1/2}dM/du$；$f_1=-\log p_{data}+\log\sqrt{|\hat H|}$，r = 0 時 $\sqrt{|\hat H|}=dM/du$、$\partial^2f_0/\partial r^2=I$，所以 $\pi=p_{data}$ | S5 |
| LO8 | 寫出 TS Langevin 與 α 的範圍 | TS Langevin 怎麼寫？梯度情形下的平穩分佈與 θ？α 的範圍與三個端點各自的理由？ | $dX=\sigma^\alpha s\,dt+\sqrt2dW$（Eq. 8）；$\tilde\pi_\sigma\propto\exp(-\sigma^\alpha f_\sigma)$，$\theta=\sigma^{2-\alpha}$；$\max\{-\beta,0\}<\alpha<2$（Thm 5.1）：α > 0 使 $O(\sigma^2)$ 的 $p_{data}$／曲率項為 $o(\theta)$，α > −β 使誤差 $o(\sigma^{2+\beta})$ 為 $o(\theta)$，α < 2 使 θ → 0（逐項理由為〔本頁推導〕，與 Thm 5.2 證明（App. B.5）的「because α > max{−β, 0}」一致） | S6 |
| LO9 | 解釋 TS 極限與 $p_{data}$ 無關 | 為什麼 TS 的極限不含 $p_{data}$？有限 σ 時切向還殘留多少？ | $\sigma^2f_\sigma=d_M+O(\sigma^2)+o(\sigma^{2+\beta})$，後兩項皆 $o(\theta)$，$f_1\equiv0$，極限 $\propto\sqrt{\det g}$；精確分數時 $\tilde\pi_\sigma\propto p_\sigma^{\sigma^\alpha}$，切向殘留 $p_{data}^{\sigma^\alpha}$，von Mises 的最大／最小比在 σ → 0 時 $\approx e^{2\kappa\sigma^\alpha}$（κ = 4、α = 1、σ = 0.01 為 1.083，C2 精確值相同），法向寬度 $\sigma^{1-\alpha/2}$〔本頁推導〕 | S6 |
| LO10 | 說明 Thm 5.2 的證明結構 | Thm 5.2 比 Thm 5.1 多了哪些假設？WKB 的 $\theta^{-1}$、$\theta^0$、$\theta^1$ 階各給出什麼？ | 多假設 $p_{data}\in C^2$、平穩分佈存在唯一且具局部 WKB 形式（Assumption B.2）；$\theta^{-1}$：Hamilton–Jacobi，$V=f_0$；$\theta^0$：輸運方程 $\langle\nabla\tilde c_0,\nabla f_0\rangle=0$（Eq. 15），只說 $\tilde c_0$ 沿法向不變；$\theta^1$ 在 M 上：Laplace–Beltrami 加一階項、無零階項（Lemma B.5–B.6），強極大值原理 ⇒ $\tilde c_0$ 為常數（正文寫 parabolic，附錄實為橢圓型）。三階結論句寫在內文，推導細節才收合 | S7 |
| LO11 | 說明逆問題與 CFG 的對應 | Thm 6.1 的極限分佈是什麼？在 CFG 中 TS 改了哪一項？極限是哪一種後驗？ | $\propto\exp(-v(\Phi(u)))\,dM/du$（Eq. 10，先驗分數乘 $\sigma^\alpha$、v 有界、假設同 Thm 5.2）；只把無條件分數乘上 $\sigma_t^\alpha$：$\tilde s_t=-\tfrac1{\sigma_t}(\sigma_t^\alpha\epsilon(x,t,\varnothing)+w(\epsilon(x,t,y)-\epsilon(x,t,\varnothing)))$（C.2）；是均勻先驗下的後驗，不是 $p_{data}$ 先驗下的後驗 | S8 |
| LO12 | 讀出並評估實驗 | P-sim、I-sim 各代表什麼？Table 1 Furniture 的 PC 與 TS 數字？實驗有哪些限制？ | P-sim：生成圖與提示詞的平均 CLIP 分數，越高越好；I-sim：同提示詞圖兩兩平均 CLIP 分數，越低越多樣；Furniture PC 29.40/81.24、TS 30.20/80.76（Table 1）；每類 1 個 prompt、每設定 512 張、無誤差棒、作者稱初步；Table 1 每格分別調參，同列兩數可能來自不同步數與 α（Sec. 7.2、C.2、C.4） | S9 |

---

## 3. 頁面大綱

單欄閱讀（文字欄 680–720px），元件放加寬圖版（最大 1080px）。章節編號 JetBrains Mono（`01 · …`），標題一律陳述句、不擬人、不用比喻。重複視覺記號：**誤差階數軸**——σ = 0.01 時的對數軸，標出 $1$（$o(1)$，還原 $p_{data}$）與 $10^4=\sigma^{-2}$（$o(\sigma^{-2})$，集中到流形），在 Hero、S4、S6 出現〔本頁推導：σ⁻² 在 σ = 0.01 的數值〕。每節所用符號都在該節或更早定義；新符號第一次出現時就地定義。

| # | id | 標題 | 內容（依序） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈小噪音時，分數的主導項只含流形的幾何〉 | 副標：「噪音很小時，加噪密度取 log 後最大的一項只與到資料曲面的距離有關，資料在曲面上的分佈只出現在小得多的項；把取樣時的分數乘上 σ 的次方，可以在較大的分數誤差下取得曲面上的均勻樣本。」三張事實卡（見下）＋誤差階數軸。連結：arXiv `https://arxiv.org/abs/2509.24912`、`../index.html#paper-237`（在關聯圖中查看）、`score-geometry-lineage-hub.html` | 1 螢幕 |
| S1 | `basics` | 01 · 分數指向密度增加的方向，Langevin 動力學用它取樣 | 分數 $\nabla\log p$ 不需要歸一化常數；Eq. 1（VE／VP 的加噪方式，密度式 Eq. 3 移到 S2，在 M、Φ、u、$p_{data}$ 定義之後）；擴散模型反向過程的目標是 $p_\sigma$、σ → 0（Sec. 2.1）；Langevin 與平穩分佈 $\propto e^{-f}$；溫度 θ 與 $e^{-f/\theta}$；平穩 Fokker–Planck 方程一句定義；非梯度場無封閉式（Sec. 2.3）；PC 採樣器＝predictor 步＋corrector（Langevin）步。LO1。延伸：`score-geometry-score-matching-sde.html`、`eps-prediction-ddpm-score.html` | 450–550 字 |
| S2 | `manifold` | 02 · 資料落在低維流形上，距離與體積由座標圖決定 | Assumption 2.1–2.2；座標圖 Φ、$g=J^\top J$、$\sqrt{\det g}$；$P_M$、$d_M$（Eq. 2）、$T_M(\epsilon)$；Eq. 3（$p_\sigma$ 的積分式，VE $\gamma=1$、VP $\gamma=\sqrt{1-\sigma^2}$）；管狀座標 $\Phi(u,r)=\Phi(u)+N(u)r$（r 是法向座標）；靜態 SVG（橢圓 (1,2)、管狀鄰域、點 x、$P_M(x)$、法向線段標「$d_M=\tfrac12\|x-P_M(x)\|^2$」）；**本頁例子**：$p_{data}(t)=e^{\kappa\cos(t-\theta_0)}/(2\pi I_0(\kappa))$（von Mises；κ 控制集中程度、θ₀ 是中心、$I_0$ 只是使總和為 1 的歸一化常數；本頁 κ = 4、θ₀ = π）；「對體積均勻 ≠ 對座標均勻」與橢圓數字（LO3 答案寫在內文）；漸近符號 o、O、Θ、Ω 小表與「o(σ⁻²) 比 o(1) 寬鬆」一句；可收合符號表（1.4）。LO2、LO3。延伸：`score-geometry-manifold-detection.html`、`x-prediction-manifold-view.html`、`score-geometry-diffusion-guarantees.html` | 600–700 字 + SVG |
| S3 | `expansion` | 03 · 小 σ 時 log p_σ 的主導項只含到流形的距離 | 一維 Laplace 直覺（$\int g\,e^{-F/\theta}$ 由 F 的極小點主導，寬度 $\sqrt\theta$）；Eq. 6 四項逐項說明、Eq. 7；內文定義 $H(x)$：VE 時 $H=-\tfrac12\log|\hat H|$，$\hat H$ 是 $\tfrac12\|x-\Phi(u)\|^2$ 對 u 的 Hessian、在 M 上等於 $g$（曲率資訊，與 σ 無關）；單位圓封閉式中 $e_0=(\cos\theta_0,\sin\theta_0)$、$R=\|x/\sigma^2+\kappa e_0\|$；作者的解讀；可收合區塊：Corollary B.1、Eq. 12、單位圓封閉式（〔本頁推導〕，$R$ 在此定義）；C1；Eq. 20 與「在離 M 固定距離的點，法向 ~σ⁻²、切向 ~1」的限定。LO4、LO5（例子數值寫在內文）。延伸：`score-geometry-laplace-expansion.html` | 550–650 字 + C1 |
| S4 | `existing` | 04 · 集中到流形只需 o(σ⁻²) 的誤差，還原 p_data 需要 o(1) | $\pi_\sigma$、$f_\sigma$、$E_\sigma$（生成分佈的 $L^\infty(K)$ 分數誤差，不是網路訓練誤差）、Assumption 4.1；記號「$E_\sigma=o(\sigma^\beta)$」在此定義 β；Thm 4.1 三結論與白話；路徑積分一句；誤差階數軸；C3a（只有 β）。LO6 | 450–550 字 + C3a |
| S5 | `framework` | 05 · 一個通用公式給出低溫極限在流形上的形狀 | Gibbs 形式 $\exp(-f_\theta/\theta)$（Eq. 11）、$f_0$（決定集中在哪裡）、$f_1$（決定 M 上的形狀）；Thm B.1；三種套法的表（Thm 4.1(1)：只需 $f_0$；4.1(3)：$f_1=-\log p_{data}+\log\sqrt{|\hat H|}$；5.1：$\theta=\sigma^{2-\alpha}$、$f_1\equiv0$）；體積因子相消一句。LO7 | 350–450 字 |
| S6 | `tempered` | 06 · 把分數乘上 σ^α，極限變成流形上的均勻分佈 | Eq. 8；梯度情形 $\tilde\pi\propto\exp(-\sigma^\alpha f_\sigma)$、$\theta=\sigma^{2-\alpha}$；Thm 5.1（含 Assumption 4.1 套到 $\tilde\pi_\sigma$、Eq. 9）；一句說明 Θ(1) 誤差屬於「任取接近 0 的 β < 0」的 $o(\sigma^\beta)$，所以任何 α > 0 都在範圍內；α 三端點理由；$p_\sigma^{\sigma^\alpha}$ 直覺、法向寬度、$\approx e^{2\kappa\sigma^\alpha}$〔本頁推導〕；C2a；C3b（β 與 α）。LO8、LO9。延伸：`score-geometry-tempered-langevin.html` | 600–700 字 + C2a + C3b |
| S7 | `wkb` | 07 · 分數不是梯度場時，WKB 展開給出同一結論 | 為何非梯度沒有封閉式；Eq. 20 的角色；Assumption B.2（標明是假設）；三階比對：每階一行式子＋一句白話結論（LO10 答案在內文），推導細節（Eq. 14、16–19）收合；強極大值原理（只敘述）；用語提醒（parabolic）；連通性注記〔本頁推導〕；Thm 5.2；C4。LO10。延伸：`score-geometry-sde-pde-wkb.html` | 600–700 字 + C4 |
| S8 | `inverse` | 08 · 逆問題中，均勻先驗對分數誤差較穩健 | Eq. 4 回顧；Thm 6.1、Eq. 10 與條件；「均勻先驗的後驗 ≠ 資料先驗的後驗」；v 有界的要求與高斯似然〔本頁推導〕；CFG 公式（C.2）與「只縮放無條件分數」；另一種解讀 $w/\sigma_t^\alpha$〔本頁推導〕；C2b。LO11。延伸：`score-geometry-inverse-problems-guidance.html` | 450–550 字 + C2b |
| S9 | `experiments` | 09 · 實驗在橢圓與 Stable Diffusion 1.5 上進行，作者稱為初步結果 | Fig. 1 是示意圖一句；Sec. 7.1 與 Fig. 2 結論（文字描述，不重繪）；C.3／Fig. 4（反向擴散；C2 的 Θ(1) 選項是固定 σ 的 Langevin 版，不是重現）；Table 1、Table 2 全表（唯一劣於 PC 的格粗體＋文字）、Table 3–4 全表（1.4 步驟 11）；P-sim／I-sim 定義；Table 1 逐格出處 callout〔本頁推導〕。LO12。延伸：`score-geometry-eval-metrics.html` | 450–550 字 + 4 表 |
| S10 | `impl` | 10 · TS 只改 corrector 的一行，實作有七個常見錯誤 | 設定表 A（C.1）、設定表 B（C.2）（1.4 步驟 11）；先定義 δ（步長）與 ξ（每步獨立的標準常態向量），再給一行修改的虛擬碼（兩欄：程式／說明，同參考頁版面）：`x = x + δ·σ^α·s(x,σ) + sqrt(2δ)·ξ`；常見錯誤 7 條（1.4 步驟 12） | 350–450 字 + 2 表 |
| S11 | `history` | 11 · 本研究建立在分數學習、收斂理論與流形偵測之上 | 時間線（6.6）＋一段話：分數學習（2019–2020）→ 收斂保證（2022）→ 流形偵測（2022–2023）→ 逆問題與 guidance → 均勻採樣（2025）→ 本文 | 時間線 + 100–150 字 |
| S12 | `next` | 12 · 論文列出五項限制，另有有限樣本的落差 | Sec. 8 限制 (a)–(e) 與有限樣本／記憶化一句（1.4 步驟 13）；App. D 混合速度一段（步驟 10）；研究筆記清單（hub 閱讀順序）；arXiv | 250–300 字 + 清單 |
| — | — | 頁尾 | 引用格式、arXiv；`#pm-bar` | — |

Hero 事實卡（不用未定義術語）：
1. 「資料落在高維空間中的低維曲面（流形）上時，加噪密度取 log 後，最大的一項只與『到曲面的距離』有關，與資料在曲面上的分佈無關。」
2. 「論文證明：若最終生成分佈的分數（以最大誤差衡量）誤差比 1/σ² 小，樣本就會集中到曲面上；若誤差比常數小，就能還原資料在曲面上的分佈。誤差只到常數量級時，論文能構造出收斂到曲面上任意分佈的例子。」
3. 「把取樣時的分數乘上 σ^α，只要 α 介於 0 與 2 之間且大於誤差階數 −β，樣本會依曲面的體積均勻分佈；誤差越接近 1/σ²，α 必須越接近 2，收斂越慢。」

誤差階數軸規格：水平對數軸 $10^{-1}$ 到 $10^5$；兩個標記「$o(1)$：還原 $p_{data}$」在 $10^0$、「$o(\sigma^{-2})$：集中到流形」在 $10^4$；第二個標記下加一行「TS 均勻：誤差 $o(\sigma^\beta)$、β > −2，且需 α > −β」；軸下註「σ = 0.01 時的分數誤差量級（示意）；o(f) 表示比 f 小的量，見 02 節」。標籤用參考頁 k 軸的分層演算法避免重疊（依左緣排序，重疊就上移一列）。

桌機（≥ 1024px）左側 sticky 章節目錄；< 1024px 頂部橫向捲動章節列（照參考頁）。

---

## 4. 互動元件規格

### 4.0 共用數學（`verify/core.js`，逐字複製）

- 設定：VE、$d=2$、$n=1$；$p_{data}(t)=e^{\kappa\cos(t-\theta_0)}/(2\pi I_0(\kappa))$，**κ = 4、θ₀ = π**（App. C.3 的值；頁面註明 C.1 的橢圓實驗用 κ = 1）。$t$ 是角度參數，$p_{data}$ 是對 $t$ 的 Lebesgue 密度。
- `logI0(x)`、`ratioOverX(x)`$=I_1(x)/(xI_0(x))$：x < 3.75 用 A&S 9.8.1/9.8.3，3.75 ≤ x < 50 用 A&S 9.8.2/9.8.4，x ≥ 50 用漸近級數（$\log I_0=x-\tfrac12\log2\pi x+\log(1+\tfrac1{8x}+\tfrac9{128x^2}+\tfrac{225}{3072x^3}+\tfrac{11025}{98304x^4})$；$I_1/I_0=1-\tfrac1{2x}-\tfrac1{8x^2}-\tfrac1{8x^3}-\tfrac{25}{128x^4}-\tfrac{13}{32x^5}$）。不直接算 $I_0$（R 可達 $10^4$ 會溢位）。精度：`ref_out.txt`，log I0 誤差 ≤ 3×10⁻⁷、比值誤差 ≤ 6×10⁻⁷（最大在 x ≈ 49）。`ratioOverX(x<1e-8)=1/2`（避免 0/0）。
- 單位圓：`circleLogP(x1,x2,σ)`（1.4 步驟 2 的封閉式）；`circleScore` 回傳 $s=-x/\sigma^2+\frac{I_1(R)}{R\,I_0(R)}\frac{x/\sigma^2+\kappa e_0}{\sigma^2}$；`circleScorePolar(ρ,φ,σ)` 回傳徑向 $s_{rad}=-\rho/\sigma^2+q(\rho/\sigma^2+\kappa\cos(\varphi-\theta_0))$ 與切向 $s_{tan}=-q\kappa\sin(\varphi-\theta_0)$，$q=$`ratioOverX(R)`$/\sigma^2$。切向分量直接由公式算，不由兩個 $10^4$ 量級向量相減。
- 一般橢圓：`makeEllipseLogP(a,b,σ)` 用週期梯形法對 $t'$ 積分，細節點 $N'=128\cdot\lceil\mathrm{clamp}(8\pi\max(a,b)/\sigma,256,6000)/128\rceil$（σ = 0.01、b = 2 時 5120）。**視窗化**：先用 128 個粗節點（每個粗區間的中點）算距離 $d_c$ 與最小值 $d_{min}$；只保留滿足 $(d_c-\Delta)^2\le d_{min}^2+2\sigma^2(40+2\kappa)$ 的粗區間（或 $d_c\le\Delta$），$\Delta=\max(a,b)\cdot2\pi/128$。理由：區間內真實距離 ≥ $d_c-\Delta$，$2\kappa$ 是 $\log p_{data}$ 的變化範圍，所以被丟掉的區間每個節點相對最大項都小於 $e^{-40}$。在保留區間的細節點上做 log-sum-exp 兩趟（先存指數與最大值，再只對比最大值大於 −40 的節點取 exp）。`check_c2_perf.js`：視窗化與完整求積的結果**逐位相同**（被丟的節點本來就在 −40 截斷之外），σ = 0.01 時一次完整更新（180 × 49 點）19.5–22.8 ms、拖曳用的 90 列 9.9–11.5 ms，完整求積約 100 ms（node 22，桌機）。a = b = 1 時與封閉式最大差 ≤ 5×10⁻⁹（`c2_out.txt` 開頭）。
- `mulberry32(seed)`；常態數用 Box–Muller 兩個分支：$r=\sqrt{-2\ln(1-u_1)}$，$(g_1,g_2)=(r\cos2\pi u_2,r\sin2\pi u_2)$。

### C1 · log p_σ 的四項與分數的兩個方向（S3）

- **教的概念**：Eq. 6 各項的階數；Eq. 7；Eq. 20；在離 M 固定距離的點，法向分量 ~σ⁻²、切向分量 ~1〔本頁推導的示例；論文的速率分離指 Eq. 6 的 log 密度階數〕。
- **簡化**（頁面寫出）：單位圓、VE、κ = 4、θ₀ = π，用封閉式計算精確 $p_\sigma$（〔本頁推導〕）。
- **輸入**：σ 離散滑桿，值 {0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5}，預設 0.1（索引 6）；φ ∈ [−180°, 180°] 步長 1°，預設 90°；ρ ∈ [0.80, 1.20] 步長 0.01，預設 1.10。
- **計算**（`eq6Terms`、`circleScorePolar`）：$A=-\tfrac12(\rho-1)^2/\sigma^2$、$B=\log p_{data}(\varphi)$、$C=-\tfrac12\log(2\pi\sigma^2)$、$D=-\tfrac12\log\rho$、殘差 $=\log p_\sigma-(A+B+C+D)$；$s_{rad}$（沿 $x/\|x\|$）、$-(\rho-1)/\sigma^2$、$s_{rad}+(\rho-1)/\sigma^2$、$s_{tan}$（沿逆時針 $(-\sin\varphi,\cos\varphi)$）、極限 $-\kappa\sin(\varphi-\theta_0)/\rho$、比值 $|s_{rad}/s_{tan}|$。
- **顯示**：
  - 左圖「沿射線的 σ²·log p_σ」：橫軸 ρ ∈ [0.6, 1.4]（120 點），縱軸固定 [−0.10, 0.05]；兩條線：$\sigma^2\log p_\sigma$（`--c-par` 實線，標「σ²·log p_σ」）、$-d_M=-\tfrac12(\rho-1)^2$（`--ink` 虛線，標「−d_M（σ → 0 的極限，Eq. 7）」）；目前 ρ 以細直線標示。超出縱軸範圍的部分裁掉，並在圖頂端標「超出範圍 ↑／↓」。
  - 右圖「各項大小對 σ」：雙對數，橫軸 σ ∈ [0.01, 0.5]（60 個對數等距點，與滑桿的離散值無關），縱軸 $[10^{-4},10^3]$；五條線以線型區分：|A|（粗實線，「幾何 −d_M/σ²」）、|B|（虛線，「密度 log p_data」）、|C|（點線，「常數項」）、|D|（點虛線，「曲率 H」）、|殘差|（細實線，「殘差 o(1)」）。寬度 ≥ 600px 用直接標籤；< 600px 改為圖下方的圖例（線型樣本＋文字），避免五個標籤重疊。目前 σ 以直線標示。值 < 10⁻⁴（ρ = 1 時 A = D = 0、φ 使 B 接近 0、σ ≈ 0.399 時 C 過零）畫在下緣，實際值見讀數表。
  - 讀數表（JetBrains Mono）：log p_σ、A、B、C、D（四位小數）、殘差（科學記號三位）、$s_{rad}$、$-(\rho-1)/\sigma^2$、$s_{rad}+(\rho-1)/\sigma^2$（三位小數）、$s_{tan}$、切向極限（四位小數）、比值（一位小數；$|s_{tan}|<10^{-9}$ 時顯示「—（切向為 0）」）。標籤欄是建置時渲染一次的靜態 KaTeX，更新時只改數字節點的 `textContent`。
- **防護**：ρ ≥ 0.8 不碰原點；σ = 0.01 時 R ≈ 10⁴，`logI0` 走漸近式；A 與 $\log I_0(R)$ 的相消在 double 下誤差約 10⁻¹²。
- **讀者應觀察到**：(1) σ 變小時左圖的實線貼近虛線（Eq. 7）；(2) 右圖 |A| 以斜率 −2 上升、|B| 與 |D| 水平、|C| 只按 log 變化、殘差下降（本例約按 σ²）；(3) 在 ρ = 1.1（距離 0.1）處，法向分量與 $-(\rho-1)/\sigma^2$ 的差維持 O(1)（約 −1/(2ρ)），切向分量趨近 $-\kappa\sin(\varphi-\theta_0)/\rho$，兩者比值約按 σ⁻² 增長；(4) 把 ρ 調成 1.00：法向分量本身變成 O(1)（−0.500），說明「法向 ~σ⁻²」只在固定距離成立。
- 圖版底註：「讀數由本頁依 Eq. 6、7、20 與單位圓封閉式即時計算」＋「本頁推導」標籤＋「封閉式由 Eq. 3 的 u 積分以 Bessel 函數寫出」。

### C2 · TS Langevin 的平穩分佈（梯度場情形）（C2a 在 S6；C2b 在 S8）

- **教的概念**：α = 0 時極限為 $p_{data}$（或被 Θ(1) 誤差改成別的分佈）；$\max\{-\beta,0\}<\alpha<2$ 時極限為 $\sqrt{\det g}$；橢圓上「均勻」在 t 上不平坦；(C2b) Thm 6.1 的 $e^{-v}\,dM/du$。
- **簡化**（頁面寫出）：分數是梯度場（Thm 5.1 的情形），所以平穩分佈有封閉式 $\tilde\pi_\sigma\propto\exp(\sigma^\alpha[\log p_\sigma+U_{err}]-v)$，$U_{err}$ 是誤差位能（誤差場 $=\nabla U_{err}$；與 S4 的誤差大小 $E_\sigma$ 不同字母）；直接數值積分，不跑粒子。非梯度場見 C4。
- **輸入**：
  - 形狀：分段按鈕「圓 (1, 1)」「橢圓 (1, 2)」，預設圓。
  - σ 離散滑桿，值 {0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2}，預設 0.05（索引 4）。
  - α ∈ [0, 1.9] 步長 0.05，預設 0（C2b 預設 1）；chip「L（α = 0）」「TS-1（α = 1）」「TS-1.5」。
  - 誤差（全部是梯度場）：「無」（預設）／「Θ(1) 誤差（App. C.3）」$U_{err}=-\tfrac12\|x-(1,0)\|^4$／「切向誤差 σ^β∇cos t」$U_{err}=\sigma^\beta\cos t(x)$，選此項時出現 β ∈ [−1.9, 1] 步長 0.1（預設 −1）。頁面寫明：這個誤差恰好是 $O(\sigma^\beta)$ 而不是定理要求的 $o(\sigma^\beta)$，刻意落在邊界階數上，所以 α = −β 時看得到殘留；$\cos t(x)$ 在原點不可微，對應的 K 取不含原點的環形區域（數值上 r ≥ 0.02，該區質量可忽略）。
  - 只在 C2b：λ ∈ [0, 3] 步長 0.1，預設 1.5，$v(x)=\lambda(1-\cos t(x))$（有界，Thm 6.1 的要求）。C2a 的 λ 恆為 0、不顯示。
- **計算**（`c2Grid` → `c2FillRows` → `c2Combine`）：廣義極座標 $x=(a\,r\cos t,\,b\,r\sin t)$，Jacobian $abr$（覆蓋整個平面，不受管狀鄰域半徑限制）；t 格點 180 個中點（2°；拖曳中為 90），r 用 49 點梯形法於 $[\max(0.02,1-8w),1+8w]$，$w=\sigma^{1-\alpha/2}$。`c2Grid`／`c2FillRows` 只算 $\log p_\sigma$ 網格（圓用封閉式、橢圓用視窗化求積），依 (形狀, σ, α, t 格點數) 快取；`c2Combine` 加上誤差位能、v、Jacobian，每個 t 對 r 做 log-sum-exp，**t 方向也在 log 域減去最大值後才取 exp 並正規化**（切向誤差 β = −1.9、σ = 0.01、α = 0 時 $\sigma^\alpha U_{err}$ 約 6300，log 範圍約 5476 個十進位數量級，結果仍有限，見 selfTest）。改誤差、β、λ 只重跑 `c2Combine`（0.2 ms）。參考曲線（同一 t 格點、各自正規化）：$p_{data}$、內蘊均勻 $\sqrt{a^2\sin^2t+b^2\cos^2t}/L$、Θ(1) 誤差時 Thm 4.1 框架的極限 $\propto p_{data}(t)\exp(-\tfrac12\|\Phi(t)-(1,0)\|^4)$〔本頁推導〕、C2b 的 $e^{-v}\sqrt{\det g}$（均勻先驗後驗）與 $e^{-v}p_{data}$（資料先驗後驗）。
- **排程**（橢圓）：輸入事件取消前一個計算（每次新輸入遞增 `token`，舊的分段看到 token 不符就停止）；先 `requestAnimationFrame` 讓「計算中」狀態畫出，再以 `setTimeout(…, 0)` 分段呼叫 `c2FillRows`，每段做到耗時約 8 ms 為止（約 60 列）。拖曳中（`input`）用 90 列網格，放開（`change`）後補算 180 列。圓不分段（< 3 ms）。實測耗時見 4.0；仍不足時才改用行內 Blob Worker（R10）。
- **顯示**：
  - 主圖：橫軸 t ∈ [−π, π]（刻度 −π, −π/2, 0, π/2, π），縱軸 [0, min(max(0.85, 1.05·峰值), 4)]，超過的部分裁掉並在峰值處標「峰值 = …」；計算結果（`--accent` 粗實線，標「TS 平穩分佈」或 α = 0 時「Langevin 平穩分佈」）、$p_{data}$（`--c-perp` 虛線）、內蘊均勻（`--c-par` 點線）、其他參考線只在對應選項出現（點虛線＋直接標籤；< 600px 改圖例）。
  - 小圖：流形曲線與寬 ±w 的法向帶（w > 0.6 時帶寬截在 0.6 並標「w = …」），讓讀者看到法向寬度 $\sigma^{1-\alpha/2}$。
  - 讀數：TV(結果, $p_{data}$)、TV(結果, 內蘊均勻)（TV 為 ½∫|p − q|dt，四位小數）、最大／最小比（由 log 範圍算；> 10⁶ 時顯示「≈ 10^k」）、w、θ = σ^{2−α}、判定行「(β, α) 在 Thm 5.1 的範圍內／定理不涵蓋」（無誤差與 Θ(1) 誤差：α > 0 即在範圍內；切向誤差：需 α > max(−β, 0)；α ≥ 2 不開放）。w ≥ 0.3 時加註：「此時法向寬度大，圖中的 t 邊際是廣義極座標角度的邊際，還不是 Thm 5.1 所說的流形上的分佈」。C2b 另顯示 TV(結果, 均勻先驗後驗)、TV(結果, 資料先驗後驗)。
  - 「計算中」用 `role="status"` 的獨立元素；讀數區的 `aria-live` 只在計算完成後更新一次，不重複宣告。
- **防護**：log-sum-exp 防止 $\exp(\sigma^\alpha\log p_\sigma)$ 下溢；r 下限 0.02 避開原點；切向誤差在 β = −1.9、α = 0 時峰值極高，只影響顯示裁切；快取最多 40 筆網格，超過清空。
- **讀者應觀察到**：(1) α = 0、無誤差：結果與 $p_{data}$ 重合（σ = 0.01 時 TV < 0.001）；(2) α = 1：結果靠近內蘊均勻，σ 越小越近（圓 σ = 0.01 時最大／最小比 1.083）；(3) 橢圓時 TS 的極限在 t 上不平坦，最大／最小比趨近 2；(4) Θ(1) 誤差：α = 0 收斂到 $\propto e^{-2\cos^2t}$ 的雙峰（圓），α = 1 仍趨近均勻；(5) 切向誤差 β = −1：對**這個邊界階數的誤差**，α = 1（= −β）時結果偏離均勻且不隨 σ 變小而改善（σ = 0.05、0.01 的 TV(均勻) 為 0.239、0.279），α = 1.5 時趨近均勻；這只說明此例在邊界上不收斂，不代表定理的邊界是緊的；(6) C2b：α = 1 趨近 $e^{-v}\sqrt{\det g}$，α = 0 趨近 $e^{-v}p_{data}$，兩者不同。
- 頁面在 C2a 下方加一句：「Θ(1) 選項是固定 σ 的 Langevin 版本；論文 Fig. 4(a) 用的是反向擴散，形狀不同，本元件不重現 Fig. 4(a)。」

### C3 · 誤差階數與定理結論（C3a 在 S4；C3b 在 S6，在 C2a 之後）

- **教的概念**：C3a：Thm 4.1 的門檻（β ≥ −2 保證集中、β ≥ 0 保證還原，對應 LO6）。C3b：再加上 Thm 5.1 的 α 範圍，為什麼 α 必須在 (max{−β,0}, 2) 內。
- **同一份程式兩個實例**：`initC3(root, {withAlpha})`。C3a 不顯示 α（固定 α = 0），平面退化成一條 β 軸（高 120px 的帶狀圖）；C3b 為完整平面。
- **輸入**：β ∈ [−3, 1] 步長 0.05（預設 −1）；（C3b）α ∈ [0, 2] 步長 0.05（預設 1）；點擊／觸控圖面直接設定（四捨五入到 0.05）；（C3b）chip「論文實驗 α = 1」「α = −β 邊界」（設為 (−1, 1)）。
- **顯示**：C3b 平面圖，橫軸 β ∈ [−3, 1]、縱軸 α ∈ [0, 2]。區域與線：β < −2 斜線填色（「Thm 4.1(1) 不保證集中」）；TS 有效區 $\{\beta>-2,\ \max(-\beta,0)<\alpha<2\}$ 淺色填滿（開邊界用虛線：α = −β、α = 0（β ≥ 0 段）、α = 2、β = −2）；直線 β = 0（「o(1)：Thm 4.1(3)」）、β = −2（「o(σ⁻²)：Thm 4.1(1)」）；目前點為 `--accent` 實心圓。C3a 只畫 β 軸上的三段（β < −2、−2 ≤ β < 0、β ≥ 0）與目前點。
- **讀數**（σ = 0.01 的數值〔本頁推導〕）：（兩者）誤差量級 σ^β；（C3b）θ = σ^{2−α}、法向寬度 σ^{1−α/2}、誤差影響因子 σ^{α+β}（< 1 才隨 σ 衰減）、$p_{data}^{\sigma^\alpha}$ 殘留比的漸近式 $\approx e^{2\kappa\sigma^\alpha}$。判定清單（每行以「定理保證／定理不保證」文字＋形狀符號 ●／○，不只靠顏色；論文只證明充分條件，定理外一律寫「不保證」，不寫「不成立」）：
  1. α = 0 時：β ≥ −2 ⇒ 保證集中到 M（Thm 4.1(1)）；β ≥ 0 ⇒ 保證還原 $p_{data}$（Thm 4.1(3)）；−2 ≤ β < 0 ⇒ 不保證還原，且論文構造了收斂到 M 上任意分佈的例子（Thm 4.1(2)）。
  2. （C3b）α > 0 時：β > −2 且 max(−β, 0) < α < 2 ⇒ 保證收斂到內蘊均勻（Thm 5.1，梯度場）；否則「定理不保證」。另附一行「分數不是梯度場的情形見 07 節」（連結，不在此提 Assumption B.2）。
  3. 註記：Thm 4.1 談最終生成分佈的分數誤差；梯度場 Langevin 的平穩分佈就是 $\pi_\sigma=e^{-f_\sigma}$，所以可直接套用〔本頁推導〕。
- **讀者應觀察到**：C3a：β 越過 −2 與 0 時保證的內容改變。C3b：誤差越接近 σ⁻²（β → −2），可用的 α 越接近 2，θ 與法向寬度越接近 1（收斂越慢）。

### C4 · 粒子模擬：一行修改與非梯度誤差（S7）

- **教的概念**：TS 只改一行（drift 乘 σ^α，等價於 C.1 時鐘下擴散係數改為 $\sqrt{2\sigma^{2-\alpha}}$）；粒子直方圖收斂到 C2 算出的平穩分佈；非梯度誤差下（Thm 5.2 的情形）標準 Langevin 偏離 $p_{data}$，TS 仍接近均勻。
- **簡化**（頁面寫出）：單位圓、精確分數（封閉式）加人工誤差場；這是數值示意，不是 Thm 5.2 的證明；也不是 Fig. 2 的重現（Fig. 2 用訓練的網路）。
- **演算法**（App. C.1 的時鐘）：$x\leftarrow x+\delta\,\sigma^2\,(s^*(x,\sigma)+e(x))+\sqrt{2\sigma^{2-\alpha}\delta}\,\xi$，δ = 0.1（C.1 的步長）。左面板 α = 0（L），右面板 α = α_TS；兩面板每步用同一組 $\xi$（同一個 PRNG 序列，每個粒子每步取 $u_1,u_2$ 一次）。初始：seed 237，每個粒子 $t\sim U(-\pi,\pi)$、放在 $(\cos t,\sin t)$。
- **誤差場**：「無」；「Θ(1) 梯度誤差（C.3）」$e=-2\|x-(1,0)\|^2(x-(1,0))$；「非梯度旋轉誤差」$e=4(1-x_1/\rho)\,\hat\tau$，$\hat\tau=(-x_2,x_1)/\rho$（沿圓的環量 ≠ 0，所以不是梯度場；量級 Θ(1)，屬於任意 β < 0 的 $o(\sigma^\beta)$，Thm 5.2 要求 α > 0）。ρ < 10⁻⁹ 時 e = 0。
- **輸入**：σ 分段 {0.05, 0.1, 0.2}（預設 0.1）；α_TS ∈ {0.25, 0.5, …, 1.5}（預設 1）；誤差分段（預設「無」）；按鈕「執行／暫停」「重設」；改變任何設定時自動重設。粒子數 N = 1000／面板，總步數 10,000（C.1 的步數）。
- **繪圖**：用 `<canvas>`（2000 個點每影格重畫，SVG 節點太多）。顏色由 `getComputedStyle(document.documentElement)` 讀 token（`--ink`、`--accent`、`--c-par`、`--c-perp`、`--line`），在初始化、`pm-theme` 切換（主題按鈕的事件與 `matchMedia('(prefers-color-scheme: dark)')` 的 change）與 resize 時重讀並重畫；canvas 尺寸乘 `devicePixelRatio`。軸刻度與標籤文字放在 canvas 外的 HTML。
- **顯示**：每個面板上半為平面散點（[−2, 2]²，點半徑 1.5 CSS px，畫上單位圓），下半為 36 格角度直方圖（長條，正規化成密度），疊上 $p_{data}$（虛線）、均勻 1/(2π)（點線）、梯度誤差選項時疊上 C2 的精確平穩分佈（同 σ、同 α，實線，標「C2 精確解」）；三種參考線在直方圖下方以 HTML 圖例（線型樣本＋文字）說明。讀數：步數、mean cos、mean sin、36 格 TV(直方圖, 均勻)（附註「N = 1000 時即使完全均勻，TV 的期望約 0.076」〔本頁推導〕）、重設次數；只更新數字節點。
- **效能**：每影格先跑 25 步，量測耗時後調整，使每影格 ≤ 12 ms；10,000 步約 3–4 秒（node 實測 20M 粒子步 1.2–2.5 s）。
- **防護**：結果非有限或 $\|x\|^2>2500$ 時把該粒子放回 (1, 0) 並計數（驗收中從未觸發）。
- **reduced-motion**：不自動執行；按執行時以 `setTimeout` 分段、每段 100 步（約 20 萬粒子步，桌機約 12–25 ms），每 5 段重畫一次。
- **讀者應觀察到**：(1) 無誤差：L 的直方圖貼近 $p_{data}$，TS 較平；(2) Θ(1) 梯度誤差：L 偏離 $p_{data}$（mean cos 由 −0.86 變為約 −0.10），兩個面板都貼近各自的 C2 精確解；(3) 旋轉誤差：L 的質量轉到 t ≈ −86°（mean sin ≈ −0.80），TS 的 TV 與無誤差時相近；(4) σ = 0.05 時 L 前 1000 步還沒混合完成（mean cos −0.44，終值 −0.85），TS 已接近終值。

---

## 5. 驗收數值表

全部由 `verify/` 腳本以 `core.js`（與頁面相同的程式）產生。容許誤差：同一演算法只差浮點加總順序與瀏覽器數學函式末位。

### 5.1 C1（`check_c1.js`）

ρ = 1.10、φ = 90°（預設點；B = −4.2628、D = −0.0477 與 σ 無關）。容許 ±1 於最後一位；殘差 ±2%。

| σ | log p_σ | A | C | 殘差 | $s_{rad}$ | $s_{rad}+(\rho-1)/\sigma^2$ | $s_{tan}$ | 比值 | σ²·log p_σ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5 | −3.1372 | −0.0200 | −0.2258 | 1.419e+0 | −1.702 | −1.3021 | 2.4526 | 0.7 | −0.78431 |
| 0.2 | −3.4563 | −0.1250 | 0.6905 | 2.887e−1 | −3.210 | −0.7096 | 3.5331 | 0.9 | −0.13825 |
| 0.1 | −3.3533 | −0.5000 | 1.3836 | 7.351e−2 | −10.521 | −0.5210 | 3.6174 | 2.9 | −0.03353 |
| 0.05 | −4.2153 | −2.0000 | 2.0768 | 1.845e−2 | −40.471 | −0.4713 | 3.6321 | 11.1 | −0.01054 |
| 0.02 | −13.8145 | −12.5000 | 2.9931 | 2.954e−3 | −250.457 | −0.4572 | 3.6357 | 68.9 | −0.00553 |
| 0.01 | −50.6235 | −50.0000 | 3.6862 | 7.386e−4 | −1000.455 | −0.4552 | 3.6362 | 275.1 | −0.00506 |

切向極限 3.6364、$-d_M=-0.00500$、$-1/(2\rho)=-0.4545$。其他驗收點（σ = 0.01）：ρ = 1.0、φ = 180°：log p_σ = 3.4232、殘差 −1.875e−4、$s_{rad}$ = −0.500、$s_{tan}$ = 0（比值顯示「—」）；ρ = 0.9、φ = 30°：log p_σ = −53.9876、殘差 4.287e−4、$s_{rad}$ = 999.444、$s_{tan}$ = 2.2230（極限 2.2222）。

### 5.2 C2（`check_c2.js`，完整輸出見 `c2_out.txt`）

容許：TV ±0.0005；最大／最小比 ±0.5%。圓與橢圓同為 κ = 4、θ₀ = π。

| 形狀 | σ | α | 誤差 | TV(p_data) | TV(均勻) | 最大／最小 | 其他 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 圓 | 0.2 | 0 | 無 | 0.0348 | 0.5980 | 2518 | |
| 圓 | 0.2 | 1 | 無 | 0.4287 | 0.2297 | 4.783 | |
| 圓 | 0.05 | 0 | 無 | 0.0023 | 0.6170 | 2948 | |
| 圓 | 0.05 | 0.5 | 無 | 0.3957 | 0.2624 | 5.968 | |
| 圓 | 0.05 | 1 | 無 | 0.5673 | 0.0633 | 1.491 | |
| 圓 | 0.05 | 1.5 | 無 | 0.6070 | 0.0142 | 1.093 | |
| 圓 | 0.01 | 0 | 無 | 0.0001 | 0.6182 | 2976 | |
| 圓 | 0.01 | 1 | 無 | 0.6082 | 0.0127 | 1.083 | |
| 圓 | 0.01 | 1.5 | 無 | 0.6173 | 0.0013 | 1.008 | |
| 圓 | 0.05 | 0 | Θ(1) | 0.7436 | 0.2763 | 7.536 | TV(Thm 4.1 極限) 0.0223 |
| 圓 | 0.01 | 0 | Θ(1) | 0.7619 | 0.2883 | 7.386 | TV(Thm 4.1 極限) 0.0009 |
| 圓 | 0.01 | 1 | Θ(1) | 0.6202 | 0.0033 | 1.022 | |
| 圓 | 0.01 | 0 | 切向 β = −1 | 0.9997 | 0.9057 | ≈ 10^83（只驗量級：log₁₀ 值 83.4 ± 0.1） | |
| 圓 | 0.01 | 1 | 切向 β = −1 | 0.8233 | 0.2791 | 6.819 | α = −β 邊界 |
| 圓 | 0.05 | 1 | 切向 β = −1 | 0.7961 | 0.2387 | 4.954 | |
| 圓 | 0.01 | 1.5 | 切向 β = −1 | 0.6424 | 0.0305 | 1.212 | |
| 圓 | 0.01 | 1 | 切向 β = 0.5 | 0.6085 | 0.0124 | 1.081 | |
| 橢圓 | 0.2 | 1 | 無 | 0.3679 | 0.2419 | 4.907 | |
| 橢圓 | 0.05 | 0 | 無 | 0.0011 | 0.5677 | 2970 | |
| 橢圓 | 0.05 | 1 | 無 | 0.5122 | 0.0698 | 2.360 | |
| 橢圓 | 0.05 | 1.5 | 無 | 0.5597 | 0.0190 | 1.999 | |
| 橢圓 | 0.01 | 1 | 無 | 0.5572 | 0.0142 | 2.067 | |
| 橢圓 | 0.01 | 1.5 | 無 | 0.5702 | 0.0060 | 1.973 | |
| 橢圓 | 0.01 | 0 | Θ(1) | 0.7304 | 0.4837 | 10293 | TV(Thm 4.1 極限) 0.0024 |
| 橢圓 | 0.01 | 1 | Θ(1) | 0.5676 | 0.0130 | 2.168 | |

C2b（λ = 1.5）：圓 σ = 0.01、α = 0：TV(資料先驗後驗) 0.0001、TV(均勻先驗後驗) 0.8354；α = 1：TV(均勻先驗後驗) 0.0077、TV(資料先驗後驗) 0.8305。橢圓 σ = 0.01、α = 1：TV(均勻先驗後驗) 0.0082；α = 0：TV(資料先驗後驗) 0.0001。參考曲線：橢圓內蘊均勻最大 0.2064、最小 0.1033（180 格中點；連續值 0.1032）；圓的 Thm 4.1 極限最大 0.3415、最小 0.0463；$p_{data}$ 最大 0.7684、最小 0.00026。一致性：圓的封閉式與求積路徑最大差 ≤ 5.1×10⁻⁹。溢位檢查：圓、σ = 0.01、α = 0、切向誤差 β = −1.9 時密度全部有限，log₁₀(最大／最小) = 5476。效能（`c2_perf_out.txt`，橢圓 σ = 0.01）：一次完整更新 19.5–22.8 ms（α = 0–1.9）、拖曳 90 列 9.9–11.5 ms、只改誤差／β／λ 0.21 ms；視窗化與完整求積結果逐位相同。

### 5.3 C3（`check_c3.js`；σ = 0.01）

論文只證明充分條件；定理外一律標「不保證」。容許：保證／不保證標籤必須完全相同；θ、w、σ^{α+β}、e^{2κσ^α} 相對誤差 ≤ 0.5%（表中為三到四位有效數字）。

| (β, α) | 集中 | 還原 p_data（α = 0） | TS 均勻 | θ | w | σ^{α+β} | e^{2κσ^α}（漸近式） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (−1, 1) | 保證 | — | 不保證（邊界） | 0.0100 | 0.100 | 1.00 | 1.083 |
| (−1, 1.5) | 保證 | — | 保證 | 0.100 | 0.316 | 0.100 | 1.008 |
| (−1, 0) | 保證 | 不保證 | — | 1.00e−4 | 0.0100 | 100 | 2981 |
| (0.5, 1) | 保證 | — | 保證 | 0.0100 | 0.100 | 0.00100 | 1.083 |
| (−2.5, 1) | 不保證 | — | 不保證 | 0.0100 | 0.100 | 1.00e+3 | 1.083 |
| (−1.9, 1.95) | 保證 | — | 保證 | 0.794 | 0.891 | 0.794 | 1.001 |
| (0, 0) | 保證 | 保證 | — | 1.00e−4 | 0.0100 | 1.00 | 2981 |

### 5.4 C4（`check_c4.js`，seed 237、N = 1000、δ = 0.1、α_TS = 1；輸出 `c4_out.txt`）

同 seed 同演算法的容許誤差 ±0.01；與 C2 精確解比較（只限梯度誤差）的容許 ±0.05（蒙地卡羅標準差約 0.022，加 Euler 偏差）。

| σ | 誤差 | 步數 | L mean cos | L mean sin | L TV | TS mean cos | TS mean sin | TS TV | C2 精確 mean cos（L / TS） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 無 | 1000 | −0.819 | −0.007 | 0.574 | −0.204 | −0.020 | 0.156 | |
| 0.1 | 無 | 10000 | −0.861 | −0.017 | 0.610 | −0.165 | −0.032 | 0.128 | −0.859 / −0.195 |
| 0.1 | Θ(1) | 10000 | −0.097 | 0.017 | 0.258 | 0.050 | −0.023 | 0.065 | −0.102 / 0.036 |
| 0.1 | 旋轉 | 10000 | 0.063 | −0.802 | 0.572 | −0.098 | −0.210 | 0.154 | — |
| 0.05 | 無 | 1000 | −0.437 | 0.000 | 0.296 | −0.083 | −0.026 | 0.083 | |
| 0.05 | 無 | 10000 | −0.853 | 0.010 | 0.613 | −0.098 | −0.026 | 0.080 | −0.862 / −0.099 |
| 0.05 | 旋轉 | 10000 | 0.049 | −0.805 | 0.576 | −0.070 | −0.133 | 0.116 | — |
| 0.2 | 無 | 10000 | −0.845 | −0.023 | 0.591 | −0.335 | −0.013 | 0.219 | −0.846 / −0.358 |
| 0.2 | Θ(1) | 10000 | −0.277 | −0.020 | 0.214 | 0.076 | −0.002 | 0.102 | −0.297 / 0.051 |

所有組合重設次數為 0。

---

## 6. 實作計畫

### 6.1 檔案與外殼

- 單一檔案 `tut/237/build/index.html`，CSS／JS／時間線資料全部內嵌，不 `fetch`。上線時複製為 `tutorials/scores-learn-geometry.html`。
- **沿用** `tutorials/prediction-targets-k-diff.html` 的外殼，不重新設計：CSS tokens（含 `--c-par`、`--c-perp` 與深色模式 `--ink-3` #9C978C）、字型與 KaTeX 0.16.9 設定（`$…$`、`$$…$$`、`.katex-display` 可橫捲、動態 DOM 呼叫 `renderMathInElement`）、appbar 與主題切換（`pm-theme`）、`#pm-bar`（`← Paper Master` → `../index.html`、`教學與筆記` → `index.html`）、章節目錄（`initToc`）、plate／callout／`.eq`＋`.eq-n` 公式編號、`<details class="check">` 檢查題、兩欄演算法格線、表格橫捲容器、時間線（`renderTimeline`）、`announce()` live region、`el()`／`widthOf()` SVG 輔助、`?test` 的 `selfTest()` 架構與「SVG 寬度＝容器寬度、resize 重畫」做法。以下只列新增或改動。
- 本頁色彩語意：`--c-par` = 幾何／法向／內蘊均勻；`--c-perp` = 密度／切向／$p_{data}$；`--accent` = 目前計算結果與目前點。各曲線另以線型與直接標籤區分。
- `eps-prediction-ddpm-score`、`x-prediction-manifold-view` 是站內既有筆記（`scripts/notes_split/`），不在 `notes/` 內，連結照樣寫 `<slug>.html`。
- 連結：論文 `../index.html#paper-237`；arXiv `https://arxiv.org/abs/2509.24912`；筆記 `<slug>.html`。

### 6.2 新增程式（同一個 IIFE）

- `core.js` 全文（4.0 列出的函式與常數）。
- `initC1()`：`state={iSigma,phiDeg,rho}`（σ 取自 `C1_SIGMAS` 陣列）；`renderC1()` 畫兩圖＋讀數；右圖 60 個 σ 的四項預先在每次 φ、ρ 改變時重算（< 1 ms）。
- `initC2(root, {withV})`：建立兩個實例 `#c2a`（withV=false）與 `#c2b`（withV=true，預設 λ = 1.5、α = 1）；`state={shape,iSigma,alpha,err,beta,lambda,token}`（σ 取自 `C2_SIGMAS`）；網格快取 `gridCache=new Map()`，key `${shape}|${σ}|${α}|${nt}`（最多 40 筆，超過清空），兩個實例共用；`scheduleC2(inst, nt)` 依 4 節「排程」分段呼叫 `c2FillRows`，完成後 `c2Combine`＋`renderC2(inst)`。
- `initC3(root, {withAlpha})`：兩個實例 `#c3a`（S4，只有 β）與 `#c3b`（S6）；`state={beta,alpha}`；`renderC3(inst)`；圖面點擊用 `pointerdown`（滑鼠與觸控同一路徑）。
- `initC4()`：`state={sigma,aTS,err,sim:c4Init(1000,237),running,stepsPerFrame,colors}`；`readColors()` 從 `getComputedStyle` 讀 token；`runC4()`（rAF＋耗時調整）在 canvas 上重畫散點，`c4Stats` 每 10 影格重畫直方圖；梯度誤差時呼叫 `c2Marginal({a:1,b:1,sigma,alpha,err})` 取得疊圖。首次捲入視窗 50% 時自動執行一次（reduced-motion 時不自動）。
- S2 靜態 SVG：橢圓 (1, 2)，管狀鄰域以兩條平行曲線 $\Phi(t)\pm\epsilon N(t)$（ε = 0.3）畫出並填淺色；點 $x=\Phi(t_0)+0.25N(t_0)$（$t_0=0.6$）、$P_M(x)=\Phi(t_0)$，法向線段與直角記號，直接標籤「x」「P_M(x)」「M」「T_M(ε)」「d_M(x) = ½‖x − P_M(x)‖²」。
- Hero 誤差階數軸：沿用參考頁 `kaxis`／`layoutKAxis` 的 HTML 寫法，把線性軸換成對數軸。

### 6.3 selfTest（`?test`）

`console.assert` 逐項，最後 `console.log('selfTest: N passed, M failed')`：
1. `logI0` 在 x = 0.5、10、51、10⁴ 與 mpmath 值（`ref_out.txt` 推回）差 < 10⁻⁶；`ratioOverX(x)*x` 同上。
2. `eq6Terms(1.1, π/2, 0.01)`：exact −50.6235、resid 7.386e−4（±1e−5）；`circleScorePolar(1.1, π/2, 0.01)`：rad −1000.455、tan 3.6362（±1e−3）。
3. `c2Marginal` 圓與 `forceQuad` 在 σ = 0.05、α = 1 最大差 < 10⁻⁶。
4. TV：圓 σ = 0.01、α = 1 → TV(均勻) 0.0127；圓 σ = 0.01、α = 0 → TV(p_data) 0.0001；橢圓 σ = 0.05、α = 1.5 → TV(均勻) 0.0190；圓 σ = 0.01、α = 0、Θ(1) → TV(Thm 4.1 極限) 0.0009（皆 ±0.0005）。
5. C4：σ = 0.1、無誤差、1000 步：L mean cos −0.819、TS −0.204（±0.01）。
6. 溢位：`c2Marginal({a:1,b:1,sigma:0.01,alpha:0,err:'tan',beta:-1.9})` 的密度全部有限、總和×Δt = 1（±1e−9）。
7. 橢圓視窗化：`makeEllipseLogP(1,2,0.01)` 與 `makeEllipseLogP(1,2,0.01,true)`（完整求積）在 (0.3, 2.05)、(1.2, 0.1)、(0, 0) 三點差 < 10⁻¹²。
8. `document.documentElement.scrollWidth <= innerWidth`。

### 6.4 數學排版與文案

- 公式旁標論文編號（「Eq. 6」）。390px（內容寬 358px）時需處理的寬內容：Eq. 3、Eq. 6、Eq. 12、Eq. 14、Thm B.1 的極限式、C.2 的兩個 CFG 公式在 < 600px 拆成兩行（以 `aligned` 在 = 或 + 處斷行）；其餘公式靠 `.katex-display` 橫捲。Table 1 在 < 600px 改為依提示詞分段的堆疊卡片（每卡 3 方法 × 2 指標）；Table 2–4 與兩張設定表放橫捲容器並加左右漸層提示。C1 右圖與 C2 主圖在 < 600px 用圖例取代直接標籤。
- SVG 內用 Unicode 與 `<tspan dy>` 做上下標（「σ²·log p_σ」「s_rad」「w = σ^(1−α/2)」），不用 TeX 的 `_`。元件讀數的標籤是頁面載入時渲染一次的靜態 KaTeX，`input` 時只改數字節點的 `textContent`，不呼叫 `renderMathInElement`。
- 頁面中的論文數字與方程式必須出自第 1 節（含步驟 11 的 Table 1–4 與兩張設定表），元件數值必須出自第 5 節；不得從別處新增。
- 繁體中文、台灣用語、字面學術語氣，不用比喻與反問。專有名詞第一次出現附英文（分數 score、流形 manifold、管狀鄰域 tubular neighborhood、速率分離 rate separation、平穩分佈 stationary distribution、強極大值原理 strong maximum principle）。用詞：「分佈」「流形上的均勻分佈（對黎曼體積）」「常見錯誤」。
- 區分三種陳述：論文結果、作者的解讀（「作者認為」「作者稱」）、本頁推導。

### 6.5 引用與標籤

- 內文每個論文數字後註明出處；元件讀數不逐一加出處，每個圖版底部一行「讀數由本頁依 Eq. ○ 即時計算」，用到〔本頁推導〕的式子時加「本頁推導」pill（`--ink-3` 邊框）與一行依據。推定值用「推定」pill（本頁目前沒有推定值；若建置時新增，必須同時寫入第 7 節）。

### 6.6 研究脈絡（`notes/timeline.json`）

- 建置時把檔案原樣貼進 `<script type="application/json" id="timeline-data">`。目前 15 筆（1907.05600、2006.11239、2011.13456、2103.04715、2112.10752、2206.01018、2208.05314、2206.00364、2207.12598、2209.11215、2212.12611、2301.07882、2506.15385、2507.15397、2509.24912）；欄位 `year, title, authors_short, arxiv, role, source`。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中歸「其他」）：

| 類別標籤 | arXiv ID | 卡片連到的筆記 |
| --- | --- | --- |
| 分數學習基礎 | 1907.05600、2006.11239、2011.13456、2206.00364 | `score-geometry-score-matching-sde.html` |
| 收斂保證 | 2208.05314、2209.11215 | `score-geometry-diffusion-guarantees.html` |
| 流形偵測 | 2206.01018、2212.12611、2301.07882 | `score-geometry-manifold-detection.html` |
| 逆問題與引導 | 2103.04715、2207.12598、2507.15397、2112.10752 | `score-geometry-inverse-problems-guidance.html` |
| 均勻採樣 | 2506.15385 | `score-geometry-tempered-langevin.html` |
| 本文 | 2509.24912 | `score-geometry-lineage-hub.html` |

- 呈現照參考頁（水平捲動卡片、類別 chip 淡化、本文卡片 `--accent` 外框、缺欄位處理、JSON 解析失敗顯示「時間線資料尚未提供」）。
- S12 延伸閱讀依 hub 的閱讀順序：`score-geometry-lineage-hub.html`、`score-geometry-score-matching-sde.html`、`eps-prediction-ddpm-score.html`、`score-geometry-diffusion-guarantees.html`、`score-geometry-manifold-detection.html`、`x-prediction-manifold-view.html`、`score-geometry-laplace-expansion.html`、`score-geometry-tempered-langevin.html`、`score-geometry-sde-pde-wkb.html`、`score-geometry-inverse-problems-guidance.html`、`score-geometry-eval-metrics.html`；每項附筆記 frontmatter 的 `title`。

### 6.7 無障礙

- 每個滑桿有 `<label>` 與 `aria-valuetext`（例「σ = 0.0500」「β = −1.00」）；SVG `role="img"` 與描述目前狀態的 `aria-label`；`aria-label` 與讀數區的 `aria-live="polite"` 只在 `change` 或輸入停止 500 ms 後更新。
- C3 平面的點擊只是捷徑，兩個滑桿可完成同樣操作；C1 右圖沒有 hover 才看得到的資訊（數值都在讀數表）。
- C4 動畫中不更新 live region，只在暫停或完成時宣告「第 N 步，L 的 mean cos = …，TS 的 mean cos = …」。
- 曲線、區域、判定一律以線型／填色樣式／形狀符號＋直接文字標籤區分，不只靠顏色；focus 樣式 2px `--accent` 外框；表格有 `<caption>`。

### 6.8 建置後自我檢查

依 skill 規則：截圖迴圈最多 1 輪，只看新元件與 390px；`?test` 全數通過；逐條核對 12 個學習目標的參考答案確實寫在指定章節內文；Hero 誤差階數軸在 358px 內容寬時標籤不重疊。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 定理是漸近、存在性結論 | Eq. 6 的 o(1)、Thm 5.2 的 WKB 與極大值原理、Assumption B.2 是否成立，都無法用數值驗證。 | 元件只示範封閉式或梯度場情形；C4 明寫「數值示意，不是證明」；S7 標明 Assumption B.2 是假設。 |
| R2 | 不能重現論文實驗 | Fig. 2 需訓練網路且 C.1 超參數互相矛盾；Fig. 4(a) 是反向擴散、排程未交代；Table 1–4 需 SD 1.5 與 CLIP。〔本頁推導〕固定 σ 的 Langevin 在 Θ(1) 誤差下的極限 $\propto e^{-2\cos^2t}$（π/2 與 3π/2 雙峰），與 Fig. 4(a) 的平台形狀不同。 | S9 只照錄數字與文字結論；C2 下方註明不重現 Fig. 4(a)。 |
| R3 | Fig. 2 是否已平穩 | 〔本頁推導〕C.1 時鐘下切向擴散變異數 $2\sigma^{2-\alpha}T$（T = 1000）：α = 0、σ = 0.01 只有 0.2 rad²；初始分佈未寫。Fig. 2(d) 的傾斜比精確分數的殘留（$e^{2\kappa\sigma}$，以 C.1 的 κ = 1 為 $e^{0.02}$）大。 | S9 不把 Fig. 2(a)(c) 說成已收斂的確證；C4 用 σ ≥ 0.05 以便在 10,000 步內混合，並在觀察 (4) 展示 α = 0 混合較慢。 |
| R4 | Table 1 的讀法 | 每格分別調參；作者稱 TS「consistently」更多樣，但 Architecture P-sim DDPM 27.36 > TS 27.32；Table 3–4 中 α = 0.1 有數格比基線差、α ≥ 0.5 也有例外（Table 4 Car α = 0.5 P-sim 26.25 vs 26.26；α = 1.0 I-sim 88.07 vs 88.06）。 | S9 照錄全表，加逐格出處 callout，不寫「全面優於」。 |
| R5 | 摘要用語與定理不一致 | 摘要寫 Θ(σ⁻²) stronger、tolerates O(σ⁻²) larger errors；定理是 o(σ⁻²) 對 o(1)。Thm 4.1 後「shape separation」應為 sharp。 | 常見錯誤 1；頁面一律用定理層級的寫法。 |
| R6 | App. D 不做元件 | App. D 自身有不一致：D.2 寫 Ω(c)（c = σ^{3−α}），D.3 寫 Ω(1/σ)；「In this response」字樣；例子的 φ 在 x₁ = 0 梯度不連續，與 $f_\sigma\in C^1$ 不合；Proposition D.1 把 σ 當 Lyapunov 常數。 | S12 只用一段文字介紹結論與適用範圍（單一例子），不展開。 |
| R7 | Thm 6.1 的假設與 CFG | v 須在 $\mathbb R^d$ 有界，高斯似然不滿足；定理用梯度記號卻說假設同 Thm 5.2；CFG 中 v 實為隨 σ_t 變的學到量。〔本頁推導〕另一種解讀：$\sigma^\alpha\epsilon_\varnothing+w(\epsilon_y-\epsilon_\varnothing)=\sigma^\alpha[\epsilon_\varnothing+(w/\sigma^\alpha)(\epsilon_y-\epsilon_\varnothing)]$，等於有效引導強度變成 $w/\sigma_t^\alpha$；論文沒有排除這個解釋。 | S8 寫明 CFG 對應是啟發式，並以一句話列出另一種解讀（標本頁推導）；C2b 用有界 v。 |
| R8 | 連通性 | Assumption 2.1 沒寫連通；多分支時 WKB 論證不決定分支間的質量比。 | S7 一句注記（本頁推導）。 |
| R9 | κ 的不一致 | C.1 寫 κ = 1，C.3 用 κ = 4、θ₀ = π；von Mises 中心在 C.1 未寫。 | 所有元件用 κ = 4、θ₀ = π，頁面註明來源是 C.3。 |
| R10 | 橢圓計算成本 | 完整求積在 σ = 0.01 約 100 ms／次（node，桌機）。 | 已改為視窗化求積＋網格快取＋8 ms 可取消分段＋拖曳時 90 列（實測 4.0：完整更新約 22 ms、拖曳約 11 ms、只改誤差 0.2 ms），σ 下限保持 0.01。低階手機若單段仍超過 16 ms，再改用行內 Blob Worker 執行 `c2FillRows`（`core.js` 為純函式，可直接放進 Worker 字串）。 |
| R11 | Euler 離散化偏差 | C4 用 δ = 0.1，TS 在 σ = 0.1 時 mean cos −0.165 對精確 −0.195（差 1.4 個蒙地卡羅標準差）。 | 驗收對 C2 精確解的容許設為 ±0.05；頁面註明「離散化誤差未分析」是論文限制 (d)。 |
| R12 | 初學者門檻 | 讀者不熟流形、漸近符號與 Fokker–Planck；S7 最難。 | S1 定義平穩 Fokker–Planck 與溫度 θ；S2 定義 von Mises、管狀座標 r 與漸近符號；原 S4 拆成 S4（結果）與 S5（通用公式，定義 $f_0,f_1$）；S3 的 Corollary B.1／Eq. 12 與 S7 的 Eq. 14、16–19 放可收合區塊，但 LO 答案句留在內文。 |
| R13 | 筆記更新 | 建置前 notes 若增減，`TL_GROUP` 與延伸閱讀需同步。 | 建置時重讀 `notes/`；新 arXiv ID 歸「其他」並在建置回報列出。 |

---

## 8. 修訂紀錄（round 1，對應 `reviews/round1.md` 編號）

| # | 變更 |
| --- | --- |
| 1 | Table 2 全表、Table 3–4 全表、C.1／C.2 設定表寫進 1.4 步驟 11；6.4 規則改為「論文數字出自第 1 節、元件數值出自第 5 節」。 |
| 2 | 橢圓改用視窗化求積（128 粗節點定位、e⁻⁴⁰ 截斷判準）、網格快取（誤差／β／λ 只重組）、rAF 後 8 ms 可取消分段、拖曳 90 列；`check_c2_perf.js` 實測 σ = 0.01：完整更新 19.5–22.8 ms、拖曳 9.9–11.5 ms、只改誤差 0.21 ms，結果與完整求積逐位相同；σ 下限保持 0.01；R10 改寫，Worker 為最後手段。 |
| 3、4、9 | Hero 標題、副標、三張卡、誤差階數軸改寫（生成分佈的 L∞ 誤差、構造是存在性、需 α > −β、按體積、不寫「強 σ⁻² 倍」）。 |
| 5 | C3 與 5.3 改「保證／不保證」；C3 不提 Assumption B.2，改連到 07 節。 |
| 6 | 原 `core.js` 已在 t 方向減最大值；規格明寫、加 selfTest 6（β = −1.9、σ = 0.01 仍有限）。 |
| 7 | S1 加平穩 Fokker–Planck 與溫度 θ；S2 定義 von Mises、I₀、κ、θ₀、管狀座標 r；S3 的 Corollary B.1／Eq. 12／封閉式收合；原 S4 拆為 S4（Thm 4.1）與 S5（Thm B.1、f₀、f₁）；其後章節順延為 S6–S12。 |
| 8 | LO10 答案句明定留在 S7 內文，推導才收合；第 2 節規則補「不能只在可收合區塊」。 |
| 10 | C4 指定 canvas、token 由 `getComputedStyle` 讀取並在主題切換與 resize 時重畫；讀數標籤為靜態 KaTeX，只更新數字。 |
| 11 | $e^{2\kappa\sigma^\alpha}$ 標為漸近式並附精確值；Thm 5.1 補 Assumption 4.1（套到 $\tilde\pi_\sigma$）；1.1(iii) 補 Thm 6.1 條件；LO8 理由改標本頁推導並對應 Thm 5.2 證明。 |
| 12 | 「法向 ~σ⁻²」限定固定距離（步驟 6、LO5、C1 觀察 4）；LO5 給切向方向；C2 觀察 (5) 收斂措辭；步驟 13／S12 加有限樣本與記憶化；Fig. 1 標示意圖。 |
| 13 | 誤差位能改記 $U_{err}$；切向誤差選項寫明是 $O(\sigma^\beta)$ 邊界階數、K 取環形區域。 |
| 14 | C1、C2 的 σ 改為含驗收點的離散值；390px 寬內容清單（6.4）；標題改陳述句、Hero 不擬人、「誤差尺」改「誤差階數軸」；reduced-motion 每段 100 步；C3a（只有 β）放 S4 對應 LO6；C2 大 w 時加註、「計算中」用 `role="status"`。 |

## 9. 修訂紀錄（readiness）

- 阻擋項 1：Eq. 3 移到 S2。阻擋項 2：5.3 加容許誤差。
- 非阻擋：S3 內文定義 H(x)、e₀、R；S10 先定義 δ、ξ；5.2 的 ≈10^83 只驗量級；S6 說明 Θ(1) 誤差對應接近 0 的 β < 0；Hero 副標改為無未定義術語；註明兩篇既有筆記位於 `scripts/notes_split/`；建置前重讀 notes/（R13）。

## 10. 建置紀錄（build round）

- 產出 `build/index.html`（約 136 KB）；由 `build/src/assemble.py` 組合參考頁外殼、`src/body.html`、`src/extra.css`、`verify/core.js`（逐字）與 `src/widgets.js`；同時產生供本機 headless 預覽的 `local.html`（KaTeX 改本機、移除 Google Fonts），預覽完已刪除。
- `core.js` 的 `refCurve` 增加可選參數 nt（拖曳時 90 列網格用）；`check_c2.js` 輸出不變。
- 與設計的差異：C1 右圖一律用圖例（不用直接標籤），避免五條線標籤在任何寬度重疊；C2 小圖的法向帶寬上限改為 0.45（橢圓內側曲率半徑 0.5）；Eq. 3、6、12 與 C.2 的 TS 公式一律以 `aligned` 寫成兩行；Table 1 在 < 480px 以參考頁的 `table.stack` 依方法堆疊；非 check 的可收合區塊用 `details.more`（驗收腳本只剝除 `details.check`）。
- 驗證：`?test` 67 項全過（390 與 1440 寬）；impeccable 0 anti-patterns（12 項 numbered-section-labels advisory，沿用參考頁設計）；390px 無水平捲動（只剩 Table 2–4 在橫捲容器內）；KaTeX 正常渲染；無 console 錯誤；`accept/page.txt` 逐條核對 12 個 LO 的關鍵詞與數字皆在內文。
