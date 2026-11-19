# 237 Goldfish 驗收作答

**LO1**
- 答：VE 是 $\mu_\sigma=\mathrm{law}(X+\sigma Z)$，資料係數 $\gamma(\sigma)=1$；VP 是 $\mu_\sigma=\mathrm{law}(\sqrt{1-\sigma^2}X+\sigma Z)$，資料係數 $\gamma(\sigma)=\sqrt{1-\sigma^2}$。漂移 $s=-\nabla f$ 時，平穩分佈正比於 $\exp(-f)$。
- 段落：「VP 先把資料縮小為」；「正比於 exp⁡(−f)」
- 信心：high

**LO2**
- 答：$d_M(x)=\tfrac12\mathrm{dist}^2(x,M)$（Eq. 2，有 ½）；$P_M(x)$ 是 $x$ 在 $M$ 上的最近點；$T_M(\epsilon)=\{x:\mathrm{dist}(x,M)<\epsilon\}$ 是距離 $M$ 不到 $\epsilon$ 的管狀鄰域。$p_{data}$ 是對座標 $u$ 所在 $U$ 上 Lebesgue 測度的密度，不是對體積測度的密度，也不是經驗分佈。
- 段落：「半倍平方距離」；「對 U 上 Lebesgue 測度的密度」
- 信心：high

**LO3**
- 答：均勻分佈（對體積）在座標 $u$ 上的密度正比於 $\sqrt{\det g(u)}=dM/du$。橢圓 $(a,b)=(1,2)$ 時，$t$ 上的密度為 $\sqrt{\sin^2t+4\cos^2t}/9.6884$；最大 0.2064（$t=0,\pi$），最小 0.1032（$t=\pm\pi/2$），比值為 2。
- 段落：「最大 0.2064（t = 0、π」；「最大與最小的比值是 2」
- 信心：high

**LO4**
- 答：Eq. 6 的四項：(1) $-d_M/\sigma^2$，階 $\Theta(\sigma^{-2})$，只含到流形的距離；(2) $\log p_{data}(\Phi^{-1}(P_M(x)))$，階 $\Theta(1)$，是最近點的資料密度；(3) $-\tfrac{d-n}{2}\log(2\pi\sigma^2)$，隨 $\sigma$ 發散，但與 $x$ 無關，對分數與歸一化後的分佈沒有影響；(4) $H(x)$，曲率項，與 $\sigma$ 無關。最後的 $o(1)$ 在管狀鄰域內一致趨近 0。極限為 $\lim_{\sigma\to0}\sigma^2\log p_\sigma(x)=-d_M(x)$（Eq. 7）。
- 段落：「四項依序是：(1) −dM/σ2」；「lim⁡σ→0σ2log⁡pσ(x)=−dM(x)」
- 信心：high

**LO5**
- 答：法向分量近似 $-(x-P_M(x))/\sigma^2$，誤差為 $O(1)$（Eq. 20）；切向分量是 $O(1)$，攜帶 $p_{data}$ 的資訊。單位圓、κ = 4、θ₀ = π、ρ = 1.1、φ = 90°、σ = 0.01 時：法向分量（沿 $x/\|x\|$）約 −1000.455，切向分量（沿逆時針單位切向量）約 +3.636（極限 3.6364）。
- 段落：「法向分量（沿 x/∥x∥）是 −1000.455」；「切向分量…是 +3.636」
- 信心：high（數值取自文字段，不是互動元件的讀數）

**LO6**
- 答：集中到 $M$ 只需 $E_\sigma=o(\sigma^{-2})$；還原 $p_{data}$ 需要 $E_\sigma=o(1)$。Θ(1) 的誤差足以得到任意分佈，因為 $p_{data}$ 只出現在 Θ(1) 項，這一項可以被 Θ(1) 的誤差換成 $\log\hat\pi$（Thm 4.1(2)）。誤差小於 $\sigma^{-2}$ 時，$-d_M/\sigma^2$ 仍主導，所以集中仍成立。頁面注明這些是充分條件，不是已證的失敗門檻。
- 段落：「只要誤差比 σ−2 小」；「Θ(1) 的誤差足以把這一項換成」
- 信心：high

**LO7**
- 答：Thm B.1 的極限密度為 $\pi(u)\propto\exp(-f_1(u,0))\,|\partial^2f_0/\partial r^2|^{-1/2}\,dM/du$。套到 Thm 4.1(3) 時，$\theta=\sigma^2$、$f_0=d_M$、$f_1=-\log p_{data}+\log\sqrt{|\hat H|}$；在 $r=0$ 時 $\sqrt{|\hat H|}=dM/du$，且 $\partial^2f_0/\partial r^2=I$，所以 $\pi(u)\propto p_{data}(u)\,(dM/du)^{-1}\,(dM/du)=p_{data}(u)$。體積因子被曲率項給出的 $(dM/du)^{-1}$ 抵消。
- 段落：「π(u)∝pdata(u) (dM/du)−1 (dM/du)=pdata(u)」
- 信心：high

**LO8**
- 答：TS Langevin 為 $dX_t=\sigma^\alpha s(X_t,\sigma)\,dt+\sqrt2\,dW_t$（Eq. 8）。梯度情形的平穩分佈為 $\tilde\pi_\sigma\propto\exp(-\sigma^\alpha f_\sigma)$，溫度 $\theta=\sigma^{2-\alpha}$。$\alpha$ 的範圍是 $\max\{-\beta,0\}<\alpha<2$。端點理由：$\alpha>0$ 使 $O(\sigma^2)$ 的 $p_{data}$／曲率項為 $o(\theta)$（$\alpha=0$ 即標準 Langevin，此項不被壓掉）；$\alpha>-\beta$ 使誤差 $o(\sigma^{2+\beta})$ 為 $o(\theta)$（$\alpha=-\beta$ 是邊界，定理要求嚴格不等式）；$\alpha<2$ 使 $\theta=\sigma^{2-\alpha}\to0$（$\alpha=2$ 時 $\theta=1$，不會集中）。
- 段落：「α > 0 使 O(σ2) 的 pdata」；「α < 2 使 θ → 0」
- 信心：high

**LO9**
- 答：三個條件成立時 $f_1\equiv0$，極限 $\propto dM/du$，與 $p_{data}$ 無關，因為 $p_{data}$ 那一項在 $\sigma^2f_\sigma$ 中是 $O(\sigma^2)=o(\theta)$。直觀上 $\tilde\pi_\sigma\propto p_\sigma^{\sigma^\alpha}$，切向的 $p_{data}^{\sigma^\alpha}\to1$，被壓平。有限 $\sigma$ 時切向殘留：von Mises 的最大／最小比約 $e^{2\kappa\sigma^\alpha}$；κ = 4、α = 1、σ = 0.01 時為 1.083。
- 段落：「切向密度變成 pdataσα→1」；「為 1.083」
- 信心：high

**LO10**
- 答：Thm 5.2 比 Thm 5.1 多了兩項假設：$p_{data}\in C^2$（Eq. 20 的證明需要 $O(\theta)$ 精度的 Laplace 方法），以及 Assumption B.2（平穩分佈存在、唯一，且有局部 WKB 形式），因為非梯度漂移沒有封閉式。WKB 三階：$\theta^{-1}$ 階是 Hamilton–Jacobi 方程 $\langle\nabla f_0,\nabla V\rangle=\|\nabla V\|^2$，取 $V=f_0=d_M$；$\theta^0$ 階是輸運方程 $\langle\nabla\tilde c_0,\nabla f_0\rangle=0$，只限制法向、不限制 $M$ 上的切向；$\theta^1$ 階在 $M$ 上求值，得到不含零階項的二階方程，由強極大值原理得 $\tilde c_0$ 在 $M$ 上為常數。
- 段落：「Thm 5.2：假設 Assumption 2.1、2.2」；「θ−1 階：⟨∇f0,∇V⟩=∥∇V∥2」
- 信心：medium（附錄推導，頁面自己也標註部分未交代處）

**LO11**
- 答：Thm 6.1 的極限為 $\pi(u)\propto\exp(-v(\Phi(u)))\,dM/du$，即在均勻（最大熵）先驗下的後驗，$\propto e^{-v}\sqrt{\det g}$，不是 $p_{data}$ 先驗下的後驗。CFG 中 TS 只把無條件分數乘上 $\sigma_t^\alpha$（$\tilde s_t=-\tfrac1{\sigma_t}(\sigma_t^\alpha\epsilon(x,t,\varnothing)+w(\epsilon(x,t,y)-\epsilon(x,t,\varnothing)))$），引導增量不變。頁面說明這個對應是啟發式的。
- 段落：「這個極限是均勻先驗下的後驗」；「只把無條件分數乘上」
- 信心：high

**LO12**
- 答：P-sim 是生成圖與提示詞的平均 CLIP 分數，越高越好（品質）；I-sim 是同一提示詞下生成圖兩兩之間的平均 CLIP 分數，越低越多樣。Table 1 的 Furniture：PC 為 29.40 / 81.24，TS 為 30.20 / 80.76（DDPM 為 29.56 / 80.78）。實驗限制：(a) 沒有追蹤軌跡上的誤差累積；(b) L∞ 假設；(c) 沒有樣本複雜度結果；(d) 連續時間，未量化離散化誤差；(e) 實驗是初步的。另外每類只有 1 個提示詞、每設定 512 張，沒有誤差棒或重複次數，各格為分別調參的最佳值，PC 的 Furniture 數字也混用了 5 步與 10 步。
- 段落：「P-sim（CLIP Prompt Similarity）」；「Furniture 的 PC 為 29.40 / 81.24」
- 信心：high

## 讀者可能卡住的地方（最多 5 點）

1. 互動元件 C1 的讀數是 σ = 0.100（法向 −10.521、切向 3.6174），而文字段給的是 σ = 0.01 的數字（−1000.455、+3.636）。兩組數字看起來對不上，但其實是不同的 σ。
2. 符號 $\theta$ 被多次重新定義：Eq. 11 的溫度、04 節取 $\sigma^2$、06 節取 $\sigma^{2-\alpha}$，符號表也沒有把這三個統一起來。
3. 表格中的 TV（total variation）從未定義，讀者不知道 TV 0.0023 或 0.6170 代表什麼。
4. Thm 5.2 的 WKB 證明依賴 Assumption B.2 與附錄的 Lemma，頁面另外指出正文寫「拋物型 PDE」而附錄是橢圓型，初學者很難判斷以哪個為準。
5. Table 1 的每一格是分別調參的結果，同一列的兩個數字可能來自不同步數（例如 Furniture PC 的 P-sim 來自 5 步、I-sim 來自 10 步），容易誤讀成同一組實驗。
