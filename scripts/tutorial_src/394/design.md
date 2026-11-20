# 設計文件：〈The Diffusion Duality, Chapter II: Ψ-Samplers〉互動教學

- 論文：Justin Deschenaux、Caglar Gulcehre、Subham Sekhar Sahoo，arXiv 2602.21185v2（ICLR 2026）。資料庫編號 No. 394。
- 原文：`tut/394/src/paper.txt`（`tut/` = `scripts/tutorial_src/`）。「式 (n)」「Sec. n」「Suppl. X」「Table n」「Fig. n」「Algo. n」皆指論文原文編號。顧問筆記 `tut/394/advisor.md` 的結論已逐條重算，改標〔本頁推導〕。
- 產出：`tut/394/build/index.html`（單一 HTML）。上線位置 `tutorials/diffusion-duality-psi-samplers.html`（slug `diffusion-duality-psi-samplers`）；站內連結以上線位置為準。
- 驗算（非頁面程式）：`verify/core.py`（共用數學與對式 (2)(3)(21)(26)、邊際保持的自我檢查）、`chain.py` 與 **JS 參考移植 `chain.js`**（W3；兩者輸出逐位相同）、`prng.js`（頁面必須照抄的 PRNG）、`topk.py`（W4、式 (43) 數值積分、Algo. 1 對暴力排序）、`accept.py`（第 5 節數值）。輸出：`out_w124.txt`、`out_w3.txt`、`out_w4c.txt`、`out_rev1.txt`；`node chain.js 0.2` 產生 5.3 第二表。
- 研究筆記：`tut/394/notes/`（總覽 `psi-samplers-lineage-hub` + 6 篇主題 + `timeline.json` 13 筆）。Duo（No. 393，`diffusion-duality.html`）先上線，可直接連結。
- 標記：〔本頁推導〕＝論文沒寫、本設計推出並驗算；**推定**＝本頁為了計算而自選、論文沒給的值或公式。頁面用同名 pill（6.5）。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

（本段是給審查者的摘要，使用了後文才定義的符號；頁面 hero 與各節開頭不得照搬，所有符號與指標在頁面上第一次出現時就定義，見第 3 節規則。）

離散擴散模型把一串 token 逐步換成雜訊，再訓練網路反向還原。常用的雜訊先驗有兩種：全部換成 [MASK]（masked diffusion，MDM），或換成均勻隨機的 token（uniform-state diffusion，USDM）。MDM 的 token 一旦解開就不再更新；USDM 允許反覆改值，因此在少步生成與 guidance 表現好，但用 ancestral sampler 增加步數時品質趨於持平（摘要、Sec. 1）。本文定義 **Ψ-posterior**：把標準反向 posterior $q_{s|t}$ 與前向邊際 $q_s$ 以係數 $\kappa_t$ 做凸組合（式 (11)）；在已知乾淨資料 $x$ 的條件下，這個組合對任何先驗 $\pi$ 都保持前向過程的邊際（Suppl. A.2）。把不可得的 $x$ 換成去噪器輸出 $x_\theta$，得到不需重新訓練的 **Ψ-sampler**（式 (12)）；換成 $x_\theta$ 之後邊際保證不再成立。$q_{s|t}$ 一項是預測步（predictor），$q_s$ 一項是注入雜訊的校正步（corrector）。在 mask 先驗、$\sigma_t\le1-\alpha_s$ 且採用 carry-over 參數化時，它與 ReMDM 的 posterior 相同（Suppl. A.4）。實驗中，Duo（USDM）配 Ψ-sampler 在 OpenWebText、nucleus $p=0.9$、rescale $\eta=0.05$ 下，Gen. PPL 在 1024 步之後仍下降（Table 13）；CIFAR-10 的 FID 由 25.63 降到 15.05（Suppl. D.1.1，逐步數挑選最佳設定）。論文另外提出 Duo 課程學習的省記憶體實作：用 order statistics 只抽 $K-1$ 個零均值高斯中最大的 $k$ 個值，並以截斷高斯的條件期望近似 softmax normalizer（Sec. 4、Suppl. B）；峰值記憶體 94.3→63.4 GiB、吞吐 81.8→121.9 samples/s（Table 4），perplexity 相近（Table 2）。

### 1.2 研究動機（Sec. 1）

1. MDM 每個 token 只更新一次，解錯的 token 無法回頭修改，錯誤會累積（Sec. 2.1.1）。ReMDM 以 remasking 解決，但只適用 mask 先驗。
2. USDM 的 predictor-corrector 方法很少；Campbell et al. (2022) 基於 CTMC rate matrix 的 PC 方法已知不如 ancestral sampler（Sec. 1）。
3. USDM 在多步區間的生成品質與 likelihood 仍落後 MDM；Duo 的課程學習可縮小 likelihood 差距，但要對每個 token 具現 K 維向量，在 $K>100{,}000$ 的詞彙下不可行（Sec. 1、Sec. 4）。

### 1.3 貢獻（Sec. 1 原文三點）

1. 一族適用任意雜訊先驗、與 Markov 離散擴散有相同邊際的非 Markov posterior（Sec. 3）。
2. 由它導出的 Ψ-samplers 在高 NFE 區間比 ancestral sampler 擴展得更好：文字生成追上配 remasking 的 MDM（Sec. 5.1），影像生成超越（Sec. 5.1.2）。
3. 重寫 Duo 的課程學習：峰值記憶體 −33%、端到端訓練時間 −25%，perplexity 與下游準確率相近（Table 2、Table 1）。

本頁主軸是第 1、2 點（A 線，S1–S8）；第 3 點（B 線，S9–S11）與 A 線互不依賴。

### 1.4 從作者的角度逐步拆解

#### 符號表（S1 末尾 `<details>` 提供同一份；S1 內文先定義 Cat、⊙、⟨⟩、q）

| 符號 | 意義 | 出處 |
| --- | --- | --- |
| $K,\ L,\ \ell$ | 類別數（MDM 時含 [MASK]）、序列長度、位置索引 | Sec. 2 |
| $x,\ z_t$ | 乾淨資料、時刻 $t$ 的離散 latent，皆為 one-hot | Sec. 2, 2.1 |
| $\mathrm{Cat}(\cdot;v)$ | 以機率向量 $v$ 為參數的類別分佈 | Sec. 2 |
| $a\odot b,\ \langle a,b\rangle$ | 逐元素乘積、內積 | Sec. 2 |
| $\pi$ | 先驗：MDM $\pi=m$（[MASK] 的 one-hot），USDM $\pi=\mathbf 1/K$ | Sec. 2.1.1–2.1.2 |
| $\alpha_t$ | noise schedule，$[0,1]$ 內隨 $t$ 遞減，$\alpha_1=0$ | 式 (1) |
| $q_t(\cdot|x)$ | 前向邊際；$q_s$ 為較早時刻 $s\lt t$ 的邊際 | 式 (1) |
| $q_{s|t}(\cdot|z_t,x)$ | 反向 posterior：已知 $z_t$ 與 $x$ 時 $z_s$ 的分佈 | 式 (2)(3) |
| $q_{0|t}(\cdot|z_t,x)$ | $s=0$ 時的 $q_{s|t}$（$\alpha_0=1$），即由 $z_t$ 推測乾淨 token 的分佈 | Suppl. A.1 |
| $\alpha_{t\mid s}$ | $=\alpha_t/\alpha_s$；論文未定義，依 Duo 慣例〔本頁推導：代入 (3) 與 Bayes 一致〕 | 式 (3) |
| $x_\theta(z_t,t)$ | 去噪器輸出，simplex 上的機率向量 | Sec. 2.1.1 |
| $\kappa_t$ | Ψ 的混合係數，$[0,1]$；$\kappa_t=1$ 為 ancestral | 式 (11) |
| $\sigma_t,\ \eta$ | ReMDM 的 remask 機率與 schedule 參數 | 式 (21)(27)(28) |
| NFE | 去噪器呼叫次數；Ψ-sampler 每步一次，NFE＝步數〔本頁推導〕 | Fig. 1 |
| $\tilde\alpha_t,\ \tilde\sigma_t,\ w_t$ | Gaussian 側 schedule、$\tilde\sigma_t=\sqrt{1-\tilde\alpha_t^2}$、latent $w_t\sim\mathcal N(\tilde\alpha_t x,\tilde\sigma_t^2I_K)$ | Sec. 2.1.2、4 |
| $\tau,\ k$ | softmax 溫度（$10^{-3}$）、保留的最大項數 | Sec. 4 |
| $\mathcal K_1\ge\dots\ge\mathcal K_k$ | $K-1$ 個零均值座標中最大的 $k$ 個值（頁面寫 $\mathcal K$，與詞彙數 $K$ 區分） | Sec. 4 |
| $o,\ \tilde w$ | 乾淨 token 的索引、該座標的值 $\sim\mathcal N(\tilde\alpha_t,\tilde\sigma_t^2)$（另外單獨抽） | Sec. 4 |

#### 步驟 1：只規定邊際的前向過程（Sec. 2.1，式 (1)）

每個位置獨立，$z_t^\ell\sim q_t(\cdot|x^\ell)=\mathrm{Cat}(\cdot;\alpha_t x^\ell+(1-\alpha_t)\pi)$：以機率 $\alpha_t$ 保留原 token，否則從 $\pi$ 抽一個。式 (1) 只規定各時刻的邊際，沒有規定不同時刻的聯合分佈；Sec. 3 第一句「Multiple joint distributions can give rise to the same marginals」利用這個自由度。

#### 步驟 2：兩種 posterior 與 ancestral sampler（式 (2)(3)）

MDM（式 (2)）：$z_t=m$ 時 $\mathrm{Cat}\big(\frac{\alpha_s-\alpha_t}{1-\alpha_t}x+\frac{1-\alpha_s}{1-\alpha_t}m\big)$，否則 $\mathrm{Cat}(x)$——已解開的 token 不再更新。USDM（式 (3)）：
$$q_{s|t}=\mathrm{Cat}\Big(\tfrac{K\alpha_t z_t\odot x+(\alpha_{t|s}-\alpha_t)z_t+(\alpha_s-\alpha_t)x+(1-\alpha_{t|s})(1-\alpha_s)\mathbf 1/K}{K\alpha_t\langle z_t,x\rangle+1-\alpha_t}\Big),$$
token 可以變成任何值。Ancestral sampler：$p^\theta_{s|t}=\prod_\ell q_{s|t}(\cdot|z_t^\ell,x^\ell=x_\theta^\ell(z_t,t))$（Sec. 2.1.1）。〔本頁推導〕兩式都是同一個 Bayes 公式：
$$q_{s|t}(\cdot|z_t,x)\ \propto\ \big(\alpha_{t|s}\,z_t+(1-\alpha_{t|s})\langle\pi,z_t\rangle\mathbf 1\big)\odot\big(\alpha_s x+(1-\alpha_s)\pi\big).$$
第一個括號是「從每個 $z_s$ 走到觀察到的 $z_t$ 的機率」，第二個是 $q_s(\cdot|x)$。分子對 $x$ 線性，所以 $x$ 換成機率向量 $x_\theta$ 仍是合法的 Bayes 更新。頁面只實作這一個 `post()`；`core.py` 在 $K=3,4,6$、三組 $(\alpha_t,\alpha_s)$ 下與 (2)(3) 相差 $\lt$ 1e-12。頁面內文只放統一形式，(3) 原式放 `<details>`（B5）。

#### 步驟 3：Ψ-posterior（Sec. 3，式 (11)）

$$\Psi_{s|t}(\cdot|x^\ell,z_t^\ell)=\kappa_t\,q_{s|t}(\cdot|z_t^\ell,x^\ell)+(1-\kappa_t)\,q_s(\cdot|x^\ell),\qquad \kappa_t\in[0,1],\ \Psi_1=\mathrm{Cat}(\pi).$$
第一項依賴 $z_t$，第二項不依賴；作者稱為 superposition posterior。由 Bayes 得到的前向過程 $\Psi(z_t|z_s,x)$ 一般不是 Markov（Sec. 3「Ψ-Forward Processes」）。

#### 步驟 4：邊際保持（Suppl. A.2，式 (16)）

對 $t$ 向下歸納。基底 $\Psi_1=q_1$。若 $\Psi_t(\cdot|x)=q_t(\cdot|x)$，則
$$\Psi_s(z_s|x)=\sum_{\tilde z_t}q_t(\tilde z_t|x)\big[\kappa_t q_{s|t}(z_s|x,\tilde z_t)+(1-\kappa_t)q_s(z_s|x)\big]=\kappa_t q_s+(1-\kappa_t)q_s\sum_{\tilde z_t}q_t=q_s(z_s|x).$$
條件：(i) 條件在真 $x$ 上；(ii) $q_{s|t}$ 是與 (1) 相容的真 posterior；(iii) 第二項不依賴 $\tilde z_t$；(iv) $\kappa_t$ 不依賴 $z_t$（否則無法提到加總外；〔本頁推導〕$\kappa(z_t{=}x)=0.5$、其他 1 時邊際偏離 0.0525，見 5.1）。〔本頁推導〕代數對 $\kappa_t\notin[0,1]$ 仍成立（`core.py` 以 $\kappa=-0.3$ 驗證），但 $\Psi$ 可能出現負分量。$\kappa_t=0$ 時每步丟棄 $z_t$、直接從 $q_s(\cdot|x)$ 重抽。

#### 步驟 5：Ψ-sampler（Suppl. A.1，式 (12)/(15)）

作者「choose to replace」兩處：$q_{s|t}$ 的 $x\to x_\theta(z_t,t)$；$q_s(\cdot|x)$ 的 $x\to q_{0|t}(\cdot|z_t,x_\theta)$：
$$[\Psi^\theta_{s|t}(\cdot|z_t)]^\ell=\kappa_t\,q_{s|t}(\cdot|z_t^\ell,x_\theta^\ell)+(1-\kappa_t)\big[\alpha_s\,q_{0|t}(\cdot|z_t^\ell,x_\theta^\ell)+(1-\alpha_s)\pi\big].$$
$\kappa_t=1$ 還原 ancestral。MDM 時 $q_{0|t}=x_\theta$（Suppl. A.1 末句；這依賴 MDLM 的 carry-over 參數化：已解開位置 $x_\theta=e_{z_t}$）。USDM 的 $q_{0|t}$ 論文沒寫；〔本頁推導〕步驟 2 取 $\alpha_s=1$ 得 $q_{0|t}\propto x_\theta\odot(\alpha_t z_t+(1-\alpha_t)\mathbf 1/K)$。步驟 4 的保證在換成 $x_\theta$ 後不再成立；Sec. 3 末句「converge to the correct distribution given sufficient samples」是對 (11) 的敘述，論文沒有 (12) 的誤差界。作者把它類比為 Gaussian 擴散的 PC（Song et al., 2021）（Sec. 3；原文該句寫 $q_t$，應為 $q_s$，見 6.6 I5）。

#### 步驟 6：offset 項（Sec. 3「Intuitive Explanation」）

作者認為關鍵是 offset 項 $(1-\kappa_t)(1-\alpha_s)\pi$。MDM：已解開的 token 以機率 $(1-\kappa_t)(1-\alpha_s)$ 變回 [MASK]〔本頁推導：由 (26) 第一列〕。USDM：每個 token 至少有 $(1-\kappa_t)(1-\alpha_s)/K$ 的機率〔本頁推導〕，即使 $x_\theta$ 給正確 token 零機率。

#### 步驟 7：與 ReMDM 的對應（Corollary；Suppl. A.4，式 (21)–(26)）

ReMDM（式 (21)）：$z_t\ne m$ 時 $\mathrm{Cat}((1-\sigma_t)x+\sigma_t m)$；$z_t=m$ 時 $\mathrm{Cat}\big(\frac{\alpha_s-(1-\sigma_t)\alpha_t}{1-\alpha_t}x+\frac{1-\alpha_s-\sigma_t\alpha_t}{1-\alpha_t}m\big)$，$0\le\sigma_t\le\sigma_t^{\max}=\min\{1,(1-\alpha_s)/\alpha_t\}$。π = m 時代入 $\kappa_t=1-\sigma_t/(1-\alpha_s)$ 即得 (21)。論文只比對第一列；〔本頁推導〕第二列 $x$ 係數 $\frac{\alpha_s-\alpha_t}{1-\alpha_t}+\frac{\sigma_t}{1-\alpha_s}\cdot\frac{\alpha_t(1-\alpha_s)}{1-\alpha_t}=\frac{\alpha_s-(1-\sigma_t)\alpha_t}{1-\alpha_t}$，一致。ReMDM 又涵蓋 FB（Campbell et al., 2022）與 DFM（Gat et al., 2024）（Suppl. A.4 末段）。A.4 的實驗佐證（Table 11）只說「similar performance」。

〔本頁推導〕範圍：$\kappa_t\in[0,1]\iff\sigma_t\le1-\alpha_s$；$\sigma_t^{\max}\ge1-\alpha_s$，所以 $1-\alpha_s\lt\sigma_t\le\sigma_t^{\max}$ 對應 $\kappa_t\lt0$，超出 Sec. 3 的範圍（MDM 下仍合法）。因此「Ψ 涵蓋 ReMDM」嚴格成立的條件是 $\sigma_t\le1-\alpha_s$。三種 schedule（Suppl. A.5）：cap $\sigma_t=\min(\eta,\sigma_t^{\max})$（式 (27)）、rescale $\sigma_t=\eta\sigma_t^{\max}$（式 (28)）、loop（$[t_{\text{off}},t_{\text{on}}]$ 內 $\alpha_t=\alpha_s=\alpha_{t_{\text{on}}}$、$\sigma_t=\eta$）。換算成 $\kappa$〔本頁推導，數值見 5.3〕：rescale 在 $\alpha_t\ge1-\alpha_s$ 時 $\sigma^{\max}=(1-\alpha_s)/\alpha_t$、$\kappa=1-\eta/\alpha_t$；否則 $\sigma^{\max}=1$、$\kappa=1-\eta/(1-\alpha_s)$；全程約在 $[1-2\eta,1-\eta]$。cap 在 $s\lt\eta$ 的最後幾步 $\kappa\lt0$。loop 為常數 $1-\eta/(1-\alpha_{\text{on}})$（$\alpha_{\text{on}}=0.9$：$\eta=0.01$ 得 0.9，$\eta=0.06$ 得 0.4）。

**USDM 上的 κ schedule**（B1）。Table 13 對 Duo 使用「κt schedules matching ReMDM」，即把上面的 $\kappa_t=1-\sigma_t(\alpha_t,\alpha_s,\eta)/(1-\alpha_s)$ 當作只依賴 $(\alpha_t,\alpha_s,\eta)$ 的函式，直接套到 uniform 先驗。$\sigma^{\max}$ 作為「分佈合法」上界的意義只屬於 MDM；USDM 的合法條件是 $\kappa_t\in[0,1]$。rescale 全程 $\kappa\ge1-2\eta\ge0$，所以在 USDM 上合法；cap 會產生負 $\kappa$，論文沒有說明 USDM 如何處理。本頁把這個讀法標為**推定**（論文的套用方式只能由表名推得），W3 的 uniform 先驗只開放 rescale 與 window。

#### 步驟 8：NELBO（Suppl. A.3，式 (17)–(20)）——放在 S14

Ψ 也有 reconstruction + prior + Σ KL 的上界，prior 項為 0。論文明說「not directly relevant for sampling」；所有實驗都用既有 checkpoint，Ψ-sampler 不引入可學習元件（Sec. 6）。

#### 步驟 9：語言實驗（Sec. 5.1.1，Table 12–14）

設定：MDLM 與 Duo checkpoint，OWT，$L=1024$，1M 步、batch 512；logits 轉 64-bit（Sec. 5.1.1、C.1.1）。指標（S7 先定義）：Gen. PPL＝以 GPT-2 Large 計算生成文字的 perplexity（越低越好）；unigram entropy＝單一樣本內 token 頻率的熵，作為多樣性代理（重複文字會壓低 Gen. PPL，也會壓低熵）；nucleus sampling $p$＝每步只從累積機率達 $p$ 的最高機率 token 中抽（$p=1$ 為不截斷）。圖中選與 ancestral 熵最接近的 schedule（Suppl. D.1.2）。Table 13（log-linear step size、未蒸餾，括號為熵）：

| 設定 | 32 步 | 512 步 | 1024 步 | 4096 步 |
| --- | --- | --- | --- | --- |
| Duo ancestral, p=1.0 | 96.76 (5.57) | 76.93 (5.54) | 75.02 (5.53) | 75.39 (5.52) |
| Duo ancestral, p=0.9 | 44.24 (5.40) | 35.77 (5.37) | 34.79 (5.35) | 34.75 (5.35) |
| Duo rescale η=0.05, p=0.9 | 44.52 (5.41) | 31.06 (5.34) | 26.94 (5.28) | 19.21 (5.17) |
| Duo rescale η=0.02, p=1.0 | 89.55 (5.58) | 69.85 (5.54) | 68.39 (5.53) | 63.70 (5.52) |
| MDLM ancestral, p=0.9 | 70.34 (5.49) | 39.02 (5.35) | 37.91 (5.34) | 36.76 (5.31) |
| MDLM rescale η=0.05, p=0.9 | 69.44 (5.48) | 34.10 (5.33) | 30.29 (5.30) | 20.85 (5.16) |
| MDLM cap η=0.005, p=1.0 | 195.83 (5.74) | 110.75 (5.66) | 112.78 (5.67) | 131.85 (5.71) |
| Duo cap η=0.01, p=0.9 | 43.05 (5.40) | 23.38 (5.26) | 18.74 (5.17) | 10.88 (4.87) |

讀法（S7 內文照寫）：(a) p=0.9 時，ancestral 在 1024 步後持平，rescale η=0.05 繼續下降（Takeaway 3）；(b) 32 步時沒有增益（44.52 對 44.24），增益在 NFE 超過序列長度後才明顯（Takeaway 2）；(c) 持續下降依賴 nucleus 與 schedule：p=1.0 時 Duo rescale η=0.02 在 4096 步只到 63.70，MDLM cap η=0.005 由 512 步 110.75 變差到 4096 步 131.85；(d) nucleus 本身由 75.39 降到 34.75；(e) cap η=0.01 的 PPL 最低但熵降到 4.87，推薦 rescale 的依據是保住熵。推薦：rescale、η=0.05、p=0.9、log-linear、線性遞減的 $t$（Sec. 5.1.1；D.1.2 寫最佳 η∈{0.01,0.02}，見 6.6 I11）。蒸餾後 MDLM 較好（Table 14，ancestral p=0.9、4096 步：MDLM 22.46、Duo 29.19）。

#### 步驟 10：影像實驗（Sec. 5.1.2，Suppl. C.1.2、D.1.1，Table 6–11）

設定：35M 參數 U-Net（Table 3 為 35.8M），CIFAR-10 每像素 256 類，1.5M 步、batch 128，class dropout 0.1（訓練時以 10% 機率丟掉類別標籤，供 CFG 使用）。CFG 強度 1（Table 6、11），依式 (10) 等於純條件取樣〔本頁推導〕；Table 6 的 ancestral 結果最後一步取 greedy（argmax）。FID＝50k 樣本與訓練集的 Fréchet Inception Distance（越低越好）；IS＝Inception Score（越高越好）。cosine／log-linear 是兩種 noise schedule 形式（論文未給公式）。結果（Suppl. D.1.1）：ReMDM 把 MDLM 由 24.73 降到 23.71（4096 步）；Ψ 在 log-linear 取樣下把 23.71 降到 20.71；Duo（訓練 log-lin、取樣 cosine、4096 步）由 25.63 到 15.05，MDLM 由 24.73 到 17.86。同條件比較（Table 7）：MDLM cosine 取樣的 ancestral 為 23.93，不是 24.73；Duo ancestral 加 p=0.9 已有 22.41（log-lin 取樣、4096 步），而 Ψ 的比較不用 nucleus。推薦（Sec. 5.1.2）：Duo++ 用 cosine、$\kappa_t=0.95$、$t_{\text{on}}\in\{0.5,0.6\}$、$t_{\text{off}}=0.1$；MDLM 用 $\kappa_t=0.99$、$t_{\text{on}}=1.0$、$t_{\text{off}}=0.1$；$\kappa_t$ 只在 $[t_{\text{off}},t_{\text{on}}]$ 內小於 1。限制（Table 9，S8 內文照寫）：Fig. 1 右是逐步數挑最佳 FID 的包絡（Table 9 caption）——4096 步的 15.05 來自 $t_{\text{on}}=0.5$；$\kappa=0.95$、$t_{\text{on}}=0.6$ 在 1024 步 16.70、4096 步 18.99；$\kappa=0.5$、$[0.4,0.6]$ 由 256 步 20.77 惡化到 4096 步 105.75。Suppl. D.1：「With a poor choice of κt, Ψ-samplers can underperform ancestral sampling」。最佳 IS 9.50 與最佳 FID 來自不同設定（Table 10）。

#### 步驟 11：快速課程學習（Sec. 2.1.2、Sec. 4、Suppl. B）

背景（S9）：USDM 可由 one-hot 上的 Gaussian 擴散取 argmax 得到，邊際 $q_t(\cdot|x;\mathcal T(\tilde\alpha_t))$（式 (6)(7)），$\mathcal T$ 把 Gaussian 側 schedule 換算成離散側。Duo 在前 50% 訓練步把 transformer 輸入的 argmax 換成 $\mathrm{softmax}(w_t/\tau)$（式 (9)），輸入 embedding 變成 $V^\top y$ 的加權和；這要求每個位置具現 $K$ 維向量，$K>100{,}000$ 時記憶體不可行。作者觀察 $\tau=10^{-3}$ 時 softmax 幾乎只有少數幾項非零，所以只用 $k\ll K$ 個 embedding（式 (13)）。

1. **抽 top-k 值**（S10；Prop. B.1、B.4；Algo. 1）。乾淨座標 $o$ 的均值不同，另外單獨抽；其餘 $N=K-1$ 個座標 i.i.d. $\mathcal N(0,\tilde\sigma_t^2)$。$N$ 個 i.i.d. $U[0,1]$ 的最大值 CDF 為 $u^N$（式 (30)），可用 $U^{(1)}=V^{1/N}$ 直接抽；給定 $U^{(i)}=u$，$U^{(i+1)}$ 是 $N-i$ 個 $U[0,u]$ 的最大值（式 (38)）；$\Phi^{-1}$ 單調，$\tilde\sigma_t\Phi^{-1}(U^{(i)})$ 是第 $i$ 大的零均值值。Algo. 1 用 log-space：$R_\ell=\log U_\ell/\ell$，$P_\ell=\sum_{m=\ell}^{N}R_m$，$V_\ell=e^{P_\ell}$。$\tilde w\sim\mathcal N(\tilde\alpha_t,\tilde\sigma_t^2)$；$\tilde w\lt\mathcal K_k$ 為 case 1（$o$ 不在 top-k），否則 case 2：插在第 $r+1$ 名（$r=\#\{j:\mathcal K_j>\tilde w\}$），丟掉 $\mathcal K_k$。索引以 Floyd 演算法無放回抽（Algo. 2：依序抽 $k$ 個整數，抽到重複就改用目前範圍的最大值，不需要洗牌整個 $K$ 元素陣列）；頁面只以文字說明索引，W4a 不抽索引、只顯示值。〔本頁推導〕這一步抽出的值與索引在分佈上精確。
2. **近似加權和與 normalizer**（S11；式 (13)(14)(40)–(45)）。兩處近似：(a) 加權和只保留 $k$ 個 embedding（式 (13) 的 ≈），丟掉的權重不再使用，$\sum_i\lambda_i\lt1$；(b) $\tilde Z$ 中未抽樣的零均值項以截斷高斯的條件期望 $\mu=\mathbb E[e^{X/\tau}\mid X\lt c]$ 代替，
$$\log\mu=\frac{\tilde\sigma_t^2}{2\tau^2}-\log\Phi\Big(\frac{c}{\tilde\sigma_t}\Big)+\log\Phi\Big(\frac{c-\tilde\sigma_t^2/\tau}{\tilde\sigma_t}\Big)\quad(式\ (44)).$$
Case 1：$\tilde Z=\sum_{i\le k}e^{\mathcal K_i/\tau}+e^{\tilde w/\tau}+(K-k-1)\mu$（式 (41)，$c=\mathcal K_k$）；case 2：$\tilde Z=\sum_{i\le k}e^{\mathcal K_i/\tau}+(K-k)\mu$（式 (42)；門檻是插入後的第 $k$ 名，即插入前的 $\mathcal K_{k-1}$，見 6.6 I3）。式 (14) 後「δ = 1 if w̃ ∈ 𝒦 (case 2)」與 (41)(42)、Algo. 3 相反，本頁以後者為準。
3. **$\mathcal T$ 的線上計算**（Suppl. B.3；S9 只用文字）：級數 (49)、導數 (55)，約 150 項，只需快取 $M_n,I_n$（取代預先計算 100k 點、GPT-2 詞彙最多兩小時），再以 9 次多項式擬合（Fig. 5）。

#### 步驟 12：課程學習的結果（Sec. 5.2，Table 1、2、4、5）——S11 末段

訓練：16 張 H100、bf16、1M 步、batch 512；$\tau=0.001$，$(\beta,\gamma)=(0.03,0.15)$，前 500K 步。Table 2 LM1B／OWT PPL：Duo 29.9／25.2；Duo++ $k=2$ 30.0／25.2、$k=3$ 30.1／25.3、$k=5$ 30.2／25.4；MDLM 27.0／23.2。Table 4（8 張 GH200、batch 32）：Duo 81.8 samples/s、94.3 GiB；Duo 課程結束後 122.4、63.3；Duo++ 121.9、63.4。〔本頁推導〕吞吐比 121.9/81.8 = 1.49（Intro 寫「2× speedup」）；假設前後各半步數（前半課程期 Duo 81.8、Duo++ 121.9，後半兩者都是 122.4），總時間比 $(0.5/121.9+0.5/122.4)/(0.5/81.8+0.5/122.4)=0.803$（省 19.7%）；記憶體 −32.8%。摘要的「訓練時間 −25%」、D.3 的「平均吞吐 +25%」、Table 1 caption 的「25% fewer flops」是三個不同的量，論文沒有量測 FLOPs（6.6 I7）。

#### 步驟 13：實作細節與限制

64-bit logits；CIFAR 的 Ψ 不用 nucleus；Sec. 5.1.1／圖例說用 Duo++ checkpoint，C.1.1、D.1 說用 Duo（6.6 I12）。限制（S14）：沒有 (12) 的收斂保證；$\kappa_t$ 需逐任務調；增益伴隨熵下降或超參數在評估指標上挑選；USDM 的 likelihood 仍不如 MDM（Table 2）。

---

## 2. 學習目標（驗收測試）

規則：每題的參考答案在指定章節的**一個內文段落**內完整寫出（不只在元件、表格或檢查題中）；每章結尾 `<details><summary>檢查理解</summary>` 照搬題目與答案。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 寫出前向邊際與兩種先驗 | 式 (1) 怎麼寫？MDM 與 USDM 的 $\pi$？ | $z_t\sim\mathrm{Cat}(\alpha_t x+(1-\alpha_t)\pi)$，$\alpha_t$ 遞減、$\alpha_1=0$；MDM $\pi=m$，USDM $\pi=\mathbf 1/K$ | S1 |
| LO2 | 說明 ancestral sampler 與 MDM 不能 remask | ancestral sampler 如何由 posterior 得到？式 (2) 中 $z_t\ne m$ 時 $z_s$ 的分佈？ | 把 $q_{s|t}$ 的 $x$ 換成 $x_\theta(z_t,t)$，每步呼叫一次去噪器（NFE＝步數）；$z_t\ne m$ 時為 $\mathrm{Cat}(x)$，已解開的 token 不再更新；USDM 的 (3) 允許改成任何值 | S2 |
| LO3 | 寫出 Ψ-posterior | 式 (11) 由哪兩項組成？$\kappa_t$ 變小時 token 被改動的機率如何變化？ | $\kappa_t q_{s|t}(\cdot|z_t,x)+(1-\kappa_t)q_s(\cdot|x)$；$\kappa_t=1$ 為標準 posterior；$\kappa_t$ 越小，$z_s=z_t$ 的機率越低（例：均勻、$K=4$、$\alpha_t=0.3$、$\alpha_s=0.6$ 時 $\kappa$=1／0.5／0 依序為 0.625／0.505／0.385），邊際不變 | S3 |
| LO4 | 說出邊際保持的關鍵與條件 | 為何任何 $\kappa_t$ 都保持 $q_s$？需要哪些條件？ | 兩項以 $q_t$ 加權後都等於 $q_s$，凸組合仍是 $q_s$；條件：用真 $x$、$q_{s|t}$ 是真 posterior、第二項不依賴 $z_t$、$\kappa_t$ 不依賴 $z_t$；換成 $x_\theta$ 後不再保證 | S3 |
| LO5 | 寫出 Ψ-sampler | 式 (12) 替換了哪兩處？MDM 時 $q_{0|t}$ 是什麼？ | $q_{s|t}$ 的 $x\to x_\theta$；$q_s$ 的 $x\to q_{0|t}(\cdot|z_t,x_\theta)$；MDM 時 $q_{0|t}=x_\theta$（carry-over 參數化下） | S4 |
| LO6 | 說明 offset 項 | $(1-\kappa_t)(1-\alpha_s)\pi$ 在 MDM 與 USDM 各有什麼作用？ | MDM：已解開 token 以 $(1-\kappa_t)(1-\alpha_s)$ 機率變回 [MASK]；USDM：每個 token 至少 $(1-\kappa_t)(1-\alpha_s)/K$ | S4 |
| LO7 | 換算 ReMDM | Ψ 等於 ReMDM 時 $\kappa_t$ 與 $\sigma_t$ 的關係？何時 $\kappa_t\lt0$？ | $\kappa_t=1-\sigma_t/(1-\alpha_s)$；$\sigma_t>1-\alpha_s$ 時 $\kappa_t\lt0$（cap 在最後幾步），此時超出 Sec. 3 的 $\kappa_t\in[0,1]$；rescale 的 $\kappa_t$ 約在 $[1-2\eta,1-\eta]$ 之間，不會為負 | S5 |
| LO8 | 解釋 toy 中步數的作用 | 在本頁 toy 的預設設定下，為何 MDM ancestral 增加步數幾乎沒有改善，Ψ 會？ | toy 假設去噪器錯誤率隨正確位置比例下降；ancestral 下早期以高錯誤率解開的 token 不再更新；Ψ 的校正步以機率 $(1-\kappa_t)(1-\alpha_s)$ 把 token 變回 [MASK]，之後由錯誤率較低的去噪器重新預測；步數越多，校正次數越多。這是 toy 假設下的結果，不對應論文的 NFE | S6 |
| LO9 | 讀出語言實驗與條件 | OWT、nucleus $p=0.9$、log-linear、4096 步的 Duo：ancestral 與 rescale η=0.05 的 Gen. PPL（熵）？持續下降需要什麼條件？ | 34.75 (5.35) → 19.21 (5.17)；需要 $p=0.9$ 與 rescale；$p=1.0$ 時 Duo rescale η=0.02 只到 63.70，MDLM cap η=0.005 由 110.75 變差到 131.85；32 步無增益 | S7 |
| LO10 | 讀出影像實驗與限制 | CIFAR-10 Duo 的 FID 改善？兩個限制？ | Duo（訓練 log-lin、取樣 cosine、4096 步）25.63 → 15.05；Fig. 1 右是逐步數挑最佳超參數的包絡；選錯 $\kappa$ 比 ancestral 差（$\kappa=0.5$、$[0.4,0.6]$ 在 4096 步 105.75）；同條件 MDLM 基準為 23.93 而非 24.73 | S8 |
| LO11 | 說明 top-k 抽樣 | 如何不具現 K 維向量抽最大的 $k$ 個值？哪些步驟是近似？ | 乾淨座標另外抽；其餘 $N=K-1$ 個：$U^{(1)}=V^{1/N}$，給定 $U^{(i)}$ 時下一個是 $N-i$ 個 $U[0,U^{(i)}]$ 的最大值，再乘 $\tilde\sigma_t\Phi^{-1}$；值與索引在分佈上精確；只保留 $k$ 個 embedding（式 (13)）與 $\tilde Z$ 是近似 | S10 |
| LO12 | 說明 normalizer 與效率數字 | case 1、2 的 $\tilde Z$ 各含幾個未抽樣項？峰值記憶體與吞吐變化？ | case 1：$K-k-1$ 個，另加 $e^{\tilde w/\tau}$；case 2：$K-k$ 個；94.3→63.4 GiB，81.8→121.9 samples/s，OWT PPL 25.2→25.2（$k=2$） | S11 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），圖版最大 1080px。章節編號 JetBrains Mono（`01 · …`）；標題用陳述句。全頁視覺記號：**token 方格**（實心＝正確 token、斜線＝錯誤 token、空心虛線框＝[MASK]），在 hero、S1、W3 重複出現。**κ 軸**第一次出現在 S5（W2b），推薦值標記只在 S7、S8 的推薦段落旁的小 κ 軸上，並寫明適用範圍（F10）。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Ψ-Samplers：讓離散擴散在取樣時重新抽樣已產生的 token〉 | 副標：「這篇論文提出一種取樣方法：每一步以可調的小機率重新抽樣已產生的 token，不需重新訓練模型。」三張卡（下）；連結 arXiv、`../index.html#paper-394`、`psi-samplers-lineage-hub.html` | 1 螢幕 |
| S1 | `basics` | 01 · 離散擴散把 token 換成 [MASK] 或隨機 token | one-hot、simplex、$\mathrm{Cat}$、⊙、⟨⟩；式 (1)、$\alpha_t$、$q_t$／$q_s$ 記號、兩種先驗；靜態 SVG（8 格方格 × $t$=0、0.5、1，MDM 與 USDM 各一組）；符號表。延伸：`psi-discrete-diffusion-interpolating-process.html`、`discrete-diffusion-foundations-d3pm.html`、`diffusion-duality.html` | 450–550 字 |
| S2 | `posterior` | 02 · Ancestral sampler 依 posterior 逐步還原 | $q_{s|t}$ 的意義；統一 Bayes 形式（〔本頁推導〕）；(2) 的兩種情形；(3) 原式在 `<details>`；$x_\theta$、ancestral、NFE；LO2 段落。延伸：`mdlm-masked-diffusion-objective.html`、`psi-uniform-state-guidance-and-self-correction.html` | 400–500 字 |
| S3 | `psi` | 03 · Ψ-posterior 是兩個分佈的凸組合，邊際不變 | 同一組邊際可有多種聯合分佈 → (11) → (16) 逐行附理由 → 四個條件 → 非 Markov；W1；LO3（含保留率數字）、LO4 段落 | 500–600 字 + W1 |
| S4 | `sampler` | 04 · Ψ-sampler 以去噪器輸出代替乾淨資料 | (12)、$q_{0|t}$、predictor／corrector、offset 項、保證失效；W2；LO5、LO6 段落。延伸：`psi-superposition-posteriors-method.html` | 450–550 字 + W2 |
| S5 | `remdm` | 05 · 在 mask 先驗下，ReMDM 是 Ψ-sampler 的特例 | (21)、換算、條件 $\sigma_t\le1-\alpha_s$、FB／DFM、三種 schedule、USDM 套用方式（推定）；W2b；LO7 段落。延伸：`psi-predictor-corrector-gaussian-to-discrete.html`、`dlm-remasking-self-correction.html`、`psi-ctmc-discrete-diffusion-samplers.html` | 400–500 字 + W2b |
| S6 | `toy` | 06 · 玩具問題：校正步讓更多步數繼續降低錯誤率 | toy 假設（與圖版第一行相同）；W3；LO8 段落；與真實模型的差異（F11） | 350–450 字 + W3 |
| S7 | `lang` | 07 · 語言實驗：p = 0.9 與 rescale 下 Gen. PPL 在 1024 步後仍下降 | 指標定義段（Gen. PPL、unigram entropy、nucleus）；步驟 9 表與 (a)–(e)；推薦設定＋小 κ 軸（「OWT：rescale η = 0.05，κ 約 0.90–0.95，隨 t 變化」）；LO9 段落 | 500–600 字 + 表 |
| S8 | `image` | 08 · 影像實驗：CIFAR-10 FID 改善依賴 κ 與時間窗 | FID／IS、class dropout、CFG=1、greedy 最後一步；步驟 10 數字、同條件基準、推薦＋小 κ 軸（「CIFAR-10：Duo 0.95、MDLM 0.99，只在 [t_off, t_on] 內」）；Table 9 限制；LO10 段落 | 450–550 字 |
| S9 | `duality` | 09 · Duo 的課程學習需要 K 維 softmax | Gaussian latent、argmax、Duality 與 $\mathcal T$（一段）、softmax 溫度與 $\tau\to0$、$V^\top y$；為何 K 維向量不可行（$K>100{,}000$）；$\tau=10^{-3}$ 的稀疏性。延伸：`psi-duo-curriculum-background.html`、`diffusion-duality.html` | 400–500 字 |
| S10 | `topk` | 10 · Order statistics 直接抽出最大的 k 個值 | 逆轉換抽樣（式 (29)）、式 (30)(38)、Algo. 1、乾淨座標與 case、Floyd（一句）；W4a；LO11 段落 | 450–550 字 + W4a |
| S11 | `normalizer` | 11 · 以截斷高斯的期望值近似未抽樣項 | 截斷高斯定義、(40)–(45) 配方 4 行、δ 寫反、log-space 必要（$\tilde\sigma^2/2\tau^2$ 達 $10^5$）；W4b；Table 2、4 與推導比值；LO12 段落 | 450–550 字 + W4b + 表 |
| S12 | `impl` | 12 · 實作設定與常見錯誤 | 6.6 標「頁面」的條目；取樣設定小表 | 300–400 字 |
| S13 | `history` | 13 · 研究脈絡 | 時間線（6.7）+ 120 字 | — |
| S14 | `next` | 14 · 限制、NELBO 與延伸閱讀 | 限制；Suppl. A.3 一段；筆記清單 | 250 字 + 清單 |

規則：hero 與每節開頭只使用已定義的符號與指標；$q_{s|t}$、$q_s$、$\kappa_t$、Ψ-posterior 在 S1–S3 首次出現處定義，Gen. PPL、nucleus 在 S7 首次出現處定義。

Hero 三張卡（每張只用卡內自己定義的詞，B4）：
1. 「離散擴散模型把一段文字的每個 token 逐步換成雜訊，再由神經網路一步步還原。若雜訊是一個特殊的『遮罩』符號，模型還原出某個 token 後就不再改動它。」
2. 「本文的取樣方法在每一步，以一個可調的小機率重新抽樣已經還原的 token，讓後續步驟有機會改正錯誤；這個方法不需要重新訓練模型。」
3. 「在 CIFAR-10 影像生成上，以 FID（一種越低越好的影像品質分數）衡量，作者把 token 換成隨機 token 的模型 Duo 由 25.63 降到 15.05；每個步數各自挑選最佳設定。」
Hero 圖：兩列 8 格 token 方格（標「示意」）：上列「不重新抽樣」，一格斜線保持到最後；下列「重新抽樣」，同一格變回空心再變實心。reduced-motion 時為靜態三格快照。

桌機（≥1024px）左側 sticky 章節目錄；<1024px 頂部橫向捲動列（沿用參考頁）。

---

## 4. 互動元件規格

共同規則：圖版第一行（`.plate-lead`）寫該元件的**假設**；讀數一律「標籤＝值」；所有「應觀察到」文字只描述預設值，或由程式即時產生（帶入目前讀數），不寫死會隨輸入改變的結論。SVG 內只用 Unicode 標籤（κ_t、q_s、α_t、ε、T），需要 TeX 的放在 HTML 疊層並在每次重繪後對該容器呼叫 `renderMath(el)`；`aria-label` 不含 TeX（B10）。聯動滑桿統一規則：改動一個時，被限制者以 clamp 更新（先改 `value` 再改 `min/max`），並在該元件的 live region 宣告一次（例「α_s 已調為 0.31」）（B14）。live region 只在 `change` 或停止輸入 500ms 後更新。

### 4.0 共用數學（`core.py` 一對一移植；W3 部分已有 `chain.js`）

- `onehot, prior`：MDM 的 [MASK] 一律是最後一個索引 $K-1$。`qMarg`＝式 (1)。`post(zt,x,at,as,pi)`＝步驟 2 統一形式（要求 $\alpha_s>0$；分母 $\lt$ 1e-300 時回傳 `onehot(zt)`）。`q0t=post(…,1,…)`。`psiTrue`＝式 (11)；`psiTheta`＝式 (12)；`marginalAfter(x,at,as,kapOf,pi)`＝式 (16) 左邊，`kapOf(z)` 回傳該列的 κ（W1 的「κ 依賴 z_t」開關用）。
- `sigmaMax(at,as)`：$\alpha_t\le0$ 回傳 1。`kappaFromSigma(sig,as)`：$1-\alpha_s\le0$（最後一步）回傳 1。`kappaSched(kind,t,at,as,p)`，kind ∈ {`ancestral`,`window`,`rescale`,`cap`}，與先驗無關（步驟 7 末段）。
- PRNG，**照抄**（`verify/prng.js`，已確認與 `core.py` 前三個值 0.795194906881、0.827687913552、0.691516105784 相同，seed 1000）：
  ```js
  function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
  function catSample(r,p){const u=r();let c=0;for(let i=0;i<p.length;i++){c+=p[i];if(u<c)return i;}for(let i=p.length-1;i>=0;i--)if(p[i]>0)return i;}
  function gauss(r){const u1=r(),u2=r();return Math.sqrt(-2*Math.log(1-u1))*Math.cos(2*Math.PI*u2);}
  ```
- 常態分佈：`erfcc`（Numerical Recipes Chebyshev，相對誤差 $\lt$1.2e-7）；`logPhi(x)`：$x>-5$ 用 $\log(\tfrac12\,\mathrm{erfcc}(-x/\sqrt2))$，否則 $-x^2/2-\ln(-x)-\tfrac12\ln2\pi+\mathrm{log1p}(-x^{-2}+3x^{-4}-15x^{-6})$；`normPpf`：Acklam + 一次 Halley，$p>0.5$ 用 $-\mathrm{ppf}(1-p)$（係數見 `core.py`）。`lse`＝log-sum-exp。
- 格式：機率 `toFixed(4)`，錯誤率 3 位。

### W1 · 邊際保持（S3）

- **假設（第一行）**：「已知乾淨 token 為 A；這裡用真 posterior，不用去噪器。」
- **輸入**：先驗 分段（均勻＊／[MASK]）；$K\in\{3..8\}$＊4（MDM 時含 [MASK]）；$\alpha_t\in[0,0.98]$ 步長 0.01＊0.30；$\alpha_s\in[\alpha_t+0.01,1]$＊0.60（聯動規則見上）；$\kappa\in[0,1]$ 步長 0.01＊0.50，chip「κ = 1」「κ = 0」；開關「κ 依賴 z_t」＊關：開啟時 $z_t=$A 的列用 $\kappa$，其他列用 1（說明：「違反條件 (iv)」）。（＊＝預設）
- **計算**：每一列 $\Psi_{s|t}(\cdot|z_t,x)$（僅 $q_t(z_t)>0$）；邊際 `marginalAfter`；目標 $q_s$；最大差；保留率 $\sum_z q_t(z)\Psi(z|z)$。
- **顯示**：左：$K\times K$ 熱圖格（列＝$z_t$，欄＝$z_s$），格內印數值；列左標「q_t = …」；$q_t=0$ 的列為 `--surface-2` 並寫「此列不會出現」。右：「邊際」實心長條與「q_s」外框長條疊在同位置。讀數：「最大差 = …」（$\lt$1e-12 時印「0（浮點誤差內）」）、「保留率 = …」。
- **與內文的連結**（B7）：保留率支撐 LO3 段落（S3 內文寫預設下 0.625／0.505／0.385）；開關支撐 LO4 條件 (iv)（內文寫偏離 0.0525）。
- **應觀察到（預設）**：拖動 $\kappa$ 時熱圖各列改變、兩組長條重合；開啟開關後兩組長條分開。即時句：「κ = {κ}：保留率 {r}，邊際與 q_s 最大差 {d}。」

### W2 · Ψ-sampler 的一步（S4）

- **假設（第一行）**：「正確 token 為 A；去噪器輸出是本頁設定的三組機率（推定），不是真實模型。」
- **輸入**：先驗（均勻＊／[MASK]）；$K=4$ 固定（USDM：A B C D；MDM：A B C [M]）；$\alpha_t$＊0.30、$\alpha_s$＊0.60；$\kappa$＊0.50（B15）；「目前 $z_t$」：USDM A–D＊B；MDM [M]＊或 B（已解開）；「去噪器」（推定）：USDM `猜對` [0.85,0.05,0.05,0.05]、`偏錯` [0.5,0.3,0.15,0.05]、`全錯`＊[0,0.6,0.3,0.1]；MDM `猜對` [0.9,0.05,0.05,0]、`全錯`＊[0,0.7,0.3,0]。MDM 且 $z_t$ 已解開時 $x_\theta=e_{z_t}$（carry-over，選單停用並說明）。
- **計算／顯示**：三組長條：「預測步 q_{s|t}」「校正步 α_s q_{0|t} + (1−α_s)π」「混合 Ψ」；長條刻度上以虛線標下限 $(1-\kappa)(1-\alpha_s)/K$（USDM）。讀數表兩欄並排：「κ = 1」與「κ = 目前值」，列為 P(z_s = A)、P(z_s = [M])（MDM）；另列「下限 = …」（USDM）。
- **應觀察到（預設）**：USDM、全錯、κ=0.5：P(A) 由 0.0352（κ=1）升到 0.0676；即時句帶入兩欄數值。

### W2b · κ 與 σ 的換算（S5，與 W2 共用 $\alpha_t,\alpha_s$）

- **假設（第一行）**：「[MASK] 先驗；σ 是 ReMDM 的 remask 機率。」
- **輸入**：$\sigma\in[0,\sigma^{\max}]$ 步長 0.01＊0.04（$\sigma^{\max}$ 隨 $\alpha$ 更新，clamp）。
- **顯示**：直線圖 $\kappa=1-\sigma/(1-\alpha_s)$，橫軸 $[0,\sigma^{\max}]$；$\sigma>1-\alpha_s$ 的區段以斜線底紋標「κ < 0：超出 Sec. 3 的範圍」；目前點。讀數：σ_max、1 − α_s、κ、Ψ([M]|z_t = B) $=(1-\kappa)(1-\alpha_s)$（＝σ）。下方 κ 軸（0–1，左端外延到負值區）標 ancestral（1）與目前 κ。
- **應觀察到（預設）**：σ = 0.04 → κ = 0.9000；σ = 0.40 → κ = 0；σ = σ_max = 1 → κ = −1.5000。

### W3 · 玩具序列（S6）

- **假設（第一行，照寫）**：「假設：資料只有一個序列；去噪器的錯誤率 ε 隨目前正確位置的比例 c 線性下降（本頁自訂）；α_t = 1 − t（推定）。結果說明機制，不重現論文數字。」
- **toy 定義**（公式與 `chain.js` 相同）：$c=\frac1L\#\{\ell: z_t^\ell=x^{*\ell}\}$，分母是全部 $L$ 個位置，[M] 位置算作不正確；window：$t_{\text{off}}\le t\le t_{\text{on}}$ 時 $\kappa_t=\kappa$，否則 1（$t$ 是該步的起點 $t_i$）；rescale 用 4.0 的 `kappaSched`。$L=32$，真實 token 數 8，目標 $x^{*\ell}=\ell\bmod8$。MDM $K=9$，$z_1$ 全 [M]；USDM $K=8$，$z_1$ 每格由 $\pi$ 抽（依位置順序 32 次 `catSample`）。$t_i=i/T$，$i=T..1$，$s=t-1/T$，$\alpha_t=1-t$。每步先算 $c$、$\varepsilon=\varepsilon_{\max}-(\varepsilon_{\max}-0.02)c$、$\kappa_t$；$x_\theta^\ell=(1-\varepsilon)e_{x^{*\ell}}+\frac{\varepsilon}{7}\sum_{j\ne x^{*\ell}}e_j$（[M] 分量 0；MDM 已解開位置 $e_{z_t}$）；依位置順序每格一次 `catSample(psiTheta(...))`。錯誤率＝$z_0\ne x^*$ 的比例，對 $R=32$ 次（seed 1000..1031，每次新 `mulberry32(seed)`）平均。參考實作：`verify/chain.js`（頁面可直接移植）。
- **輸入**：先驗（[MASK]＊／均勻）；schedule（兩種先驗都可選以下三項；`ancestral` 即 $\kappa_t=1$，作為基準，與 R4 不衝突——R4 只排除 cap）：`ancestral`、`window`＊（$\kappa\in[0.5,1]$＊0.90；$t_{\text{on}}$＊0.6、$t_{\text{off}}$＊0.1，各 [0,1] 步長 0.05，聯動保持 $t_{\text{off}}\lt t_{\text{on}}$）、`rescale`（$\eta\in[0.005,0.1]$＊0.05）；$\varepsilon_{\max}\in[0.1,0.6]$ 步長 0.05＊0.5（F11：讓讀者看到持平程度取決於這個假設）。按鈕「計算」；參數變動時標「結果已過期」並中止進行中的計算。
- **顯示**：(1)「錯誤率對步數」：$T\in\{8,16,32,64,128,256,512\}$（log₂ 軸），縱軸 [0,0.5]；目前先驗的 ancestral（灰色實線＋圓點）與所選 schedule（`--accent` 實線＋方點），直接標籤；圖下註「每點 1024 個 token，統計誤差約 ±0.014」。(2)「κ_t 曲線」：$T=128$，橫軸 $t$ 由 1 到 0（標「取樣方向 →」）；rescale 時加同 η 的 cap 細虛線，$\kappa\lt0$ 段斜線底紋。(3)「一次執行的軌跡」：seed 1000、$T=64$（<600px 為 $T=32$），**單一 `<canvas>`** 畫 32 欄 × T 列方格（三種格子記號，hatch 以 canvas pattern 繪製，底色取 `getComputedStyle` 的 `--surface`）；canvas 外層容器 `tabindex=0`、`role="img"`（含動態 `aria-label`）＋固定 readout：點選或 ↑／↓ 鍵移動選取列，readout 顯示「第 i 步：t = …，c = …，ε = …，κ_t = …，[M] 數 = …」（B12）。
- **即時句**（不寫死）：「目前設定：T = 8 → 512，錯誤率 {a} → {b}；ancestral {c} → {d}。」
- **執行**：預配置 `Float64Array` 緩衝（`post` 的 num、`psi` 的輸出），避免每格配置陣列；以 `setTimeout` 分段，每段 8 次執行（$T\ge256$）或一個 $T$（其餘）；每段檢查 `token` 是否過期（B13）。`?test` 時以同一函式同步執行（不經 setTimeout）。
- **守衛**：`catSample` 捨入；$s=0$ 時 $\kappa=1$；$t=1$ 時 $\sigma^{\max}=1$；cap 不開放執行。

### W4 · Top-k 抽樣與 normalizer（W4a 在 S10、W4b 在 S11）

- **假設（第一行）**：W4a「單一位置的 Gaussian latent，乾淨 token 索引 o = 0。」W4b「同一次具現的 K 維向量 w 上比較三個 normalizer。」
- **共用輸入**（B9）：$K=\mathrm{round}(10^v)$，$v\in[2,\log_{10}50257]$ 步長 0.05，chip「50257（GPT-2）」設 $v=\log_{10}50257$ 並顯示 50257；$k\in\{1..8\}$＊3；$\tilde\alpha_t\in[0,0.99]$ 步長 0.01＊0.34（讀數 $\tilde\sigma_t$、log SNR $=\log(\tilde\alpha_t^2/\tilde\sigma_t^2)$）；$\tau=10^u$，$u\in[-3,0]$ 步長 0.1＊−3；seed＊（W4a 3、統計 11、W4b 7；「重新抽樣」+1）。上限 50257，所以 W4b 不會空白。
- **W4a**：`efficient(r,K,k,at,tau)`（`topk.py`；JS 參考移植 `verify/topk.js`，輸出與 Python 逐位相同，見 `out_topkjs.txt`）。**抽樣順序**：同一個 `r` 先抽 $k$ 個均勻數（$\ell=N,N-1,\dots,N-k+1$，每個一次 `r()`），再以 `gauss(r)`（兩次 `r()`）抽 $\tilde w$；「重複 400 次」以單一 `mulberry32(11)` 連續呼叫 400 次 `efficient`。Algo. 1（抽取順序 $\ell=N,N-1,\dots$，$N=K-1$；$U=0$→1e-300；$1-V$ 以 `-Math.expm1(P)`，$1-V\lt0.5$ 時用 $-\tilde\sigma\,\mathrm{ppf}(1-V)$）；`gauss` 抽 $\tilde w$；case、插入；$\lambda_i=e^{\mathcal K_i/\tau-\log\tilde Z}$。顯示 $k$ 個長條（乾淨 token 以星號），讀數「case = …」「Σλ = …」。「重複 400 次」：$P(\text{case 2})$、$\bar{\mathcal K}_1$、$\bar\lambda_1$、「λ₁ > 0.99 的比例」；$K\le10^4$ 時另跑 400 次暴力法（具現 $K$ 個、排序），top-1 值直方圖疊畫（實心＝高效、外框＝暴力），註「兩者用不同亂數，只應在統計誤差內一致」。連結：重複統計支撐 LO11（分佈精確）與 S9 的稀疏性陳述（內文寫 0.9725）。
- **W4b**：`naiveW`（seed 7，依索引順序 `gauss`，再 $w_0$ += $\tilde\alpha_t$）；真值 $\log Z$；$\log Z_{\text{top}}$（top-k 項，含乾淨 token 若在其中）；$\log\tilde Z$（`ztilde_from`：零均值座標以單次掃描維護前 k 大）。三條水平長條「Z_top / Z」「Z̃ / Z」「1」，讀數另列「Σλ = Z_top/Z̃」；$\tau$ 掃描小圖（$u=-3..0$ 共 31 點，重用同一 $w$）。連結：τ 掃描支撐 S11 內文「τ 越大，只取 top-k 的誤差越大」。
- **數值守衛**：normalizer 一律在 log 空間；$\tau=10^{-3}$ 時 $\tilde\sigma^2/2\tau^2\approx4.4\times10^5$ 與 $\log\Phi(\cdot)\approx-4.4\times10^5$ 相消，必須逐項在 log 空間相加；$k=1$ 的 case 2 沒有 $\mathcal K_{k-1}$，用 $\mathcal K_k$（本頁選擇）；$K-k-1=0$ 時略去該項。
- **應觀察到（預設）**：$\tau=10^{-3}$ 時 $\lambda_1$ 幾乎總是 > 0.99，三個 normalizer 相同；即時句帶入目前 Z_top/Z、Z̃/Z、Σλ。

---

## 5. 驗收數值表

由 `verify/accept.py`（與 `chain.js`）產生，演算法、PRNG、seed 與頁面相同。容許誤差：確定性機率 ±1e-4；W3 錯誤率在相同 PRNG 下應逐位相同（`chain.js` 已確認），容許 ±0.02 以防 `Math.log/cos` 末位差異；W4 統計值 ±0.02；W4b 比值 ±1e-3。

### 5.1 W1（$K=4$、$\alpha_t=0.30$、$\alpha_s=0.60$、$x$＝A）

| 先驗 | 量 | 值 |
| --- | --- | --- |
| 均勻 | q_t ／ q_s | q_t = [0.4750, 0.1750, 0.1750, 0.1750]；q_s = [0.7000, 0.1000, 0.1000, 0.1000] |
| 均勻 | 列 z_t = A | q_{s\|t} = [0.9211, 0.0263, 0.0263, 0.0263]；Ψ(κ=0.5) = [0.8105, 0.0632, 0.0632, 0.0632] |
| 均勻 | 列 z_t = B | q_{s\|t} = [0.5000, 0.3571, 0.0714, 0.0714]；Ψ(κ=0.5) = [0.6000, 0.2286, 0.0857, 0.0857] |
| 均勻 | 任一列 κ = 0 | Ψ = [0.7000, 0.1000, 0.1000, 0.1000] |
| 均勻 | 邊際 | κ = 1、0.5、0 皆 [0.7000, 0.1000, 0.1000, 0.1000]；最大差 < 1e-12 |
| 均勻 | 保留率 | κ=1：0.6250；κ=0.5：0.5050；κ=0：0.3850 |
| 均勻 | 開關開（κ(A)=0.5） | 邊際 = [0.6475, 0.1175, 0.1175, 0.1175]；最大差 = 0.0525 |
| [MASK] | q_t ／ q_s（A,B,C,[M]） | q_t = [0.3, 0, 0, 0.7]；q_s = [0.6, 0, 0, 0.4] |
| [MASK] | 列 A | q_{s\|t} = [1, 0, 0, 0]；Ψ(κ=0.5) = [0.8000, 0, 0, 0.2000] |
| [MASK] | 列 [M] | q_{s\|t} = [0.4286, 0, 0, 0.5714]；Ψ(κ=0.5) = [0.5143, 0, 0, 0.4857] |
| [MASK] | 保留率 | κ=1：0.7000；κ=0.5：0.5800；κ=0：0.4600 |
| [MASK] | 開關開（κ(A)=0.5） | 邊際 = [0.5400, 0, 0, 0.4600]；最大差 = 0.0600 |

### 5.2 W2、W2b（$\alpha_t=0.30$、$\alpha_s=0.60$）

| 設定 | 值 |
| --- | --- |
| 均勻、z_t=B、全錯：q_{0\|t} | [0.0000, 0.8028, 0.1479, 0.0493] |
| 均勻、z_t=B、全錯：Ψ | κ=1：[0.0352, 0.8099, 0.0986, 0.0563]；κ=0.9：[0.0417, 0.7870, 0.1076, 0.0637]；κ=0.5：[0.0676, 0.6958, 0.1437, 0.0930] |
| 均勻、全錯：下限 | κ=0.9：0.0100；κ=0.5：0.0500 |
| 均勻、z_t=B、偏錯 | q_{0\|t} = [0.3302, 0.5377, 0.0991, 0.0330]；P(A)：κ=1 0.1887、κ=0.9 0.1996、κ=0.5 0.2434 |
| 均勻、z_t=B、猜對 | q_{0\|t} = [0.7829, 0.1250, 0.0461, 0.0461]；P(A)：κ=1 0.4013、κ=0.9 0.4182、κ=0.5 0.4855 |
| [MASK]、z_t=[M]、全錯 | κ=1：Ψ = [0.0000, 0.3000, 0.1286, 0.5714]；κ=0.9：Ψ = [0.0000, 0.3120, 0.1337, 0.5543] |
| [MASK]、z_t=[M]、猜對、κ=0.9 | Ψ = [0.4011, 0.0223, 0.0223, 0.5543] |
| [MASK]、z_t=B | κ=1：[0, 1, 0, 0]；κ=0.9：[0, 0.9600, 0, 0.0400]（P([M]) = 0.0400） |
| W2b（0.30, 0.60） | σ_max = 1.0000；1 − α_s = 0.4000；κ(σ=0.04) = 0.9000；κ(σ=0.40) = 0.0000；κ(σ=1.00) = −1.5000 |
| W2b（0.70, 0.75） | σ_max = 0.3571；κ(σ=0.04) = 0.8400；κ(σ=0.25) = 0.0000；κ(σ=0.3571) = −0.4286 |

### 5.3 W3

κ_t 曲線（$\alpha_t=1-t$、$T=128$；±1e-4）：

| schedule | κ(t=1) | κ(0.5) | κ(0.25) | κ(1/T) | 最小值 | κ < 0 |
| --- | --- | --- | --- | --- | --- | --- |
| rescale η=0.05 | 0.9496 | 0.9000 | 0.9333 | 1.0000 | 0.9000（t = 0.5078） | 無 |
| rescale η=0.01 | 0.9899 | 0.9800 | 0.9867 | 1.0000 | 0.9800 | 無 |
| cap η=0.01 | 0.9899 | 0.9797 | 0.9587 | 1.0000 | −0.0159（t = 0.0156） | 1 步 |
| cap η=0.05 | 0.9496 | 0.8984 | 0.7935 | 1.0000 | −0.0579（t = 0.0547） | 6 步（t ≤ 0.0547） |

錯誤率，$\varepsilon_{\max}=0.5$（預設；window κ=0.9、[0.1, 0.6]；rescale η=0.05）：

| 先驗 / schedule | T=8 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [MASK] / ancestral | 0.3691 | 0.3828 | 0.3818 | 0.3672 | 0.3330 | 0.3750 | 0.3408 |
| [MASK] / window | 0.3672 | 0.3848 | 0.3701 | 0.3145 | 0.3086 | 0.2422 | 0.1953 |
| [MASK] / rescale | 0.3740 | 0.3857 | 0.3623 | 0.3037 | 0.3135 | 0.2568 | 0.2461 |
| 均勻 / ancestral | 0.3154 | 0.2842 | 0.2373 | 0.2178 | 0.2432 | 0.2451 | 0.2324 |
| 均勻 / window | 0.3193 | 0.2715 | 0.2051 | 0.1816 | 0.1924 | 0.1416 | 0.1260 |
| 均勻 / rescale | 0.3125 | 0.2725 | 0.2324 | 0.2090 | 0.1865 | 0.1387 | 0.1309 |

錯誤率，$\varepsilon_{\max}=0.2$（`node chain.js 0.2`）：

| 先驗 / schedule | T=8 | 16 | 32 | 64 | 128 | 256 | 512 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [MASK] / ancestral | 0.1240 | 0.1230 | 0.1338 | 0.1289 | 0.1123 | 0.1289 | 0.1289 |
| [MASK] / window | 0.1270 | 0.1074 | 0.1084 | 0.0986 | 0.1016 | 0.0693 | 0.0596 |
| 均勻 / ancestral | 0.0879 | 0.0723 | 0.0752 | 0.0615 | 0.0488 | 0.0527 | 0.0488 |
| 均勻 / window | 0.0850 | 0.0596 | 0.0576 | 0.0410 | 0.0498 | 0.0254 | 0.0342 |

$\kappa=0.8/0.9/0.95$、$\varepsilon_{\max}\in\{0.5,0.6\}$ 的調參紀錄見 `chain.py` 歷次輸出（定性相同）；預設值是**推定**的展示設定。

### 5.4 W4

W4b（seed 7、同一個 $w$、$o=0$；±1e-3）：

| 設定 | log Z | Z_top/Z | Z̃/Z | Σλ |
| --- | --- | --- | --- | --- |
| K=50257、α̃=0.34、k=3、τ=0.001 | 3748.5352 | 1.0000 | 1.0000 | 1.0000 |
| K=50257、α̃=0.34、k=3、τ=0.05 | 75.1210 | 0.9916 | 1.0297 | 0.9631 |
| K=50257、α̃=0.34、k=3、τ=0.3 | 15.5058 | 0.1129 | 1.0662 | 0.1059 |
| K=50257、α̃=0.34、k=3、τ=1 | 11.2657 | 0.0015 | 1.0010 | 0.0015 |
| K=50257、α̃=0.34、k=1、τ=0.05 | 75.1210 | 0.8605 | 1.2715 | 0.6767 |
| K=50257、α̃=0.90、k=3、τ=0.05 | 35.4024 | 0.8312 | 1.1702 | 0.7104 |
| K=50257、α̃=0.90、k=3、τ=0.3 | 11.8753 | 0.0060 | 1.0025 | 0.0060 |
| K=1000、α̃=0.34、k=3、τ=0.3 | 10.9693 | 0.3053 | 1.0411 | 0.2933 |
| K=1000、α̃=0.90、k=1、τ=0.05 | 25.5214 | 0.4386 | 0.9638 | 0.4550 |

讀法：$\tau\ge0.3$ 時 $Z_{\text{top}}/Z\lt0.31$，$\tilde Z/Z$ 在 0.96–1.07；$\tau=0.05$ 時 $\tilde Z$ 可高估到 1.27（尾項是期望值，不是逐次精確值）；Σλ 隨 $\tau$ 增大而遠小於 1，說明式 (13) 只保留 $k$ 項本身也是近似（F2）。

W4a（seed 11、400 次、K=50257、k=3；±0.02）：

| 設定 | P(case 2) | 平均 𝒦₁ | 平均 λ₁ | λ₁ > 0.99 比例 |
| --- | --- | --- | --- | --- |
| α̃=0.34、τ=0.001 | 0.0000 | 3.9628 | 0.9950 | 0.9725 |
| α̃=0.34、τ=0.05 | 0.0000 | 3.9628 | 0.8328 | 0.2650 |
| α̃=0.90、τ=0.001 | 0.0300 | 1.8396 | 0.9907 | 0.9450 |
| α̃=0.97、τ=0.001 | 0.5150 | — | 0.9896 | — |
| α̃=0.99、τ=0.001 | 1.0000 | — | 0.9997 | — |

單次抽樣（`efficient(mulberry32(3), 50257, 3, 0.34, 0.001)`；±1e-4，JS 與 Python 相同）：𝒦 = [4.0994, 3.5772, 3.5308]；w̃ = 0.3703；case = 1；λ = [1.0000, 0.0000, 0.0000]。補充（`topk.js`）：α̃=0.97 平均 𝒦₁ = 1.1007、λ₁>0.99 比例 0.9425；α̃=0.99 為 0.9852、0.9975。其他：式 (43) 閉式 1.211645556、數值積分 1.211645556（$\tilde\sigma=0.5,\tau=0.2,c=0.7$）；Algo. 1 前三名均值 3.236／2.950／2.796、暴力 3.244／2.953／2.800（$N=1000$、4000 次）；`normPpf(0.975)` = 1.959964。

---

## 6. 實作計畫

### 6.1 檔案與沿用

單一 `tut/394/build/index.html`。**複製參考頁 `tutorials/prediction-targets-k-diff.html` 的外殼**：`<head>`（Google Fonts、KaTeX 0.16.9 + auto-render、`renderMath()`）、CSS tokens 與明暗主題（`pm-theme`）、appbar、`#toc`／`#tocbar`、`.ch`／`.prose`、`.eq`／`.eq-n`、`.plate`／`.plate-h`／`.plate-lead`／`.plate-f`、`.ctl`／`.rng`／`.seg`／`.chip`／`.small-btn`、`.readout`／`.legend`、`.fact`、`.check`、`.tscroll`、`.tag`、`#pm-bar`、頁尾、`selfTest()` 框架。以下只列新增。資料系列色沿用 `--c-par`（預測步／正確 token）、`--c-perp`（校正步／雜訊）；`--accent` 只用於 Ψ 結果與介面強調。

### 6.2 新增 CSS／元件

- token 方格（SVG `<rect>` 14px；W3 軌跡用 canvas 同一記號）：正確＝實心 `var(--ink)`；錯誤＝`var(--surface)` 底＋45° 斜線 pattern（線色 `var(--ink)`）＋實線框；[MASK]＝無填色＋虛線框（`stroke-dasharray:2 2`）。不寫死白色（B11）。
- W1 熱圖：背景 `color-mix(in oklab, var(--c-par) calc(p*100%), var(--surface))`；字色：$p\ge0.5$ 用 `var(--surface)`、否則 `var(--ink)`——兩個 token 在明暗主題下都是相對色，深色主題下同樣成立；驗收時在深色模式截圖確認每格對比 ≥ 4.5:1，不足則把門檻改 0.6 並回寫。
- κ 軸：SVG，支援負值外延（W2b）；S7、S8 的小 κ 軸共用同一繪製函式，標記文字含適用範圍。
- 對數滑桿：`<input type=range>` 存指數，顯示值另算。
- TeX 中的小於號一律寫 `\lt`（避免 HTML 解析問題）；表格儲存格先標籤後數值。

### 6.3 程式結構（單一 IIFE）

- 共用數學：`onehot, prior, qMarg, post, q0t, psiTrue, psiTheta, marginalAfter, sigmaMax, kappaFromSigma, kappaSched, mulberry32, catSample, gauss, erfcc, logPhi, normPpf, lse`。
- `initW1/renderW1`、`initW2/renderW2/renderW2b`。
- W3：`runChain(kind,sched,T,p,seed,emax,trace)`（移植 `chain.js`，加預配置緩衝；`trace` 時回傳 `Uint8Array(T*L)` 與每步 $c,\varepsilon,\kappa$）；`errCurve(params, token, onDone, sync)`：`sync=true`（`?test`）時直接迴圈。快取 key 含全部參數。
- W4：`topkZeroMean, logMu, efficient, naiveW, ztildeFrom, renderW4a, renderW4b`。
- `renderTimeline(data)`。
- `selfTest()`（`?test`）：
  - `post` 對 (2)(3) 原式（$K=4$、5.1 參數）$\lt$1e-12；`marginalAfter` 在 $\kappa\in\{0,0.5,1,-0.3\}$ 對 $q_s$ $\lt$1e-12；ReMDM (21) 兩列 vs $\Psi(\kappa=1-\sigma/(1-\alpha_s))$，$\sigma=0.1$。
  - 5.1：[0.9211…]、保留率 0.5050、開關最大差 0.0525；5.2：κ=0.5 的 P(A) 0.0676、P([M]) 0.0400、W2b κ(σ=1) −1.5000（±1e-4）。
  - `kappaSched('cap')` $T=128$、$\eta=0.05$ 最小值 −0.0579。
  - mulberry32(1000) 前三值（±1e-12）；`errCurve` 同步：[MASK] window $T=512$ 0.1953、ancestral $T=512$ 0.3408（±0.02）。
  - `logMu(0.7,0.5,0.2)` 1.21165（±1e-5）；W4b 第 2、3 列比值（±1e-3）；`normPpf(0.975)` 1.959964（±1e-6）。

### 6.4 版面與響應式

沿用參考頁（390–1440px、16px 側邊留白、<768px 控制列堆疊、觸控區 ≥44px、圖上字 ≥12px）。W1 熱圖 <480px 只顯示選取列（chip 切換 $z_t$），長條移到下方；W3 軌跡 <600px 為 $T=32$；W4b τ 掃描 <480px 只留端點刻度。

### 6.5 引用與標籤

內文每個論文數字附出處；第 1 節以外不得新增論文數字。「本頁推導」「推定」兩種 pill（`.tag`）＋一行依據。區分論文結果、作者解釋（「作者認為」）、本頁推導。圖版底部：「讀數由本頁依式 (…) 即時計算」。

### 6.6 論文不一致與常見錯誤（S12 取「頁面」者）

| # | 內容 | 位置 |
| --- | --- | --- |
| I1 | 式 (14) 後 δ 的 case 寫反；以 (41)(42)、Algo. 3 為準 | 頁面 |
| I2 | $\mu$ 在 (40) 為期望、(43) 寫成 $\log\mathbb E[e^X]$（少 $/\tau$）；自洽讀法：(43)(44) 右邊為 $\log\mathbb E[e^{X/\tau}]$ | 頁面 |
| I3 | case 2 門檻：(45) 的 $\mathcal K_k$ 是插入後的編號，等於 Algo. 3 的 $\mathcal K_{k-1}$，兩者一致；但已知被丟掉的值是舊 $\mathcal K_k$，以 $\mathcal K_{k-1}$ 為門檻較寬鬆，是作者的設計選擇（F9） | 頁面 |
| I4 | B.1.4 寫 $K-i+1$，Prop. B.4／Algo. 1 為 $K-i$；且實際的變數個數是 $N=K-1$ | 頁面 |
| I5 | Sec. 3「$q_t$ plays the role of the corrector」應為 $q_s$ | 頁面 |
| I6 | Sec. 4「$\tau$ in (8)」應為 (9) | — |
| I7 | 「2× speedup」對 Table 4 的 1.49 倍；「25%」有三種定義 | 頁面 |
| I9–I10 | Table 1 caption「6 個任務中 4 個」實為 7 個中 5 個；Table 5 caption「7 個中 6 個」實為 5 個〔本頁推導〕 | S11 註腳 |
| I11 | 最佳 η：Sec. 5.1.1 為 0.05，D.1.2 為 {0.01, 0.02} | S7 |
| I12 | 取樣 checkpoint：Duo++ 或 Duo 前後不一 | S7 |
| I13 | Table 8「Best (p=0.9)」64 步 81.91 疑為 35.56 | — |
| I14 | D.1.1 的 MDLM 基準 24.73 是 log-lin 取樣；cosine 取樣為 23.93（Table 7） | S8 |
| P1 | 換成 $x_\theta$ 後仍宣稱邊際正確——保證只對 (11) | 頁面 |
| P2 | MDM 的 $q_{0|t}=x_\theta$ 依賴 carry-over 參數化 | 頁面（W2 註） |
| P3 | 把 cap schedule 直接套到 USDM 可能出現負 $\kappa$ | 頁面 |

### 6.7 研究脈絡（`notes/timeline.json`）

建置時原樣貼進 `<script type="application/json" id="timeline-data">`；渲染沿用參考頁（水平卡片列、篩選 chip、本文卡片 `--accent` 外框、缺欄位處理）。`TL_GROUP`：

| 類別 | arXiv | 筆記 |
| --- | --- | --- |
| 基礎 | 1503.03585、2107.03006 | `psi-discrete-diffusion-interpolating-process.html` |
| Gaussian PC | 2011.13456 | `psi-predictor-corrector-gaussian-to-discrete.html` |
| CTMC | 2205.14987、2310.16834、2407.15595 | `psi-ctmc-discrete-diffusion-samplers.html` |
| MDM | 2406.07524 | `mdlm-masked-diffusion-objective.html` |
| Guidance／USDM | 2207.12598、2406.01572、2412.10193 | `psi-uniform-state-guidance-and-self-correction.html` |
| Remasking | 2503.00307 | `psi-predictor-corrector-gaussian-to-discrete.html` |
| Duo | 2506.10892 | `diffusion-duality.html`（No. 393 教學）與 `psi-duo-curriculum-background.html` |
| 本文 | 2602.21185 | `psi-superposition-posteriors-method.html` |

S14 延伸閱讀依總覽順序：`psi-samplers-lineage-hub`、`psi-discrete-diffusion-interpolating-process`、`psi-ctmc-discrete-diffusion-samplers`、`psi-predictor-corrector-gaussian-to-discrete`、`psi-uniform-state-guidance-and-self-correction`、`psi-duo-curriculum-background`、`psi-superposition-posteriors-method`（各加 `.html`）；站內 `diffusion-duality.html`、`discrete-diffusion-foundations-d3pm.html`、`mdlm-masked-diffusion-objective.html`、`dlm-remasking-self-correction.html`、`uniform-noise-gidd-clean-supervision.html`。每項附 frontmatter `title`。

### 6.8 文案、無障礙、自我檢查

沿用參考頁 5.8–5.10。字面敘述，不用比喻（B8）：寫「已解開的 token 不再更新」「加入未抽樣項的期望值」「以機率 (1−κ_t)(1−α_s) 把 token 變回 [MASK]／重新抽樣」「去噪器錯誤率較低」；不寫「鎖住」「補上」「放回雜訊」「讓…重新預測」以外的擬人說法。名詞首次出現：「重新遮罩（remask）」「預測步（predictor）」「校正步（corrector）」「Floyd 演算法（無放回抽整數的方法，一句說明）」。token 方格與曲線以形狀＋直接標籤區分。W3 計算中不更新 live region，完成時宣告一次即時句。建置後跑 `?test`、逐條核對 12 個學習目標（答案在單一內文段落）、截圖 390／768／1024／1440＋深色＋focus＋reduced-motion。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | W3 是人為 toy | 「去噪器錯誤率隨正確比例下降」是本頁假設；真實 MDM ancestral 在 NFE ≤ L 時仍會改善（Table 13 MDLM p=0.9：70.34→37.91 至 1024 步），toy 從 T=8 就持平，來自 $\varepsilon_{\max}$ 的選擇 | 圖版第一行寫假設；開放 $\varepsilon_{\max}$；S6 明寫 toy 的 T 不對應論文 NFE，並連到 S8 的反例 |
| R2 | $\alpha_t=1-t$ 為推定 | 論文沒給 log-linear／cosine 公式 | 標「推定」 |
| R3 | USDM 的 $q_{0|t}$ 未寫出 | 由 (3) 取 $\alpha_s=1$ | 標「本頁推導」 |
| R4 | USDM 的 κ schedule | 論文只以表名表示「matching ReMDM」；cap 會出現 $\kappa\lt0$ | 步驟 7 的推定讀法；W3 不開放 cap 執行 |
| R5 | normalizer 版本 | δ 寫反；case 2 門檻是設計選擇 | 依 Algo. 3；S11 寫明 |
| R6 | 效率數字定義 | 25%／2×／33% 說法不一 | 照錄原文並給推導比值 |
| R7 | Fig. 1 左的藍底列 | 文字檔看不到底色 | 只說「Table 13 中 rescale η = 0.05、p = 0.9 的列」 |
| R8 | 效能 | W3 一條曲線約 1e7 次運算（node 約 0.25 秒）；W4a 暴力法 400×$10^4$ 個高斯 | 分段＋中止＋同步測試路徑；暴力法限制 $K\le10^4$ |
| R9 | 浮點差異 | 抽樣以 `<` 比較累積和 | 驗收 ±0.02 |
| R10 | 篇幅 | 15 個章節 | B 線（S9–S11）自成一段；hero 只談 A 線 |

---

## 8. 修訂紀錄（round 1）

採納：F1（rescale 兩種情形對調）、F2（式 (13) 也是近似，加 Σλ 讀數）、F3（$N=K-1$）、F4（hero 與 1.1 區分 (11)／(12)）、F5（ReMDM 等價條件）、F6／F7（p、schedule、η 寫進敘述與 LO9）、F8（同條件基準 23.93、22.41，I14）、F9（I3 改寫）、F10（推薦值標記移到 S7、S8 並寫範圍）、F11（toy T 不對應 NFE；加 $\varepsilon_{\max}$）、F12（greedy 最後一步；去噪器向量標推定）、B1（USDM κ schedule 的推定讀法）、B2（S8 拆成 S9–S11）、B3（語言／影像分兩節並先定義指標）、B4（hero 卡只用自定義詞）、B5（S1 補記號；(3) 放 details）、B6（W2b；NELBO 移到 S14）、B7（W1 開關與保留率、W4 讀數各連到 LO）、B8（字面用詞）、B9（K 滑桿規格）、B10（SVG 標籤）、B11（深色）、B12（canvas＋鍵盤）、B13（緩衝、分段、中止）、B14（clamp 規則）、B15（預設 κ=0.5、兩欄讀數）。

未採納：無。

## 9. 修訂紀錄（readiness）

阻擋項：(1) 1.1 加註「頁面不得照搬」，第 3 節規則要求頁面第一次出現就定義；(2) W4a 寫出抽樣順序、新增 `verify/topk.js`（`efficient`、`logMu`、`normPpf`、`erfcc` 移植，輸出與 `topk.py` 一致；`logMu(0.7,0.5,0.2)` JS 1.211645625 對 Python 1.211645556，差異來自 erfcc，容許 1e-5），單次抽樣列加容許誤差。非阻擋 6 項：W3 的 $c$ 與 window 公式；Floyd 只在文字說明；ancestral 兩種先驗皆可選；符號表加 $q_{0|t}$；LO7 改「約在 $[1-2\eta,1-\eta]$」；Table 4 寫明前後各半假設，0.803／19.7%。

## 10. 建置紀錄（build）

- 來源在 `build/src/`（head、css_base（參考頁 CSS + No. 255 共用補充）、css_add、body1–3、app.js），`python3 build/src/assemble.py` 產生 `build/index.html`（143 KB，無 file:// 路徑）；`build/src/lo.py` 為 LO 自查；`build/shot.sh` 以本機 KaTeX 截圖，`?shot=<id>` 只顯示含該元素的章節（僅供截圖）。
- 驗收：`?test` 49 passed、0 failed（1440 與 390px iframe 內皆同；W3 走同步路徑）；LO 自查 12／12；`impeccable detect` 0 anti-patterns（advisory：章節編號 14 則、`.tk-bad` 圖例的斜線 repeating-gradient 1 則，屬功能性圖例記號）。
- 視覺一輪修正：式 (16) 改兩行；SVG／純文字中的 `q_{s|t}` 改為 Unicode；W2 圖例改色塊、下限標籤移到左側；W3 兩張圖的軸標避讓；W4a 改用「第 i 大」標籤（𝒦 字形在內文字型缺字）、統計區未計算時隱藏；hero 方格在 390px 依寬度縮小；W1 熱圖列標籤加寬。深色模式截圖確認 canvas 與 SVG 皆取主題色。
- 與設計的差異：W2b 未另畫 κ 軸（折線圖的縱軸即 κ，含負值區）；hero 圖為靜態三格快照（不做動畫）。
