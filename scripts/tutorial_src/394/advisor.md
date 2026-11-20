# 394 顧問筆記：The Diffusion Duality, Chapter II: Ψ-Samplers（arXiv 2602.21185v2, ICLR 2026）

給教學設計者，不是給讀者。凡標 〔顧問推導〕 者不是論文原文，是我由論文公式推出或用數值驗過的結論（K=4 與 K=100 的小例已用 numpy/scipy 實算核對）；其餘數字與式號都直接取自 paper.txt。

符號：K 詞彙大小、L 序列長、x one-hot 乾淨 token、z_t 時刻 t 的離散 latent、π 先驗（MDM：π=m；USDM：π=1/K）、α_t 遞減的 noise schedule、s<t、x_θ denoiser 輸出（simplex 上的機率向量）。論文沒有定義 α_{t|s}；依 Duo 前作慣例是 α_{t|s}=α_t/α_s 〔顧問推導：用此定義數值驗證 (3) 滿足 Σ_{z_t} q_t(z_t|x) q_{s|t}(z_s|z_t,x)=q_s(z_s|x)，成立〕。

---

## 1. 論證骨架（9 步）

論文其實是兩篇獨立的東西釘在一起：A 線＝取樣器（Sec. 3, 5.1, Suppl. A, D.1），B 線＝訓練 curriculum 的省記憶體實作（Sec. 4, 5.2, Suppl. B, D.2–D.3）。兩線互不依賴，教學時應分開。

**A 線：Ψ-sampler**

1. **前提：插值型前向邊際**（Sec. 2.1, 式 (1)）。假設每個 token 獨立、邊際為 q_t(·|x)=Cat(α_t x+(1−α_t)π)。只規定了「邊際」，沒規定聯合分佈——這是後面全部自由度的來源。
2. **兩種標準 posterior**（式 (2) MDM、式 (3) USDM）。由 Bayes 得 q_{s|t}(·|z_t,x)。MDM 的 (2)：已解開的 token 永遠不變（不能 remask）。USDM 的 (3)：token 可反覆改值。這一步「證明」的只是：ancestral sampler＝把 x 換成 x_θ 的 q_{s|t}。
3. **定義 Ψ-posterior**（Sec. 3, 式 (11)）：Ψ_{s|t}(·|x,z_t)=κ_t q_{s|t}(·|z_t,x)+(1−κ_t) q_s(·|x)，κ_t∈[0,1]，Ψ_1=Cat(π)。假設：只需要 q_{s|t} 是與 (1) 相容的真 posterior。沒有對 π 做任何假設，所以對 mask、uniform、混合先驗都成立。
4. **邊際保持定理**（Suppl. A.2, 式 (16) 加五行歸納）。證明：對任意 κ_t 序列，Ψ_t(·|x)=q_t(·|x) 對所有離散化時刻成立。注意這是「以真 x 為條件」的敘述。
5. **可取樣的近似**（Suppl. A.1, 式 (12)/(15)）：把 q_{s|t} 裡的 x 換成 x_θ(z_t,t)；把 q_s(·|x) 裡的 x 換成 q_{0|t}(·|z_t,x_θ)，得
   Ψ^θ_{s|t}=κ_t q_{s|t}(·|z_t,x_θ)+(1−κ_t)[α_s q_{0|t}(·|z_t,x_θ)+(1−α_s)π]。
   這一步是「選擇」不是「推導」（原文用 "we choose to replace"）。每步只用一次 x_θ(z_t,t)，所以 NFE＝步數〔顧問推導〕。
6. **涵蓋 ReMDM**（Corollary；Suppl. A.4, 式 (21)–(26)）。π=m 時代入 κ_t=1−σ_t/(1−α_s) 即得 ReMDM posterior (21)；ReMDM 又涵蓋 FB（Campbell 2022）與 DFM（Gat 2024）corrector。Suppl. A.5 給 cap (27)、rescale (28)、loop 三種 σ_t schedule。
7. **NELBO 存在**（Suppl. A.3, 式 (17)–(20)）。只是說明 Ψ 也能寫成 reconstruction+prior+ΣKL；論文明說「與取樣無直接關係」，沒有用它訓練任何模型。所有實驗都用既有 checkpoint，Ψ 純粹是 training-free 的推論期改動。
8. **實驗主張**（Sec. 5.1；Tables 6–14, Figs. 1, 6）。OWT：nucleus p=0.9 下 Gen. PPL 隨 NFE 持續下降，ancestral 則持平；CIFAR-10：Duo FID 25.63→15.05、MDLM 24.73→17.86、ReMDM 最佳 23.71（D.1.1）。

**B 線：快速 curriculum**

9. **稀疏化 Duo 的 curriculum**（Sec. 4；Suppl. B）。前提：Duo 在前 50% 訓練步用 softmax(w_t/τ) 取代 argmax(w_t) 當 transformer 輸入（式 (9)），τ=0.001，需要對每個 token 具現 K 維向量。Duo++ 三個零件：(a) 用 order statistics 直接抽 K−1 個零均值高斯中最大的 k 個（B.1，Algo. 1）加一個獨立抽的「乾淨 token」值 w̃~N(α̃_t,σ̃_t²)，比大小決定 case 1/2；(b) 用截斷高斯的條件期望補上未抽樣項的 normalizer（式 (14)/(45)，推導 (43)–(44)）；(c) 用級數 (49)、(55) 與 9 次多項式擬合取代 100k 點查表算 𝒯（B.3）。結果：Table 4 的 81.8→121.9 samples/s、94.3→63.4 GiB；Table 2 PPL 幾乎不變（LM1B 29.9→30.0，OWT 25.2→25.2，k=2）。

---

## 2. 必須教對的 5 個推導

### D1. 邊際保持（Suppl. A.2）——全篇的核心，只有五行
對 t 向下歸納。基底：Ψ_1=Cat(π)=q_1。假設 Ψ_t(·|x)=q_t(·|x)，則
```
Ψ_s(z_s|x) = Σ_{z̃_t} Ψ_t(z̃_t|x) Ψ_{s|t}(z_s|z̃_t,x)                       (16)
           = Σ_{z̃_t} q_t(z̃_t|x) [κ_t q_{s|t}(z_s|x,z̃_t) + (1−κ_t) q_s(z_s|x)]
           = κ_t Σ_{z̃_t} q_t(z̃_t|x) q_{s|t}(z_s|x,z̃_t) + (1−κ_t) q_s(z_s|x) Σ_{z̃_t} q_t(z̃_t|x)
           = κ_t q_s(z_s|x) + (1−κ_t) q_s(z_s|x) = q_s(z_s|x)
```
成立條件：(i) 條件在真 x 上；(ii) q_{s|t} 是真 posterior（第一項才會邊際化成 q_s）；(iii) q_s(·|x) 不依賴 z_t（第二項才能把 Σq_t=1 提出來）；(iv) κ_t 可隨 t 變，但不能依賴 z_t。κ_t 在 [0,1] 之外時代數仍成立，只是 Ψ 未必是合法機率〔顧問推導〕。
教學重點：這是「兩個邊際都等於 q_s 的分佈做凸組合，邊際仍是 q_s」。κ=0 的極端＝每步丟掉 z_t、直接從 q_s(·|x) 重抽。
〔顧問推導〕K=4、α_t=0.3、α_s=0.6、x=e_1 的數值例：(3) 給出 z_t=x 時 q_{s|t}=[0.9211, 0.0263, 0.0263, 0.0263]；z_t=e_2≠x 時 q_{s|t}=[0.5, 0.3571, 0.0714, 0.0714]（x 佔 0.5、留在原 token 0.357）。以 q_t=[0.475,0.175,0.175,0.175] 加權後正好得 q_s=[0.7,0.1,0.1,0.1]；κ=0、0.5、0.95 三種混合後也仍是 q_s。

### D2. 取樣器 (12) 與「offset 項」
把 (12) 的第二項展開：(1−κ_t)α_s q_{0|t}(·|z_t,x_θ) + (1−κ_t)(1−α_s)π。論文稱 (1−κ_t)(1−α_s)π 為 offset（Sec. 3 "Intuitive Explanation"）：
- MDM：這一項把質量放到 [MASK] 上 → 已解開的 token 可被 remask。
- USDM：這一項給每個 token 至少 (1−κ_t)(1−α_s)/K 的機率 → denoiser 給近零機率的正確 token 仍可能出現。
q_{0|t} 的形式：MDM 時 q_{0|t}(·|z_t,x_θ)=x_θ（A.1 末句）。USDM 時論文沒有寫出來。〔顧問推導〕在 (3) 取 s=0（α_s=1、α_{t|s}=α_t）得
```
q_{0|t}(·|z_t,x) = [Kα_t z_t⊙x + (1−α_t)x] / [Kα_t⟨z_t,x⟩ + 1−α_t]  ∝  x ⊙ (Kα_t z_t + (1−α_t)1)
```
即「以 x_θ 為先驗、q_t(z_t|·) 為 likelihood 的 Bayes 更新」：現有 token 那一格被放大 (Kα_t+1−α_t)/(1−α_t) 倍。x 為 one-hot 時退化成 x。數值例：α_t=0.3、K=4、x_θ=[.5,.3,.15,.05]、z_t=e_2 → q_{0|t}=[0.330,0.538,0.099,0.033]。
成立條件／注意：D1 的邊際保證在換成 x_θ 後不再成立（除非 x_θ 是完美的）。論文 Sec. 3 末句 "hence we converge to the correct distribution given sufficient samples" 是對真 posterior (11) 的敘述，不是對 (12) 的定理。

### D3. ReMDM 對應（Suppl. A.4）
π=m，Ψ 展開後（式 (26)）：
- z_t≠m（此時 z_t=x）：Cat([κ_t+(1−κ_t)α_s]x + (1−κ_t)(1−α_s)m)
- z_t=m：Cat([κ_t(α_s−α_t)/(1−α_t)+(1−κ_t)α_s]x + [κ_t(1−α_s)/(1−α_t)+(1−κ_t)(1−α_s)]m)

與 (21) 比對第一列：σ_t=(1−κ_t)(1−α_s) ⟺ κ_t=1−σ_t/(1−α_s)。論文在這裡只寫「then the equation reduces to the ReMDM posterior」，沒驗第二列。〔顧問推導〕第二列的 x 係數：
```
(α_s−α_t)/(1−α_t) + [σ_t/(1−α_s)]·[α_s − (α_s−α_t)/(1−α_t)]
 = (α_s−α_t)/(1−α_t) + [σ_t/(1−α_s)]·α_t(1−α_s)/(1−α_t) = [α_s−(1−σ_t)α_t]/(1−α_t)   ✓ 與 (21) 一致
```
條件與漏洞〔顧問推導〕：κ_t∈[0,1] ⟺ σ_t∈[0,1−α_s]。但 ReMDM 允許到 σ_t^max=min{1,(1−α_s)/α_t}，而 (1−α_s)/α_t ≥ 1−α_s。所以 σ_t∈(1−α_s, σ_t^max] 對應 κ_t<0，超出 Sec. 3 宣告的 κ_t∈[0,1]。「Ψ 涵蓋 ReMDM」嚴格說要允許 κ_t<0（此時 Ψ 在 MDM 仍合法，因 σ≤σ^max）。具體：
- rescale（σ=ησ^max）：α_t<1−α_s 時 κ=1−η/(1−α_s)；α_t≥1−α_s 時 κ=1−η/α_t。兩端 κ≈1−η，中段（α≈0.5）最低約 1−2η。η=0.05 → κ 約在 0.90–0.95，全程合法。
- cap（σ=min(η,σ^max)）：後段 (1−α_s)/α_t<η 時 κ=1−1/α_t<0（略負）。
- loop：區間內 α_t=α_s=α_on，σ=η → κ=1−η/(1−α_on)；α_on=0.9 時 κ=1−10η（η=0.01→0.9；η=0.06→0.4）。
論文沒說把 cap 這類 schedule 搬到 USDM（κ 可能為負）時如何處理。

### D4. 不具現 K 維向量抽 top-k（Sec. 4 Step 1；B.1；Algo. 1）
(a) K 個 i.i.d. U[0,1] 的最大值 CDF 為 u^K（Prop. B.1, 式 (30)）→ U^(1)=V^{1/K}，V~U[0,1]。
(b) 給定 U^(i)=u^(i)，U^(i+1) 分佈同「K−i 個 i.i.d. U[0,u^(i)] 的最大值」（Prop. B.4, 式 (38)：密度 (K−i)(u^(i+1))^{K−i−1}/(u^(i))^{K−i}）→ U^(i+1)=u^(i)·V^{1/(K−i)}。
(c) Φ^{-1} 單調 → 第 i 大的高斯＝σ̃_t·Φ^{-1}(U^(i))。
(d) Algo. 1 的 log-space 寫法：N=K−1；R_ℓ=log(U_ℓ)/ℓ，P_ℓ=Σ_{m=ℓ}^{N} R_m，V_ℓ=exp(P_ℓ)，ℓ=N,…,N−k+1。
(e) 併入乾淨 token：另抽 w̃~N(α̃_t,σ̃_t²)。𝒦_k>w̃ → case 1（o 不在 top-k）；w̃>𝒦_k → case 2，r=#{j:𝒦_j>w̃}，插在第 r+1 名並丟掉原最小值。索引：case 1 從 [K]\{o} 無放回抽 k 個；case 2 抽 k−1 個再把 o 放在第 r+1 位（Floyd 演算法，Algo. 2）。
成立條件：其餘 K−1 個座標必須 i.i.d.（one-hot 上加各向同性高斯才成立）；w̃ 與它們獨立。這是「分佈上精確」的抽樣，不是近似；近似只出現在 D5 的 normalizer。
〔顧問推導〕已用 N=1000、k=3 的 Monte Carlo 核對 Algo. 1 的遞迴與暴力排序的前三名均值一致（0.9990/0.9980/0.9970）。

### D5. normalizer 的尾項（B.2, 式 (40)–(45)）
未抽樣的零均值項在給定門檻 c 後是 i.i.d. 截斷高斯 X~N(0,σ̃_t²) | X<c。令 σ=σ̃_t/τ（對 X̄=X/τ 配方）：
```
E[e^{X/τ} | X<c] = e^{σ²/2} · Φ((c/τ−σ²)/σ) / Φ(c/σ̃_t)
log E[·] = σ̃_t²/(2τ²) − log Φ(c/σ̃_t) + log Φ((c − σ̃_t²/τ)/σ̃_t)                 (43)/(44)
```
Case 1（o 不在 top-k，式 (41)）：Z̃ ≈ Σ_{i≤k} e^{𝒦_i/τ} + e^{w̃/τ} + (K−k−1)·E[·]
Case 2（o 在 top-k，式 (42)）：Z̃ ≈ Σ_{i≤k} e^{𝒦_i/τ} + (K−k)·E[·]
成立條件：以每項的條件期望代替實際值（Z̃ 是近似，不是無偏的 softmax）；必須用 logΦ 在 log 空間算（τ=10⁻³ 時 σ̃_t²/(2τ²)~10⁵ 量級，直接算會溢位）〔顧問推導〕。
〔顧問推導〕(43) 已用數值積分核對（σ̃=0.5、τ=0.2、c=0.7：兩邊皆 1.21165）。
這個推導在原文有三處會讓老師踩雷，見 §3 的 I1–I3。

（次要，若要教 𝒯：式 (6)、(49)、(55)。〔顧問推導〕𝒯 的意義：P(argmax(w_t)=o)=∫φ(z−ν_t)Φ^{K−1}(z)dz，ν_t=α̃_t/√(1−α̃_t²)；令它等於 α_t+(1−α_t)/K 解出 α_t 即得 (6)。𝒯(0)=0、𝒯(1)=1。K=100 時實算 𝒯(0.3)=0.0113、𝒯(0.6)=0.0428、𝒯(0.9)=0.3384：高度非線性，高斯側訊號很強時離散側仍幾乎全是雜訊。M_0=1/K、I_n=M_{n+1}。）

---

## 3. 常見誤讀、內部不一致、未交代之處

### 3.1 看摘要／圖會誤信、但論文沒有證明的事
- M1「Ψ-sampler 保持正確邊際，所以收斂到正確分佈」：只對真 posterior (11)、條件在真 x 上成立（A.2）。實際取樣器 (12) 用 x_θ，沒有任何誤差界或收斂定理。
- M2「步數越多越好」：只對挑過的設定成立。Table 9 固定設定下常在高 NFE 變差：Duo κ=0.95、cos、t_on=0.6、t_off=0.1 在 2048 步 16.30、4096 步 18.99；κ=0.5、t_on=0.6、t_off=0.4 由 256 步 20.77 惡化到 4096 步 105.75。Fig. 1 右是「每個步數各取最佳 FID」的包絡線（Table 9 caption），不同步數用的是不同超參數（4096 步的 15.05 來自 t_on=0.5；1024 步最佳 16.70 來自 t_on=0.6）。D.1 原文也說選錯 κ_t 會輸給 ancestral。
- M3「Ψ 是 PPL 下降的主因」：OWT 上 nucleus 的貢獻更大。Table 13 Duo ancestral 4096 步：p=1.0 為 75.39，p=0.9 為 34.75；再加 rescale η=0.05 才到 19.21。且 32 步時 Ψ 沒有幫助（44.52 對 ancestral 44.24）；增益在 NFE≥512 才明顯（1024 步 26.94 對 34.79）。
- M4「matched unigram entropy」：〔顧問推導〕由 Fig. 1 左的端點與 (5.2) 標註推斷圖中畫的是 Table 13 的 rescale η=0.05、p=0.9 兩列（Duo 19.21 (5.17)、MDLM 20.85 (5.16)；文字檔看不到藍底，無法直接確認）。熵是 Duo 與 MDLM 彼此相近，但比 ancestral 的 5.35 低。更激進的 schedule 換來的低 PPL 伴隨熵崩：cap η=0.01、p=0.9、4096 步 Duo 10.88 (4.87)。Gen. PPL 低不等於品質好。
- M5「USDM 超越 MDM」：只在「未蒸餾、p=0.9、rescale」的 Gen. PPL 與 CIFAR FID 上。likelihood 仍輸：Table 2 OWT PPL MDLM 23.2 對 Duo++ 25.2；Table 1 多數下游任務輸 MDLM（原文承認）。蒸餾後的模型 MDLM 較好（Table 14 ancestral p=0.9、4096 步：MDLM 22.46 對 Duo 29.19）。結論段自己的用詞也只是 "matches the performance of MDMs"。
- M6「CIFAR 的 baseline 很公平」：(i) D.1.1 的「MDLM 24.73 (ancestral)」是 log-linear 取樣；同 checkpoint 用 cosine 取樣是 23.93（Table 7），比 ReMDM 最佳 23.71 只差 0.22。(ii) Duo ancestral 加 nucleus p=0.9 就有 22.21（256 步，Table 7），已勝 ReMDM 最佳；Ψ 的比較則不用 nucleus。(iii) FID 是對 training set、50k 樣本計算，超參數直接在這個指標上挑。
- M7「FID 與 IS 同時最佳」：最佳 IS 與最佳 FID 來自不同設定。Duo 最佳 IS 9.50（κ=0.5、log-lin、t_on=0.6、t_off=0.4、4096 步，Table 10）對應的 FID 是 27.97（Table 9）。
- M8「用了 CFG」：Sec. 5.1.2 說以 discrete CFG 取樣，但 Table 6、Table 11 寫強度為 1。〔顧問推導〕依式 (10)，γ=1 時無條件項係數為 0，等於純條件取樣。
- M9「Ψ-posterior 需要重新訓練／是新模型」：不需要。A.3 的 NELBO 沒被用來訓練。
- M10「curriculum 的 softmax 是很糊的混合」：〔顧問推導〕τ=10⁻³ 時絕大多數情況幾乎是 one-hot。K=50257、σ̃=0.94 的模擬：top-1 與 top-2 的平均差約 0.20，即 200τ；約 98% 的樣本 top-1 權重>0.99。這是 k=2 就夠用的原因；Figs. 7–14 是 log 尺度的直方圖，容易誤看成 top-2 常有 0.2–0.4 的權重。
- M11「top-k 是近似抽樣」：值與索引在分佈上是精確的（D4）；近似只在 Z̃。D.2 自己也只驗了邊際直方圖，並明說邊際相符不保證聯合相符。
- M12「recommend loop 以外的 schedule 是因為 PPL 較低」：不是。Table 13 loop η=0.01、p=0.9、4096 步 Duo 13.61 (5.05) 的 PPL 比 rescale η=0.05 的 19.21 (5.17) 低；推薦 rescale 的依據是保住熵。

### 3.2 內部不一致（原文對原文）
- I1 式 (14) 後的 δ 定義寫反：正文「δ=1 if w̃∈𝒦 (case 2)」；但 (41)/(42) 與 Algo. 3 是 case 1（o 不在 top-k）才另加 e^{w̃/τ} 且未抽樣項為 K−k−1，即 δ=1 應對應 case 1。以 (41)/(42)/Algo. 3 為準。
- I2 μ 的定義：(40) 寫 μ=E[exp(X/τ)|X<𝒦_k]，(43) 卻寫 μ=log E[exp(X)|X<𝒦_k]（多了 log、少了 /τ），(45) 又以 exp(…) 使用。自洽的讀法：(43)/(44) 右邊是 log E[exp(X/τ)|·]，(41)/(42) 裡的 μ 是其 exp。(40) 下一行的文字 "pmf"、"the expectation that … is less than" 也不通。
- I3 case 2 的門檻：(45) 一律寫 𝒦_k；Algo. 3 的 case 2 用 𝒦_{k−1}（插入前的編號），case 1 用 𝒦_k。
- I4 B.1.4「Sampling the k-th largest」寫 U_{i+1} 是 K−i+1 個中的最大值；Prop. B.4 與 Algo. 1 是 K−i。以後者為準。
- I5 Sec. 3 寫「q_t plays the role of the corrector」；(11)、Fig. 2 用的是 q_s。
- I6 Sec. 4 寫 τ=10⁻³ "in (8)"；(8) 沒有 τ，應為 (9)。Intro 的 "(Fig. 1, right, Table 5)" 被拿來支持 perplexity，但 Fig. 1 右是 CIFAR FID。
- I7 加速數字：Intro 寫 curriculum "2× speedup"，Sec. 5.2 寫 "doubles the speed"；Table 4 是 81.8→121.9 samples/s 〔顧問推導：1.49 倍〕。摘要寫「training time 減 25%」，D.3 寫「average throughput 高 25%」，Table 1 caption 寫「25% fewer flops」，三者不是同一個量。〔顧問推導〕以 Table 4 計：一半步數 81.8、一半 122.4 對全程 121.9，總時間比 0.804（省約 19.6%），平均吞吐高約 24.3%；記憶體 94.3→63.4 為 −32.8%。FLOPs 沒有任何量測。
- I8 量測環境：訓練用 16 張 H100、batch 512（Sec. 5.2）；Table 4 的效率量測是 8 張 GH200、batch 32。
- I9 Table 1 caption：「k=2 在 6 個任務中 4 個高於 Duo」；表內有 7 個任務，k=2 高於 Duo 的是 Arc-c、WinoG、PIQA、MathQA、OQA 共 5 個〔顧問推導：逐欄比對〕。
- I10 Table 5 caption：「k=2 在 7 個資料集中 6 個低於 Duo」；逐欄比對只有 5 個（PTB 94.96>89.35、Wiki 34.05>33.57 較差）〔顧問推導〕。
- I11 最佳 η：Sec. 5.1.1 寫 rescale η=0.05 最好；D.1.2 寫最佳是 rescale η∈{0.01,0.02}。
- I12 用哪個 checkpoint 取樣：Sec. 5.1.1 與圖例寫 Duo++（efficient curriculum 訓練）；C.1.1、D.1 寫用 Duo 原始 pre-trained checkpoint，Tables 12–14 的列名也是 Duo。
- I13 Table 8「Best (p=0.9)」列的 64 步寫 81.91（與 32 步同值），依上方各列應為 35.56。
- I14 CIFAR 模型大小：正文 35M，Table 3 為 35.8M。
- I15 B.3.1「Advantages」把查表 100k 點歸給 Sahoo et al. (2024)（那是 MDLM），其他處是 (2025a)。
- I16 (56)/(62) 左邊寫 d/dν_t，(58) 稱為 ∂f/∂t 卻沒有 α̃_t′ 因子；最終 (55) 是對的（dν_t/dt=α̃_t′/(1−α̃_t²)^{3/2}）〔顧問推導〕。
- I17 B.3.1 說級數「as t→1 收斂變慢」。〔顧問推導〕項的大小由 ν_t=α̃_t/√(1−α̃_t²) 決定，ν_t 在 α̃_t→1（即 t→0）時發散；原文方向存疑。
- I18 符號撞名：𝒦（top-k 值）對 K（詞彙）；m=K−1 對 mask m；B.1.5 "excluding i" 應為 o；Prop. B.4 證明裡的 k、l 與 top-k 的 k。

### 3.3 未交代的超參數與定義
- α_{t|s} 沒定義；USDM 的 q_{0|t} 沒寫出（見 D2）。
- (β,γ)=(0.03,0.15) 的意義（(9) 中 t~U[β,γ]）在本文沒有解釋，只引 Duo。
- "log-linear"、"cosine" schedule 的公式沒給。
- OWT 實驗的 κ_t 是由 ReMDM 的 σ_t 換算；用在 USDM 時 σ^max 的界（原本是為了 MDM 合法性）是否仍採用、κ<0 如何處理，都沒說。
- Table 11「Duo with the ReMDM rescale schedule」沒給 η。
- CIFAR 每一步是否每個像素獨立抽、最後一步 greedy（Table 6 caption 提到）在 Ψ 表中是否沿用，沒說。
- 9 次多項式擬合的係數與誤差只有 Fig. 5（約 10⁻⁵–10⁻⁸ 量級的圖），無數值表。級數截斷 150 項、「300 個係數」＝150 個 M_n 加 150 個 I_n。
- Gen. PPL 的樣本數、OWT 取樣是否 64-bit 之外的細節沒列。

---

## 4. 哪些可以在瀏覽器現場演示

**可以（閉式或小型 toy，全部是機率向量運算）**
1. **邊際保持（D1）**：K=3–8、單一 token、自選 α_t、α_s、κ。精確計算 Σ_{z_t} q_t·Ψ_{s|t}，拉 κ 滑桿顯示邊際不動、但轉移矩陣在變。可同時切 π=mask／uniform。
2. **(3) 的 posterior 長相**：顯示 z_t=x 與 z_t≠x 兩種情況的長條圖（§2 D1 的 K=4 數字可當測試值）。
3. **offset 的作用（D2）**：給一個「故意錯」的 x_θ（正確 token 機率≈0），比較 ancestral（κ=1）與 Ψ 的轉移機率；USDM 下限 (1−κ)(1−α_s)/K、MDM 的 remask 機率 (1−κ)(1−α_s)。
4. **ReMDM 對應（D3）**：σ↔κ 換算器；畫 rescale／cap／loop 的 κ_t 曲線並標出 κ<0 的區段。這是〔顧問推導〕，演示時要標明非論文圖。
5. **鏈式 toy 取樣**：L 約 16–64、K 約 4–8、用一個帶固定錯誤率的手寫「假 denoiser」，跑 ancestral 對 Ψ，展示 MDM 的錯 token 卡死、Ψ 可修。屬示意，不能宣稱重現論文數字，也不要把它畫成「步數越多越好」的證據（見 M2）。
6. **order-statistics 抽 top-k（D4）**：K=50257、k=5 即時抽；與暴力「抽 K 個再排序」的直方圖對比（暴力版在 JS 每次 5 萬個高斯，幾百次以內可行）。Φ^{-1} 需自帶近似（如 Acklam）；V 很接近 1，用 log-space 與 float64。
7. **softmax 稀疏性（M10）**：τ 滑桿，顯示 top-1 權重分佈；τ=10⁻³ 時幾乎 one-hot。
8. **normalizer 尾項（D5）**：比較精確 Z、只取 top-k 的 Z、加上尾項的 Z̃。需 logΦ 的穩定實作；τ 極小時要在 log 空間。建議 demo 用 τ≥0.05 才看得出尾項貢獻。
9. **𝒯 曲線**：小 K（≤1000）用數值積分 (6) 畫 α̃_t→α_t；可加 Monte Carlo argmax 對照。
10. **Table 4 的算術（I7）**：1.49 倍、19.6%、24.3%、32.8% 的來源，一個小計算器即可。

**不可以**
- Gen. PPL／unigram entropy 曲線（Fig. 1 左、Tables 12–14）：需要 138M 參數 checkpoint、L=1024、GPT-2 Large 評分。只能把表中數字畫成靜態圖。
- CIFAR FID／IS（Tables 6–11、Fig. 6）：需 35.8M U-Net、50k 樣本、Inception 網路。
- 訓練吞吐與峰值記憶體（Table 4）、perplexity 與下游準確率（Tables 1, 2, 5）：需實際訓練。
- 「efficient 與 naive 的聯合分佈一致」：論文自己也只驗邊際（D.2）；瀏覽器可驗小 K 的邊際，不能替論文補上聯合的主張。
- 級數 (49) 取 150 項於 K=50257：M_n 隨 n 快速成長、需高精度積分與 log 空間，不適合現場算；改用小 K 的直接積分。

---

## 5. 先備知識（教學順序）

1. 類別分佈、one-hot、simplex；凸組合仍是機率分佈。
2. 聯合／邊際／條件、Bayes；「同一組邊際可對應很多聯合」（全篇的支點）。
3. 離散擴散的前向邊際 (1)：α_t 與「保留原 token 對換成先驗樣本」的混合解讀。
4. 兩種先驗：mask（吸收態，一去不回）對 uniform（可反覆改）；各自的 posterior (2)、(3) 與 ancestral sampling。
5. denoiser x_θ 是什麼、為何把 x 換成 x_θ 是近似；NFE 的定義。
6. Predictor-corrector 的直覺（先往前一步，再注入雜訊重修），以及為何 MDM 沒有 remask 會累積錯誤。
7. （到此可教 A 線：(11)、A.2、(12)、ReMDM 對應。）
8. 評估指標的讀法：Gen. PPL 可被重複文字作弊 → 要配 unigram entropy；nucleus sampling；FID／IS。
9. 高斯擴散 w_t=α̃_t x+σ̃_t ε 與 argmax pushforward（Duality，(6)–(7)）；𝒯 是把高斯側 schedule 換算成離散側 schedule。
10. softmax 溫度、τ→0 趨近 argmax；embedding lookup＝one-hot 乘 embedding 矩陣，soft 版是加權和。
11. 逆轉換抽樣；order statistics（最大值 CDF 為 u^K；給定第 i 大後其餘仍 i.i.d.）。
12. 截斷高斯與配方求 E[e^{aX}|X<c]；logΦ 的數值穩定性。
13. （選修）Taylor 級數、積分與求和／微分交換（dominated convergence），只在教 B.3 時需要。
