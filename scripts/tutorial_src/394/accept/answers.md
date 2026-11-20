# 394 Ψ-Samplers 驗收答案（Goldfish，僅依頁面文字）

**LO1**
- 答：式 (1) 為 $z_t\sim q_t(\cdot|x)=\mathrm{Cat}(\cdot;\alpha_t x+(1-\alpha_t)\pi)$。MDM 的 $\pi=m$（[MASK] 的 one-hot）；USDM 的 $\pi=\mathbf 1/K$（均勻分佈）。
- 段落：「π=m（MDM）、π=1/K（USDM）」
- 信心：high

**LO2**
- 答：生成時沒有 x，所以訓練去噪器 $x_\theta(z_t,t)$ 輸出 simplex 上的機率向量，把 $q_{s|t}$ 裡的 x 換成 $x_\theta$，就得到 ancestral sampler $p^\theta_{s|t}$。式 (2)：MDM 下 $z_t\ne m$ 時 $z_s\sim\mathrm{Cat}(x)$，已解開的 token 不再更新；$z_t=m$ 時以機率 $\frac{\alpha_s-\alpha_t}{1-\alpha_t}$ 解開成 x，否則留在 [MASK]。
- 段落：「就得到 ancestral sampler」
- 信心：high

**LO3**
- 答：式 (11) 為 $\Psi_{s|t}=\kappa_t q_{s|t}(\cdot|z_t,x)+(1-\kappa_t)q_s(\cdot|x)$。第一項依賴目前狀態 $z_t$，第二項不依賴。κ 變小時 token 被改動的機率變高（保留率下降）：頁面例子 K=4、均勻先驗、$\alpha_t=0.3$、$\alpha_s=0.6$ 時，κ=1、0.5、0 的保留率依序為 0.625、0.505、0.385。
- 段落：「保留率依序是 0.625、0.505、0.385」
- 信心：high

**LO4**
- 答：頁面以對 t 向下的歸納證明（Suppl. A.2）。基底 $\Psi_1=\mathrm{Cat}(\pi)=q_1$；若 $z_t$ 邊際為 $q_t$，則 $\Psi_s=\kappa_t q_s+(1-\kappa_t)q_s\sum q_t=q_s$，與 κ 值無關。需要四個條件：(i) 用真 x；(ii) $q_{s|t}$ 是與式 (1) 相容的真 posterior；(iii) 第二項不依賴 $z_t$；(iv) κ 不依賴 $z_t$。頁面的 W1 例子顯示若 κ 依賴 $z_t$（違反 iv），邊際會偏移（最大差 0.0525）。
- 段落：「這個證明需要四個條件」
- 信心：high

**LO5**
- 答：式 (12) 替換兩處：(a) $q_{s|t}$ 裡的 x 換成 $x_\theta(z_t,t)$；(b) $q_s(\cdot|x)$ 裡的 x 換成 $q_{0|t}(\cdot|z_t,x_\theta)$（即 s=0 的 posterior）。MDM 時 $q_{0|t}=x_\theta$，但前提是 carry-over：已解開位置的 $x_\theta$ 必須等於 $z_t$。
- 段落：「MDM 時 q0∣t=xθ」
- 信心：high

**LO6**
- 答：偏移項為 $(1-\kappa_t)(1-\alpha_s)\pi$。MDM（π=m）：已解開的 token 以 $(1-\kappa_t)(1-\alpha_s)$ 的機率變回 [MASK]，之後重新解開，稱為 remask。USDM（π=1/K）：每個 token 至少得到 $(1-\kappa_t)(1-\alpha_s)/K$ 的機率，即使去噪器給正確 token 零機率，它仍可能出現。W2 例：$P(z_s=A)$ 從 κ=1 的 0.0352 升到 κ=0.5 的 0.0676，下限 0.0500（本頁推導）。
- 段落：「在 USDM，每個 token 至少得到」
- 信心：high

**LO7**
- 答：$\kappa_t=1-\sigma_t/(1-\alpha_s)$（Suppl. A.4；論文只比對第一列，其餘由頁面推導）。$\kappa_t<0$ 發生於 $\sigma_t>1-\alpha_s$。頁面以 cap schedule 在 128 步、$\alpha_t=1-t$、η=0.05 算出最小 κ 為 −0.0579（只出現在最後幾步）；rescale schedule 的 κ 不為負。
- 段落：「當 σt>1−αs 時」
- 信心：high

**LO8**
- 答：在 toy 預設 $\varepsilon_{\max}=0.5$ 下，MDM ancestral 從 8 步到 512 步的錯誤率只從 0.369 到 0.341，停在約 0.33–0.38，因為早期以高錯誤率解開的 token 不再更新。Ψ（window κ=0.9、t∈[0.1,0.6]）錯誤率從 8 步的 0.367 降到 512 步的 0.195，因為校正步把 token 以 $(1-\kappa_t)(1-\alpha_s)$ 機率變回 [MASK] 重新預測，步數越多校正次數越多。步數少時兩者相同。頁面另註明 toy 的 T 不對應論文的 NFE。
- 段落：「增加步數幾乎沒有改善」
- 信心：high

**LO9**
- 答：OWT、nucleus p=0.9、log-linear、4096 步的 Duo：ancestral 為 Gen. PPL 34.75（unigram entropy 5.35）；rescale η=0.05 為 19.21（5.17）。持續下降需要 p=0.9 與 rescale 兩者：p=1.0 時 Duo rescale η=0.02 在 4096 步只到 63.70；MDLM cap η=0.005 則由 512 步的 110.75 變差到 4096 步的 131.85。ancestral 在 1024 步後持平（34.79→34.75）。
- 段落：「持續下降需要 p=0.9」
- 信心：high

**LO10**
- 答：CIFAR-10 Duo（訓練 log-linear、取樣 cosine、4096 步）的 FID 由 ancestral 的 25.63 降到 15.05（MDLM 由 24.73 降到 17.86）。兩個限制：(1) Fig. 1 右是逐步數挑最佳超參數的包絡，15.05 來自 $t_{\text{on}}=0.5$，同樣 κ=0.95 但 $t_{\text{on}}=0.6$ 時 1024 步為 16.70、4096 步為 18.99；(2) 選錯 κ 會比 ancestral 差，κ=0.5、[0.4,0.6] 由 256 步的 20.77 惡化到 4096 步的 105.75。
- 段落：「由 ancestral 的 25.63 降到 15.05」
- 信心：high

**LO11**
- 答：不具現 K 維向量的做法：乾淨 token 座標另抽 $\tilde w\sim\mathcal N(\tilde\alpha_t,\tilde\sigma_t^2)$，其餘 N=K−1 個座標為 i.i.d. $\mathcal N(0,\tilde\sigma_t^2)$。N 個 i.i.d. U[0,1] 的最大值 CDF 為 $u^N$，故 $U^{(1)}=V^{1/N}$；$U^{(i+1)}$ 是 N−i 個 $U[0,U^{(i)}]$ 的最大值；再乘 $\tilde\sigma_t\Phi^{-1}$ 得第 i 大的值。Algo. 1 在 log 空間計算，索引以 Floyd 演算法無放回抽出。
- 近似步驟：只保留 k 個 embedding（式 13），以及用期望值近似未抽樣項的正規化分母 $\tilde Z$（下一節）。頁面另註明「值與索引在分佈上精確」是本頁推導。
- 段落：「只保留 k 個 embedding（式 13）」
- 信心：high

**LO12**
- 答：$\tilde Z$ 中，case 1（乾淨 token 不在前 k 名）有 K−k−1 個未抽樣項（另加乾淨 token 的 $e^{\tilde w/\tau}$）；case 2 有 K−k 個。效率（Table 4，8 張 GH200、batch 32）：峰值記憶體 94.3→63.4 GiB（−32.8%，本頁推導）；吞吐 81.8→121.9 samples/s（約 1.49 倍，本頁推導）。頁面 Intro 寫的「2× speedup」與表格不符。
- 段落：「94.3→63.4 GiB」
- 信心：high

---

## 初學者可能卡住的地方

1. **符號密度**：one-hot、simplex、$\mathrm{Cat}(\cdot;v)$、⊙、內積在 01 節一次出現，而 π 與 m 的關係（m 是 [MASK] 的 one-hot）只有一句話帶過。
2. **「邊際不變」的證明**：式 (16) 的歸納證明需要 (i)–(iv) 四個條件，頁面只給摘要，沒說明為什麼 κ 依賴 $z_t$ 會破壞證明。
3. **$q_{0|t}$ 的來源**：MDM 的 $q_{0|t}=x_\theta$ 依賴 carry-over，USDM 版本是頁面自行推導（令 $\alpha_s=1$），初學者看不出推導依據。
4. **κ 與 σ 的換算與三種 schedule**：cap、rescale、loop 只給數字（如 −0.0579），$\sigma_t^{\max}$ 與 $\alpha_t=1-t$ 的換算沒有展開。
5. **頁面內部不一致**：式 (14) 後的 δ 條件寫反（頁面以式 (41)(42) 為準）、式 (43) 少了 /τ、推薦 η 在 §5.1.1 與 Suppl. D.1.2 不同（0.05 對 {0.01, 0.02}）、Duo 與 Duo++ checkpoint 說法不一，讀者不知道哪個數字才是依據。
