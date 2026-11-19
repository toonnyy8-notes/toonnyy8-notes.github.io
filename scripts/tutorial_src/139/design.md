# 設計文件：〈Value Forcing: Aligning Return Distributions at All Noise Levels〉互動教學

- 論文：Chongyi Zheng、Zhiyuan Zhou、Qiyang Li、Benjamin Eysenbach、Sergey Levine，arXiv 2611.02160（v1，2026-11-01）。資料庫編號 No. 139。程式碼 `https://github.com/chongyi-zheng/value-forcing`；專案頁 `https://chongyi-zheng.github.io/value-forcing`。
- 原文：`tut/139/src/paper.txt` 與 `paper.pdf`（`tut/` = `scripts/tutorial_src/`）。本文件的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」「Alg. 1」皆指論文原文編號。顧問筆記 `tut/139/advisor.md`（以下稱「顧問」）。
- 產出：`tut/139/build/index.html`（單一 HTML）。上線位置 `tutorials/value-forcing.html`（slug `value-forcing`）；站內相對連結一律以上線位置為準。
- 驗算腳本：`tut/139/verify/common.py`（mulberry32、Box–Muller、expectile、Acklam 反常態 CDF、兩種 W₁，含自我檢查）、`c1.py`、`c2.py`、`c3.py`、`c4.py`、`numbers.py`；輸出 `*_out.txt`。第 5 節所有數字取自這些輸出。`verify/fig/` 存 PDF 第 8、20 頁的渲染圖（Fig. 2、Fig. 6 數值已以目視核對，見 R1）。
- 研究筆記：`tut/139/notes/`（總覽 `value-forcing-lineage-hub` + 5 篇主題 + `timeline.json` 15 筆）；另引用既有筆記 `rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`（No. 568）、`flow-matching-velocity-target`（No. 238）、`expectile-regression-value-learning`、`iql-lineage-overview`（No. 569）、`diffusion-policy-class`（No. 443）。
- 標記慣例：〔本頁推導〕＝論文沒有寫出、本頁自行推導（已以腳本驗算）；「推定」＝為玩具元件選定、論文沒有給出的數值；「本頁解讀」＝對論文未說明處的解讀。頁面用同樣三種 pill。**所有玩具元件的標題列都帶「玩具」pill，說明文字寫明「不是重現論文實驗」。**

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

強化學習的價值函數只能用「一步獎勵 + 自己的估計」當目標（TD bootstrapping），資料少時監督很弱。分布式 RL（distributional RL）改學回報 $Z$ 的整個分佈，近期方法用 flow matching 把 noise level $\tau=0$ 的高斯分佈逐步轉換成 $\tau=1$ 的回報分佈。作者指出先前方法要不是只在路徑終點 $\tau=1$ 施加 Bellman 目標（FlowCritic 等），就是在每個 $\tau$ 套用「乾淨」的 Bellman 方程而在 $\tau=0$ 不成立（Value Flows），或只沿共用起點雜訊的路徑施加（PCBF）。Value Forcing 提出依 $\tau$ 而定的 **flow Bellman 算子**（Eq. 5）：$Z(s,a,\tau)\overset d=\tau r+\gamma Z(S',A',\tau)+(1-\tau)\sqrt{1-\gamma^2}\,N$，多出的高斯修正項讓 $\tau=0$ 兩側都是 $\mathcal N(0,1)$、$\tau=1$ 回到分布式 Bellman 方程（Eq. 1）；對每個 $\tau$ 它都是 $\gamma$-contraction（Remark 1、Prop. 1），因此每個 noise level 都有唯一不動點。再用 Lemma 1 把它轉成 V、Q 兩個速度場互為目標的回歸損失（Eq. 12–13），得到兩個演算法：VF（從 flow 行為複製策略抽 $K$ 個動作取 Q 最大者，Eq. 14）與 IVF（以純量 Q、V 決定的 expectile 權重，Eq. 15–16）。在 4 個 OGBench 狀態的 policy evaluation 中，VF 的平均 1-Wasserstein 距離低於 FlowCritic、Value Flows、PCBF（Fig. 2）；在 20 個 OGBench 任務上，VF 在 10%／25% 資料是 explicit 方法中總平均最高者（48、54；同資料量 IVF 為 43、61），IVF 在所有縮減資料量上領先 implicit 方法（Table 2–4），完整資料時 IVF 65 對最佳 implicit 基線 66（Table 5）。

### 1.2 研究動機（Sec. 1）

1. 價值的監督「有限且間接」：每筆轉移只給一個即時獎勵，沒有真實的期望回報；TD 用自己的估計當目標，可能有偏，資料少時沒有修正的機會（Sec. 1）。
2. 分布式 RL 以整個回報分佈當 TD 目標，可提供較豐富的學習訊號；例：一個動作以相同機率得 −1 與 +1，另一個動作恆得 0，期望相同但分佈不同（Sec. 1）。
3. Flow matching 可以在不同 noise level 學習，提供更多監督的機會；但先前 flow 回報估計器多半只用終點監督，或在 $\tau=0$ 施加不正確的一致性、或只沿同一起點雜訊的路徑施加（Fig. 1 中）。作者「經驗上發現」稀疏或不正確的密集監督會使學到的分佈不準、常集中在接近單一值（Fig. 2）。

### 1.3 貢獻（Sec. 1）

1. Flow Bellman 算子（Eq. 5）：含高斯修正項；每個 noise level 都有唯一不動點，保留初始高斯並在 $\tau=1$ 回到分布式 Bellman 方程（Sec. 4.1，App. A.1）。
2. 由 Lemma 1 推出的實用損失（Eq. 10–13）。
3. 兩個演算法 VF、IVF（Sec. 5，Alg. 1）；理論上 IVF 在 expectile $\mu\to1$ 時趨近資料支撐內的最佳值（App. A.2，前提見步驟 8）。
4. 實驗：26 個長視野任務（20 OGBench + 6 OCBench）的 offline RL（Sec. 6）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S1 末尾以可收合表格提供；每章開頭有「本章符號」小表）

| 符號 | 意義 | 首次定義 |
| --- | --- | --- |
| $s,a,r,s'$；$\gamma\in[0,1)$ | 狀態、動作、確定性有界獎勵 $r(s,a)\in[r_{\min},r_{\max}]$、下一狀態；折扣 | S1 |
| $t$ | RL 時間步（整數） | S1 |
| $\pi$、$\pi_\beta$ | 策略；收集資料的行為策略（behavior policy） | S1 |
| $Z^\pi(s,a)=\sum_t\gamma^tr(S_t,A_t)$ | 回報隨機變數；$Q^\pi=\mathbb E[Z^\pi]$ | S1 |
| $\overset d=$ | 分佈相等（不是逐點相等） | S1 |
| $W_p$、$W_p^{\sup}$ | $p$-Wasserstein 距離；對所有 $(s,a)$ 取上確界 | S1 |
| $\tau\in[0,1]$ | flow 的 noise level；0 = 純高斯雜訊，1 = 乾淨回報（**不是** IQL 的 expectile） | S2 |
| $x^0,x^1,x^\tau$；$u$、$v$、$\phi$ | 雜訊、目標樣本、線性內插；真實速度、參數化速度、flow 路徑 | S2 |
| 耦合（coupling）、邊際（marginal） | 兩個隨機變數的一個聯合分佈且其兩個邊際分佈為指定者；聯合分佈中單一變數的分佈 | S2 |
| $m',\sigma'$；對 1、對 2 | 玩具中下一狀態乾淨回報的平均與標準差（C1、C3）；C2 玩具的兩個 (s,a) 對 | S5、S6 |
| $Z^\pi(s,a,\tau)$ | 第 $\tau$ 層的 noisy return 隨機變數 | S4 |
| $N,\ \varepsilon$ | 與其他一切獨立的 $\mathcal N(0,1)$（$\varepsilon$ 是損失中的取樣值） | S4 |
| $c\triangleq\sqrt{1-\gamma^2}$ | 單步修正係數（〔本頁記號〕，論文直接寫根號） | S4 |
| $G$ | 不動點閉式中的獨立 $\mathcal N(0,1)$（〔本頁記號〕，避免與 $N$ 混淆） | S5 |
| $Z_V,Z_Q$；$\phi_V,\phi_Q$；$u_V,u_Q$；$v_V,v_Q$ | V／Q 回報、其 flow、真實速度、參數化速度 | S6（V 部分 S7） |
| $p_Q(z^\tau|s,a,\tau)$、$p_V(z^\tau|s',\tau)$ | 由 Q／V flow 得到的第 $\tau$ 層 noisy return 分佈 | S6 |
| $K$、$M$ | 候選動作數、估計 Q／V 的回報樣本數 | S8 |
| $\mu\in[0.5,1)$；$w_\mu$；$\pi^\mu$ | expectile；Eq. 15 權重；隱式策略（Eq. 24） | S8、S9 |

#### 步驟 1：問題設定與分布式 Bellman 方程（Sec. 3，Eq. 1）

- 無限期 MDP $\mathcal M=(\mathcal S,\mathcal A,p,r,\rho,\gamma)$，獎勵確定且有界；offline：只有固定資料集 $\mathcal D=\{(s,a,r,s')\}$；目標是最大化 $\mathbb E_\pi[\sum_t\gamma^tr(s_t,a_t)]$。
- Eq. 1：$Z^\pi(s,a)\overset d=\mathcal T^\pi Z^\pi(s,a)=r(s,a)+\gamma Z^\pi(S',A')$。$\mathcal T^\pi$ 在 max-Wasserstein 下是 $\gamma$-contraction，唯一不動點為 $Z^\pi$（引 Bellemare et al. 2017，本文不重證）；取期望回到 Q 的 Bellman 方程。
- 一維 $W_p$ 的分位數公式（Eq. 17）：$W_p(Z_1,Z_2)=\big(\int_0^1|F^{-1}_{Z_1}(u)-F^{-1}_{Z_2}(u)|^pdu\big)^{1/p}$；兩組等權樣本時就是「排序後逐一配對」的平均差（S1 以一行文字說明，C2 使用）。

#### 步驟 2：Flow matching 與 Lemma 1（Sec. 3，Eq. 2–4）

- Eq. 2：$\frac{d}{d\tau}\phi(x^0|\tau)=u(\phi(x^0|\tau)|\tau)$，$\phi(x^0|0)=x^0$。Eq. 3（CFM）：$\mathbb E\|v(x^\tau|\tau)-(x^1-x^0)\|^2$，$x^\tau=(1-\tau)x^0+\tau x^1$，$x^0\sim\mathcal N(0,I)$，$\tau\sim\mathrm{Unif}[0,1]$。
- Lemma 1（非正式，Liu et al. 2022 Sec. 2.3、3.3）：若過程 $X(\tau)$ 在給定隨機變數集合 $\mathcal X$ 後是 $\tau$ 的確定函數，則 $u(x|\tau)\triangleq\mathbb E_{\mathcal X}[\frac{d}{d\tau}X(\tau)\mid X(\tau)=x]$ 生成 $X(\tau)$ 的邊際密度路徑；並可用 Eq. 4 $\mathbb E\|v(x^\tau|\tau)-\frac{d}{d\tau}X(\tau)\|^2$ 擬合。取 $\mathcal X=(X^0,X^1)$、$X(\tau)=(1-\tau)X^0+\tau X^1$ 即回到 Eq. 3。
- 這是全文唯一的工具：之後每個損失都是「定義過程 → 對 $\tau$ 微分 → 套 Lemma 1」（顧問 §1 步驟 2）。
- 頁面補充（引 D.5 對 Liu et al. 2022 Lemma 3.9 與 Bansal et al. 2026 Lemma 7 的轉述）：一維 flow 路徑不交叉，因此保分位數；C3、C4 的高斯玩具用到這點。

#### 步驟 3：先前方法在哪裡施加一致性（Sec. 2，App. D.2，Table 1）

| 方法 | 論文的描述 | Table 1：初始高斯／中間層 Bellman／終點 Bellman |
| --- | --- | --- |
| FlowCritic（DFC、Dual-Flow RL 同為終點類；FlowIQN 另有分位數耦合，標記見下） | TD 目標 $y=r+\gamma z^1$，$z^1$ 取自下一步的乾淨回報樣本，再以 Eq. 3 擬合（Eq. 25）；不在中間層施加一致性 | ✓／✗／✓ |
| Value Flows | 速度場直接當 TD 目標，在所有 noise level「天真地」套用乾淨 Bellman 方程；在 $\tau=0$ 給出不正確的監督 | ✗†／✓／✓ |
| PCBF | 兩條路徑共用起點雜訊 $z^0$，沿耦合路徑套 Bellman；邊際路徑因耦合而缺乏 Bellman 一致性 | ✓／✗§／✓ |
| VF／IVF | 每個 noise level 都對齊 | ✓／✓／✓ |

- Table 1 另列 Floq（純量終點，✗／✗／✓）、EVOR（✗／✗‡／✓）、FlowIQN（✗／✗／✓）；FlowIQN、EVOR 不在實驗中。
- 頁面註記（本頁推導，顧問 3.1-11）：在頁面的模型中（把乾淨方程當成對邊際分佈的算子，見步驟 5 的但書），乾淨方程在所有 $\tau<1$ 都與正確的邊際不一致，越靠近 $\tau=1$ 差越小；論文強調的是 $\tau=0$。C1 的第二張圖顯示這點。PCBF 的恆等式（D.2；頁面放 `<details>`，先定義 $z'$ 為下一步以同一 $z^0$ 生成的乾淨回報、$\tilde z^\tau=(1-\tau)z^0+\tau z'$）：$(1-\tau)z^0+\tau(r+\gamma z')=\tau r+\gamma\tilde z^\tau+(1-\tau)(1-\gamma)z^0$——用同一個 $z^0$，係數 $1-\gamma$；VF 用獨立的 $N$，係數 $\sqrt{1-\gamma^2}$。兩者在 $\tau=0$ 都保持單位高斯（PCBF：$\gamma z^0+(1-\gamma)z^0=z^0$）。

#### 步驟 4：Flow Bellman 方程（Sec. 4.1，Eq. 5）

- Eq. 5：$Z^\pi(s,a,\tau)\overset d=\mathcal T^\pi_\tau Z^\pi(s,a,\tau)=\tau r(s,a)+\gamma Z^\pi(S',A',\tau)+(1-\tau)\sqrt{1-\gamma^2}\,N$，$N\sim\mathcal N(0,1)$ 與初始雜訊分佈獨立；$Z^\pi(s,a,1)\overset d=Z^\pi(s,a)$。
- 兩端（論文）：$\tau=0$ 時 $Z^\pi(s,a,0)\overset d=\gamma Z^\pi(S',A',0)+\sqrt{1-\gamma^2}N$，兩側都是標準高斯（初始雜訊），修正項不可少；省略它（Value Flows）得到錯誤的方程。$\tau=1$ 回到 Eq. 1。論文對修正項的解釋：「accounts for the variance in the discounted return at the next time step」。
- 〔本頁推導〕變異數：$\gamma^2\cdot1+(1-\gamma^2)=1$。由內插反推（顧問推導 A(i)）：令 $Z(\tau)=(1-\tau)Z^0+\tau Z^1$、$Z^1=r+\gamma Z^{1\prime}$、$Z^0\overset d=\gamma Z^{0\prime}+cN$，代入得 $Z(\tau)=\tau r+\gamma Z'(\tau)+(1-\tau)cN$，即 Eq. 5。
- 量級（〔本頁推導〕，`c1_out.txt`）：$c=0.4359$（γ = 0.9）、$0.0447$（0.999，OGBench）、$0.0141$（0.9999，OCBench）。單步修正很小；有效視界約 $1/(1-\gamma)$ 步。頁面不說「修正項很大」。
- 腳註 1：換成 $\mathrm{Unif}[0,1]$ 初始雜訊會有「不同的修正項」，論文沒給式子；頁面只照引，不展開（顧問 3.1-12）。

#### 步驟 5：每個 τ 的唯一不動點（Remark 1，Eq. 6，Prop. 1，App. A.1）

- Remark 1／Prop. 1：給定 $\pi$、$\gamma\in[0,1)$、$\tau\in[0,1]$、$p\in[1,\infty)$，$W_p^{\sup}(\mathcal T^\pi_\tau Z_1,\mathcal T^\pi_\tau Z_2)\le\gamma W_p^{\sup}(Z_1,Z_2)$（Eq. 6），$W_p^{\sup}(Z_1,Z_2)\triangleq\sup_{s,a}W_p(Z_1(s,a,\tau),Z_2(s,a,\tau))$。由 Banach 不動點定理得唯一不動點；反覆套用 $\mathcal T^\pi_\tau$ 會收斂到它。
- 證明骨架（App. A.1，頁面以 `<details>` 四步呈現）：(1) 用同一個 $U\sim\mathrm{Unif}[0,1]$ 經兩個分位數函數造出 $C_i=F^{-1}_{Z_i}(U|S',A',\tau)$，$C_i\overset d=Z_i$；(2) 這個耦合給出 Eq. 17；(3) $T_i=\tau r+\gamma C_i+(1-\tau)cN$ 共用 $(S',A',U,N)$，是 $\mathcal T_\tau Z_1,\mathcal T_\tau Z_2$ 的耦合；(4) $W_p\le(\mathbb E|T_1-T_2|^p)^{1/p}=\gamma(\mathbb E|C_1-C_2|^p)^{1/p}\le\gamma W_p^{\sup}$——相同的 $\tau r$ 與相同的 $N$ 相消，所以係數是 $\gamma$、與 $\tau$ 無關。
- 前提（Prop. 1 與顧問推導 B、3.1-8）：固定 $\pi$（policy evaluation 算子）、固定 τ、$\gamma<1$、$p\in[1,\infty)$、$Z_1,Z_2\in\mathcal Z$（一致有界動差，Sec. 3）；回報一維才有分位數公式；Banach 定理需要 $(\mathcal Z,W_p^{\sup})$ 完備，論文未驗證；對神經網路訓練沒有收斂保證。
- 〔本頁推導〕不動點閉式（顧問推導 A(ii)）：沿軌跡展開 Eq. 5，$Z^\pi(s,a,\tau)\overset d=\tau\sum_t\gamma^tr_t+(1-\tau)c\sum_t\gamma^tN_t$；$\sum_t\gamma^tN_t$ 是變異數 $1/(1-\gamma^2)$ 的高斯，所以 $Z^\pi(s,a,\tau)\overset d=\tau Z^\pi(s,a)+(1-\tau)G$，$G\sim\mathcal N(0,1)$ 與回報獨立——正是 Eq. 3「獨立耦合、線性內插」的邊際路徑。條件：每步的 $N$ 與未來全部獨立、$\gamma<1$。推論：平均 $\tau Q^\pi$，變異數 $\tau^2\mathrm{Var}Z^\pi+(1-\tau)^2$。
- 〔本頁推導〕「乾淨算子（本頁模型）」：若在每個 $\tau$ 都把 $r+\gamma Z'(\tau)$ 當成對邊際分佈的算子，它就是 $\mathcal T^\pi$，在每個 $\tau$ 的不動點都是 $Z^\pi$ 本身，與「$\tau=0$ 必須是 $\mathcal N(0,1)$」矛盾。兩個算子都是 $\gamma$-contraction，差別是收斂到哪裡；兩者在 $\tau<1$ 不同、越靠近 $\tau=1$ 越接近、$\tau=1$ 重合。C1、C2 示範。**但書（頁面正文必寫）**：這是頁面的模型，不是 Value Flows 實際學到的分佈——Value Flows 的 DCFM 目標是 $\bar v_Q(z^\tau|s',a',\tau)$、沒有 γ 係數（D.2；顧問 3.2-6），而且任何 flow 在 τ = 0 的邊際依構造就是輸入高斯。論文對 Value Flows 的說法只到「τ = 0 的監督不一致，offering an explanation」（Sec. 6.1）。

#### 步驟 6：拆成 V 與 Q，套 Lemma 1（Sec. 4.2，Eq. 7–11）

- Eq. 7：$Z^\pi_V(s,\tau)\overset d=Z^\pi_Q(s,A,\tau)$，$Z^\pi_Q(s,a,\tau)\overset d=\tau r+\gamma Z^\pi_V(S',\tau)+(1-\tau)cN$。Eq. 8–9：用兩條 flow 把**獨立**的 $N_V,N_Q$ 轉換成這兩個分佈：$\phi^\pi_V(N_V|s,\tau)\overset d=\phi^\pi_Q(N_Q|s,A,\tau)$；$\phi^\pi_Q(N_Q|s,a,\tau)\overset d=\tau r+\gamma\phi^\pi_V(N_V|S',\tau)+(1-\tau)cN$。等號都是分佈相等，不是同一顆粒子的軌跡（顧問 3.1-10；那是 PCBF 的做法）。
- V：$\mathcal X_V=\{N_Q,A\}$，$X_V(s,\tau)\triangleq\phi_Q(N_Q|s,A,\tau)$，$\frac{d}{d\tau}X_V=u_Q(\phi_Q(\cdot)|s,A,\tau)$ → Eq. 10：$\mathbb E_{a\sim\pi,\tau,z^0,z^\tau=\phi_Q(z^0|s,a,\tau)}[(v_V(z^\tau|s,\tau)-u_Q(z^\tau|s,a,\tau))^2]$。同一個 $z^\tau$、沒有獎勵、沒有 $\varepsilon$。〔本頁推導〕最小解是以 $z$ 為條件的後驗加權平均 $\sum_a\pi(a|s)p_Q(z|s,a,\tau)u_Q(z|s,a,\tau)/\sum_a\pi(a|s)p_Q(z|s,a,\tau)$，不是對 $\pi$ 的單純平均。
- Q：$\mathcal X_Q=\{N_V,N,S'\}$，$X_Q\triangleq\tau r+\gamma\phi_V(N_V|S',\tau)+(1-\tau)cN$，$\frac{d}{d\tau}X_Q=r+\gamma u_V(\phi_V(N_V|S',\tau)|S',\tau)-cN$ → Eq. 11：$\mathbb E\big[\big(v_Q(\tau r+\gamma z^\tau+(1-\tau)c\epsilon\mid s,a,\tau)-(r+\gamma u_V(z^\tau|s',\tau)-c\epsilon)\big)^2\big]$，$z^0,\epsilon\overset{iid}\sim\mathcal N(0,1)$，$z^\tau=\phi_V(z^0|s',\tau)$。
- 抄寫時易犯的三個錯誤（顧問推導 C；本頁假設，不是論文或實作中觀察到的錯誤；頁面放在 C3 之後的框中）：(1) 輸入點是 $+(1-\tau)c\epsilon$、目標是 $-c\epsilon$，同一個 $\epsilon$；(2) 輸入點乘 $\tau$ 的是 $r$，目標裡的 $r$ 不乘 $\tau$；(3) $u_V$ 在未變換的 $z^\tau$、下一狀態 $s'$ 求值，$v_Q$ 在變換後的點、目前 $(s,a)$ 求值。C3 把 (1)(2) 做成可切換的「抄寫錯誤」模式。

#### 步驟 7：實用損失（Sec. 4.2 末，Eq. 12–13）

- 真實速度未知，改用學到的 $v_V,v_Q$ 互為目標（類比 Fitted Q-Evaluation），交替最小化、更新一方時固定另一方；獎勵乘 $(1-\gamma)$ 以正規化回報尺度（Sec. 6.3 顯示有幫助）。
- Eq. 12：$\mathcal L(v_V)=\mathbb E_{a\sim\pi(a|s),\tau,z^\tau\sim p_Q(z^\tau|s,a,\tau)}[(v_V(z^\tau|s,\tau)-v_Q(z^\tau|s,a,\tau))^2]$。
- Eq. 13：$\mathcal L(v_Q)=\mathbb E_{s',\tau,\epsilon,z^\tau\sim p_V(z^\tau|s',\tau)}\big[\big(v_Q(\tau(1-\gamma)r+\gamma z^\tau+(1-\tau)c\epsilon\mid s,a,\tau)-((1-\gamma)r+\gamma v_V(z^\tau|s',\tau)-c\epsilon)\big)^2\big]$。
- 注意：取樣 $z^\tau\sim p_Q$ 需要把 flow 從 0 積分到 $\tau$，所以 VF 不是 simulation-free（顧問推導 D）；論文沒有說取樣用幾步 Euler（顧問 3.3）。這一步沒有收斂證明（Sec. 7 自己承認 IVF 分析的前提）。

#### 步驟 8：VF 與 IVF（Sec. 5，Eq. 14–16，Alg. 1，App. A.2，D.5）

- 純量估計：$Q(s,a)\triangleq\mathbb E_{z^0}[\int_0^1v_Q(z^\tau|s,a,\tau)d\tau]$、$V(s)$ 同理，以 $M$ 個樣本平均。
- VF（explicit）：Eq. 14 $\pi(a|s)\overset d=\arg\max_{a_1..a_K:\,a_i\sim\pi_\beta}Q(s,a_i)$（rejection sampling，引 Chen et al. 2023），$\pi_\beta$ 以 Eq. 3 學成 flow BC 策略；$\pi$ 也用來抽 Eq. 12 的動作。
- IVF（implicit）：Eq. 15 $w_\mu(s,a)\triangleq\mu\mathbb 1(Q(s,a)\ge V(s))+(1-\mu)\mathbb 1(Q(s,a)<V(s))$；Eq. 16 $\mathcal L_\mu(v_V)=\mathbb E_{a\sim\mathcal D,\tau,z^\tau\sim p_Q}[w_\mu(s,a)(v_V(z^\tau|s,\tau)-v_Q(z^\tau|s,a,\tau))^2]$。只用資料中的動作。權重由**純量** Q、V 決定，再乘在速度殘差平方上。
- Alg. 1（頁面逐行對照）：初始化 $v^\theta_V,v^\psi_Q,v^\xi_\beta$ 與目標參數 $\bar\psi\leftarrow\psi$；每次迭代抽 $(s,a,r,s')$、$\tau$；VF：$a^\pi$ 由 Eq. 14、$z^\tau\sim p^{\bar\psi}_Q(\cdot|s,a^\pi,\tau)$、以 Eq. 12 更新 θ；IVF：算 $w_\mu$、$z^\tau\sim p^{\bar\psi}_Q(\cdot|s,a,\tau)$、以 Eq. 16 更新 θ；$z^\tau\sim p^\theta_V(\cdot|s',\tau)$、$\epsilon$，以 Eq. 13 更新 ψ；以 Eq. 3 更新 ξ；$\bar\psi$ 取 ψ 的 EMA。V 沒有目標網路（顧問 3.3）。
- App. A.2（假設：類 Fitted-Q 迭代、每輪精確最小化、$k\to\infty$ 收斂到自洽不動點；分析時獎勵不乘 $1-\gamma$）：隱式策略 Eq. 24 $\pi^\mu_{k-1}(a|s)=\pi_\beta(a|s)w^\mu_{k-1}(s,a)/\mathbb E_{\pi_\beta}[w^\mu_{k-1}]$（$w\in[1-\mu,\mu]$ 故非負）；不動點滿足 $\pi^\mu_\infty$ 的 flow Bellman 方程；取期望得 $V^\infty(s)=\mathbb E_{a\sim\pi^\mu_\infty}[Q^\infty(s,a)]$、$Q^\infty=r+\gamma\mathbb E[V^\infty(s')]$，再引 Hansen-Estruch et al. 2023 Theorem 4.1 得 expectile Bellman 方程；引 IQL Theorem 3：$\lim_{\mu\to1}V^\infty_\mu(s)=\max_{a:\pi_\beta(a|s)>0}Q^\star_\beta(s,a)$。
- 本頁解讀（顧問 3.1-8、推導 E）：A.2 另外隱含兩個前提——取期望時用了 $\mathbb E[\int_0^1v\,d\tau]=\mathbb E[\phi(z^0|1)]$，需要 $\mathbb E[z^0]=0$；「唯一不動點」只在給定 $\pi^\mu_\infty$ 時成立，而 $\pi^\mu_\infty$ 本身依賴解，自洽不動點的存在與唯一性沒有證明。
- 〔本頁推導〕為什麼是 expectile：$V=\mathbb E_\beta[wQ]/\mathbb E_\beta[w]\iff\mathbb E_\beta[w_\mu(Q-V)]=0$，正是 expectile 回歸的一階條件。C4(a) 示範。
- D.5：若直接對速度殘差做 expectile，$\mu\to1$ 時得逐點的 $\max_av_Q$，積分後 $\mathbb E_{z^0}[\int\max_a v_Q\,d\tau]\neq\max_a\mathbb E_{z^0}[\int v_Q\,d\tau]=\max_aQ(s,a)$。〔本頁推導〕在一維、各動作共用同一個 $z^0$ 時，max 速度場的路徑不低於每個動作自己的路徑（ODE 比較定理），所以左邊 ≥ 右邊；論文只寫「≠」。C4(b) 量出差距。
- 頁面註記（顧問 3.2-1）：A.2 p.18 的 Q 最小解式子把 $v_V$ 的引數與取樣條件寫錯（應為 $z^\tau$ 與 $s'$）；頁面以 Eq. 11、21 為準並以小字註明。

#### 步驟 9：實作設定（App. D.2–D.3，Table 6–7）

- Adam、lr $3\times10^{-4}$、500K 步、batch 1024；OGBench：MLP [1024]×4；OCBench：residual MLP（輸入層 + 4 個寬 1024 的兩層殘差塊）；GELU、LayerNorm；noise-level 16 維正弦嵌入；$\tau\sim\mathrm{Unif}[0,1]$；目標網路更新率 0.005；$\gamma$ = 0.999（OGBench）／0.9999（OCBench）；action chunk $H=5$；$K=16$；BC flow 積分 10 步（VF 訓練時 4）；return flow 積分 10 步（Table 6 的預設；VF 訓練時用一步估計，評估時 10 步，見下）。
- VF：V、Q 速度各為兩成員 ensemble；訓練時 $M=1$、一步估計 $Q(s,a)\approx z^0+v_Q(z^0|s,a,0)$（D.2，論文稱 Tweedie's formula）；評估時 $M=16$、10 步積分、候選共用起始雜訊；VF 不調參。IVF：一個網路同時當 V、Q 速度（仿 classifier-free guidance），兩成員；$\mu$ 調自 {0.7, 0.8, 0.9}，$M=16$。
- Table 7：$\mu$ = 0.9（cube-triple）、0.8（cube-quadruple）、0.9（puzzle-4x4）、0.8（humanoidmaze-giant）、0.9（OCBench）；$[r_{\min},r_{\max}]$ = [−3,0]、[−4,0]、[−16,0]、[−1,0]、[−1,0]。
- 本頁解讀：VF 訓練時用一步估計、評估時用 10 步積分與 M = 16，論文沒有說明兩者差異的影響；頁面標「本頁解讀」pill 呈現這段。
- 〔本頁推導〕一步估計為何合理：獨立耦合線性內插下 $u(z^0|0)=\mathbb E[Z^1]-z^0$，所以 $z^0+u(z^0|0)=Q$ 對任何 $z^0$ 成立（C3 在 τ = 0 顯示斜率 −1、截距 = Q 的平均）；只在速度學準時成立。
- 調參：每個 domain 用 task 2、4 與完整資料調，再套用到 5 個任務與所有資料量（D.3）。未說明處（顧問 3.3、3.2-3）：Table 6 的「Return-flow integration steps 10」與 D.3 的 VF 訓練一步如何對應（本頁解讀：10 步是評估與其他方法的設定）、Table 6 沒有 M 這一列、softmax 抽樣溫度、取樣 $z^\tau$ 的步數、twin 成員如何合併、IVF 的動作遮蔽方式、chunk 內 $\gamma$ 的處理、baseline 是否也做 reward scaling。頁面列成「論文未說明」清單。

#### 步驟 10：實驗（Sec. 6，Fig. 2–8，Table 2–5，App. B–D）

- **回報估計（Sec. 6.1，D.4）**：每個 OGBench domain 一個狀態、固定起始 action chunk，收集 1000 條訓練與 1000 條評估軌跡；各方法以資料中的動作 bootstrap（SARSA）訓練 100K 步；抽 5000 個回報樣本，對 1000 個評估回報算 1-Wasserstein。Oracle 用 Eq. 3 直接擬合同批軌跡的實際回報，代表 bootstrapping 以外的誤差。Fig. 2（Oracle／VF／FlowCritic／Value Flows／PCBF）：cube-triple 92±13／115±16／212±14／295±16／204±15；cube-quadruple 57±6／179±6／203±9／289±11／184±9；puzzle-4x4 20±5／22±5／125±6／134±7／118±6；humanoidmaze-giant 25±3／13±2／127±3／462±3／122±3。正文：相對 PCBF 下降 81%（puzzle-4x4）、89%（humanoidmaze-giant）；cube-triple 與 puzzle-4x4 接近 Oracle。Fig. 6（Oracle／IVF／Implicit C51／Implicit IQN／DIVL）：92／116／299／251／287；57／174／497／233／290；20／18／279／141／128；25／13／205／88／255（± 見第 5 節）。Fig. 7：VF 在 τ ≈ 0.25 就展開路徑，基線在 τ ≈ 0.9 才集中（圖說）。
- 讀圖注意（頁面寫成正文）：只有 4 個狀態；「接近 Oracle」只對 cube-triple、puzzle-4x4 成立；cube-quadruple VF 179 對 Oracle 57、與 PCBF 184 接近；humanoidmaze-giant VF 13 低於 Oracle 25，Oracle 不是下界；圖軸（如 −800…−400）是未縮放的回報單位，論文沒說明與訓練時 $(1-\gamma)$ 縮放的換算（本頁解讀）。論文對 Value Flows 的改善用語是 “offering an explanation”，對其他是 “suggest”；沒有任何消融單獨拿掉高斯修正項。
- **成功率（Sec. 6.2，Table 2–5）**：OGBench 4 domains × 5 tasks，完整資料 100M transitions，子集 10%／25%／50%（以每條軌跡平均獎勵為分數、softmax 不放回抽樣，溫度未說明）；OCBench 6 tasks、25%／100%。4 seeds；OGBench 每次評估 10 episodes／task，OCBench 500。總平均（`numbers_out.txt`）：10%：VF 48 對最佳 explicit 基線 QC 43；IVF 43 對最佳 implicit 基線 Implicit IQN 27。25%：VF 54 對 IQN 48；IVF 61 對 Implicit IQN 46。50%：VF 53 對 C51 51；IVF 66 對 Implicit C51 58。100%：VF 57 對 C51、QC 52；IVF 65 對 Implicit C51 66。OCBench 25%：VF 67±2 對 Value Flows 60±3；IVF 58±2 對 Implicit QC 57±3；100%：VF 92±1 對 C51 91±2；IVF 91±1 對 Implicit C51 86±2。
- 差距序列（`numbers_out.txt`）：VF 對最佳 explicit 基線 +5、+6、+2、+5（10%→100%），不隨資料量單調變化；IVF 對最佳 implicit 基線 +16、+15、+8、−1，隨資料變少而擴大。
- 讀表注意：humanoidmaze-giant 上所有 explicit 方法接近 0（VF domain 平均 0／0／0／1），只有 implicit 方法有成績（IVF 40／50／49／55）；puzzle-4x4 100%：IVF 82±11，Implicit C51 100±0。「limited data」的 10% 仍約 10M transitions。正文 “VF barely performs worse as the dataset size decreases” 對應總平均 57→53→54→48。
- **消融（Sec. 6.3，B.2，D.5；cube-triple 與 puzzle-4x4 的 task 2、4，最後三次評估平均）**：只學 $v_Q$（不用 V 速度）與 VF 相近；one-step actor 使成功率「reduces … by 44%」（原句，沒說是相對或百分點）；IVF 以純量優勢決定權重遠優於對速度殘差做 expectile 或用分位數權重；拿掉 reward scaling：VF 約 86%→11%、IVF 76%→11%；flow 步數（1–20）與 $M$ 沒有顯著影響，1 步也相近（Fig. 8）。Fig. 5 的「約 86%」與 Table 5 同 4 任務的最終評估（平均 93.5，IVF 77）協定不同，不混引。

#### 步驟 11：限制（Sec. 7 + 顧問 §3）

- 作者：IVF 分析假設精確最佳化與收斂到自洽不動點，神經網路下的收斂保證是開放問題；延伸到 online 與真實機器人是未來工作。
- 本頁整理：(1) 沒有拿掉高斯修正項的消融；(2) 回報估計只在 4 個狀態；(3) contraction 是固定 π 的理想算子；(4) 子集偏向高獎勵軌跡；(5) 單任務數字粒度粗（10 episodes × 4 seeds）。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details>` 呈現該章檢查題（題目與參考答案照搬）。**每一條參考答案都必須寫在指定章節的正文段落中**（不是只在元件、表格、圖或答案框裡）。玩具元件的數字在正文與參考答案中只寫定性結論或兩位有效數字；精確值只出現在讀數框與 selfTest。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明分布式 Bellman 方程與「分佈相等」 | Eq. 1 的 $\overset d=$ 是什麼意思？為什麼 $\mathcal T^\pi$ 有唯一不動點？ | 左右兩邊的隨機變數分佈相同，不是逐點相等；$\mathcal T^\pi$ 在 max-Wasserstein 距離下是 γ-contraction（Bellemare et al. 2017），所以唯一不動點是 $Z^\pi$ | S1 |
| LO2 | 寫出 CFM 與 Lemma 1 | Eq. 3 的回歸輸入與目標？Lemma 1 說了什麼？ | 輸入 $x^\tau=(1-\tau)x^0+\tau x^1$，目標 $x^1-x^0$；Lemma 1：若過程 $X(\tau)$ 給定 $\mathcal X$ 後是 τ 的確定函數，$u(x|\tau)=\mathbb E[\frac{d}{d\tau}X(\tau)\mid X(\tau)=x]$ 生成其邊際密度，可用 Eq. 4 對 $\frac{d}{d\tau}X$ 回歸來擬合 | S2 |
| LO3 | 分辨先前方法施加一致性的位置 | FlowCritic、Value Flows、PCBF 各在哪裡施加 Bellman 一致性？論文指出的問題？ | FlowCritic 只在終點（目標 $r+\gamma z^1$）；Value Flows 在所有 noise level 套乾淨 Bellman 方程，論文稱其在 τ = 0 不正確；PCBF 沿共用起點雜訊的兩條路徑施加，邊際路徑缺乏 Bellman 一致性 | S3 |
| LO4 | 寫出 Eq. 5 並解釋修正項 | 寫出 flow Bellman 方程；τ = 0 時為何需要修正項？γ = 0.999 時單步係數多大？ | $Z(s,a,\tau)\overset d=\tau r+\gamma Z(S',A',\tau)+(1-\tau)\sqrt{1-\gamma^2}N$；τ = 0 兩側都是 $\mathcal N(0,1)$，$\gamma N'$ 的變異數只有 $\gamma^2$，加上 $\sqrt{1-\gamma^2}N$ 才回到 1；$\sqrt{1-\gamma^2}=0.0447$ | S4 |
| LO5 | 說明 contraction 的係數、證明關鍵與前提 | Remark 1 的係數是多少？證明中為什麼與 τ 無關？結論的前提？ | 係數 γ；用同一個均勻變數的分位數耦合與同一個 $N$，$\tau r$ 與高斯項相消，只剩 $\gamma(C_1-C_2)$；前提：固定策略 π、固定 τ、γ < 1、$p\in[1,\infty)$、回報屬於一致有界動差的集合 $\mathcal Z$、一維回報；再由 Banach 定理得每個 τ 唯一不動點（需完備性，論文未驗證） | S5 |
| LO6 | 說出不動點的閉式並對照乾淨算子（本頁模型） | 修正後算子在 τ 的不動點是什麼分佈？若在每層用乾淨方程（本頁模型）呢？C2 預設（γ = 0.9、τ = 0）看到什麼？ | 〔本頁推導〕$\tau Z^\pi+(1-\tau)G$（$G$ 獨立標準高斯），平均 $\tau Q^\pi$、變異數 $\tau^2\mathrm{Var}Z^\pi+(1-\tau)^2$；乾淨算子（本頁模型）在每個 τ 都收斂到 $Z^\pi$ 本身，這不是 Value Flows 實際學到的分佈；C2：修正版收斂到平均約 0、標準差約 1，乾淨版收斂到平均約 −3.7、標準差約 1.3（對 1），兩者每步距離比值都不超過 γ = 0.9 | S5 |
| LO7 | 正確寫出 Q 損失的輸入點與目標 | Eq. 11 的回歸輸入點與目標？列出兩個抄寫時易犯的錯誤。 | 輸入 $\tau r+\gamma z^\tau+(1-\tau)\sqrt{1-\gamma^2}\epsilon$（在 $(s,a)$ 求值），目標 $r+\gamma u_V(z^\tau|s',\tau)-\sqrt{1-\gamma^2}\epsilon$；易犯錯誤（本頁假設）：輸入 +、目標 −（同一個 ε）；輸入的 r 乘 τ、目標的 r 不乘；C3 中正確版的條件平均恰好等於 $u_Q$，兩種錯誤版都不等 | S6 |
| LO8 | 說明 V 損失與實用化步驟 | Eq. 12 讓什麼回歸到什麼？真實速度未知時怎麼辦？實作多了哪個縮放？ | 在同一個 $z^\tau\sim p_Q$ 上讓 $v_V(z^\tau|s,\tau)$ 回歸 $v_Q(z^\tau|s,a,\tau)$（$a\sim\pi$，沒有獎勵與 ε）；以 $v_V$、$v_Q$ 互為目標交替最小化（類比 Fitted Q-Evaluation）；獎勵乘 $(1-\gamma)$ | S7 |
| LO9 | 比較 VF 與 IVF，說明 IVF 的理論 | VF 如何選動作？IVF 的權重由什麼決定？A.2 的結論與假設？為何不對速度殘差做 expectile？ | VF：從 flow BC 策略抽 K 個動作取 Q 最大者（Eq. 14）；IVF：$w_\mu$ 由純量 Q、V 決定（Eq. 15），只用資料動作；A.2：在精確最佳化與收斂到自洽不動點的假設下，V 是 Q 的 expectile，μ → 1 時趨近資料支撐內的最佳值；逐點 expectile 在 μ → 1 給出 $\max_a v_Q$，積分後不等於 $\max_aQ$（論文只寫 ≠；C4 的玩具中一維共用起點時偏高，〔本頁推導〕） | S8（VF、IVF）、S9（A.2、D.5） |
| LO10 | 列出實作設定 | γ、K、H、VF 訓練與評估時的 M 與積分步數？一步估計式？ | γ = 0.999（OGBench）／0.9999（OCBench）；K = 16；H = 5；VF 訓練 M = 1、一步估計 $Q\approx z^0+v_Q(z^0|s,a,0)$，評估 M = 16、10 步、候選共用起始雜訊；IVF 固定 M = 16 | S10 |
| LO11 | 正確讀出回報估計實驗 | Fig. 2 中 VF 與 PCBF、Oracle 的比較？這個實驗能說明什麼、不能說明什麼？ | VF 在 4 個狀態都低於三個先前方法；相對 PCBF 下降 81%（puzzle-4x4）與 89%（humanoidmaze-giant）；只在 cube-triple（115 對 92）與 puzzle-4x4（22 對 20）接近 Oracle；只有 4 個狀態、沒有單獨拿掉高斯修正項的消融，不能證明修正項是改善的原因 | S11 |
| LO12 | 正確讀出成功率與消融 | 10% 與 100% 資料下 VF、IVF 與同類最佳基線的總平均？拿掉 reward scaling 後成功率變成多少？Fig. 4 的 one-step actor 結果怎麼引用？ | 10%：VF 48 對 43（QC），IVF 43 對 27（Implicit IQN）；100%：VF 57 對 52，IVF 65 對 66（Implicit C51）；VF 的差距不隨資料量單調變化，IVF 的差距隨資料變少而擴大；拿掉 reward scaling：VF 約 86% → 11%、IVF 76% → 11%；one-step actor 只能照原句「reduces success rates by 44%」引用，論文沒說明是相對或百分點 | S12 |

---

## 3. 頁面大綱

外殼與 No. 569 頁相同：單欄閱讀（680–720px），圖版加寬（最大 1080px），桌機左側 sticky 章節目錄，章節編號 JetBrains Mono（`01 · …`），章節標題用陳述句。視覺記號：一條水平 τ 軸（左端「τ = 0：N(0,1) 雜訊」，右端「τ = 1：回報分佈」），在 hero、S2、S4、S5 重複出現；S3 在軸上標各方法施加一致性的位置。< 600px 時兩端標籤移到軸下方、各佔一行，避免與刻度重疊。

| # | id | 標題 | 內容（依序；每個符號在使用前定義） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Value Forcing：在每個 noise level 對齊回報分佈〉 | 副標（先定義再用）：「回報是之後所有獎勵的折扣和；flow matching 是一種生成模型，它學一個速度場，把 τ = 0 的高斯分佈逐步轉換成 τ = 1 的資料分佈，τ 稱為 noise level。Value Forcing 用它表示回報分佈，並在每個 τ 都要求 Bellman 一致性，方法是在 Bellman 方程中加一項高斯修正。論文據此提出兩個演算法：VF（explicit：另外有策略，從候選動作中挑 Q 最大者）與 IVF（implicit：不學策略，只用資料動作加權學 V）。」τ 軸。三張事實卡（下）。連結：arXiv、`../index.html#paper-139`、總覽 `value-forcing-lineage-hub.html`、程式碼 | 1 螢幕 |
| S1 | `distrl` | 01 · 分布式 RL 把整個回報分佈當成 TD 目標 | MDP、offline、$t$、$Z^\pi$、$Q^\pi$、TD 一句話、±1 對 0 例子、$\overset d=$、Eq. 1、$W_p$ 與一維分位數公式（排序配對）、contraction 與 Banach 一句話；LO1。末尾符號表。延伸：`rl-mdp-q-learning-basics.html`、`offline-rl-distribution-shift.html`、`distributional-rl-return-distributions.html` | 550–650 字 |
| S2 | `flow` | 02 · Flow matching 用速度場把高斯分佈逐步轉換成回報分佈 | τ 的定義（另寫一句：「本頁的 expectile 參數記為 μ」）；Eq. 2；Eq. 3；**耦合**（例：Eq. 3 讓 $x^0$ 與 $x^1$ 獨立抽樣，這是一種耦合；PCBF 讓兩條路徑用同一個 $x^0$，是另一種）與**邊際**（例：$x^\tau$ 單獨的分佈 $p(x^\tau|\tau)$）各一句；Lemma 1、Eq. 4；條件期望＝邊際速度；一維路徑不交叉；LO2。延伸：`flow-matching-velocity-target.html` | 500–600 字 |
| S3 | `prior` | 03 · 先前方法只在部分 noise level 施加正確的一致性 | 兩行說明：τ = 0 時 $Z(s,a,0)$ 與 $Z(S',A',0)$ 都是標準高斯，但 $\gamma Z(S',A',0)$ 的變異數是 $\gamma^2<1$，所以 $r+\gamma Z'$ 在 τ = 0 不可能等於標準高斯；靜態圖 F1；Table 1 摘錄（1.4 步驟 3）；PCBF 恆等式（`<details>`，含 $\tilde z^\tau$ 定義）；LO3。延伸：`flow-based-return-critics.html`、`flow-bellman-operator-all-noise-levels.html` | 450–550 字 + F1 + 表 |
| S4 | `fbe` | 04 · Flow Bellman 方程多一項高斯修正 | Eq. 5；兩端檢查；變異數 $\gamma^2+(1-\gamma^2)=1$；由內插反推〔本頁推導〕（寫在正文，三行）；量級表（γ = 0.9／0.999／0.9999）；結尾一句：「下一章會看到：加不加修正，兩個算子的不動點在 τ < 1 不同，越靠近 τ = 1 差越小，τ = 1 重合」；LO4 | 400–500 字 |
| S5 | `fixed` | 05 · 每個 noise level 都有唯一不動點 | Remark 1、Eq. 6；證明四步（`<details>`）；前提清單；不動點閉式〔本頁推導〕；乾淨算子（本頁模型）的不動點與但書（1.4 步驟 5）；**C1**（一次 backup 的差別，現在可以引用閉式）；C2 前先定義玩具（兩個 (s,a) 對）與「粒子表示」；**C2**；C1、C2 圖說：「預設 γ = 0.9 是為了看得見；論文 γ = 0.999 時單步係數只有 0.0447」；LO5、LO6 | 700–800 字 + C1 + C2 |
| S6 | `qloss` | 06 · Q 損失：把 flow Bellman 方程變成回歸 | Eq. 7–9（強調分佈相等、$N_V,N_Q$ 獨立）；Lemma 1 的 $\mathcal X$ 條件再提一次；Q 過程 $X_Q$、$\mathcal X_Q=\{N_V,N,S'\}$、導數、Eq. 11；C3 前定義高斯玩具；C3；C3 之後「抄寫時易犯的錯誤（本頁假設）」框；LO7 | 450–550 字 + C3 |
| S7 | `vloss` | 07 · V 損失與實用的交替訓練 | V 過程 $X_V$、$\mathcal X_V=\{N_Q,A\}$（先用一句話說明「A 也是隨機變數，所以 V 的分佈是對動作混合後的分佈」）；Eq. 10；最小解是後驗加權平均〔本頁推導〕，配一個兩動作的文字例子；真實速度未知 → 互為目標（Fitted Q-Evaluation 一句定義：用上一輪的估計當回歸目標反覆擬合）；Eq. 12–13；$(1-\gamma)$ 縮放；非 simulation-free；LO8 | 450–550 字 |
| S8 | `algos` | 08 · VF 抽樣挑選動作，IVF 改用資料動作加權 | 純量 Q／V；Eq. 14（rejection sampling 一句定義）；先以 C4(a) 的數字口述 expectile：「四個資料動作，μ = 0.5 時 V 是平均，μ 越大 V 越靠近最大的 Q」，再給 Eq. 15–16；Alg. 1（≥ 768px 兩欄：左原文虛擬碼、右中文逐行；< 768px 每行下接說明）；LO9 前半 | 450–550 字 |
| S9 | `ivf-theory` | 09 · IVF 的 V 是 Q 的 expectile，但只能在 Q 與 V 的純量上加權 | expectile 定義（不對稱平方損失 $|\mu-\mathbb 1(u<0)|u^2$ 的最小解）；隱式策略 Eq. 24；A.2 假設與結論；本頁解讀的兩個隱含前提；一階條件〔本頁推導〕；D.5；**C4**；A.2 筆誤小字；LO9 後半。延伸：`value-forcing-vf-ivf-algorithms.html`、`expectile-regression-value-learning.html` | 450–550 字 + C4 |
| S10 | `impl` | 10 · 實作設定與論文未說明的細節 | 「本章名詞」小表（目標網路：參數取線上網路 EMA 的副本，用來算目標；EMA：$\bar\psi\leftarrow(1-0.005)\bar\psi+0.005\psi$；ensemble／twin：同時訓練兩個網路成員；action chunk：一次輸出 H 步動作；Tweedie's formula：論文對一步估計的稱呼；classifier-free guidance：以同一網路、有無條件輸入兩種用法，IVF 借來讓一個網路兼任 V 與 Q 速度）；超參數表（Table 6 + Table 7 的 μ 與回報範圍）；VF／IVF 專屬設定；一步估計〔本頁推導〕；調參協定；「論文未說明」清單；6 條常見錯誤（Eq. 11 的兩個抄寫陷阱標「本頁假設」；V 損失用資料動作或策略動作；reward scaling；Q 與 V 速度的條件 (s,a) 與 s′；取樣 $z^\tau$ 需積分）；LO10。延伸：`flow-policies-offline-rl-value-learning.html` | 400–500 字 + 表 |
| S11 | `estimation` | 11 · 在 4 個狀態上，VF 的回報分佈比先前 flow 方法準 | 先定義 SARSA（bootstrap 時用資料中實際的下一個動作，所以評估的是收集資料的策略）與 Oracle（直接以 Eq. 3 擬合同批軌跡的實際折扣回報）；實驗流程（D.4）；F2；正文句子：「相對 PCBF，VF 的平均 1-Wasserstein 距離在 puzzle-4x4 降低 81%（22 對 118），在 humanoidmaze-giant 降低 89%（13 對 122）；在 cube-triple 與 puzzle-4x4 接近 Oracle（115 對 92、22 對 20）」；Fig. 6 表格；Fig. 7 文字描述；讀圖注意（1.4 步驟 10）；LO11 | 450–550 字 + F2 + 表 |
| S12 | `success` | 12 · 在 10%、25% 資料下，VF 與 IVF 的總平均高於同類基線 | OGBench 總平均表（4 資料量 × 13 方法；VF、IVF 欄加粗，同類最佳基線加底線）；兩條差距序列（VF +5／+6／+2／+5 不單調；IVF +16／+15／+8／−1）；OCBench 表；humanoidmaze-giant 與 puzzle-4x4 100% 的例外；子集抽樣方式；消融段落：正文寫明「拿掉 reward scaling，VF 的平均成功率約從 86% 降到 11%，IVF 從 76% 降到 11%（Fig. 5）；這是 4 個任務、最後三次評估的平均，與 Table 5 的最終評估協定不同，所以 86% 不能與 Table 5 的數字直接比較」，one-step actor 的 44% 照原句；LO12 | 450–550 字 + 2 表 |
| S13 | `history` | 13 · 研究脈絡 | 時間線（6.4）+ 一段：分布式 RL（C51、QR-DQN、IQN）→ flow matching → 終點監督的 flow critic → 路徑上的一致性（Value Flows、PCBF）→ Value Forcing；IQL／IDQL 是 IVF 與抽樣挑選的背景 | 時間線 + 120 字 |
| S14 | `next` | 14 · 限制與延伸閱讀 | 1.4 步驟 11；筆記清單（6.4）；arXiv、專案頁、程式碼 | 200 字 + 連結 |
| — | — | 頁尾 | 引用格式、`#pm-bar` | — |

Hero 三張事實卡（KaTeX 靜態渲染可用；卡內每個符號先以文字說明）：
1. 「γ 是折扣因子，$N$ 是獨立標準高斯。乾淨的 Bellman 方程 $Z=r+\gamma Z'$ 在 τ = 0 不成立：兩側都應是標準高斯，但 $\gamma Z'$ 的變異數只有 $\gamma^2$。Value Forcing 補上 $(1-\tau)\sqrt{1-\gamma^2}N$。」
2. 「對每個 noise level，新算子都是係數 γ 的 contraction（每套用一次，兩個分佈的距離至少縮小為 γ 倍），因此每一層都有唯一不動點（Remark 1）。」
3. 「VF 是本文的 explicit 版本（另外學一個策略，從候選動作中挑 Q 最大者），IVF 是 implicit 版本（只用資料中的動作，以權重學 V）。20 個 OGBench 任務、10% 資料：VF 總平均成功率 48%，最佳 explicit 基線 43%；IVF 43%，最佳 implicit 基線 27%（Table 2）。」

**靜態圖規格**：
- F1（S3）：寬 = 欄寬（≤ 720px），三列小圖，每列高 64px（< 600px 時 80px，標籤移到列上方）。每列一條 τ 軸（0、0.5、1 刻度，字級 13px），列標籤直接寫方法名（「FlowCritic 等：只在終點」「Value Flows：每層乾淨方程」「Value Forcing：每層修正方程」）；施加一致性的位置以實心圓、論文稱不正確的 τ = 0 以 × 標示，旁邊寫文字「不正確（τ = 0）」。不靠顏色。
- F2（S11）：四個 domain 各一張小圖（≥ 768px 2×2，< 768px 單欄堆疊，每張高 180px）。**每個 domain 獨立 y 軸**（從 0 起、刻度依該 domain 最大值），圖上方標「各圖刻度不同」；五根長條下方直接寫方法名（13px，< 600px 旋轉 −35°）；Oracle 為虛線外框、VF 實心 `--accent`、三個先前方法以 `currentColor` 描邊的斜線／點／橫線三種 pattern 加上方法名；每根長條上方標「數值 ± 標準差」。`aria-label` 列出全部 20 個數字。

**390px 公式**：全域 `.katex-display{overflow-x:auto;overflow-y:hidden;max-width:100%}`，每個 display 外包 `role="region" tabindex="0" aria-label="公式 Eq. n，可水平捲動"`。下列式子另有 < 600px 的 `aligned` 手動斷行版：Eq. 5（在 $+\gamma Z$ 前斷行）、Eq. 8–9、Eq. 10（期望下標改寫在下一行）、Eq. 11、Eq. 12、Eq. 13（第一行到 $\mid s,a,\tau\big)$，第二行 $-\big((1-\gamma)r+\gamma v_V(z^\tau|s',\tau)-\sqrt{1-\gamma^2}\epsilon\big)\big)^2\big]$）、Eq. 14（argmax 下標改為文字說明）、Eq. 16、Eq. 24、Hero 卡 1 的修正項。兩版以 media query 搭配 `display:none` 切換（不用 `visibility`），確保螢幕閱讀器只讀一份。

390px 版面（全頁通則）：單欄、16px 左右邊距；圖版內所有多欄子圖垂直堆疊；控制列每列一個滑桿（標籤在上、數值在右）；chip 列可換行；表格放 `.tscroll` 橫捲並以 `tabindex="0"`、`aria-label` 標示；Alg. 1 改為每行下接說明。

## 4. 互動元件規格

### 4.0 共用數學與規則

- PRNG：`mulberry32(seed)` 與 `gauss(rnd)`（Box–Muller，每次取兩個均勻數，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$）——直接沿用 No. 569 頁函式；Python 版在 `common.py`，`mulberry32(1)` 第一個輸出 0.6270739405881613。
- `invNorm(p)`：Acklam 有理近似（係數照 `common.py` 逐字複製：a = [−39.69683028665376, 220.9460984245205, −275.9285104469687, 138.3577518672690, −30.66479806614716, 2.506628277459239]，b = [−54.47609879822406, 161.5858368580409, −155.6989798598866, 66.80131188771972, −13.28068155288572]，c = [−0.007784894002430293, −0.3223964580411365, −2.400758277161838, −2.549732539343734, 4.374664141464968, 2.938163982698783]，d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]；三段：$p<0.02425$、中段、$p>0.97575$；任何最大誤差 < 1e-8 的反常態 CDF 都能在容許誤差內重現 C4），對 $p\in(0,1)$ 最大絕對誤差 < 3e-9（`common.py` 以 `statistics.NormalDist` 核對）。只在 C4(b) 用。
- `w1Sorted(x,y)`：兩組等長、已排序樣本的 $\frac1n\sum|x_i-y_i|$。`w1Weighted(xa,wa,xb,wb)`：把兩組加權原子合併排序，累加 $|F_a-F_b|\cdot\Delta x$（精確的 $\int|F_a-F_b|$）。
- 高斯速度閉式（C3、C4 共用，〔本頁推導〕）：乾淨回報 $\mathcal N(m,s^2)$、獨立耦合線性內插時，$\sigma^2(\tau)=\tau^2s^2+(1-\tau)^2$，$u(z|\tau)=m+k(\tau)(z-\tau m)$，$k=(\tau s^2-(1-\tau))/\sigma^2$。保護：$s$ 下限 0.05；$\sigma^2$ 對 τ 的最小值是 $s^2/(1+s^2)$（在 $\tau=1/(1+s^2)$），$s=0.05$ 時約 0.0025，所以除法永遠安全。
- 動態文字（讀數、`aria-label`、`aria-valuetext`、live region、SVG 標籤）一律純 Unicode，**不得使用組合字元**（U+0300–U+036F，如上橫線、波浪號、hat）：目標網路寫「目標 v_Q」，$\bar N$ 一律改用 $G$，$\tilde z$ 不出現在動態文字。下標用 `<tspan baseline-shift="sub" font-size="75%">`，上標（如 z 的 τ 上標）用 `<tspan baseline-shift="super" font-size="75%">`；`aria-label` 中寫成「z（τ 層）」。數字格式化函式 `fmt(v,d)` 把 $|v|<0.5\cdot10^{-d}$ 顯示為 `0`（避免「−0.0000」），負號用 U+2212。selfTest 掃描所有 `.plate` 的 `textContent` 與 `aria-label`，斷言不含 `/[\u0300-\u036f]/`。
- `input` 事件只更新狀態，以 `requestAnimationFrame` 合併重畫（`schedule()`）；live region 與 SVG `aria-label` 只在 `change` 或停止輸入 500ms 後更新（`announce()`）。
- 無 hover 專屬資訊：所有資訊都在固定讀數框；SVG 無拖曳。線條以線型＋標記形狀＋直接文字標籤區分：「正確／修正後」實線＋圓點，「乾淨方程／錯誤版」虛線＋三角，「參考／精確值」點線。顏色只作第二編碼。
- 所有元件沒有自動播放；C2 的「播放」按鈕在 `prefers-reduced-motion: reduce` 時直接跳到最終狀態。全域 `@media (prefers-reduced-motion: reduce){*{transition:none!important;animation:none!important}}`（含進度、chip、details）。
- chip 一律是 `<button aria-pressed>`；旗標值（C1 的 γ = 0.9999）只能由 chip 取得，鍵盤可用 Tab 到 chip 再按 Enter／Space。
- 斜線等 pattern 的描邊用 `currentColor`，由外層 `color` 設定，深色模式自動跟隨。
- 「乾淨算子（本頁模型）」一詞在 C1、C2 的標題列、圖例與讀數框都完整寫出；兩個元件的圖說都附這句：「這是把乾淨方程當成對每層邊際分佈的算子所得的模型，不是 Value Flows 實際學到的分佈。」

### C1 · 高斯修正項：一次 backup 在每個 τ 的差別（S5，接在不動點閉式之後）〔玩具〕

- **教的概念**：在 noise level τ，若下一步的 noisy return 已是正確分佈，則 Eq. 5 的右邊恰好給出正確分佈；乾淨算子（本頁模型）$r+\gamma Z'(\tau)$ 的平均差 $(1-\tau)r$、變異數差 $(1-\tau)^2(1-\gamma^2)$，在 τ = 0 最大、τ = 1 消失（〔本頁推導〕，顧問 A(iv)）。
- **情境**（頁面寫出，標「推定」）：下一狀態的乾淨回報 $Z'\sim\mathcal N(m',\sigma'^2)$；依本章剛推出的閉式，它在 τ 的正確分佈是 $\tau Z'+(1-\tau)G$。目前 $(s,a)$ 的獎勵 $r$。
- **輸入**：γ 滑桿 [0, 0.999] 步長 0.001，預設 0.9，chip「0.9」「0.99」「0.999」「0.9999」（0.9999 以旗標 `gamma4` 傳入，不寫回滑桿，同 No. 569 的 `tauIsOne` 機制；拖動滑桿清除旗標）；τ 滑桿 [0,1] 步長 0.01，預設 0.5；$r$ ∈ [−2, 2] 步長 0.1，預設 −1；$m'$ ∈ [−4, 2] 步長 0.1，預設 −2；$\sigma'$ ∈ [0.05, 2] 步長 0.05，預設 0.5；勾選框「獎勵乘 (1−γ)（論文的實作）」預設不勾，勾選時以 $r_{\rm eff}=(1-\gamma)r$ 計算。$m'$、$\sigma'$ 不隨勾選改變，視為讀者自選的單位；頁面寫明：「τ = 0 的 W₂ 只依 r 與 γ（平均差 r、標準差 1 對 γ），與 m′、σ′ 無關，所以勾選只用來看 τ = 0 的單步差距在論文設定下有多小。」γ chip 為 `aria-pressed` 按鈕。圖說：「預設 γ = 0.9 讓差別看得見；論文 γ = 0.999 時單步係數 $\sqrt{1-\gamma^2}$ 只有 0.0447」，並緊鄰 S4 的量級表連結。
- **計算**（`c1.py` 的 `c1`，閉式）：$c=\sqrt{\max(0,1-\gamma^2)}$；正確：平均 $\tau(r+\gamma m')$、標準差 $\sqrt{\tau^2\gamma^2\sigma'^2+(1-\tau)^2}$；修正版：平均 $\tau r+\gamma\tau m'$、標準差 $\sqrt{\gamma^2(\tau^2\sigma'^2+(1-\tau)^2)+(1-\tau)^2c^2}$（恆等於正確版，selfTest 檢查差 < 1e-12）；乾淨版：平均 $r+\gamma\tau m'$、標準差 $\gamma\sqrt{\tau^2\sigma'^2+(1-\tau)^2}$；$W_2$（兩高斯的 2-Wasserstein 閉式 $\sqrt{\Delta\mu^2+\Delta\sigma^2}$，頁面在元件前一句定義）。
- **顯示**（≥ 768px 左右兩圖，< 768px 上下堆疊，每圖高 200px）：
  1. 「目前 τ 的分佈」：三條密度曲線，x 範圍取三者 $\mu\pm4\sigma$ 的聯集（σ 下限 1e-3；若 γ = 0、τ = 1 使乾淨版 σ = 0，畫成垂直線並標「點質量」）；正確（點線、粗）、修正版（實線＋每 25 點一個圓點）、乾淨版（虛線＋三角）。曲線用 200 點取樣。y 軸上限 = 正確分佈峰值 × 1.3；乾淨版若超出則截斷並在曲線頂端標「已截斷」。
  2. 「乾淨算子（本頁模型）與正確分佈的距離 $W_2$ 隨 τ」：τ ∈ [0,1] 101 點，目前 τ 以垂直細線標示；y 軸從 0 起。
  - 讀數表：三者平均與標準差（4 位小數）、$W_2$、$c=\sqrt{1-\gamma^2}$、$1/(1-\gamma)$（γ = 0 時顯示 1）。
- **數值保護**：σ = 0 的密度不計算（畫垂直線）；γ 上限 0.9999 時 $1-\gamma^2$ 以 `(1-g)*(1+g)` 計算避免相消。
- **讀者應觀察到**：預設（γ = 0.9、τ = 0.5）正確平均 −1.4、標準差 0.5483，乾淨版 −1.9、0.5031，$W_2$ = 0.5020；τ = 0 時正確為 N(0,1)、乾淨版平均 −1、標準差 0.9；$W_2$ 曲線在 τ = 0 最大（1.0050）、τ = 1 為 0；勾選縮放且 γ = 0.999 時 τ = 0 的 $W_2$ 只有 0.0014。
- **無障礙**：圖 1 `aria-label` 例：「τ = 0.50：正確分佈平均 −1.40、標準差 0.548；乾淨方程平均 −1.90、標準差 0.503；距離 W₂ = 0.502」。

### C2 · 兩個算子都會收斂，但收斂到不同地方（S5）〔玩具〕

- **教的概念**：(1) Eq. 6：反覆套用 $\mathcal T^\pi_\tau$，兩個任意起點之間的 $W_1^{\sup}$ 每步至少乘以 γ；乾淨算子也是。(2) 修正版的不動點平均 $\tau Q$、標準差 $\sqrt{\tau^2\mathrm{Var}Z+(1-\tau)^2}$，形狀接近 $\tau Z+(1-\tau)G$；τ = 0 時是 N(0,1)。(3) 乾淨算子（本頁模型）在每個 τ 都收斂到 $Z^\pi$；兩者在 τ < 1 不同，τ = 1 重合。
- **玩具**（全部「推定」，頁面以小狀態圖畫出）：兩個 (s,a) 對，稱為「對 1」「對 2」（程式索引 0、1；策略固定，所以每個對的下一個對只依轉移機率）。獎勵 R = [0, −1]；轉移 P = [[0.7, 0.3], [0.4, 0.6]]。每個對以 400 個等權原子表示分佈。起點 A：所有原子為 0；起點 B：對 1 為 [−10, 0] 均勻格點、對 2 為 [−2, 6] 均勻格點（第 k 個原子 $lo+w(k+0.5)/400$）。
- **一步**（`c2.py` 的 `step`／`project`，逐式相同）：對每個對 i，混合分佈由（下一對 i′, j）組成，權重 P[i][i′]/400，值為修正版 $\tau R[i]+\gamma Z[i'][j]+(1-\tau)c\,g_{i,i',j}$ 或乾淨版 $R[i]+\gamma Z[i'][j]$；雜訊表 $g$ 每次迭代 k 用 `mulberry32(1000+k)` 產生 **1600** 個 `gauss`（2 個對 × 2 個下一對 × 400），順序為 i 外層、i′ 中層、j 內層：前 800 個給對 1（其中前 400 個 i′ = 對 1、後 400 個 i′ = 對 2），後 800 個給對 2。起點 A 與 B、以及修正版的兩條序列共用同一張表（乾淨版不用雜訊）。k = 1 的前幾個值：$g_{1,1,0}$ = −0.060209008271024565、$g_{1,2,0}$ = −1.29139894185053、$g_{2,1,0}$ = −0.855905775769241（`c2_out.txt`）。比值：$W_1^{\sup}$（投影**前**的混合分佈，用 `w1Weighted`）÷ 上一步的 $W_1^{\sup}$（`w1Sorted`）；理論上 ≤ γ（共用 (x′, j, g) 就是 A.1 的耦合；頁面寫出這句）。投影：原子排序後，第 k 個新原子取累積權重首次 ≥ (k+0.5)/400 的值（容許 1e-15）。
- **參考**：每個 γ 計算一次「乾淨算子、τ = 1、起點 A、120 次迭代」的粒子 $Z_{\rm ref}$；τ 層參考樣本 $\mathrm{sort}(\tau z_j+(1-\tau)\,\mathrm{gauss})$，雜訊用 `mulberry32(7)`，先對 1 的 400 個、再對 2 的 400 個。精確動差：$Q=(I-\gamma P)^{-1}R$，$M_2=(I-\gamma^2P)^{-1}(R^2+2\gamma R\circ PQ)$，$\mathrm{Var}=M_2-Q^2$（2×2 以克拉瑪公式）。
- **輸入**：γ 滑桿以**整數 index** i ∈ {10, …, 19}（步長 1，預設 18），γ = i/20，`aria-valuetext` 與讀數顯示 `(i/20).toFixed(2)`，快取 key 用 i；τ 滑桿同樣以整數 index t ∈ {0, …, 20}、τ = t/20，預設 0；顯示的對（radiogroup 對 1／對 2，預設對 1）；迭代 k 滑桿 0…80，預設 60；按鈕「播放」（k 從 0 走到 80，每幀 +1）。
- **計算排程**：γ、τ 滑桿的 `input` 只更新數字標籤；`change`（放開或鍵盤調整）才啟動工作。一個工作依序做：(1) 若快取沒有該 γ 的 $Z_{\rm ref}$，先算 120 次迭代（每個 rAF 回呼 10 次）；(2) 80 次迭代 × 4 條序列（修正 A、修正 B、乾淨 A、乾淨 B），每個 rAF 回呼 4 次迭代。進度文字「計算中：參考 k/120」「計算中 k/80」寫在讀數框上方（`aria-live` 只在完成時朗讀「完成」）。新工作開始時以世代計數取消舊工作。工作進行中：圖與讀數保留上一個結果、不透明度 0.5、k 滑桿停用；首次載入且尚無結果時，三個圖位顯示同尺寸的空框與文字「計算中」。保存每次迭代 A 序列兩個對的粒子快照（81 × 2 × 2 × 400 個 Float64，含 k = 0）與 $W_1^{\sup}$、比值紀錄。k 滑桿只讀快照，不重算。
- **顯示**（≥ 1024px 三圖並排；768–1023px 上兩圖下一圖；< 768px 三圖堆疊）：
  1. 「第 k 次迭代的分佈（所選對）」：兩個直方圖（40 個 bin，範圍取兩組粒子與參考的聯集；範圍寬 < 1e-9 時以 ±0.5 展開）：修正版實心長條、乾淨版只畫外框虛線；τ = 0 時疊 N(0,1) 密度點線；兩條垂直標線：精確平均 $\tau Q(x)$（實線，標「修正版的目標平均」）與 $Q(x)$（虛線，標「Q」）。
  2. 「兩個起點之間的距離 $W_1^{\sup}$」：對數 y 軸，k = 0…80，修正版實線圓點、乾淨版虛線三角，參考線 $5\gamma^k$（點線，標「γ^k 速率」）；值 < 1e-12 時截在 1e-12。
  3. 讀數表（所選對）：修正版與乾淨版的平均、標準差（4 位）、各自的精確值；「修正版與 τZ+(1−τ)G 參考樣本的 W₁」；最後一步比值與 1…k 中的最大比值（6 位小數），旁邊寫「≤ γ = 0.90」。
- **數值保護**：上一步 $W_1^{\sup}=0$ 時比值顯示「—」；k = 0 時只顯示起點。
- **讀者應觀察到**（預設 γ = 0.9、τ = 0、k = 60、對 1；正文只寫「約 0、約 1」「約 −3.7、約 1.3」，下列四位數只在讀數框）：修正版平均 −0.0237、標準差 0.9998（精確 0、1）；乾淨版 −3.6924、1.3262（精確 −3.6986、1.3255）；所有比值 ≤ 0.9（最大 0.899927）；τ 拉到 1 時兩者重合；τ = 0.5 時修正版 −1.8592、0.8266（精確 −1.8493、0.8302）而乾淨版不變。粒子數有限，與精確值的差約 0.01–0.05（對 2 在 τ = 0 的標準差 0.9570 對 1 是最大的例子），頁面在讀數表下註明「400 個原子、有限雜訊樣本與投影造成的誤差」。圖說另附 γ 預設誇大單步效果的句子（同 C1）。
- **效能**：Python 版 6 組 × 60 次迭代 3.4 秒；JS 預估每次迭代 < 2ms，整個工作 < 0.2 秒並分散在約 20 幀。

### C3 · Q 損失的回歸目標：雜訊很大，條件平均恰好是速度（S6）〔玩具〕

- **教的概念**：Lemma 1 的意思是「目標本身有雜訊，但以輸入點為條件的平均就是真實速度」；Eq. 11 的輸入點與目標各自的正負號與 τ 係數；兩個抄寫時易犯的錯誤（本頁假設）會讓條件平均偏離 $u_Q$。
- **玩具**（「推定」，頁面寫出）：下一狀態的乾淨 V 回報 $\mathcal N(m',\sigma'^2)$，所以 V flow 在 τ 的路徑是 $z^\tau=\tau m'+\sigma_V(\tau)z^0$（一維高斯 flow 是這個仿射映射，保分位數）；$u_V$ 用 4.0 閉式；Q 的乾淨回報因此是 $\mathcal N(r+\gamma m',\gamma^2\sigma'^2)$，$u_Q$ 同樣閉式。獎勵不縮放。
- **輸入**：γ 滑桿 [0.5, 0.999]，預設 0.9；τ [0, 0.999] 步長 0.001，預設 0.5（上限避開 τ = 1 的退化，頁面寫明）；$r$ [−2, 2] 預設 −1；$m'$ [−4, 2] 預設 −2；$\sigma'$ [0.05, 2] 預設 0.6；模式 radiogroup：「正確（Eq. 11）」「抄寫錯誤（本頁假設）：目標用 +ε」「抄寫錯誤（本頁假設）：目標的 r 乘 τ」；選取樣本的 $z^0$ 與 ε 滑桿各 [−2.5, 2.5] 步長 0.05，預設 1.0 與 −1.0。
- **計算**（`c3.py`）：$x=a_0+a_1z^0+a_2\epsilon$，$a=(\tau r+\gamma\tau m',\ \gamma\sigma_V,\ (1-\tau)c)$；$y=b_0+b_1z^0+b_2\epsilon$，$b=(r'+\gamma u_V(\tau m'),\ \gamma k_V\sigma_V,\ \mp c)$（$r'$ 在「乘 τ」模式為 $\tau r$；「+ε」模式 $b_2=+c$）。精確條件平均（兩者聯合高斯）：斜率 $(a_1b_1+a_2b_2)/(a_1^2+a_2^2)$、截距 $b_0-\text{斜率}\cdot a_0$。300 個樣本 $(z^0,\epsilon)$ 由 `mulberry32(3)` 依序取 `gauss` 兩次產生（樣本與模式、參數無關，只隨參數變換）；樣本 OLS 線。
- **顯示**（≥ 768px 左右兩圖，< 768px 堆疊）：
  1. 「下一狀態的 V flow 路徑」：τ ∈ [0,1] 為橫軸，11 條路徑（起點 $z^0\in\{-2,-1.6,\dots,2\}$），選取樣本的路徑加粗，目前 τ 為垂直線並標點 $z^\tau$。
  2. 「回歸資料 (輸入點, 目標)」：300 個散點（小圓，`--ink-3`），$u_Q(x)$ 實線（標「u_Q：Q 回報的真實速度」）、精確條件平均虛線（標「目標的條件平均」；正確模式時與實線重合，圖上加文字「重合」）、OLS 細點線；選取樣本以大方塊標出，並從 $x$ 畫一條垂直細線到 $u_Q(x)$。
  - 讀數：選取樣本的 $z^\tau$、$u_V(z^\tau)$、輸入點 $x$、目標 $y$、$u_Q(x)$；條件平均與 $u_Q$ 的斜率、截距（4 位）；OLS 斜率、截距；「條件平均 − u_Q」在 $x=\mathbb E[x]$ 處的差。
- **數值保護**：$\sigma'\ge0.05$（見 4.0）；$a_1^2+a_2^2>0$ 恆成立（τ < 1 時 $a_2>0$ 或 γσ_V > 0）。
- **讀者應觀察到**：正確模式下條件平均與 $u_Q$ 完全重合（預設斜率 −1.0969、截距 −4.3357）；「+ε」使斜率變為 −0.5085；「r 乘 τ」使截距變為 −3.8357（增加 $-(1-\tau)r=0.5$）；τ = 0 時 $u_Q$ 斜率 −1、截距 $=r+\gamma m'$（即 $z^0+u(z^0|0)=Q$，對應 S10 的一步估計）；γ 接近 1 時「+ε」造成的斜率偏差變小（γ = 0.99、τ = 0.5 斜率 −0.8979 對 −0.9568），頁面寫明：「這只是這個玩具中條件平均的偏差，不是訓練結果；偏差小不代表無害。」

### C4 · IVF 的 expectile 權重與「對速度取 max」的偏差（S9）〔玩具〕

- **教的概念**：(a) Eq. 15 權重、Eq. 24 隱式策略；$V$ 是 Q 在 $\pi_\beta$ 下的 μ-expectile，等於 $\mathbb E_{\pi^\mu}[Q]$；μ → 1 趨近資料內最大 Q。(b) D.5：逐點對速度取 max 再積分，結果不等於 $\max_aQ$；在這個一維、兩動作共用起點 $z^0$ 的玩具中一定 ≥（〔本頁推導〕，ODE 比較定理），論文只寫 ≠。
- **(a) 輸入**：μ 滑桿 [0.5, 0.999] 步長 0.001，預設 0.9，chip 0.5／0.7／0.9／0.99；四個資料動作 Q = [−3, −2, −1.2, −0.5]、$\pi_\beta$ = [0.4, 0.3, 0.2, 0.1]（推定，固定不可調）。
- **(a) 計算**：`expectile(Q, πβ, μ)`（No. 569 的函式，逐行相同）；$w_i=\mu$ 若 $Q_i\ge V$ 否則 $1-\mu$；$\pi^\mu_i=\pi_{\beta,i}w_i/\sum\pi_\beta w$；$\mathbb E_{\pi^\mu}[Q]$（selfTest：與 V 差 < 1e-12）。
- **(a) 顯示**：每個動作兩條並排橫向長條（$\pi_\beta$ 外框、$\pi^\mu$ 實心），長條末端標權重 μ 或 1−μ 的數字；Q 軸上的水平線：平均（μ = 0.5）、V（粗實線）、資料內最大（點線）。
- **(b) 輸入**：兩個動作的 Q 回報 $\mathcal N(m_a,s_a^2)$：$m_1,m_2$ ∈ [−4, 0] 步長 0.1（預設 −2.0、−2.2），$s_1,s_2$ ∈ [0.05, 2] 步長 0.05（預設 0.3、1.2）。
- **(b) 計算**（`c4.py` 的 `envelope`）：J = 101 個起點 $z^0_j=$ invNorm((j+0.5)/101)；Euler S = 400 步，第 i 步在 $\tau_i=i/400$ 求速度；三條 ODE 同時積分：每個動作自己的 $u_a$，以及 $\max_au_a$；結果取 J 個終點的平均。計算量 101 × 400 × 3 次閉式求值，直接在主執行緒（< 5ms），以 `schedule` 合併。
- **(b) 顯示**：路徑圖（τ 為橫軸）：取 j = 10, 30, 50, 70, 90 五個起點，畫兩個動作各自的路徑（細線，動作 1 實線、動作 2 虛線）與 max 場路徑（粗實線＋圓點）；右側讀數：$\max_aQ=\max(m_1,m_2)$、max 場終點平均、差距（4 位）、兩個動作自己的終點平均（應等於 $m_a$，Euler 對仿射場的平均是精確的）。
- **讀者應觀察到**：(a) μ = 0.5 時 V = −2.09（平均）、0.9 時 −1.2971、0.99 時 −0.6472，μ → 1 趨近 −0.5；$\pi^\mu$ 把機率移向 Q ≥ V 的動作。(b) 預設差距 0.2787；兩個 $s$ 相同時差距為 0（一個動作的速度處處較大）；差距來自兩個分佈寬度不同、速度在不同 z 處交叉。讀數旁固定文字：「≥ 只在一維、共用 z⁰ 時成立（本頁推導）；論文只寫 ≠。」頁面寫明這只說明 D.5 的不等式，不是 Fig. 4 成功率差異的解釋。正文只寫「差距約 0.28」，四位數留在讀數框。

---

## 5. 驗收數值表

腳本與頁面同一算法、雙精度。容許誤差：4 位小數者 ±0.0005；6 位小數的比值 ±1e-6；C2 粒子統計量（頁面對腳本）±0.002（同一 PRNG 與算法，只差 libm 最後一位）；C2「粒子對精確值」±0.06（標準差）／±0.05（平均）；C4(b) ±0.003；論文數字照抄、不容許誤差。

**C1**（`c1_out.txt`；r = −1、m′ = −2、σ′ = 0.5；欄：正確＝修正版 平均／標準差 ｜ 乾淨版 平均／標準差 ｜ $W_2$）

| γ | τ = 0 | 0.25 | 0.5 | 0.75 | 1 |
| --- | --- | --- | --- | --- | --- |
| 0.9 | 0／1.0000 ｜ −1.0000／0.9000 ｜ 1.0050 | −0.7000／0.7584 ｜ −1.4500／0.6843 ｜ 0.7536 | −1.4000／0.5483 ｜ −1.9000／0.5031 ｜ 0.5020 | −2.1000／0.4200 ｜ −2.3500／0.4056 ｜ 0.2504 | −2.8000／0.4500 ｜ 同左 ｜ 0 |
| 0.999 | 0／1.0000 ｜ −1.0000／0.9990 ｜ 1.0000 | −0.7495／0.7603 ｜ −1.4995／0.7596 ｜ 0.7500 | −1.4990／0.5589 ｜ −1.9990／0.5585 ｜ 0.5000 | −2.2485／0.4504 ｜ −2.4985／0.4502 ｜ 0.2500 | −2.9980／0.4995 ｜ 同左 ｜ 0 |

勾選 (1−γ) 縮放：γ = 0.9、τ = 0 → 乾淨版 −0.1000／0.9000，$W_2$ 0.1414；γ = 0.999、τ = 0 → −0.0010／0.9990，$W_2$ 0.0014；γ = 0.999、τ = 0.5 → 正確 −0.9995／0.5589、乾淨 −1.0000／0.5585，$W_2$ 0.0007。$c$：0.4359（0.9）、0.1411（0.99）、0.0447（0.999）、0.0141（0.9999）。γ = 0、τ = 0.3：正確 −0.3／0.7，乾淨 −1.0／0（點質量）。

**C2**（`c2_out.txt`；k = 60；欄：修正版 平均／標準差／對參考 W₁ ｜ 乾淨版 平均／標準差）

| γ（index） | τ | 精確 Q；sd(Z) | 對 1 | 對 2 | 最大比值 |
| --- | --- | --- | --- | --- | --- |
| 0.9（18） | 0 | −3.6986, −5.0685；1.3255, 1.3481 | −0.0237／0.9998／0.0528 ｜ −3.6924／1.3262 | −0.0159／0.9570／0.0628 ｜ −5.0665／1.3537 | 0.899927 |
| 0.9 | 0.5 | 同上（修正版精確：−1.8493／0.8302；−2.5342／0.8392） | −1.8592／0.8266／0.0445 ｜ 同上 | −2.5407／0.8217／0.0307 ｜ 同上 | 0.899947 |
| 0.9 | 1 | 同上 | −3.6924／1.3262／0.0076 ｜ 同左 | −5.0665／1.3537／0.0076 ｜ 同左 | 0.899751 |
| 0.7（14） | 0 | −0.8861, −2.1519；0.5755, 0.5991 | −0.0299／1.0025／0.0508 ｜ −0.8866／0.5758 | −0.0179／0.9504／0.0583 ｜ −2.1523／0.5991 | 0.699993 |
| 0.7 | 0.5 | 修正版精確：−0.4430／0.5769；−1.0759／0.5829 | −0.4583／0.5740／0.0285 ｜ 同上 | −1.0851／0.5647／0.0205 ｜ 同上 | 0.699993 |

γ = 0.9、τ = 0：比值 k = 1, 2, 3, 10 = 0.7180、0.8051、0.8482、0.8594；$W_1^{\sup}$（步前）k = 1, 10, 30, 60：修正版 5、1.046、0.09512、0.004021，乾淨版 5、0.9452、0.09498、0.003932（相對誤差 ±0.1%）。τ = 1 時兩算子逐位相同（selfTest 斷言差 < 1e-12）。

**C3**（`c3_out.txt`；r = −1、m′ = −2、σ′ = 0.6；欄：條件平均 斜率／截距；$u_Q$ 斜率／截距；OLS（300 樣本，seed 3））

| γ, τ | 正確 | +ε | r 乘 τ | $u_Q$ | OLS（正確） |
| --- | --- | --- | --- | --- | --- |
| 0.9, 0 | −1.0000／−2.8000 | −0.6200／−2.8000 | −1.0000／−1.8000 | −1.0000／−2.8000 | −1.0000／−2.8000 |
| 0.9, 0.25 | −1.1660／−3.6162 | −0.6752／−3.2726 | −1.1660／−2.8662 | 同正確 | −1.1607／−3.6123 |
| 0.9, 0.5（預設） | −1.0969／−4.3357 | −0.5085／−3.5119 | −1.0969／−3.8357 | 同正確 | −1.0736／−4.3024 |
| 0.9, 0.75 | −0.1382／−3.0902 | 0.2812／−2.2095 | −0.1382／−2.8402 | 同正確 | −0.0968／−3.0014 |
| 0.99, 0.5 | −0.9568／−4.4056 | −0.8979／−4.3179 | −0.9568／−3.9056 | 同正確 | −0.9527／−4.3991 |

輸入點的標準差恆等於 $\sigma_Q(\tau)$（例：γ = 0.9、τ = 0.5 → 0.5682）。OLS 容許 ±0.002。

**C4**（`c4_out.txt`）

(a) μ = 0.5／0.7／0.8／0.9／0.99／0.999 → V = −2.0900／−1.7690／−1.5579／−1.2971／−0.6472／−0.5158；μ = 0.9 的 $\pi^\mu$ = [0.1176, 0.0882, 0.5294, 0.2647]；μ = 0.99 → [0.0370, 0.0278, 0.0185, 0.9167]。

(b)（J = 101、S = 400）：預設 m = (−2.0, −2.2)、s = (0.3, 1.2) → max 場平均 −1.7213，差距 0.2787（J = 201、S = 1600 為 −1.7202，所以容許 ±0.003）；m = (−2, −2)、s = (0.5, 0.5) → 差距 0；m = (−2, −3)、s = (0.3, 0.3) → 差距 0；m = (−2, −2.5)、s = (0.2, 1.5) → −1.6536，差距 0.3464。每個動作自己的終點平均 = $m_a$（±1e-9）。

**論文數字的算術**（`numbers_out.txt`）：1 − 22/118 = 0.814、1 − 13/122 = 0.893（正文 81%、89%）；1 − 115/204 = 0.436、1 − 179/184 = 0.027（頁面不寫成「顯著下降」）；Table 5 同 4 個消融任務的最終評估平均 VF 93.5、IVF 77.0；未縮放回報範圍（γ = 0.999）[−3000, 0]、[−4000, 0]、[−16000, 0]、[−1000, 0]；OGBench 10% ≈ 10M transitions。Fig. 6 的 ±：cube-triple 13／16／15／13／13；cube-quadruple 6／6／10／7／7；puzzle-4x4 5／4／4／6／6；humanoidmaze-giant 3／2／3／2／3。

**S12 的完整表格（照抄 Table 2–5 的 Overall average 列與 OCBench 列；mean ± std，4 seeds）**。方法名：Floq；FlowCritic；Value Flows；PCBF（Path-Coupled Bellman Flows）；C51（51 個原子的類別分佈 critic）；IQN（Implicit Quantile Networks）；QC（Q-chunking：對 action chunk 學純量 critic 的標準 TD，Li et al. 2025）；VF（本文，explicit）；DIVL；Implicit C51、Implicit IQN（作者自建的 implicit 版本）；Implicit QC（IQL 式純量 Q、V + 抽樣挑選）；IVF（本文，implicit）。

| OGBench 資料量 | Floq | FlowCritic | Value Flows | PCBF | C51 | IQN | QC | **VF** | DIVL | Impl. C51 | Impl. IQN | Impl. QC | **IVF** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10%（Table 2） | 13±2 | 34±3 | 36±2 | 29±2 | 39±4 | 39±4 | 43±2 | **48±3** | 14±2 | 26±3 | 27±4 | 25±3 | **43±3** |
| 25%（Table 3） | 19±2 | 34±4 | 47±3 | 26±3 | 47±3 | 48±3 | 44±3 | **54±2** | 15±2 | 39±4 | 46±2 | 38±3 | **61±4** |
| 50%（Table 4） | 19±3 | 32±2 | 46±4 | 23±2 | 51±5 | 50±6 | 45±3 | **53±2** | 15±2 | 58±2 | 57±2 | 51±4 | **66±3** |
| 100%（Table 5） | 16±3 | 31±4 | 48±3 | 23±2 | 52±5 | 48±5 | 52±3 | **57±5** | 18±3 | 66±3 | 61±2 | 62±4 | **65±5** |

| OCBench 資料量 | Value Flows | C51 | QC | **VF** | Impl. C51 | Impl. QC | **IVF** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25%（Table 3） | 60±3 | 58±3 | 45±5 | **67±2** | 51±3 | 57±3 | **58±2** |
| 100%（Table 5） | 85±2 | 91±2 | 62±5 | **92±1** | 86±2 | 80±1 | **91±1** |

同類最佳基線（加底線）：10% QC 43／Impl. IQN 27；25% IQN 48／Impl. IQN 46；50% C51 51／Impl. C51 58；100% C51 與 QC 52／Impl. C51 66；OCBench 25% Value Flows 60／Impl. QC 57；100% C51 91／Impl. C51 86。Domain 例外（照抄）：humanoidmaze-giant domain 平均 VF 0／0／0／1、IVF 40／50／49／55；puzzle-4x4 100% IVF 82±11、Impl. C51 100±0。


---

## 6. 實作計畫

### 6.1 檔案與共用外殼

- 單一檔案 `tut/139/build/index.html`，CSS／JS／時間線資料全部內嵌，不 `fetch`；上線時複製為 `tutorials/value-forcing.html`。
- **直接複製** `tut/569/build/index.html` 的外殼，不重寫：CSS tokens（含深色模式三段覆寫、`pm-theme`）、字型、appbar、`#pm-bar`、章節目錄、`.plate`／`.plate-h`／`.plate-f`、callout `.note`、`.qa` 檢查題、pill（「本頁推導」「推定」「本頁解讀」，新增「玩具」）、「本章符號」小表、`.tscroll`、KaTeX 0.16.9 設定與 `renderMath()`、`?test` selfTest 框架（`ok()`、`#selftest`、`scrollWidth <= innerWidth`）、`renderTimeline()`、SVG 輔助（`el`、`tx`、`lin`、`lgs`、`lp`、`frame`、`arrow`、`niceTicks`、`widthOf`、`sub`）、`announce()`、`schedule()`、radiogroup、`bindTau` 的旗標模式（改名 `bindRange`，供 C1 的 γ = 0.9999 chip）、`expectile`／`bisectExpectile`、`mulberry32`／`gauss`、`ResizeObserver` 重畫。刪除 IQL 專屬的 C1–C4 程式與迷宮常數。
- 以下只列新增或變更。

### 6.2 新增 tokens 與連結

- 圖表色：`--c-ok`（修正版／正確；沿用 569 的 `--c-beh` 值 light #3F6E8C／dark #8DB4D3）、`--c-clean`（乾淨方程／錯誤版；沿用 `--c-naive` 值 light #6B7A3A／dark #A9BC7C）；目前值與選取樣本用 `--accent`。建置時確認在兩種 `--surface` 上圖形 ≥ 3:1、文字 ≥ 4.5:1，不足則調整並回寫此處。
- 「玩具」pill：`--surface-2` 底、`--ink-2` 字，與「推定」同形狀，文字「玩具：不是論文實驗」。
- 站內連結：Hero「在關聯圖中查看」→ `../index.html#paper-139`；arXiv `https://arxiv.org/abs/2611.02160`；專案頁與程式碼見文首；筆記 → `<slug>.html`。

### 6.3 程式結構（單一 IIFE）

- 共用：`invNorm(p)`、`w1Sorted`、`w1Weighted`、`gaussVel(m,s,tau)` → `{k,b,sig}`、`fmt(v,d)`。
- C1：`c1Values(g,tau,r,m1,sig1)` → `{c,tgt,cor,cln,w2}`；`initC1()`、`renderC1(final)`。
- C2：常數 `R2=[0,-1]`、`P2=[[.7,.3],[.4,.6]]`、`NAT=400`、`KMAX=80`；`c2Init(kind)`、`c2Noise(k)`、`c2Step(Z,g,tau,corr,G)`、`c2Project(xs,ws)`、`c2Moments(g)`、`c2Ref(g)`（快取）、`c2Job(gi,ti)`（以整數 index 傳入；含 $Z_{\rm ref}$ 的 rAF 分段、世代計數）、`c2Run(gi,ti)`（同步版，只給 selfTest）、`renderC2(final)`。逐式對應 `c2.py`。
- C3：`c3Values(g,tau,r,m1,sig1,mode)` → `{slope,icpt,kQ,bQ,a,b,kV,bV,sV}`；`c3Samples()`（seed 3 的 300 組 $(z^0,\epsilon)$，只產生一次）；`renderC3(final)`。
- C4：`implicitPolicy(Q,pb,mu)`、`envelope(ms,ss,J=101,S=400)`；`renderC4a`、`renderC4b`。
- `selfTest()` 斷言（容許誤差見第 5 節）：
  - `mulberry32(1)()` = 0.6270739405881613；`invNorm(0.975)` 與 1.959963985 差 < 1e-8；`w1Weighted([0,1],[.5,.5],[.5,1.5],[.5,.5])` = 0.5。
  - C1：預設三組平均／標準差與 $W_2$ = 0.5020；τ = 0 正確版 = (0, 1)；修正版與正確版在 20 組隨機參數（`mulberry32(42)`）下差 < 1e-12；γ = 0.9999 旗標時 $c$ = 0.0141。
  - C2：k = 1 雜訊表長度 1600、前三個值見 4 C2；γ index 18 → γ = 0.9 且 `(18/20).toFixed(2)` = "0.90"；γ = 0.9、τ = 0 跑完 60 次後對 1 修正版平均 −0.0237、標準差 0.9998、乾淨版 −3.6924、1.3262；1…60 的最大比值 ≤ 0.9 + 1e-9 且 = 0.899927 ± 1e-6；精確動差 −3.6986、1.3255；τ = 1 時修正與乾淨序列逐位相同。selfTest 直接同步呼叫 `c2Run`（不經 rAF），測試模式可接受約 0.2 秒。
  - C3：預設三種模式的斜率／截距；正確模式在 τ ∈ {0, 0.25, 0.5, 0.75, 0.999} 與 $u_Q$ 差 < 1e-12；OLS 預設 −1.0736／−4.3024。
  - C4：V 在 μ = 0.5／0.9／0.99 = −2.09／−1.2971／−0.6472，且 = $\mathbb E_{\pi^\mu}[Q]$；`envelope` 預設 −1.7213、兩個 s 相同時差距 < 1e-9。
  - 所有 `.plate` 文字與 `aria-label` 不含 U+0300–U+036F。
  - 390px：`document.documentElement.scrollWidth <= innerWidth`（外殼既有）。

### 6.4 研究筆記與時間線

- 來源 `tut/139/notes/timeline.json`（15 筆，`year` 為數字），原樣貼進 `<script type="application/json" id="timeline-data">`；`Number(year)` 排序，同年保持檔案順序。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中歸「其他」並在建置回報列出）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 分布式 RL | 1707.06887、1710.10044、1806.06923 | `distributional-rl-return-distributions.html` |
  | Flow matching | 2210.02747、2209.03003 | `flow-matching-velocity-target.html` |
  | 以 flow 表示價值（終點目標） | 2503.09817、2509.06863、2509.23087、2510.22686、2605.08515 | `flow-based-return-critics.html` |
  | 路徑上的 Bellman 一致性 | 2510.07650、2605.08253 | `flow-bellman-operator-all-noise-levels.html` |
  | 隱式價值學習與抽樣挑選 | 2110.06169、2304.10573 | `expectile-regression-value-learning.html`（IQL）、`value-forcing-vf-ivf-algorithms.html`（IDQL） |
  | 本文 | 2611.02160 | `value-forcing-lineage-hub.html` |

- S14 延伸閱讀（依總覽閱讀順序）：`value-forcing-lineage-hub`、`rl-mdp-q-learning-basics`、`offline-rl-distribution-shift`、`distributional-rl-return-distributions`、`flow-matching-velocity-target`、`flow-based-return-critics`、`flow-bellman-operator-all-noise-levels`、`value-forcing-vf-ivf-algorithms`、`expectile-regression-value-learning`、`iql-lineage-overview`、`flow-policies-offline-rl-value-learning`、`diffusion-policy-class`。標題取各筆記 frontmatter 的 `title`（跨教學的五篇從 `tut/568|238|569|443/notes/` 讀取）。

### 6.5 文案與排版（新增部分）

- 用詞：「noise level（雜訊層級）τ」「回報（return）」「回報分佈」「分佈相等」「乾淨（clean）Bellman 方程」「高斯修正項」「flow Bellman 算子」「速度場（velocity field）」「邊際（marginal）」「耦合（coupling）」「不動點」「行為複製（behavioral cloning）」「抽樣挑選（rejection sampling）」「隱式策略（implicit policy）」。第一次出現附英文。
- 三種陳述分開：論文結果（附出處）、「作者認為／作者稱」、〔本頁推導〕；玩具元件一律「玩具」pill + 「推定」。
- 過度概括的禁句（建置後以 grep 檢查）：「全面優於」「高斯修正項造成／導致改善」「先前方法必然塌縮」「搬到／搬成」（改為「逐步轉換成」）「塌縮」（改為「集中在接近單一值」）「VF 接近 Oracle」（沒有限定 domain 時）「小資料」（未註明 10% ≈ 10M transitions 時）。改寫為第 1.4 步驟 10 的限定說法。
- 第 1 節以外不新增論文數字；頁面數字須與第 1、5 節一致。

### 6.6 建置後自我檢查

沿用流程：390／768／1024／1440 截圖（只看新元件與 390px，最多 1 輪）＋深色、鍵盤 focus、reduced-motion；`?test` 全數通過；impeccable 0 項；`extract_page_text.py` 確認 12 條 LO 參考答案的關鍵詞與數字出現在正文。無頭 Chrome 無法連 CDN：截圖時把 KaTeX 指向本機 `scratchpad/katex/0.16.9`，上線檔仍指向 cdnjs。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | Fig. 2／Fig. 6 數值來源 | 文字層把 ± 印成「§」，數值以欄位位置對應。已渲染 PDF 第 8、20 頁（`verify/fig/p8-08.png`、`f6crop-20.png`）目視核對：五根長條順序與數值、長條高度都與顧問 §3.4 一致。 | 已關閉；頁面照用並標「讀自 Fig. 2／Fig. 6 長條上的數字」。 |
| R2 | 高斯修正項的因果 | 論文沒有拿掉修正項的消融；與 Value Flows 的比較同時混入 BCFM、confidence weight、return clipping 等差異。 | S11、S12 寫成正文；禁句檢查（6.5）。 |
| R3 | 玩具不是論文實驗 | C1–C4 全是自選小例子；C2 的粒子數、投影會造成 0.01–0.05 的誤差。 | 「玩具」pill；C2 讀數框寫明誤差來源；不在元件中「演」出塌縮。 |
| R4 | 不動點閉式是本頁推導 | 論文沒寫 $Z(s,a,\tau)\overset d=\tau Z+(1-\tau)G$。 | 〔本頁推導〕pill；推導步驟以 `<details>` 全列；C2 以動差與 W₁ 數值佐證。 |
| R5 | 原文筆誤 | A.2 p.18 Q 最小解式子的引數與條件；D.2 對 Value Flows DCFM 目標沒有 γ（無法判斷是原方法或轉述）；Sec. 5 的 Q 定義沒有 $z^0$ 項而 D.2 的一步估計有（期望相同）。 | 第一項小字註明以 Eq. 11、21 為準；第二項不細講 DCFM 公式，但在乾淨算子但書中說明其目標沒有 γ；第三項在 S10 一句話說明。 |
| R6 | 圖軸尺度 | Fig. 2 圖軸是未縮放回報（幾百），訓練時獎勵乘 $(1-\gamma)$；論文未說明換算。 | 標「本頁解讀」；只引 W₁ 數字，不解讀其絕對大小。 |
| R7 | 內部敘述不一致 | OCBench：Fig. 3 圖說與 Sec. 6.2 正文對 VF 100% 的說法不同（92±1 對 91±2）；「one-step actor 44%」未說明單位；Fig. 5 與 Table 5 協定不同。 | 照原句引用並附表格數字；不換算。 |
| R8 | 命名 | 論文引 Diffusion Forcing 但沒解釋「Forcing」的命名。 | 頁面不編典故。 |
| R9 | 腳註 1（均勻雜訊） | 論文說會有「不同修正項」但沒給；顧問推導顯示一般 γ 下不存在獨立加性修正。 | 只照引腳註，不展開，也不放顧問的推論。 |
| R10 | 初學者門檻 | 讀者可能不熟 Wasserstein、ODE、expectile。 | S1 以排序配對解釋一維 $W_1$；S2 以「速度 × 小步長」解釋 Euler；S8 先用 C4(a) 的數字例子口述 expectile，S9 再給定義（不對稱平方損失的最小解）並連到 No. 569。 |
| R11 | 未指明的超參數 | 見 1.4 步驟 9 清單。 | 頁面列「論文未說明」，不補值。 |
| R12 | 相對連結 | 建置位置與上線位置不同。 | 預覽只檢查外部連結與頁內錨點。 |

---

## 8. 修訂紀錄（round 1，對應 `reviews/round1.md` 28 項）

| # | 變更 |
| --- | --- |
| 1–2（P0） | C2 雜訊表改寫為 1600 個並寫明切分與前三個值（`c2.py` 新增輸出）；γ、τ 滑桿改整數 index，快取 key 用 index |
| 3–4 | S12（原 S10）標題改為「在 10%、25% 資料下，VF 與 IVF 的總平均高於同類基線」，寫出兩條差距序列；1.1 改為 explicit 方法中最高並附 IVF 數字 |
| 5、8 | 乾淨算子一律稱「本頁模型」，加 DCFM 無 γ、τ = 0 邊際依構造為高斯的但書；S4 結尾改為「τ < 1 不同、τ = 1 重合」 |
| 6、24 | LO12 改問 reward scaling 數字並照引 44%；LO6、LO9 與正文只寫兩位有效數字 |
| 7 | C3 錯誤模式與步驟 6 改稱「抄寫時易犯的錯誤（本頁假設）」 |
| 9–10 | C1／C2 圖說註明 γ 預設誇大單步效果；C1 縮放勾選的單位說明 |
| 11 | 標準差改 σ′；C2 的對改稱對 1、對 2 |
| 12、16 | LO5／步驟 5 補全前提；步驟 8 加 A.2 兩個隱含前提（本頁解讀） |
| 13、15、17 | FlowIQN 標記分開寫；Table 6 的 10 步列入未說明清單；C4(b) 的 ≥ 標為一維共用 z⁰ 的本頁推導 |
| 14 | C2 誤差改為 0.01–0.05 |
| 18–19 | S6／S7 各拆兩章（共 S0–S14）；C1 移到 S5 閉式之後；S4 的內插反推移入正文；expectile 先以數字例子引入 |
| 20–21 | S2 補耦合、邊際；S3 補 γZ′ 變異數說明、PCBF 恆等式入 details；S10 本章名詞表；SARSA、Oracle 在 S11 定義；刪除 IQL τ 提示 |
| 22 | C2 只在 change 重算，Z_ref 併入 rAF 工作，定義工作中與空狀態 |
| 23、25 | 390px 公式全域規則與手動斷行清單、display:none 切換；F1／F2 規格與 τ 軸 390px 標籤 |
| 26–28 | 用詞改字面；C1 y 軸截斷、上標規則、aria-pressed chip；全域 reduced-motion 與 pattern currentColor |
| 其他 | verify 腳本加 `sys.path.insert`，可用 `python3 -I` 執行（輸出與修訂前相同，C2 另加雜訊表與 γ index 檢查行） |

無拒絕項目。

## 9. readiness 修正紀錄

| # | 變更 |
| --- | --- |
| B1 | §5 貼上 OGBench 4 資料量 × 13 方法與 OCBench 完整總平均表，展開 PCBF、C51、IQN、QC 等縮寫 |
| B2 | S12 列寫入 reward scaling 86% → 11%、76% → 11% 與協定不同的說明 |
| N1 | invNorm 係數貼入 4.0，並寫精度要求 |
| N2 | Hero 副標與事實卡 3 加 VF／IVF／explicit／implicit 一句說明 |
| N3 | S11 列寫入 81%／89% 句子 |
| N4 | 一步訓練 vs 10 步評估標「本頁解讀」 |


## 10. 建置紀錄

- 產出 `build/index.html`（單檔，約 159 KB），外殼取自 No. 569 建置頁（同源於 k-diff 外殼）；原始片段與重建腳本放在 scratchpad `build139-src/`（不上架）。
- `?test`（本機 KaTeX 副本，1440 與 390 寬）：70 passed, 0 failed。涵蓋 C1–C4 驗收表、C2 雜訊表與 γ index、τ = 1 兩算子逐位相同、無組合字元、390px 無橫向捲動。
- C2 的 rAF 工作改為每幀以 12ms 時間預算推進（headless 下每幀固定迭代數過慢）。
- impeccable：0 anti-patterns（14 則 numbered-section-label advisory）；`.warn` 由單側色條改為完整邊框。
- LO 自查：12 條參考答案的關鍵詞與數字都在 `accept/page.txt` 正文中（KaTeX 轉成 Unicode 後比對）。
- 截圖：`build/caps/`（1440 的 C1–C4、F2；390 的 hero、S3、C1–C4、F2、S12；深色 C2）。
