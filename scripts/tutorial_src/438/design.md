# 設計文件：〈Flow-GRPO: Training Flow Matching Models via Online RL〉互動教學

- 論文：Jie Liu*、Gongye Liu*、Jiajun Liang、Yangguang Li、Jiaheng Liu、Xintao Wang、Pengfei Wan、Di Zhang、Wanli Ouyang†（* 同等貢獻，† 通訊作者），arXiv 2505.05470（v5），NeurIPS 2025。資料庫編號 No. 438。程式碼 `https://github.com/yifan123/flow_grpo`。
- 原文：`tut/438/src/paper.txt`（PDF 同目錄）。`tut/` = `scripts/tutorial_src/`。本文件的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」皆指論文原文編號。Sec. 4 有兩個未編號的式子：$r^i_t(\theta)$ 的定義（Eq. 5 下方）與 KL 閉式（Eq. 9 下方），本文件稱為「Eq. 5 的 $r$」「Sec. 4 KL 閉式」。
- 產出：`tut/438/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/flow-grpo.html`（slug `flow-grpo`）。站內相對連結一律以上線位置為準。
- 驗算腳本（與頁面同一演算法，純 Python 浮點）：`tut/438/verify/core.py`（共用數學）、`check.py`（驗收表，輸出 `check_out.txt`）。
- 研究筆記：`tut/438/notes/`（撰寫中；本文件寫作時資料夾尚不存在）。第 6.4 節為佔位，建置前依實際筆記補上。
- 標記慣例：**〔本頁推導〕** = 本設計自行推導、論文沒有寫出；**推定** = 為玩具可視化或為補足論文未寫明處而選定的值；「作者認為」= 論文作者的解釋，不是實驗結果；**玩具** = 本頁在瀏覽器中即時計算的一維模型，不是論文的 SD3.5-M，也不重現論文的任何數字。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

流匹配（flow matching）模型以一個確定性的常微分方程（ODE）把噪聲搬運成影像。作者想用線上強化學習（online RL，邊用目前的模型產生樣本、邊以獎勵更新模型）改進這類模型，但遇到兩個障礙：ODE 取樣沒有隨機性，無法探索，也難以計算策略機率；每張影像要很多步去噪，收集樣本很慢（Sec. 1）。Flow-GRPO 用兩個策略處理：(1) **ODE 轉 SDE**：把 ODE 換成一個在每個時刻都與原 ODE 有相同邊際分布的隨機微分方程（SDE）（Eq. 7–8，App. A）；以 Euler–Maruyama 離散化（Eq. 9）後，每一步是等向高斯分布，機率比與 KL 散度都有閉式（Sec. 4 KL 閉式）；(2) **去噪步數縮減（Denoising Reduction）**：訓練時只用 $T=10$ 步產生樣本，推論仍用預設的 $T=40$ 步（Sec. 4）。更新規則採用 GRPO（Group Relative Policy Optimization）：同一個提示詞產生 $G$ 張影像，以組內獎勵的標準化值當作優勢（Eq. 4），不需要價值網路（Sec. 2）。在 SD3.5-M 上，GenEval 由 0.63 提升到 0.95，文字渲染準確率由 0.59 提升到 0.92（w/ KL；w/o KL 為 0.93），PickScore 由 21.72 提升到 23.31（w/ KL）（Table 1–2）。

### 1.2 研究動機（Sec. 1）

1. 流匹配模型是目前影像生成的主流，但在多物體、屬性、相對位置的組合場景，以及影像中的文字渲染上仍常出錯（Sec. 1）。
2. 線上 RL 在大型語言模型的推理能力上效果顯著（DeepSeek-R1、OpenAI-o1，Sec. 2）。先前影像生成的 RL 研究多針對早期擴散模型（DDPO [12]），而流匹配模型多用離線方法（如 DPO）；線上 RL 用於流匹配模型的潛力「largely unexplored」（Sec. 1）。
3. 兩個挑戰（Sec. 1）：(1) 流匹配依賴確定性的 ODE 生成程序，取樣時沒有隨機性，而 RL 需要隨機取樣來探索；(2) 線上 RL 需要有效率地收集樣本，流模型每個樣本要多步迭代，大模型更嚴重。

### 1.3 貢獻（Sec. 1 條列）

1. 第一個把 GRPO 引入流匹配模型的方法：把確定性的 ODE 取樣轉成 SDE 取樣，顯示線上 RL 對文字生成影像（T2I）任務有效；SD3.5-M 的 GenEval 準確率由 63% 提升到 95%，影像品質沒有明顯下降。
2. 發現流匹配模型的線上 RL 收集樣本時不需要完整的長步數：訓練用較少的去噪步、測試保留原步數，可顯著加速訓練。
3. 顯示 KL 約束能有效防止 reward hacking（獎勵上升但影像品質或多樣性下降）；KL 正則化在實驗上不等同於提早停止：適當的 KL 能達到與無 KL 版本相同的高獎勵並保持品質，代價是訓練較久。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S10 開頭以可收合「符號總表」提供，作為回查用；S1 只放 S1–S2 用到的 $x_0,x_1,t,x_t,v,c$。每個符號在頁面中第一次使用前於內文定義）

| 符號 | 意義 | 出處 |
| --- | --- | --- |
| $x_0$、$x_1$ | 資料樣本、噪聲樣本 $x_1\sim\mathcal N(0,I)$ | Sec. 3 |
| $t\in[0,1]$ | 連續時間；$t=0$ 為資料端、$t=1$ 為噪聲端 | Eq. 1 |
| $x_t$ | $(1-t)x_0+t\,x_1$ | Eq. 1 |
| $v_\theta(x_t,t)$、$v_t$ | 模型預測的速度場；目標 $v=x_1-x_0$ | Eq. 2、Eq. 6 |
| $c$ | 文字提示詞（prompt） | Sec. 3 |
| $s_t,a_t,\pi$ | MDP 的狀態 $(c,t,x_t)$、動作 $x_{t-1}$、策略 $p_\theta(x_{t-1}\mid x_t,c)$ | Sec. 3 |
| $T$ | 去噪步數；離散步索引也寫成 $t$（$T\to0$） | Sec. 3–4 |
| $R$、$r(x_0,c)$ | 獎勵；只在最後一步給 | Sec. 3 |
| $G$ | 組大小：同一提示詞產生的影像數 | Eq. 4 |
| $\hat A^i_t$ | 第 $i$ 張影像的優勢（對所有 $t$ 相同） | Eq. 4 |
| $r^i_t(\theta)$ | 機率比 $p_\theta/p_{\theta_\text{old}}$（與獎勵 $r$ 同字母，頁面寫成 $\rho$ 並註明） | Eq. 5 |
| $\varepsilon$、$\beta$ | 裁剪範圍、KL 係數 | Eq. 5 |
| $\pi_\text{ref}$、$\pi_{\theta_\text{old}}$ | 參考策略（預訓練模型）、收集樣本時的策略 | Eq. 3、5 |
| $\sigma_t$、$a$ | SDE 擴散係數 $a\sqrt{t/(1-t)}$、噪聲等級 | Eq. 7–9 |
| $\nabla\log p_t$ | 分數（score）：邊際密度對數的梯度 | Eq. 7 |
| $\Delta t$、$\epsilon$ | 離散時間步長、標準常態噪聲 | Eq. 9 |

頁面寫法決定：符號總表的表頭註明「各符號的定義見對應章節；此表供回查」。Eq. 5 的機率比在頁面以 $\rho^i_t$ 表示（避免與獎勵 $r$ 混淆），第一次出現時寫「論文記為 $r^i_t(\theta)$」。裁剪範圍用 $\varepsilon$，SDE 噪聲用 $\epsilon$，兩者在 S4 定義時並列說明。

#### 步驟 1：流匹配與 Rectified Flow（Sec. 3，Eq. 1–2）

$x_t=(1-t)x_0+t\,x_1$，$t\in[0,1]$（Eq. 1）。以 Transformer 回歸速度場，損失 $\mathcal L(\theta)=\mathbb E_{t,x_0,x_1}\|v-v_\theta(x_t,t)\|^2$，$v=x_1-x_0$（Eq. 2）。取樣時解 ODE $dx_t=v_t\,dt$（Eq. 6），由 $t=1$ 走到 $t=0$；常用離散化在相鄰時間步之間是一對一映射（Sec. 4）。頁面說明：除了初始噪聲 $x_1$ 之外沒有任何隨機性。

#### 步驟 2：把去噪寫成 MDP（Sec. 3，引自 [12]）

狀態 $s_t\triangleq(c,t,x_t)$；動作 $a_t\triangleq x_{t-1}$；策略 $\pi(a_t\mid s_t)\triangleq p_\theta(x_{t-1}\mid x_t,c)$；轉移是確定性的 $(\delta_c,\delta_{t-1},\delta_{x_{t-1}})$；初始分布 $(p(c),\delta_T,\mathcal N(0,I))$；獎勵只在最後一步：$t=0$ 時 $R=r(x_0,c)$，否則 0。頁面說明這裡的 $t$ 是整數步索引（$T,T-1,\dots,0$），與 Eq. 1 的連續時間 $t$ 同字母；Eq. 9 改用連續時間寫成 $x_{t+\Delta t}$。頁面另寫一句：在 ODE 取樣下 $x_{t-1}$ 由 $x_t$ 唯一決定，策略退化為確定性映射，沒有可用的機率密度；S5 起把每一步改成隨機的，這個 MDP 才有真正的機率分布。

#### 步驟 3：GRPO 目標（Sec. 4，Eq. 3–5）

RL 目標（Eq. 3）：$\max_\theta\mathbb E\big[\sum_{t=0}^T\big(R(s_t,a_t)-\beta D_\text{KL}(\pi_\theta(\cdot\mid s_t)\|\pi_\text{ref}(\cdot\mid s_t))\big)\big]$。GRPO 以組相對方式估計優勢：給定 $c$，取樣 $G$ 張影像 $\{x^i_0\}$ 與軌跡 $\{(x^i_T,\dots,x^i_0)\}$，

$$\hat A^i_t=\frac{R(x^i_0,c)-\mathrm{mean}(\{R(x^i_0,c)\}_{i=1}^G)}{\mathrm{std}(\{R(x^i_0,c)\}_{i=1}^G)}\quad\text{(Eq. 4)}$$

目標 $\mathcal J_\text{Flow-GRPO}(\theta)=\mathbb E_{c,\{x^i\}\sim\pi_{\theta_\text{old}}}f(r,\hat A,\theta,\varepsilon,\beta)$（Eq. 5），

$$f=\frac1G\sum_{i=1}^G\frac1T\sum_{t=0}^{T-1}\Big(\min\big(r^i_t\hat A^i_t,\ \mathrm{clip}(r^i_t,1-\varepsilon,1+\varepsilon)\hat A^i_t\big)-\beta D_\text{KL}(\pi_\theta\|\pi_\text{ref})\Big),\quad r^i_t=\frac{p_\theta(x^i_{t-1}\mid x^i_t,c)}{p_{\theta_\text{old}}(x^i_{t-1}\mid x^i_t,c)}.$$

作者選 GRPO 而非 PPO 的理由：不需要價值網路，較省記憶體；PPO 也能以類似方式用於流匹配（Sec. 2）。

〔本頁推導〕兩個性質寫在頁面 S3：(i) 只要組內獎勵不全相同，組內 $\sum_i\hat A^i=0$；若 std 取母體標準差（除以 $G$），$\sum_i(\hat A^i)^2=G$。所以這時一組中同時有優勢為正（更新使其機率增大）與優勢為負（機率減小）的樣本。(ii) 若組內獎勵全相同，分子為 0，整組優勢為 0，這組提示詞的替代目標項（Eq. 5 的 min 項）梯度為 0；KL 項 $-\beta D_\text{KL}$ 仍有梯度。若組內 $G$ 個樣本的二元獎勵獨立、成功率都是 $p$（本頁假設，論文未寫），這種情況的機率是 $p^G+(1-p)^G$。論文沒有寫 std 用母體或樣本標準差，也沒有寫分母加多少小常數；頁面玩具取母體標準差，std $<10^{-12}$ 時整組優勢設為 0（推定）。

〔本頁推導〕裁剪項：對單一樣本，$\min(\rho\hat A,\mathrm{clip}(\rho)\hat A)$ 對 $\rho$ 的導數在 $\hat A>0,\rho>1+\varepsilon$ 或 $\hat A<0,\rho<1-\varepsilon$ 時為 0，其餘為 $\hat A$。也就是：機率比已經朝優勢的方向改變超過 $\varepsilon$ 的樣本，替代目標項不再提供梯度（KL 項仍有）；機率比朝相反方向改變時，裁剪不限制這個方向的更新。字面目的（頁面 S4）：$\pi_{\theta_\text{old}}$ 是收集這批樣本時的模型；同一批樣本做多次更新時，$\rho$ 衡量目前模型與收集時模型對同一動作的機率差多少；第一次更新前 $\rho=1$；裁剪讓單批資料造成的機率改變不超過約 $\varepsilon$。論文沒有寫 $\varepsilon$ 的值（App. B.3 未列）；頁面預設 0.2，標推定。

#### 步驟 4：確定性取樣為什麼不符合 GRPO 的需要（Sec. 4）

兩點（Sec. 4 原文）：(1) Eq. 5 的 $r^i_t$ 需要計算 $p(x_{t-1}\mid x_t,c)$，在確定性動力學下因為需要估計散度（divergence），計算昂貴（頁面字面解釋：確定性映射下要追蹤密度，必須計算速度場各分量對自身座標的偏導數總和，即散度，影像維度很高時代價很大）；(2) 更重要的是 RL 依賴探索，Sec. 5.3 顯示隨機性減少會大幅降低訓練效率；只有初始種子帶隨機性的確定性取樣特別有問題。

#### 步驟 5：ODE 轉 SDE（Sec. 4 Eq. 7–8，App. A Eq. 10–28）

目標：構造一個 SDE，使它在每個 $t$ 的邊際密度與 Eq. 6（＝Eq. 10）相同。App. A 的推導鏈：

1. 一般 SDE $dx_t=f_\text{SDE}\,dt+\sigma_t\,dw$（Eq. 11），邊際密度服從 Fokker–Planck 方程（Eq. 12；頁面字面解釋：描述密度 $p_t$ 如何隨漂移與噪聲隨時間改變的偏微分方程）；ODE 的邊際服從連續方程（Eq. 13）。
2. 令兩者相等（Eq. 14），利用 $\nabla^2[\sigma_t^2p_t]=\sigma_t^2\nabla\cdot(p_t\nabla\log p_t)$（Eq. 15），得 $f_\text{SDE}=v_t+\frac{\sigma_t^2}2\nabla\log p_t$（Eq. 16），即前向 SDE（Eq. 17）。
3. 前向 SDE $dx=f\,dt+g\,dw$ 的反向時間 SDE 是 $dx=[f-g^2\nabla\log p_t]dt+g\,dw$（Eq. 18–19，[75, 23]）。代入得 Eq. 20，化簡為 $dx_t=\big[v_t(x_t)-\frac{\sigma_t^2}2\nabla\log p_t(x_t)\big]dt+\sigma_t\,dw$（Eq. 21＝Eq. 7）。
4. 分數與速度的關係：對 $x_t=\alpha_tx_0+\beta_tx_1$，條件分數 $-x_1/\beta_t$（Eq. 23），邊際分數 $-\mathbb E[x_1\mid x_t]/\beta_t$（Eq. 24）；Eq. 25 推出一般式，代入 $\alpha_t=1-t$、$\beta_t=t$ 得 $v_t(x)=-\frac{x}{1-t}-\frac{t}{1-t}\nabla\log p_t(x)$（Eq. 26），反解 $\nabla\log p_t(x)=-\frac xt-\frac{1-t}t v_t(x)$（Eq. 27）。（頁面註明：此處的 $\beta_t$ 是插值係數，與 KL 係數 $\beta$ 不同。）
5. 代入得最終 SDE（Eq. 28＝Eq. 8）：$dx_t=\big[v_t(x_t)+\frac{\sigma_t^2}{2t}\big(x_t+(1-t)v_t(x_t)\big)\big]dt+\sigma_t\,dw$。

頁面重點：(a) 推導的前提：$v_t$ 是真實邊際 $p_t$ 對應的速度場（等於一個完全正確的模型；學到的 $v_\theta$ 有誤差時，SDE 與 ODE 的邊際不再保證相同），$p_t$ 足夠平滑使 Fokker–Planck 成立，且 Eq. 26–27 與 $\sigma_t=a\sqrt{t/(1-t)}$ 在 $t=1$ 奇異。在這些前提下，等式對 $t\in(0,1)$ 上的任何 $\sigma_t$ 成立，$\sigma_t$ 只控制隨機性的大小；$\sigma_t=0$ 時退回 ODE。(b) 邊際相同是連續時間的結論；離散成有限步後會有誤差（玩具 C3 顯示，〔本頁推導〕）。(c) 分數不必另外學習，由速度場直接換算（Eq. 27）。

#### 步驟 6：Euler–Maruyama 離散化與高斯策略（Eq. 9，Sec. 4 KL 閉式）

$$x_{t+\Delta t}=x_t+\Big[v_\theta(x_t,t)+\frac{\sigma_t^2}{2t}\big(x_t+(1-t)v_\theta(x_t,t)\big)\Big]\Delta t+\sigma_t\sqrt{\Delta t}\,\epsilon,\quad\epsilon\sim\mathcal N(0,I)\quad\text{(Eq. 9)}$$

$\sigma_t=a\sqrt{t/(1-t)}$，$a$ 是噪聲等級（Sec. 4；App. B.3 $a=0.7$）。

〔本頁推導〕兩個實作上的注意點（頁面以「實作注意」callout 寫出，整段標〔本頁推導〕）：
- 取樣方向是 $t$ 由 1 往 0，所以 $\Delta t<0$；論文 Eq. 9 寫 $\sigma_t\sqrt{\Delta t}$，沒有明寫正負號，應讀成 $\sqrt{|\Delta t|}$（反向時間 SDE 的 Wiener 增量變異數是 $\sigma_t^2|\Delta t|$）。
- $\sigma_t$ 在 $t=1$ 發散（分母 $1-t=0$）。論文沒有說明 $t=1$ 的處理。頁面玩具在 $t=1$ 時以 $t'=1-1/T$（下一個格點）代入 $\sigma$，標推定。

Eq. 9 顯示策略 $\pi_\theta(x_{t-1}\mid x_t,c)$ 是等向高斯分布：平均是 Eq. 9 去掉噪聲項，標準差 $\sigma_t\sqrt{|\Delta t|}$（Sec. 4）。因此 Eq. 5 的機率比可直接計算，KL 有閉式：

$$D_\text{KL}(\pi_\theta\|\pi_\text{ref})=\frac{\|\bar x_{t+\Delta t,\theta}-\bar x_{t+\Delta t,\text{ref}}\|^2}{2\sigma_t^2\Delta t}=\frac{\Delta t}{2}\Big(\frac{\sigma_t(1-t)}{2t}+\frac1{\sigma_t}\Big)^2\|v_\theta(x_t,t)-v_\text{ref}(x_t,t)\|^2$$

（論文寫成 $x_{t+\Delta t,\theta}$，指兩策略的平均；同樣 $\Delta t$ 取絕對值。）`check.py` 驗證兩個等號右邊在所有測試點相同（相對差 $<10^{-12}$）。

〔本頁推導〕KL 係數 $c(t)=\frac{|\Delta t|}2\big(\frac{\sigma_t(1-t)}{2t}+\frac1{\sigma_t}\big)^2$ 的形狀：$t\to1$ 時 $\sigma_t\to\infty$、兩項都趨近 0，$c\to0$；$t\to0$ 時 $\sigma_t\approx a\sqrt t$，$c\approx\frac{|\Delta t|}{2t}\big(\frac1a+\frac a2\big)^2$。在均勻格點上，最後一步（最接近資料）的係數最大；第一步（$t=1$）的係數最小，但那一步的標準差 $\sigma_t\sqrt{|\Delta t|}$ 最大，所以這不表示第一步不重要。$\big(\frac1a+\frac a2\big)^2$ 在 $a=\sqrt2$ 時最小（值 2），$a=0.7$ 時為 3.163：$a<\sqrt2$ 時 $a$ 越小，$1/\sigma_t$ 項越大，KL 對速度差越敏感；$a>\sqrt2$ 時 $\sigma_t(1-t)/(2t)$ 項變大，係數又上升（`check_out.txt`：$T=10$ 時 $\sum_jc_j$ 在 $a=1.4$ 為 1.932，$a=2.0$ 為 2.171）。機率比 $\rho$ 的對數則與 $\delta/\text{std}$ 成正比，$a$ 越小越極端（C4）。

#### 步驟 7：去噪步數縮減（Sec. 4，Sec. 5.3，App. C.2，App. D）

訓練時 $T=10$，推論用 SD3.5-M 預設 $T=40$（Sec. 4、App. B.3）。Sec. 5.3（未加 KL）：收集樣本的步數由 40 降到 10，三個任務都有超過 4 倍的加速，最終獎勵不受影響（GenEval 見 Fig. 7a；文字渲染與 PickScore 見 App. C.2 的 Fig. 9）；再降到 5 不一定更快，有時更慢，因此採用 10。App. D（Fig. 19）：40 步 ODE 與 40 步 SDE 的影像肉眼無法區分；10 步與 5 步 SDE 出現明顯瑕疵（顏色漂移、細節模糊），但作者認為因為 Flow-GRPO 依賴組內相對比較，低品質樣本仍提供有用的獎勵訊號。定量評估的步數是 $T=40$（B.3），論文沒有寫定量評估用 ODE 或 SDE；C.7 的訓練過程質性圖以 40 步 ODE 取樣產生。

〔本頁推導〕每個提示詞每次迭代的模型前向次數 $G\times T$：$24\times10=240$，對照 $24\times40=960$。這只是取樣次數的比例；論文報告的是 GPU 小時的加速（「over 4×」），兩者不是同一個量。

#### 步驟 8：任務與獎勵（Sec. 5.1）

- **GenEval 組合生成**：六類任務；訓練提示詞由官方腳本產生；測試集嚴格去重（只差物體順序的提示詞視為相同並自訓練集移除）。提示詞比例 Position : Counting : Attribute Binding : Colors : Two Objects : Single Object = 7 : 5 : 3 : 1 : 1 : 0，依基礎模型在六類上的初始準確率決定。獎勵是規則式：計數 $r=1-|N_\text{gen}-N_\text{ref}|/N_\text{ref}$；位置／顏色：物體數正確時給部分獎勵，位置或顏色也正確時給其餘部分。
- **文字渲染**：提示詞模板「A sign that says "text"」；GPT-4o 產生 20K 訓練、1K 測試提示詞；$r=\max(1-N_e/N_\text{ref},0)$，$N_e$ 為最小編輯距離、$N_\text{ref}$ 為引號內字元數；此獎勵同時是準確率指標。
- **人類偏好**：以 PickScore 為獎勵模型。
- **品質評估**：在 DrawBench 上計算 Aesthetic、DeQA、ImageReward、UnifiedReward（App. B.1）以偵測 reward hacking。

#### 步驟 9：實驗結果（Sec. 5.2–5.3，Table 1–4，App. C）

- **Table 1（GenEval）**：SD3.5-M 0.63 → 0.95；分項 Single 1.00、Two 0.99、Counting 0.95、Colors 0.92、Position 0.99、Attr. Binding 0.86。GPT-4o 0.84、SANA-1.5 4.8B 0.81、SD3.5-L 0.71、FLUX.1 Dev 0.66。
- **Table 2**：

| 模型 | GenEval | OCR | PickScore | Aesthetic | DeQA | ImgRwd | PickScore(DB) | UniRwd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SD3.5-M | 0.63 | 0.59 | 21.72 | 5.39 | 4.07 | 0.87 | 22.34 | 3.33 |
| GenEval w/o KL | 0.95 | — | — | 4.93 | 2.77 | 0.44 | 21.16 | 2.94 |
| GenEval w/ KL | 0.95 | — | — | 5.25 | 4.01 | 1.03 | 22.37 | 3.51 |
| OCR w/o KL | — | 0.93 | — | 5.13 | 3.66 | 0.58 | 21.79 | 3.15 |
| OCR w/ KL | — | 0.92 | — | 5.32 | 4.06 | 0.95 | 22.44 | 3.42 |
| PickScore w/o KL | — | — | 23.41 | 6.15 | 4.16 | 1.24 | 23.56 | 3.57 |
| PickScore w/ KL | — | — | 23.31 | 5.92 | 4.22 | 1.28 | 23.53 | 3.66 |

  讀法（Sec. 5.3）：GenEval 與 OCR 去掉 KL 時 DrawBench 品質與偏好分數明顯下降；PickScore 任務去掉 KL 不影響品質指標（作者認為可能因 PickScore 與評估指標重疊），但視覺多樣性崩潰，不同種子產生幾乎相同的影像（Fig. 6）。頁面不說「w/ KL 所有指標都較高」：w/ KL 的 Aesthetic 5.25 仍低於 SD3.5-M 的 5.39，PickScore 任務 w/ KL 的 Aesthetic 5.92 低於 w/o KL 的 6.15。
- **對照方法（Fig. 4、App. C.1，Fig. 8，Table 5–6）**：同組大小下比較 SFT、Flow-RWR、Flow-DPO 與其線上版本（線上版每 40 步更新收集模型）；Flow-GRPO 優於全部。線上 DPO 優於離線 DPO；DPO 的 $\beta$ 太小可能導致訓練崩潰。RWR 沒有隨訓練改善。DDPO（經 ODE 轉 SDE 改用於流模型）獎勵上升較慢且後期崩潰；ReFL（反傳到 $t\in[30,40]$ 的隨機晚期步）被超越，且 GRPO 不需要可微分獎勵。ORW：Table 5 在 step 960 時 23.05 對 29.89；Table 6 CLIP 28.40 對 30.18、Diversity 0.97 對 1.02（SD3.5-M 27.99、0.96）。
- **噪聲等級 $a$（Fig. 7b，只有 OCR 任務）**：$a=0.1$ 探索不足、獎勵上升慢；增加到 0.7 加快上升；0.7 到 1.0 沒有額外好處；再增加會使影像品質下降，獎勵為零、訓練失敗。Fig. 7 圖說：$a=0.7$ 使 OCR 準確率最高。
- **組大小 $G$（Fig. 5，PickScore）**：$G=12$ 與 $G=6$ 訓練不穩定並最終崩潰，$G=24$ 全程穩定；作者認為小組產生不準確的優勢估計、變異增加。
- **初始噪聲（App. C.3，Fig. 10）**：每個 rollout 用不同初始噪聲，比相同初始噪聲持續得到較高獎勵。
- **泛化（Table 3–4）**：Table 4 未見物體類別 Overall 0.64 → 0.90；計數訓練 2–4 個，5–6 個由 0.13 → 0.48、12 個由 0.02 → 0.12。Table 3（T2I-CompBench++）：七項中六項提升（例：2D-Spatial 0.2850 → 0.5447），Texture 由 0.7338 降為 0.7236。頁面照實寫出 Texture 下降。
- **FLUX.1-Dev（App. C.4，Table 7）**：PickScore 22.62 → 23.97，DeQA 4.31 → 4.24（略降），其餘上升。
- **KL 學習曲線（App. C.5，Fig. 12）**：KL 在訓練初期使獎勵上升較慢，但能達到無 KL 版本的高獎勵並保持品質。

#### 步驟 10：實作細節（App. B.3–B.4）與常見錯誤

| 項目 | 值 | 出處 |
| --- | --- | --- |
| 訓練取樣步數 / 評估步數 | 10 / 40（評估取樣器未寫；C.7 質性圖為 40 步 ODE） | B.3、C.7 |
| 組大小 $G$ | 24 | B.3 |
| 噪聲等級 $a$ | 0.7 | B.3 |
| 解析度 | 512 | B.3 |
| KL 係數 $\beta$ | GenEval、文字渲染 0.04；PickScore 0.01 | B.3 |
| LoRA | $\alpha=64$、$r=32$ | B.3 |
| 硬體 | 24 張 NVIDIA A800 | B.4 |
| 裁剪範圍 $\varepsilon$、學習率、批次大小、時間格點 | 論文未寫 | — |

常見錯誤（頁面 S10，各一兩句）：
1. 把 Eq. 9 的 $\sqrt{\Delta t}$ 用在負的 $\Delta t$ 上（應取絕對值）。
2. 在 $t=1$ 直接計算 $\sigma_t$（除以零）。
3. 混淆兩個參考策略：機率比的分母是收集樣本的 $\pi_{\theta_\text{old}}$，KL 的對象是預訓練的 $\pi_\text{ref}$（Eq. 5）。Fig. 2 的圖中把 KL 寫成 $D_{KL}(\pi_\theta\mid\pi_{\theta_\text{old}})$（單豎線；已以 PDF 第 4 頁 300dpi 影像確認），與 Eq. 5 的 $\pi_\text{ref}$ 不同；頁面依 Eq. 5，並以註腳照圖指出。
4. 組內獎勵全相同時 std 為 0，沒有保護會產生 NaN。
5. 用訓練時的 10 步去評估（論文定量評估用 40 步，B.3；取樣器未寫）。
6. 同一組的影像用相同初始噪聲（App. C.3 顯示不同初始噪聲較好）。
7. 把 $\beta$ 設為 0 以求快：獎勵仍可上升，但 GenEval／OCR 品質下降、PickScore 多樣性崩潰（Table 2、Fig. 6）。

#### 步驟 11：限制（Sec. 6）

T2I 之外，影片生成仍有三個方向：獎勵設計、多個獎勵的平衡、規模化（影片資源需求大得多）。KL 正則化有幫助，但需要更長訓練，且某些提示詞仍偶爾出現 reward hacking。

---

## 2. 學習目標（驗收測試）

每章結尾以 `<details>`（摘要「檢查理解」）呈現該章檢查題；參考答案必須**在該章內文段落**寫出，不能只出現在元件、表格、圖或答案框。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明線上 RL 用在流匹配模型的兩個障礙 | 為什麼不能直接把線上 RL 套到流匹配模型上？ | ODE 取樣是確定性的，沒有探索用的隨機性，策略機率也難算；每個樣本要很多步去噪，收集資料慢（Sec. 1） | S1 |
| LO2 | 寫出 Rectified Flow 的插值與訓練目標 | $x_t$ 怎麼定義？速度的回歸目標是什麼？哪一端是資料？ | $x_t=(1-t)x_0+t\,x_1$；目標 $v=x_1-x_0$，損失 $\|v-v_\theta(x_t,t)\|^2$；$t=0$ 資料、$t=1$ 噪聲（Eq. 1–2） | S2 |
| LO3 | 把去噪對應到 MDP | 狀態、動作、轉移、獎勵各是什麼？ODE 取樣下策略是什麼？ | 狀態 $(c,t,x_t)$；動作 $x_{t-1}$；轉移確定性；獎勵只在 $t=0$ 給 $r(x_0,c)$，其餘為 0（Sec. 3）；ODE 下策略退化為確定性映射，沒有機率密度 | S2 |
| LO4 | 計算組相對優勢並說出它的性質 | Eq. 4 怎麼算？組內獎勵全相同時會怎樣？ | 同一提示詞 $G$ 張影像的獎勵減去組平均、除以組標準差（Eq. 4）；獎勵不全相同時組內優勢加總為 0；全相同時整組優勢為 0，替代目標項梯度為 0，KL 項仍有梯度；若組內二元獎勵獨立且成功率為 $p$，機率 $p^G+(1-p)^G$〔本頁推導〕 | S3 |
| LO5 | 說明 GRPO 不需要價值網路的取捨與組大小的作用 | 為什麼用 GRPO 而不是 PPO？$G$ 太小會怎樣？ | GRPO 以組統計取代價值網路，較省記憶體（Sec. 2）；$G=12$、6 時訓練不穩並崩潰，$G=24$ 穩定，作者認為小組使優勢估計不準（Sec. 5.3、Fig. 5） | S3 |
| LO6 | 解讀機率比與裁剪替代目標 | $\rho$ 是什麼？Eq. 5 的 min 與 clip 何時使替代目標項的梯度為 0？ | $\rho$ 是目前模型與收集樣本時的模型 $\pi_{\theta_\text{old}}$ 對同一動作的機率比，第一次更新前為 1；$\hat A>0$ 且 $\rho>1+\varepsilon$，或 $\hat A<0$ 且 $\rho<1-\varepsilon$ 時替代目標項梯度為 0（KL 項仍有），相反方向裁剪不限制〔本頁推導〕；KL 對象是 $\pi_\text{ref}$（Eq. 5） | S4 |
| LO7 | 說明確定性取樣不符合 GRPO 的兩個理由 | 為什麼 ODE 取樣不能直接用？ | (1) 機率比需要 $p(x_{t-1}\mid x_t,c)$，確定性下要計算速度場的散度（各分量對自身座標偏導數的總和），高維時昂貴；(2) RL 需要探索，隨機性減少大幅降低訓練效率（Sec. 4） | S5 |
| LO8 | 說明 ODE 轉 SDE 為什麼保持邊際分布，以及前提 | 新 SDE 的漂移怎麼來？分數從哪裡來？前提是什麼？ | 讓描述密度隨時間變化的 Fokker–Planck 方程與 ODE 的連續方程相等，得漂移 $v+\frac{\sigma^2}2\nabla\log p$，再取反向時間得 Eq. 7；分數由速度換算 $\nabla\log p_t=-x/t-(1-t)v/t$（Eq. 27）。前提：$v_t$ 是真實邊際的速度（完美模型）、$p_t$ 足夠平滑、$t=1$ 奇異；在此前提下 $t\in(0,1)$ 上任何 $\sigma_t$ 都保持邊際，$\sigma_t=0$ 退回 ODE〔本頁推導〕 | S5 |
| LO9 | 寫出 Eq. 9 的更新與高斯策略，並說出離散化的影響 | 一步 SDE 取樣的平均與標準差是什麼？步數少時會怎樣？ | 平均 $x_t+[v+\frac{\sigma^2}{2t}(x_t+(1-t)v)]\Delta t$，標準差 $\sigma_t\sqrt{|\Delta t|}$，$\sigma_t=a\sqrt{t/(1-t)}$（Eq. 9）；$\Delta t<0$ 時開根號取絕對值、$t=1$ 時 $\sigma$ 發散需處理〔本頁推導〕；有限步會使分布偏離精確邊際，本頁玩具中 $T=5$ 的 ODE 已明顯偏離，$a=2.5$、$T=5$ 時 SDE 數值發散 | S6 |
| LO10 | 寫出 KL 閉式並說明它的權重 | 兩個高斯策略的 KL 是什麼？哪些步的權重最大？$a$ 如何影響？ | $\frac{|\Delta t|}2\big(\frac{\sigma_t(1-t)}{2t}+\frac1{\sigma_t}\big)^2\|v_\theta-v_\text{ref}\|^2$（Sec. 4）；均勻格點上最接近資料端的步權重最大；$(1/a+a/2)^2$ 在 $a=\sqrt2$ 最小，$a$ 小於 $\sqrt2$ 時越小越敏感，大於時又變大〔本頁推導〕；論文 $\beta$：GenEval／OCR 0.04、PickScore 0.01（App. B.3） | S7 |
| LO11 | 說明去噪步數縮減與噪聲等級的實驗結論 | 訓練與推論各用幾步？效果如何？$a$ 怎麼選？ | 訓練 10、推論 40；40→10 三任務超過 4 倍加速（GPU 小時）、最終獎勵不受影響，5 步不一定更快（Sec. 5.3、Fig. 7a、Fig. 9）；OCR 任務中 $a=0.1$ 太慢、0.7 最好、到 1.0 無額外好處，再大會失敗（Fig. 7b） | S8 |
| LO12 | 讀出主要結果與 KL 的作用 | 三個任務的提升？去掉 KL 會怎樣？ | GenEval 0.63→0.95、OCR 0.59→0.92（w/ KL）、PickScore 21.72→23.31（w/ KL）；去掉 KL：GenEval／OCR 的 DrawBench 品質下降，PickScore 多樣性崩潰；KL 不等同提早停止（Table 2、Sec. 5.3、C.5） | S9 |
| LO13 | 辨識實作上的常見錯誤 | 說出三個實作 Flow-GRPO 時容易犯的錯誤與正確做法 | 例：$\sqrt{\Delta t}$ 應取 $|\Delta t|$；$t=1$ 的 $\sigma_t$ 會除以零；比值分母是 $\pi_{\theta_\text{old}}$、KL 對象是 $\pi_\text{ref}$；std 為 0 時需保護；評估用 40 步；同組用不同初始噪聲；$\beta=0$ 會造成 reward hacking（步驟 10） | S10 |

---

## 3. 頁面大綱

單欄閱讀（文字欄 680–720px），元件放在加寬圖版（最大 1080px）。章節編號以 JetBrains Mono 顯示（`01 · …`），標題用陳述句。視覺記號：**時間軸條**（左端 $t=1$ 噪聲、右端 $t=0$ 資料，均勻刻度 $T$ 格），在 Hero、S6、S8、C3、C4 重複出現，固定「噪聲在左、資料在右」。

| # | id | 標題 | 內容（依序；每個術語與符號在使用前定義） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Flow-GRPO：用線上強化學習訓練流匹配模型〉 | 副標「把確定性的取樣改成每個時刻分布不變的隨機取樣，再以同一提示詞的一組影像互相比較來更新模型。」三張事實卡，每張自帶術語解釋：①「流匹配模型以常微分方程（ODE）把噪聲確定地變成影像；本文把它換成隨機微分方程（SDE），每個時刻的分布不變，但每一步多了隨機性，可用來探索。」②「GRPO（組相對策略最佳化）：同一提示詞產生 24 張影像，以『獎勵減組平均、除以組標準差』作為每張的優劣分數，不需要另外訓練價值網路。」③「在 GenEval（以物體偵測自動檢查物體數量、顏色、位置是否符合提示詞的測驗）上，SD3.5-M 由 0.63 提升到 0.95；訓練取樣只用 10 步，推論仍用 40 步。」靜態時間軸條（10 格，無動畫）。連結：arXiv、`../index.html#paper-438`、研究筆記總覽 `flowgrpo-lineage-hub.html`、程式碼 | 1 螢幕 |
| S1 | `problem` | 01 · 線上 RL 需要隨機取樣與便宜的樣本 | 線上 RL 的字面定義（用目前模型產生樣本、依獎勵更新、再產生）；T2I 的兩個弱點（組合、文字）；兩個障礙（LO1 答案）；先前工作：DDPO（把 PPO 式 RL 用在擴散模型的方法）、DPO（用「較好／較差」成對資料直接調整模型、不需線上取樣的方法），各以括號字面定義；三個貢獻以字面寫：「把取樣改成隨機」「訓練時少用去噪步」「加入使模型不偏離原模型太遠的懲罰（KL 散度，衡量兩個機率分布差多少的量），避免 reward hacking（獎勵上升但影像品質或多樣性下降）」；小符號表（$x_0,x_1,t,x_t,v,c$）。延伸閱讀：`flowgrpo-lineage-hub`、`rl-finetuning-text-to-image-diffusion` | 400–500 字 |
| S2 | `flow` | 02 · 流匹配沿直線插值並回歸速度 | Eq. 1、Eq. 2、Eq. 6；$t$ 的方向；ODE 的一對一映射；MDP 對應表（Sec. 3）與「ODE 下策略退化為確定性，S5 起改為隨機」（LO2、LO3 答案）；步索引 $t$ 與連續 $t$ 同字母。延伸閱讀：`flowgrpo-flow-matching-backbone`、`flow-matching-velocity-target`、`rl-mdp-q-learning-basics`、`flowgrpo-denoising-mdp-and-grpo` | 400–500 字 |
| S3 | `advantage` | 03 · GRPO 以同組影像的相對獎勵作為優勢 | 頁面明寫前向引用：「本節先假設每一步取樣已經是隨機的；第 05 節說明如何做到」；Eq. 3（KL 與 $\pi_\text{ref}$＝預訓練模型 在此定義，$\beta$＝懲罰的權重）；Eq. 4 與兩個性質（LO4）；GRPO 與 PPO 的差別（PPO 需要另一個網路估計每個狀態的期望獎勵，即價值網路）、組大小實驗（LO5）；C1；C1 後一段寫玩具觀察。延伸閱讀：`rlvr-grpo-credit-assignment` | 450–550 字 + C1 |
| S4 | `clip` | 04 · 機率比與裁剪限制單批資料造成的改變 | $\pi_{\theta_\text{old}}$、$\rho$ 的字面說明；頁面明寫前向引用：「這裡先把 ρ 當作一個數；第 06 節說明每一步的機率如何計算」；第一次更新前 $\rho=1$；$\rho\hat A$ 增大代表優勢為正的動作機率增大；Eq. 5（`aligned`）、$\varepsilon$；裁剪的導數規則與 KL 項仍有梯度（LO6）；C2 | 400–500 字 + C2 |
| S5 | `sde` | 05 · 邊際分布相同的 SDE 帶來隨機性 | 確定性取樣的兩個問題，散度一句字面定義（LO7）；在 Eq. 7 之前定義擴散係數 $\sigma_t$（控制每一小段時間加入多少噪聲）與論文的選擇 $\sigma_t=a\sqrt{t/(1-t)}$、噪聲等級 $a$（純量超參數，論文 0.7）；Wiener 過程（每一小段時間加入獨立高斯噪聲的隨機過程）、分數 $\nabla\log p_t$、Fokker–Planck 各一句字面定義；推導鏈五步（Eq. 12–21 放 `<details>`，正文保留 Eq. 7、Eq. 27、Eq. 8）；前提與結論（LO8） | 450–550 字 |
| S6 | `discrete` | 06 · 離散化後每一步是高斯分布 | Eq. 9；回指 S5 的 $\sigma_t$、$a$；「實作注意」callout（$|\Delta t|$、$t=1$，〔本頁推導〕）；高斯策略的平均與標準差，回答 S4 的「$\rho$ 怎麼算」（兩個高斯密度的比）；玩具說明（一維、兩個高斯混合的資料分布，速度場有閉式，等於完全正確的模型）；$W_1$ 字面定義「兩條累積分布曲線之間的面積，越小代表兩個分布越接近」；C3；C3 後一段寫玩具觀察（LO9 答案中的玩具句）與「不重現論文」 | 500–600 字 + C3 |
| S7 | `kl` | 07 · KL 懲罰有閉式，權重集中在接近資料的步 | Sec. 4 KL 閉式；$c(t)$ 的形狀、$a$ 的非單調影響、$t=1$ 標準差最大（LO10，含 $\beta$ 值）；C4 | 350–450 字 + C4 |
| S8 | `reduction` | 08 · 訓練只用 10 步取樣，推論保留 40 步 | Denoising Reduction 定義；$G\times T$ 的取樣次數〔本頁推導〕與「GPU 小時加速不是同一量」；Fig. 7a、Fig. 9、App. D 的結論；噪聲等級（OCR，Fig. 7b）（LO11）；回指 C3 的 $T$ 選項（玩具只顯示步數少時分布誤差變大，不顯示訓練速度）；靜態 SVG `#s8-cost` | 350–450 字 + 靜態 SVG |
| S9 | `results` | 09 · 實驗：三個任務都提升，KL 抑制 reward hacking | 三個任務的獎勵定義（步驟 8，含 OCR、PickScore、DrawBench 的字面定義）；Table 1 摘錄：GPT-4o、FLUX.1 Dev、SD3.5-L、SANA-1.5 4.8B、SD3.5-M、SD3.5-M+Flow-GRPO 六列的全部七欄（全表 `<details>`）；Table 2 全表；KL 的讀法與兩個反例；對照方法（SFT、RWR、DPO、DDPO、ReFL、ORW 各以括號一句定義）、組大小、初始噪聲、泛化（含 Texture 下降）、FLUX（LO12） | 600–700 字 + 2 表 |
| S10 | `impl` | 10 · 實作設定與常見錯誤 | 符號總表（可收合）；步驟 10 的表；7 條常見錯誤（LO13）；延伸閱讀：`flow-ode-to-sde-for-rl`、`flowgrpo-design-choices-and-results` | 350–450 字 + 表 |
| S11 | `history` | 11 · 研究脈絡 | 時間線（6.4）＋一段話：擴散模型與分數 SDE → 流匹配／Rectified Flow 與統一框架 → 擴散模型的 RL 與偏好對齊 → LLM 的 GRPO → 本文 | 時間線 + 100–150 字 |
| S12 | `next` | 12 · 限制與延伸閱讀 | 步驟 11；筆記清單（6.4）；arXiv、程式碼 | 200 字 + 清單 |

---

## 4. 互動元件規格

### 4.0 共用：PRNG、高斯、玩具資料分布（C1、C3 共用，與 `core.py` 逐式相同）

- **PRNG**：`mulberry32(seed)`，標準寫法 `a=(a+0x6D2B79F5)|0; t=Math.imul(a^(a>>>15),1|a); t=(t+Math.imul(t^(t>>>7),61|t))^t; return ((t^(t>>>14))>>>0)/4294967296;`。`gauss(r)`：Box–Muller，取 $u_1,u_2$，回傳 $\sqrt{-2\ln(1-u_1)}\cos(2\pi u_2)$（不快取 sin 分支）。預設 seed 1；「重新抽樣」每按一次 seed 加 1。
- **玩具資料分布**（推定，為可視化而選）：一維 $x_0\sim0.4\,\mathcal N(-1.2,0.3^2)+0.6\,\mathcal N(1.2,0.3^2)$，噪聲 $x_1\sim\mathcal N(0,1)$。精確邊際 $p_t(x)=\sum_kw_k\mathcal N\big(x;(1-t)\mu_k,(1-t)^2s_k^2+t^2\big)$〔本頁推導：高斯的線性組合〕。精確統計（$t=0$）：平均 0.2400、標準差 1.2134、$P(x>0)=0.6000$。
- **〔本頁推導〕精確速度場**：$v_t(x)=\mathbb E[x_1-x_0\mid x_t=x]=\big(x-\mathbb E[x_0\mid x_t=x]\big)/t$。各成分的後驗平均 $\mu_k+\frac{(1-t)s_k^2}{(1-t)^2s_k^2+t^2}\big(x-(1-t)\mu_k\big)$，以責任值（log-sum-exp 計算）加權。這個形式在 $t=1$ 有定義（不使用 Eq. 26 的 $1/(1-t)$），只有 $t=0$ 時無定義；模擬中速度只在 $t=j/T$、$j\ge1$ 計算，不會碰到 $t=0$。
- **頁面文字**：「本頁的速度場由玩具分布直接算出，等於一個完全正確的模型；論文的 $v_\theta$ 是學出來的，有誤差。」

### C1 · 組相對優勢與組大小（S3）

- **教的概念**：Eq. 4；組內優勢加總為 0；全相同組的替代目標項沒有梯度；$G$ 越小，優勢估計誤差越大（Sec. 5.3 的現象，玩具版）。
- **輸入**：分段按鈕「獎勵類型」：連續（玩具 $R\sim\mathcal N(0.5,0.15^2)$）／二元（成功率 $p$；二元獎勵的簡化，例如單一物件是否畫對；論文的 GenEval 獎勵有部分給分，不是二元），預設連續；滑桿 $G\in\{2,3,4,6,8,12,16,24,32,48\}$（以索引 0–9 儲存，預設索引 7 即 24；`aria-valuetext`「G = 24」）；滑桿 $p$，整數 `pp`∈[2,98]，$p=pp/100$（預設 30，只在二元模式啟用，連續模式 `disabled` 並標示「連續模式不使用 p」）；按鈕「重新抽樣」。
- **假設**（頁面寫出）：組內 $G$ 個獎勵獨立同分布。
- **計算**（全部即時）：
  1. **單組**：`r=mulberry32(seed)`，依序抽 $G$ 個獎勵（連續：$0.5+0.15\,$gauss；二元：$u<p$ 為 1 否則 0），算平均、母體標準差、$\hat A$；std $<10^{-12}$ 時 $\hat A\equiv0$（推定）。
  2. **全相同的機率**：二元 $p^G+(1-p)^G$（閉式）；連續為 0。
  3. **估計誤差曲線**：對 10 個 $G$ 值各做 $K=2000$ 組（每個 $G$ 重新以 `mulberry32(7)` 開始），$\mathrm{RMSE}=\sqrt{\frac1{KG}\sum(\hat A^i-z^i)^2}$，$z$ 為以真實分布標準化的值（連續 $(R-0.5)/0.15$；二元 $(R-p)/\sqrt{p(1-p)}$），以及模擬的全相同比例。總抽樣數 $3.1\times10^5$；在切換類型或 $p$ 的 `change` 後以 `setTimeout` 分 10 片（每片一個 $G$）計算，新的請求以 token 取消舊的。拖動 $p$ 時只更新單組與閉式機率，曲線改為淡色並標「p 已變更，曲線更新中」，完成後恢復。
- **顯示**（SVG）：
  - 上：$G$ 根長條，縱軸 $[0,1]$（刻度 0、0.5、1），超出範圍的連續獎勵以三角形標在邊緣。
  - 中：同序 $G$ 根發散長條為 $\hat A$，縱軸固定 $[-4,4]$（刻度 −4、−2、0、2、4；母體 std 下 $|\hat A|\le\sqrt{G-1}<7$，超出 4 時以三角形標在邊緣）。正值 `--accent` 實心、負值 `--c-ref` 斜線圖樣（方向＋圖樣雙重編碼）；零線標「組平均」。全相同時中圖顯示「整組獎勵相同：std = 0，所有優勢為 0；替代目標項沒有梯度，KL 項仍有。」
  - 下：RMSE 對 $G$，橫軸對數 $[2,48]$（刻度 2、6、12、24、48），左縱軸 $[0,1]$（刻度 0、0.5、1）；10 點折線＋圓點，目前 $G$ 以空心大圓標示；二元模式另畫「全相同機率」（方形點、虛線），右縱軸 $[0,1]$。
  - 長條最小寬 5px；$G=48$ 在容器寬 < 300px 時，上、中兩圖改為橫向捲動容器。
  - 讀數（JetBrains Mono）：組平均、組標準差、ΣA（$|x|<10^{-12}$ 顯示 0）、ΣA²、目前 $G$ 的 RMSE、全相同機率（閉式與模擬）。live region 只放一句：「G = 24，RMSE 0.254，全相同機率 0.000」。
- **數值保護**：$p\in[0.02,0.98]$，$z$ 的分母 $\ge0.14$；std 保護如上。
- **讀者應觀察到**（皆為玩具結果）：(1) 獎勵不全相同時 ΣA = 0、ΣA² = G；(2) 連續模式 RMSE 由 $G=2$ 的 0.923 降到 $G=24$ 的 0.254、$G=48$ 的 0.179；(3) 二元 $p=0.05$ 時，$G=24$ 仍有約 0.29 的組整組優勢為 0（閉式 0.2920）；$p=0.3$ 時 $G=24$ 只有 0.0002。頁面接著寫：論文 Fig. 5 的崩潰是真實訓練的結果，玩具只顯示估計誤差隨 $G$ 的變化，不模擬訓練。
- **390px**：三圖縱向堆疊（高 120／140／160px），控制在圖下。

### C2 · 裁剪替代目標（S4）

- **教的概念**：Eq. 5 的 $\min(\rho\hat A,\mathrm{clip}(\rho,1-\varepsilon,1+\varepsilon)\hat A)$ 與它的導數。
- **輸入**（全部以整數儲存，避免浮點累加）：$\hat A$＝`ai/10`，`ai`∈[−20,20]（預設 10）；$\varepsilon$＝`ei/100`，`ei`∈[5,50]（預設 20，標「論文未寫 ε，0.2 為推定」）；$\rho$＝`ri/100`，`ri`∈[50,150]（預設 100）。$\varepsilon$ 是頁面共用狀態，C4 讀同一個值。
- **計算**：$L(\rho)=\min(\rho\hat A,\mathrm{clip}(\rho)\hat A)$；導數：若 $\rho\hat A\le\mathrm{clip}(\rho)\hat A+10^{-9}$ 則 $\hat A$，否則 0。曲線取 $[0.5,1.5]$ 的 201 個等距點，另插入 $1-\varepsilon$、$1+\varepsilon$ 兩個精確點（排序後繪製），使折點準確。$\hat A=0$ 時曲線為 0 線、導數 0。
- **顯示**：橫軸 $\rho\in[0.5,1.5]$（刻度 0.5、0.8、1、1.2、1.5 以外依 ε 標出 1±ε）；縱軸固定 $[-3,3]$（刻度 −3…3）；未裁剪 $\rho\hat A$ 細虛線、$L$ 粗實線；$[1-\varepsilon,1+\varepsilon]$ 淡底色帶並標「裁剪範圍」；替代目標項梯度為 0 的區段下方加文字「替代目標項梯度 0」。目前 $\rho$ 以圓點標示。讀數：$L$、∂L/∂ρ、狀態文字（「未裁剪」／「已裁剪：替代目標項梯度 0，KL 項不受影響」）。
- **讀者應觀察到**：$\hat A=1,\varepsilon=0.2$：$\rho=1.3$ 的 $L=1.2$、梯度 0；$\rho=0.7$ 的 $L=0.7$、梯度 1。$\hat A=-1$：$\rho=0.7$ 的 $L=-0.8$、梯度 0；$\rho=1.3$ 的 $L=-1.3$、梯度 −1（裁剪不限制這個方向的更新）。
- **390px**：圖寬 100%、高 200px；三個滑桿縱向堆疊。

### C3 · ODE 與 SDE：同樣的邊際分布（玩具，S6）

- **教的概念**：Eq. 8／9 在連續時間保持邊際（前提見 S5）；$a=0$ 時與 ODE 完全相同；有限步離散化產生誤差；$a$ 很大且 $T$ 很小時數值發散（玩具現象）。
- **輸入**：分段按鈕「取樣器」：ODE／SDE（預設 SDE）；滑桿 $a$＝`ak/20`，`ak`∈[0,50]（$a\in[0,2.5]$，預設 14 即 0.70，標「論文 a = 0.7」；ODE 模式 `disabled`）；分段按鈕 $T\in\{5,10,20,40\}$（預設 10）；滑桿「顯示時刻」$j\in[0,T]$ 整數（$t_j=1-j/T$，預設 $j=T$ 即 $t=0$；改 $T$ 時重設為 $T$）；勾選框「同時顯示 ODE」（預設開）；「重新抽樣」。
- **計算**（4.0 的玩具；$M=4000$ 個粒子）：
  1. `r=mulberry32(seed)`；$x^{(m)}=$ gauss，$m=0..M-1$。
  2. 對 $j=0..T-1$：$t=1-j/T$，$\Delta t=-1/T$；$\sigma=a\sqrt{t'/(1-t')}$，$t'=t$（$t<1$）或 $1-1/T$（$t=1$，推定）；漂移中的 $\sigma^2/(2t)$ 與 $(1-t)$ 用真實 $t$。ODE 時 $\sigma=0$ 且不抽噪聲。對每個粒子 $m$（依序）：$v=v_t(x)$，$x\leftarrow x+\big[v+\frac{\sigma^2}{2t}(x+(1-t)v)\big]\Delta t+\sigma\sqrt{|\Delta t|}\,\epsilon$，$\epsilon=$ gauss（只在 SDE）。
  3. 保存每個 $j$ 的快照（$T+1$ 個長度 $M$ 的 `Float64Array`）。ODE 對照用同一 seed 的初始值（重新建立 `mulberry32(seed)`，只抽前 $M$ 個），快取 key `${T}|${seed}`。
  4. 指標：快照 $j$ 的經驗平均、標準差、$P(x>0)$；$W_1=\int_{-4}^{4}|F_\text{emp}(x)-F_{t_j}(x)|\,dx$（801 點梯形；$F$ 用 erf 的 Abramowitz–Stegun 7.1.26，誤差 $<1.5\times10^{-7}$）。
  5. 成本與排程：$T=40$ 時 SDE＋ODE 共 $3.2\times10^5$ 次速度計算。拖動以 `requestAnimationFrame` 合併；拖動 $a$ 時只重算 SDE 並畫直方圖，$W_1$ 讀數改淡色並標「已變更，更新中」，`change` 後才計算 $W_1$ 與更新 live region。建置時量測單次重算；> 30ms 時改為 `setTimeout` 分片（每片 5 步），圖版顯示「計算中」（不設 live）。
- **數值保護**：每步檢查粒子是否非有限或 $|x|>10^6$；成立時停止，圖上顯示「數值發散：第 j 步有粒子超過 10⁶」（本頁網格內不會發生，保護仍保留）。直方圖範圍 $[-4,4]$、64 格，範圍外粒子數以文字顯示「範圍外 n 個」。$W_1$ 積分範圍外的質量忽略（精確分布在範圍外的質量 $<10^{-4}$）。
- **顯示**：
  - 左（寬螢幕 60%）：快照 $j$ 的密度直方圖，縱軸為 count/(M·格寬)，固定 $[0,1]$（刻度 0、0.5、1；精確 $p_0$ 峰值約 0.80；超過 1 的格子截頂並加三角形）；SDE 實心淡色長條、ODE 對照 `--ink-2` 階梯輪廓、精確 $p_{t_j}$ `--c-ref` 粗虛線並直接標「精確邊際 p_t」。
  - 右（40%）：前 12 個粒子的軌跡，橫軸左 $t=1$ 到右 $t=0$，縱軸 $x\in[-4,4]$（刻度 −4、−2、0、2、4）；SDE `--accent` 細實線、ODE `--ink-2` 細虛線；目前 $j$ 以垂直線標示。
  - 讀數：平均、標準差、$P(x>0)$（SDE、ODE、精確三欄）；W₁（SDE、ODE）；參考句「4000 個精確樣本本身的 W₁ 約 0.009–0.043（平均 0.027），低於此範圍的差異來自抽樣」。live region 一句：「t = 0，SDE W₁ 0.043，ODE W₁ 0.071」。
- **讀者應觀察到**（皆為玩具結果，seed 1，$t=0$）：(1) $a=0$ 的 SDE 與 ODE 每個粒子完全相同；(2) $T=40$ 時 ODE（W₁ 0.0175）與 $a\le2.5$ 的 SDE（0.017–0.041）都在抽樣誤差範圍內，與「連續時間邊際相同」一致；(3) 步數少時偏離：$T=5$ 的 ODE 標準差 1.071（精確 1.213）、W₁ 0.143，10 個種子 0.139–0.159，明顯高於抽樣誤差；(4) $T=5$、$a=2.5$ 時 SDE 標準差 15.5、W₁ 2.02：數值發散，原因是漂移中 $\frac{\sigma^2}{2t}x\,\Delta t$ 項在大步長下的放大（第 2 步只看這一項，$x$ 的係數為 $1-a^2/2=-2.125$），加上 $t=1$ 的 σ 處理（推定，第 1 步 $\sigma=a\sqrt{T-1}$）；這是本頁玩具與推定處理的結果。頁面另寫：$T=10$ 時 SDE（$a=0.7$）的 W₁ 在 10 個種子中為 0.031–0.059，與抽樣誤差上緣重疊，本頁不據此比較 SDE 與 ODE 的優劣；論文中 $a$ 過大的失敗（Fig. 7b）發生在學習到的模型上，玩具速度場沒有誤差，不重現那個現象。
- **390px**：直方圖在上、軌跡圖在下（各寬 100%、高 180px）；讀數改為「SDE／ODE／精確」三行的兩欄 `<dl>`。

### C4 · 一步的高斯策略、機率比與 KL（S7）

- **教的概念**：Eq. 9 的一步是高斯分布；兩個策略只差速度 $\Delta v$ 時，平均差 $\delta=(1+\frac{\sigma_t^2(1-t)}{2t})\Delta v\,\Delta t$；KL 閉式與係數 $c(t)$；$a$ 小時機率比極端；$a$ 對 KL 係數的非單調影響。
- **輸入**：滑桿 $a$＝`ak/20`，`ak`∈[2,40]（$a\in[0.1,2.0]$，預設 14）；分段按鈕 $T\in\{10,40\}$（預設 10）；滑桿「步 $j$」$\in[0,T-1]$（由 $t=1-j/T$ 走到 $t-1/T$；預設 5；改 $T$ 時設為 $\lfloor T/2\rfloor$）；滑桿 $\Delta v$＝`dk/100`，`dk`∈[−100,100]（預設 20，標「一維；影像中為所有維度的平方和」）；滑桿 $z$＝`zk/10`，`zk`∈[−30,30]（預設 10；樣本 $x=\bar x'+\text{std}\cdot z$，$\bar x'$ 為另一策略的平均）。
- **計算**（全部閉式，`core.step_policy`、`core.kl_coeff`）：$t=1-j/T$；$\sigma$ 用 $t'$（$t=1$ 時 $1-1/T$），$\frac{\sigma(1-t)}{2t}$ 與 $\delta$ 中的 $(1-t)/(2t)$ 用真實 $t$（$t=1$ 時為 0）；std $=\sigma\sqrt{1/T}$；$\delta=(1+\frac{\sigma^2(1-t)}{2t})\Delta v\cdot(-1/T)$；$c_j=\frac1{2T}\big(\frac{\sigma(1-t)}{2t}+\frac1\sigma\big)^2$（**直接用閉式，不除以 $\Delta v^2$**，$\Delta v=0$ 時仍有定義）；$\mathrm{KL}_2=c_j\Delta v^2$；$\mathrm{KL}_1=\delta^2/(2\sigma^2/T)$（兩者顯示，selfTest 驗證相等；$\Delta v=0$ 時都為 0）；$\log\rho=\frac{(\text{std}\,z)^2-(\text{std}\,z-\delta)^2}{2\,\text{std}^2}$。
- **數值保護**：$a\ge0.1$、$t\ge1/T$ 保證 $\sigma>0$、std $>0$；顯示 $\rho$ 時 $\log\rho$ 截到 $[-50,50]$ 再取 exp，超出時顯示「> e⁵⁰」或「< e⁻⁵⁰」。
- **顯示**：
  - 上：兩個高斯密度（$\pi_\theta$ 實線 `--accent`、另一策略 $\pi'$ 虛線 `--c-ref`，各有文字標籤），橫軸以 $\bar x'$ 為原點、範圍 ±(4·std＋|δ|)，刻度顯示實際值；縱軸 $[0,1.1\cdot\text{峰值}]$，峰值 $=1/(\text{std}\sqrt{2\pi})$ 以讀數顯示（不畫縱軸刻度）；樣本 $x$ 以垂直線＋菱形標記。
  - 下：$c_j$ 長條圖（$T$ 根），縱軸對數 $[10^{-5},10^{2}]$（刻度每十倍一格），橫軸左 $t=1$、右 $t=0$；目前 $j$ 的長條加粗外框並標數值。
  - 讀數：t、σ、std、δ、KL₁、KL₂、log ρ、ρ，以及「與裁剪範圍比較」：ρ 是否在 [1−ε, 1+ε]（ε 讀 C2 的共用狀態，讀數寫出目前 ε）。live region 一句：「第 j 步，KL 0.00633，ρ 0.888」。
  - 頁面說明 $\pi'$ 可以是 $\pi_{\theta_\text{old}}$（算機率比）或 $\pi_\text{ref}$（算 KL），兩者數學相同。
- **讀者應觀察到**（〔本頁推導〕的閉式，非論文數字）：(1) 長條在最右（$j=T-1$）最高：$T=10,a=0.7$ 時 $c_0=0.0113$、$c_5=0.158$、$c_9=1.423$；最左一根最低，但那一步的 std 最大（0.664，對照 $j=9$ 的 0.0738）；(2) $a$ 由 0.7 減到 0.1，$j=9$ 的 KL 由 0.0569 增為 1.818，log ρ（$z=1$）由 −0.394 變為 −3.72（ρ ≈ 0.024，遠在裁剪範圍外）；(3) $T=10$ 時 $\sum_jc_j$：$a=0.1$ 97.97、0.7 3.062、1.4 1.932、2.0 2.171：$a$ 超過約 $\sqrt2$ 後係數又上升。
- **390px**：密度圖在上（高 160px）、長條圖在下（高 140px，$T=40$ 時每根最窄 4px，不足時橫向捲動）；五個滑桿縱向堆疊。

### 4.5 共同的無障礙、排版與互動規則

- **KaTeX**：靜態正文、`<label>`、表頭用 KaTeX；display 數學放在 `.katex-display{overflow-x:auto;overflow-y:hidden}`。Eq. 5、Eq. 9、KL 閉式各只有一份 DOM，以 `aligned` 寫成兩行（寬螢幕亦兩行），不用 CSS 切換兩份。`renderMathInElement` 的輸出加 `aria-hidden="true"`，並在外層 `span` 加 `aria-label`（Unicode 讀法，如「ρ 等於 p θ 除以 p θ old」），避免 MathML 與 HTML 重複朗讀。KaTeX 載入失敗（`window.katex` 不存在）時，`.math-fallback` 顯示每個式子預先寫好的 Unicode 純文字版本，原始 `$…$` 隱藏。
- **動態文字的 Unicode 對照**（讀數、SVG `<text>`、`aria-*`、live region；只用預組字元，**不得**出現 U+0300–U+036F）：

| 數學 | 動態文字 |
| --- | --- |
| $\hat A^i$ | A（標頭寫「優勢 A」） |
| $\rho$、$\varepsilon$、$\sigma_t$、$\delta$ | ρ、ε、σ、δ |
| $\Delta t$、$\Delta v$ | Δt、Δv |
| $\pi_\theta$、$\pi_{\theta_\text{old}}$、$\pi_\text{ref}$、$\pi'$ | πθ、πθ,old、πref、π′ |
| $p_t$、$W_1$、$c_j$ | p_t、W₁、c_j |
| $\bar x'$ | 平均 x′ |
| $\sum\hat A$、$\sum\hat A^2$ | ΣA、ΣA² |

  selfTest 以 `/[̀-ͯ]/` 檢查所有 `.readout`、`svg text`、`[aria-label]`、`[aria-valuetext]`。
- **非顏色編碼**：C1 正負優勢以方向＋斜線圖樣；C2 裁剪區段有文字標籤；C3 SDE 實心長條、ODE 階梯輪廓、精確虛線；C4 實線／虛線＋標籤；所有線有直接文字標籤。
- **對比**：見 6.2 色表（light／dark 數值）。
- **觸控**：不使用 hover；所有查值以滑桿完成；觸控區 ≥ 44px。
- **live region**：每個元件的 `aria-live="polite"` 只放 4.x 指定的一句摘要，在 `change` 或停止輸入 500ms 後更新（共用 debounce）；完整讀數不放 live；「計算中」「更新中」不設 live。`aria-valuetext` 隨滑桿即時更新。
- 分段按鈕 `role="radiogroup"`＋`role="radio"`／`aria-checked`，roving tabindex，方向鍵切換。
- `prefers-reduced-motion: reduce`：取消 transition；本頁沒有自動播放動畫。
- **390px 共同規則**：圖在上、控制在中、讀數在下；SVG 寬度等於容器（`ResizeObserver` 重畫），圖上文字 12–13px；Table 1／2 與符號總表放在 `<div class="table-scroll" tabindex="0" role="region" aria-label="表 2（可橫向捲動）">`，表上方加小字「← 左右捲動 →」（只在 `scrollWidth > clientWidth` 時顯示）；selfTest 檢查 `document.documentElement.scrollWidth <= innerWidth`。

---

## 5. 驗收數值表

全部取自 `verify/check_out.txt`（`check.py`，與頁面同一演算法、雙精度）。**容許誤差**：表中所有值 ±1e-4（表值四捨五入到 4 位；JS 與 Python 只差浮點加總順序與 `Math.exp`／`Math.log` 的最後一位）；C3 的 $W_1$ ±5e-4（801 點梯形對最後一位敏感）；C4 的值以 6 位有效數字列出，相對誤差 ±1e-5（6 位有效數字的捨入誤差最大 5e-6）；整數與二元序列必須完全相同。

**PRNG 與玩具基本量**

| 項目 | 值 |
| --- | --- |
| `mulberry32(1)` 前三個均勻數 | 0.627074、0.002736、0.527447 |
| `gauss` seed 1 前兩個 | 1.4043、1.2158 |
| $v_{0.5}(0.5)$、$v_1(-1)$、$v_{0.1}(1.0)$ | −1.0646、−1.2400、−1.2183 |
| $p_0(0)$、$p_0(1.2)$、$p_1(0)$ | 0.0004、0.7979、0.3989 |
| 精確 $t=0$：平均、標準差、$P(x>0)$ | 0.2400、1.2134、0.6000 |
| $\sigma$：$t=0.5,a=0.7$；$t=1,T=10$；$t=1,T=40$ | 0.7000、2.1000、4.3715 |

**C1**（seed 1，$G=24$）

| 項目 | 值 |
| --- | --- |
| 連續：$R_{0..2}$；平均；std；$\hat A_{0..2}$；$\sum\hat A^2$ | 0.7107、0.6824、0.4234；0.5272；0.1235；1.4858、1.2568、−0.8398；24.0000 |
| 二元 $p=0.3$：序列；平均；std；$\hat A(1)$、$\hat A(0)$ | `010001000000101101001110`；0.3750；0.4841；1.2910、−0.7746 |
| $G=2$、獎勵 (1,1) | 優勢 (0, 0) |

RMSE（seed 7，$K=2000$）與全相同比例（模擬／閉式）：

| $G$ | 連續 RMSE | 二元 0.3 RMSE | 0.3 全相同 模擬／閉式 | 二元 0.05 RMSE | 0.05 全相同 模擬／閉式 |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.9228 | 0.7022 | 0.5720 / 0.5800 | 0.8450 | 0.9005 / 0.9050 |
| 6 | 0.5156 | 0.4167 | 0.1160 / 0.1184 | 0.5751 | 0.7385 / 0.7351 |
| 12 | 0.3576 | 0.3204 | 0.0165 / 0.0138 | 0.4238 | 0.5425 / 0.5404 |
| 24 | 0.2536 | 0.2276 | 0.0015 / 0.0002 | 0.3150 | 0.2820 / 0.2920 |
| 48 | 0.1788 | 0.1583 | 0.0000 / 0.0000 | 0.2795 | 0.0760 / 0.0853 |

（其餘 $G$ 與 $p=0.5$ 見 `check_out.txt`，selfTest 以表中各列檢查。）

**C2**（$\varepsilon=0.2$）：$\hat A=1$：$\rho=0.7,0.9,1.0,1.1,1.3\to L=0.7,0.9,1.0,1.1,1.2$，梯度 1,1,1,1,0。$\hat A=-1$：$L=-0.8,-0.9,-1.0,-1.1,-1.3$，梯度 0,−1,−1,−1,−1。

**C3**（seed 1，$M=4000$，$t=0$；平均／標準差／$P(x>0)$／$W_1$）

| 設定 | 平均 | 標準差 | P(x>0) | $W_1$ |
| --- | --- | --- | --- | --- |
| ODE $T=5$ | 0.2720 | 1.0705 | 0.6178 | 0.1426 |
| ODE $T=10$ | 0.2554 | 1.1402 | 0.6075 | 0.0712 |
| ODE $T=40$ | 0.2438 | 1.1953 | 0.5998 | 0.0175 |
| SDE $T=10$ $a=0.1$ | 0.2602 | 1.1411 | 0.6100 | 0.0725 |
| SDE $T=10$ $a=0.7$ | 0.2315 | 1.1756 | 0.5975 | 0.0429 |
| SDE $T=10$ $a=1.5$ | 0.1986 | 1.1995 | 0.5830 | 0.0481 |
| SDE $T=10$ $a=2.5$ | 0.1093 | 1.2343 | 0.5487 | 0.1375 |
| SDE $T=5$ $a=0.7$ | 0.2401 | 1.1368 | 0.5988 | 0.0841 |
| SDE $T=5$ $a=2.5$ | 0.0085 | 15.4839 | 0.5030 | 2.0233 |
| SDE $T=40$ $a=0.1$ | 0.2438 | 1.1963 | 0.6020 | 0.0170 |
| SDE $T=40$ $a=0.7$ | 0.2086 | 1.2075 | 0.5877 | 0.0356 |
| SDE $T=40$ $a=1.5$ | 0.1992 | 1.2140 | 0.5830 | 0.0410 |
| SDE $T=40$ $a=2.5$ | 0.2167 | 1.2174 | 0.5903 | 0.0237 |

其他：$a=0$ 的 SDE 與 ODE（$T=10$）最大差 0.0（第一個粒子 1.4485）；SDE $a=0.7,T=10$ 第一個粒子 $j=0..3$：1.4043、1.1430、0.5522、−0.4426；$j=5$（$t=0.5$）的 $W_1$ 0.0297；ODE $T=40$、$j=20$ 的 $W_1$ 0.0147；seed 2：SDE 0.0368、ODE 0.0867；seed 3：0.0308、0.0693。精確樣本 $W_1$ 下限（seed 101–110，`check.py` 以混合分布直接抽樣）：平均 0.0269、範圍 0.0087–0.0428（頁面只引用這組數字，不重算）。

**C4**（$(j,T,a,\Delta v,z)\to\sigma$、std、$\delta$、KL、$\log\rho$）

| 輸入 | $\sigma$ | std | $\delta$ | KL | $\log\rho$ |
| --- | --- | --- | --- | --- | --- |
| (5,10,0.7,0.2,1) | 0.7 | 0.221359 | −0.0249 | 0.00632663 | −0.118813 |
| (0,10,0.7,0.2,1) | 2.1 | 0.664078 | −0.02 | 0.000453515 | −0.0305704 |
| (9,10,0.7,0.2,1) | 0.233333 | 0.0737865 | −0.0249 | 0.0569397 | −0.3944 |
| (9,10,0.1,0.2,1) | 0.0333333 | 0.0105409 | −0.0201 | 1.81805 | −3.7249 |
| (9,10,1.4,0.2,1) | 0.466667 | 0.147573 | −0.0396 | 0.0360037 | −0.304346 |
| (20,40,0.7,0.2,1) | 0.7 | 0.11068 | −0.006225 | 0.00158166 | −0.057825 |
| (39,40,0.7,0.2,1) | 0.11209 | 0.0177229 | −0.006225 | 0.0616847 | −0.412924 |

$c_j$（$\Delta v=1$）：$T=10$：$a=0.7$ 時 $c_0,c_5,c_9=0.0113379,0.158166,1.42349$，總和 3.06231；總和 $a=0.1$ 97.9709、$a=1.4$ 1.932、$a=2.0$ 2.17148。$T=40,a=0.7$：$c_0,c_{20},c_{39}=0.000654108,0.0395415,1.54212$，總和 5.18619。所有設定的最大值都在 $j=T-1$。$(1/a+a/2)^2$：$a=0.7$ 3.1633、$a=\sqrt2$ 2.0000。閉式 `kl_coeff` 與 KL₂/Δv² 的最大差 3.6e-15；$\Delta v=0$ 時 $c_5$（$T=10,a=0.7$）仍為 0.158166，KL₁＝KL₂＝0、log ρ＝0。KL 兩種寫法的最大相對差 7.0e-16。

**C3 多種子**（seed 1–10，$t=0$ 的 W₁ 平均／標準差／最小／最大；`check_out.txt` 末段）：ODE $T=5$ 0.1462／0.0068／0.1393／0.1587；SDE $T=5,a=0.7$ 0.0868／0.0050／0.0779／0.0959；SDE $T=5,a=2.5$ 2.0231／0.0170／2.0021／2.0589；ODE $T=10$ 0.0767／0.0062／0.0690／0.0867；SDE $T=10,a=0.7$ 0.0419／0.0091／0.0308／0.0591；SDE $T=10,a=2.5$ 0.1273／0.0121；ODE $T=40$ 0.0298／0.0089；SDE $T=40,a=0.7$ 0.0218／0.0095；SDE $T=40,a=2.5$ 0.0283／0.0093。這組數字只用於撰寫 C3 的觀察句，頁面不重算。

---

## 6. 實作計畫

### 6.1 檔案與共用外殼

- 單一檔案 `tut/438/build/index.html`；CSS、JS、時間線資料全部內嵌。上線時複製為 `tutorials/flow-grpo.html`。
- **直接沿用**最近的建置頁 `tut/447/build/index.html`（其外殼來自 `tutorials/prediction-targets-k-diff.html`）：CSS tokens（含深色模式三段覆寫、`pm-theme`）、字型載入、appbar、`#pm-bar` 導覽、章節目錄（sticky 左欄／窄螢幕橫列）、`.plate` 圖版、callout、`<details>` 檢查題、「本頁推導」「推定」pill、表格橫捲容器、KaTeX 0.16.9 設定與 `renderMath()`、`?test` selfTest 框架（`ok()` 計數、`#selftest` 輸出、`scrollWidth` 檢查）、時間線卡片列、SVG 輔助（`el`、`scaleLin`、`linePath`）、`schedule(fn)`、`debounce(fn,500)`、`radiogroup(el,onChange)`、`mulberry32`、`gauss`。
- 以下只列新增或變更。

### 6.2 新增 tokens、元件與連結

- 圖表色（沿用 447，值已由 447 build 取出；對比為在 `--surface` 上的計算值）：

| token | light | dark | 對比（light / dark） | 用途 |
| --- | --- | --- | --- | --- |
| `--accent` | #B4532A | #E58C66 | 4.99 / 6.57 | SDE、πθ、正優勢、L 曲線 |
| `--c-ref` | #3F6E8C | #8DB4D3 | 5.50 / 7.62 | 精確邊際、π′、負優勢 |
| `--ink-2` | #57534B | #B5B0A5 | 7.65 / 7.71 | ODE、樣本、次要線 |
| `--surface` | #FFFFFF | #1F1E1B | — | 圖版底 |
| `--surface-2` | #EEEBE3 | #2A2925 | — | 裁剪範圍底色帶 |

  SVG `<pattern id="hatch">` 的斜線 `stroke="currentColor"`，外層 `<g style="color:var(--c-ref)">`，深色模式自動換色。直方圖淡色填色用 `color-mix(in oklab, var(--accent) 35%, var(--surface))`，另以 1px `--accent` 描邊使長條邊界在兩種模式都可辨識（圖形對比靠描邊，≥ 3:1）。
- 新元件：時間軸條 `timeStrip(svg, T, jSel)`（Hero 靜態 10 格、S8 兩列、C3／C4 軸）；直方圖 `histogram(svg, data, lo, hi, bins)`；發散長條 `divBars(svg, values)`。
- S8 靜態 SVG `#s8-cost`（viewBox 0 0 640 160，< 600px 改為兩列上下堆疊、文字 12px）：上列 10 格「訓練取樣：T = 10（SDE）」，下列 40 格「推論：T = 40」，右側文字「每個提示詞 24 × 10 = 240 次 vs 24 × 40 = 960 次模型前向」。
- 連結：Hero「在關聯圖中查看」→ `../index.html#paper-438`；arXiv `https://arxiv.org/abs/2505.05470`；程式碼 `https://github.com/yifan123/flow_grpo`；筆記 → `<slug>.html`。

### 6.3 程式結構（單一 IIFE）

- 共用：`TOY={w:[0.4,0.6],mu:[-1.2,1.2],s:[0.3,0.3]}`；`comp(t)`、`density(x,t)`、`velocity(x,t)`、`sigmaT(t,a,T)`、`erf(x)`、`cdf(x,t)`、`w1(xs,t)`、`stats(xs)`；`advantages(R)`、`drawReward(r,kind,p)`、`trueZ`、`advMC(G,kind,p,seed,K)`、`pDegenerate`；`clipObj(rho,A,eps)`；`stepPolicy(j,T,a,dv,z)`、`klCoeff(j,T,a)`。函式與 `core.py` 一一對應，註解寫出對應的 Python 名稱。
- C1：`initC1()`，`state={kind,gi,pp,seed,token,stale}`；RMSE 曲線快取 key `${kind}|${pp}`。
- 頁面共用 `shared={ei:20}`（ε＝ei/100），C2 寫入、C4 讀取並在變更時重畫 C4。
- C2：`initC2()`，`state={ai,ri}`（整數）。
- C3：`initC3()`，`state={mode,ak,T,j,showOde,seed,stale}`；`simulate(a,T,seed,sde)` 回傳快照陣列與 `divergedAt`；ODE 快取 `${T}|${seed}`。
- C4：`initC4()`，`state={ak,T,j,dk,zk}`（整數）。
- `selfTest()` 新增斷言（容許誤差同第 5 節）：PRNG 前三個值；`velocity` 三點；`sigmaT` 三點；C1 seed 1 連續組的平均、std、$\hat A_0$、$\sum\hat A^2$，二元序列字串；`advMC(24,'continuous',.3,7)` 的 RMSE 0.2536、`advMC(24,'binary',.05,7)` 的全相同比例 0.2820；`pDegenerate(24,'binary',.05)=0.2920`；C2 第 5 節十個點；C3：$a=0$ 與 ODE 完全相同、SDE $a=0.7,T=10$ 的 $W_1$ 0.0429 與標準差 1.1756、ODE $T=5$ 的 $W_1$ 0.1426；C4 表中七列與 $c_j$ 總和；`stepPolicy(5,10,.7,0,1)` 的 KL₁、KL₂、log ρ 皆為 0 且 `klCoeff(5,10,.7)=0.158166`（Δv=0 不產生 NaN）；C2 在 `ri=120, ei=20, ai=10` 時狀態為「未裁剪」（邊界取未裁剪分支）；組合字元檢查（4.5）；`#s8-cost` 含「T = 10」「T = 40」文字；KaTeX 後備：移除 `window.katex` 後呼叫 `renderMath()`，`.math-fallback` 可見。輸出 `selfTest: N passed, M failed`。

### 6.4 研究筆記與時間線

- **本篇筆記**（`tut/438/notes/`，7 篇；括號為 frontmatter `title`）：總覽 `flowgrpo-lineage-hub`（Flow-GRPO 的研究脈絡總覽）；`flowgrpo-flow-matching-backbone`（Flow matching 與 rectified flow 作為生成骨幹）；`flowgrpo-diffusion-sde-background`（從 DDPM 到 score SDE 與 stochastic interpolants）；`flowgrpo-denoising-mdp-and-grpo`（Denoising 作為 MDP，以及 Flow-GRPO 的目標函數）；`rl-finetuning-text-to-image-diffusion`（文字生成圖像擴散模型的 RL 微調脈絡）；`flow-ode-to-sde-for-rl`（Flow 模型 RL 的 ODE-to-SDE 轉換）；`flowgrpo-design-choices-and-results`（Denoising Reduction、KL 與 Flow-GRPO 的實驗結果）。
- **hub 引用的既有筆記**（在 `scripts/notes_split/`，已上線於 `tutorials/`）：`flow-matching-velocity-target`（Flow matching 的速度目標與 rectified flow 的大規模使用）、`rlvr-grpo-credit-assignment`（RLVR、GRPO 與信用分配瓶頸）、`rl-mdp-q-learning-basics`（MDP、Q 函數與 Bellman backup：value-based RL 的基本記號）。
- 所有筆記連結為 `<slug>.html`（與 `tutorials/flow-grpo.html` 同目錄）。hub 內文「時間軸見 `timeline.json`」一句是筆記內部說明，頁面不引用、不轉錄 hub 內文。
- **延伸閱讀依 hub 的 6 步閱讀順序**：

| hub 步驟 | 筆記 | 放在哪一章 |
| --- | --- | --- |
| 1 生成骨幹 | `flowgrpo-flow-matching-backbone`；既有 `flow-matching-velocity-target` | S2 |
| 2 SDE 背景 | `flowgrpo-diffusion-sde-background` | S5 |
| 3 MDP 與 GRPO | `flowgrpo-denoising-mdp-and-grpo`（S2 的 MDP、S3 的優勢）；既有 `rl-mdp-q-learning-basics`（S2）、`rlvr-grpo-credit-assignment`（S3） | S2、S3 |
| 4 T2I 的 RL 微調 | `rl-finetuning-text-to-image-diffusion` | S1、S9 |
| 5 ODE 轉 SDE | `flow-ode-to-sde-for-rl` | S5、S6、S10 |
| 6 設計選擇與結果 | `flowgrpo-design-choices-and-results` | S8、S9、S10 |

  S12 列出 hub 加上表中全部筆記，順序同上，每項附 `title`。Hero 的筆記連結指向 `flowgrpo-lineage-hub.html`。
- **時間線**：`timeline.json`（13 筆，`year` 為整數，已包含 2505.05470）貼進 `<script type="application/json" id="timeline-data">`，只取 `year`、`title`、`authors_short`、`arxiv` 四欄；**不顯示** `role`（英文）與 `source`（內部的 `paper.txt §` 出處）。卡片的中文說明由 `TL_GROUP` 的類別名與連到的筆記提供。排序依 `year`，同年依 `arxiv` 字串。解析失敗時顯示「時間線資料尚未提供」。
- **`TL_GROUP`**（13 個 ID 全部指定，不留「其他」）：

| 類別 | arXiv ID | 卡片連到的筆記 |
| --- | --- | --- |
| 擴散模型與分數 SDE | 2006.11239（DDPM）、2010.02502（DDIM）、2011.13456（Score SDE） | `flowgrpo-diffusion-sde-background` |
| 流匹配與統一框架 | 2209.03003（Rectified Flow）、2210.02747（Flow Matching）、2303.08797（Stochastic Interpolants）、2403.03206（SD3，本文基礎模型） | `flowgrpo-flow-matching-backbone` |
| 生成模型的 RL 與偏好對齊 | 1910.00177（AWR，Flow-RWR 對照組的方法家族）、2305.13301（DDPO）、2305.18290（DPO）、2409.08861（Adjoint Matching） | `rl-finetuning-text-to-image-diffusion` |
| LLM 的 GRPO | 2402.03300（DeepSeekMath） | `flowgrpo-denoising-mdp-and-grpo` |
| 本文 | 2505.05470 | `flowgrpo-design-choices-and-results` |

  建置時以程式檢查 `timeline.json` 的每個 ID 都在 `TL_GROUP` 中；遺漏者歸入「其他」並在建置回報中列出。S11 的敘事段落依上表五類的順序。

### 6.5 文案規則（新增部分）

- 用詞固定：「流匹配（flow matching）」「去噪步數縮減（Denoising Reduction）」「組相對優勢」「裁剪」「噪聲等級 $a$」「邊際分布」；第一次出現附英文。「步」指離散去噪步，「時刻」指連續 $t$。
- 三種陳述分開：論文結果（附出處）、「作者認為」、〔本頁推導〕；玩具數字一律以「本頁玩具中」開頭，不與論文數字同句。
- C1、C3 圖版底部各一行：「讀數由本頁的一維玩具依 4.0 的公式即時計算，不是論文的模型，也不重現論文數字。」C4 底部：「閉式由 Eq. 9 與 Sec. 4 的 KL 式即時計算；一維、Δv 為示意值。」
- 不寫「Flow-GRPO 在所有指標上都更好」；Table 2 與 Table 3 的反例照實寫出。

### 6.6 建置後自我檢查

依 447 頁流程：390／768／1024／1440 截圖＋深色、鍵盤 focus、reduced-motion（headless Chrome 無法連 CDN，截圖時以本機 KaTeX `scratchpad/katex/0.16.9` 代換）；逐項評分並修正；`?test` 全數通過；逐條核對 13 個學習目標的參考答案出現在指定章節內文；以 Performance 面板確認 C3 $T=40$ 單次重算時間（> 30ms 則啟用分片）；深色模式截圖人工檢查 hatch、直方圖描邊、虛線在 `--surface` dark 上可辨識（selfTest 無法檢查深色對比）。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具與論文的差距 | 一維、精確速度場、無學習誤差；「a 大時 T=40 仍準」是玩具現象，與 Fig. 7b 的 a 過大失敗無關；T=10 的 SDE／ODE 差距與抽樣誤差重疊（10 個種子），已不作比較；C1 不模擬訓練。 | 每個元件標「玩具」；觀察項加限定；C3 後段落明寫不重現論文。 |
| R2 | Eq. 9 的 $\sqrt{\Delta t}$ 與正負號 | 論文寫 $\sqrt{\Delta t}$、KL 分母 $2\sigma_t^2\Delta t$，取樣方向 $\Delta t<0$。 | 頁面照錄原式，〔本頁推導〕註明取絕對值。 |
| R3 | $t=1$ 的 $\sigma_t$ | 論文未說明處理方式。 | 玩具用 $t'=1-1/T$（推定）；常見錯誤 2。若 notes 或程式碼研究確認官方做法，只在註腳寫「官方程式碼」，不寫成論文內容。 |
| R4 | 未寫明的超參數 | $\varepsilon$、學習率、批次、std 定義、std 保護常數、時間格點（SD3 預設可能非均勻）。 | 表中寫「論文未寫」；玩具用均勻格點、$\varepsilon=0.2$、母體 std（皆推定）。 |
| R5 | 記號不一致 | Eq. 3 求和 $t=0..T$、Eq. 5 平均 $t=0..T-1$；獎勵 $r$ 與比值 $r^i_t$ 同字母；插值 $\beta_t$ 與 KL $\beta$ 同字母；Fig. 2 的 KL 對象寫 $\pi_{\theta_\text{old}}$（已以 PDF 影像確認）；Sec. 3 的 MDP 用整數 $t$、Eq. 9 用連續 $t$；App. A 稱 Eq. 17 為「forward SDE」但它與 Eq. 21 都在描述同一組邊際。 | 照錄並以註腳指出；頁面以 $\rho$ 表示比值。 |
| R6 | 加速倍數的意義 | 「over 4×」是 GPU 小時；$G\times T$ 的 4 倍是取樣次數。 | S8 分開寫，不把兩者等同。 |
| R7 | Table 結論的精確性 | Table 3 Texture 下降；Table 7 DeQA 略降；Table 2 w/ KL 的 Aesthetic 仍低於基礎模型。 | S9 照實寫出。 |
| R8 | 圖表只有曲線、無數據 | Fig. 4、5、7、8–12 無數據表。 | 只引用文字結論，不重繪曲線、不讀圖取值。 |
| R9 | 效能 | C3 $T=40$ 每次約 $3.2\times10^5$ 次速度計算；C1 RMSE 曲線 $3.1\times10^5$ 次抽樣。 | C1 曲線分片計算；C3 ODE 快取、rAF 合併，實測 > 30ms 改分片（C3 計算第 5 點）。 |
| R10 | 時間線涵蓋不足 | notes 已完成（6.4），但 `timeline.json` 只有 13 筆，擴散 RL／偏好對齊只有 DDPO 與 DPO；筆記提到的 DPOK、ReFL、Diffusion-DPO、PPO、DeepSeek-R1 不在時間線。 | 不自行增補時間線條目（單一事實來源在 notes）；S11 的敘事只提時間線中有的論文，其餘以筆記連結帶出。建置前重新讀取 notes，若時間線更新則依 TL_GROUP 規則歸類。 |
| R11 | 初學者門檻 | 讀者不熟 RL、SDE。 | S1 字面定義線上 RL、KL、reward hacking、DDPO、DPO；S3 優勢、S4 裁剪分節，各有元件；S5 把 Fokker–Planck 推導收進 details；S6 先教 Eq. 9 再放 C3；Hero 卡片自帶術語解釋（含 GenEval）。 |
| R12 | 相對連結 | 建置位置與上線位置不同。 | 預覽只檢查外部連結與頁內錨點。 |

---

## 8. 修訂紀錄（round 1）

依 `reviews/round1.md`（fidelity 15 項＋pedagogy 15 項合併為 M1–M28）全部處理，無拒絕項。主要變更：C4 的 $c_j$ 改用閉式（Δv=0 不再除以零）；6.4 依 notes 補齊；章節重排為 13 節（S3 優勢／S4 裁剪／S5 SDE 構造／S6 離散化與 C3／S7 KL 與 C4）；LO10 改為非單調敘述；所有「梯度為 0」限定為替代目標項並註明 KL 項仍有梯度；LO8 補前提；C3 發散改歸因於漂移放大與 t=1 處理，刪除 SDE 優於 ODE 的比較；新增 LO13；圖表引用改為 Fig. 7a＋Fig. 9、Fig. 7b 限 OCR、評估取樣器未寫；Fig. 2 的 π_old 以 PDF 影像確認後保留；`verify/check.py` 可用 `python3 -I` 執行，新增 `kl_coeff`、多種子 W₁ 與 KL 兩式一致性檢查。

## 9. 修訂紀錄（readiness）

阻擋項：$\sigma_t$、$a$ 的定義移到 S5 的 Eq. 7 之前（S6 改為回指）。非阻擋項：第 5 節加入 SDE $T=40$ 的 $a=0.1$、$a=1.5$ 兩列；S9 逐列寫出 Table 1 摘錄；S3、S4 的前向引用寫成頁面文字；Hero 錨點 `#paper-438` 於建置時檢查；Δt 正負號與 $t=1$ 的 σ 處理維持〔本頁推導〕／推定標記。

## 10. 建置紀錄（build round）

- 產出 `build/index.html`（單檔，約 110 KB），由 `build/parts/assemble.py`（`python3 -I` 執行）組合：外殼 CSS 取自 `tut/447/build`（k-diff 外殼）＋ `parts/extra.css`、`parts/body.html`、`parts/app.js`，並內嵌 `notes/timeline.json`（只保留 year、title、authors_short、arxiv）。
- `?test`：67 passed, 0 failed（1024 與 390 寬）；console 無錯誤；390 寬無水平捲動。selfTest 改在 `load` 事件後執行，以便檢查 KaTeX 已排版。
- LO 自查：`extract_page_text.py` 產生 `accept/page.txt`，13 個學習目標的關鍵詞與數字都在內文（檢查答案框已剔除）。
- `npx impeccable detect`：0 項 anti-pattern；12 則 numbered-section-labels advisory（允許）。原先 C1 圖例以 repeating-gradient 畫斜線被標記，已改為內嵌 SVG。
- 一輪視覺檢查（`build/caps/`：1440 四個元件、390 各元件與 clip／discrete／kl／results 章、深色 C1／C3）：修正 C1 軸標題重疊（改為 HTML 圖說）、「組平均」標籤移到右側邊界、停用滑桿的視覺狀態、C2 刻度重疊、C4 的 πθ／π′ 標籤位置與對數刻度改為上標。
- 與設計的差異：(1) KaTeX 載入失敗時顯示一則說明（TeX 原始碼仍可讀），沒有為每個式子另寫 Unicode 版本；(2) KaTeX 預設已對 `.katex-html` 加 `aria-hidden`，未再另加 `aria-label`；(3) 長式（Eq. 5、Eq. 9、KL 閉式）在 390 寬以 `.katex-display` 橫向捲動顯示。
