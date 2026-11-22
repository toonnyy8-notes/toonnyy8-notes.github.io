# 設計文件：〈Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in MLLMs〉互動教學

- 論文：Hyeontaek Hwang、Nguyen Dinh Son、Daeyoung Kim（KAIST），arXiv 2602.04509（v7，ICML 2026）。資料庫編號 No. 586。
- 原文：`tut/586/src/paper.txt`（PDF 同目錄）。本文件中的「Eq. n」「§n」「App. X」「Fig. n」「Table n」皆指論文原文編號。`tut/` = `scripts/tutorial_src/`。
- 顧問筆記：`tut/586/advisor.md`（論證鏈、常見誤讀、內部不一致）。本文件第 1 節已吸收其內容，建置者不需另讀。
- 產出：`tut/586/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/model-dowser-importance-probing.html`（slug `model-dowser-importance-probing`）。站內連結以上線位置為準。
- 驗算腳本（僅 Python 標準函式庫，演算法與頁面相同）：`tut/586/verify/common.py`（erf、GELU、mulberry32）、`c1_taylor.py`、`c2_scores.py`、`c3_hutch.py`、`c4_tables.py`；腳本只印到 stdout；凍結的驗收來源存於 `verify/expected/c{1,2,3,4}_out.txt`（不會被重跑覆蓋），`verify/check.sh` 重跑全部腳本並與 `expected/` 比對。第 5 節驗收表取自 `expected/`。
- 研究筆記：`tut/586/notes/`（總覽 `dowser-lineage-hub` + 7 篇主題筆記 + `timeline.json`），上線為 `tutorials/<slug>.html`。
- 標記：〔本頁推導〕＝本文件自行推導、論文沒寫；**推定**＝為了配合論文數字而選定、論文沒寫明的值；「作者認為」＝論文的解釋而非結果。頁面使用相同標籤（6.6）。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

多模態大型語言模型（MLLM）由視覺編碼器、連接模組與語言解碼器組成。用下游資料微調語言解碼器後，預訓練任務的 zero-shot 表現會下降（catastrophic forgetting，災難性遺忘）。作者發現微調的層數往輸入端延伸時遺忘更嚴重，而既有方法不是在深層微調時失效（post-merging：Grafting、DARE、Model Tailor），就是記憶體隨模型變大而過高（SPIDER，O(3|P|)）。Model-Dowser 在微調**之前**為每個權重算一個分數 $S_{ij}=\|J_i\|_2\cdot|W_{ij}|\cdot|h_j|$（輸出敏感度 × 連接強度 × 輸入活動量，Eq. 3）：Theorem 3.1 以一階 Taylor 展開給出單一權重擾動的輸出位移，作者再把其中未知的 $|\Delta W_{ij}|$ 代換為 $|W_{ij}|$ 來**定義**分數（App. B：不是要預測真實位移）；$\|J_i\|_2$ 用 Rademacher 投影估計（Eq. 4），探測輸入由模型自己從隨機 token 生成（Eq. 6），因此不需要任何資料。每層內分數最低的 ρ 比例權重可更新、其餘凍結（Eq. 7–8），遮罩固定，記憶體與一般微調同為 O(|P|)。在 LLaVA-1.5-7B 與 NVILA-Lite-2B、四個下游任務、只微調最後 20 層、ρ = 0.1 的 8 個設定中，H-score 全部最高（Tables 1–2）。

### 1.2 研究動機（§1、§2）

1. 遺忘與微調深度：先前方法多只微調最後幾層（Model Tailor 最後 12 層、SPIDER 少於最後 5 層，§1），但近期研究指出較早的解碼器層對多模態理解很重要，所以作者檢查更深的微調（Fig. 1(c)(d)、Fig. 3）。
2. Post-merging 的限制：作者認為它們「本質上是事後補救」，微調已破壞潛空間後，融合權重無法恢復（§4.2）。Grafting 需要 2 倍記憶體（App. E.2）。
3. Sparse fine-tuning 的限制：SPIDER 在訓練中累積每個參數的梯度歷史與軟遮罩，記憶體 O(3|P|)，LLaVA-1.5-7B 在 L > 20 時無法訓練（§4.1）。HFT 等方法缺少有原則的重要性準則（§2.1）。
4. 重要性準則的限制：magnitude 準則假設齊次激活（ReLU 滿足 $\phi(\alpha z)=\alpha\phi(z),\ \alpha>0$），GELU、SiLU、GLU 不滿足，權重大小不再對應輸出影響（§3.1）；Wanda 等經驗準則缺少輸出層級的理論分析，且在 massive activations 下可能不可靠（§2.2）。

### 1.3 貢獻（§1 列點）

1. 遺忘診斷：微調延伸到較早的解碼器層時遺忘加劇，既有方法失效或不穩定。
2. 可擴展的 sparse fine-tuning：無資料的重要性分數（輸入激活 × 輸出敏感度 × 權重大小），微調前凍結關鍵參數。
3. 理論：分數刻畫了輸出對單一參數擾動的敏感度（Theorem 3.1、Corollary 3.2、Theorem D.1）。
4. 實驗：兩個 MLLM、四個下游任務上的最佳結果，且記憶體與一般微調同階。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面 S2b 末尾以可收合表格提供）

| 符號 | 意義 |
| --- | --- |
| $l$ | 層索引；$W^{(l)}\in\mathbb R^{d_{out}\times d_{in}}$ 為該線性層權重 |
| $h^{(l-1)}\in\mathbb R^{d_{in}}$ | 該層的輸入激活（前一層輸出）；$h_j$ 為第 $j$ 個分量 |
| $z^{(l)}=W^{(l)}h^{(l-1)}$ | 預激活（pre-activation），$z_i=\sum_jW_{ij}h_j$ |
| $f\in\mathbb R^{d_{final}}$ | 網路輸出（論文稱 final representation；具體是哪個張量論文未寫明，見 R3） |
| $F$ | 把 $z^{(l)}$ 映到 $f$ 的函數（該層之後的所有運算） |
| $J_i=\partial f/\partial z_i\in\mathbb R^{d_{final}}$ | Jacobian 的**第 $i$ 個欄向量**：$d_{final}$ 個偏導數 $\partial f_r/\partial z_i$ 排成的向量。全頁一律稱「第 $i$ 個欄向量」，不用「行／列」 |
| $i$、$j$ | $i$ 為輸出節點（$W$ 的第 $i$ 列）、$j$ 為輸入分量（第 $j$ 欄）。同一列 $i$ 的所有 $W_{ij}$ 共用同一個 $\|J_i\|$ |
| 玩具網路 | $f=V\,\mathrm{GELU}(Wh+b)$：$b$ 為 bias、$a=\mathrm{GELU}(z)$ 為激活、$V$ 為輸出矩陣（C1、C2 使用；論文的 $z=Wh$ 不含 bias） |
| $\mathrm{GELU}$、$\Phi$、$\varphi$ | $\mathrm{GELU}(z)=z\,\Phi(z)$；$\Phi$、$\varphi$ 為標準常態的累積分布與密度 |
| $\Delta W_{ij}$、$\Delta f$ | 權重擾動、對應的輸出位移 |
| $S_{ij}$、$\bar S_{ij}$ | 單一輸入的重要性分數（Eq. 3）、對 $N$ 個合成輸入平均後的分數（Eq. 5） |
| $\xi\in\{\pm1\}^{d_{final}}$ | Rademacher 向量（各分量獨立、各以 1/2 機率取 ±1） |
| $R$、$N$ | 每個輸入的 Rademacher 向量數、合成探測輸入數（實驗 $R=8,\ N=64$） |
| $\rho$、$M$ | 可更新比例、二元遮罩（$M_{ij}=1$ 表示**更新**） |
| $\lambda$、$L$、$\theta$ | 學習率、訓練目標、模型參數 |
| $A_{up}$、$A_{down}$ | 6 個上游基準的平均準確率、下游分數（CIDEr 或 EM） |
| $|P|$ | 可學習（被微調的）參數個數；Table 3 的記憶體以它的倍數表示（S1 首次使用時定義） |
| $T$、$A_{t,i}$ | 持續學習的任務數（Table 13 為 5）、學完第 $t$ 個任務後在任務 $i$ 的測試準確率（App. L） |
| MFT、MFN | Mean Finetune Accuracy $=\frac1T\sum_iA_{i,i}$（剛學完各任務時的平均，Eq. 28）；Mean Final Accuracy $=\frac1T\sum_iA_{T,i}$（全部學完後的平均，Eq. 29） |
| BWT | Backward Transfer：Eq. 31 定義為 $\frac1{T-1}\sum_{i=1}^{T-1}(A_{T,i}-A_{i,i})$，負值表示遺忘。本頁採 Eq. 31 的定義，並說明 Table 13 的數值等於 MFN − MFT，也就是同一個和除以 $T$（$=\frac{T-1}T\times$Eq. 31）；依 Eq. 31 應為表值的 1.25 倍 |

#### 步驟 1：問題設定與評估（§4.1、Eq. 9）

- 只微調語言解碼器最後 $L$ 層（從輸出端往輸入端數，Fig. 3 圖說），其餘凍結。LLaVA 的 $L$ 為 4–32、NVILA 為 4–28，間隔 4。主表 $L=20$、$\rho=0.1$。
- 下游：COCO-Caption、Flickr30k（CIDEr）；ImageNet-R、IconQA（Exact Match）。上游 zero-shot：TextVQA、OKVQA、OCRVQA、MMBench、MMBench（CN）、GQA，$A_{up}$ 為 6 者算術平均。
- $$\text{Avg}=\frac{A_{up}+A_{down}}2,\qquad \text{H-score}=\frac{2A_{up}A_{down}}{A_{up}+A_{down}}\tag{Eq. 9}$$
- 頁面在公式**之前**先用一句話定義調和平均：兩數倒數的平均再取倒數，$2ab/(a+b)$；例：90 與 10 的算術平均 50、調和平均 18〔本頁推導：$2\cdot90\cdot10/100=18$〕，所以較小的一項主導結果。接著明寫 $A_{down}$ 在 captioning 是 CIDEr，可超過 100（最大可到 200 附近的尺度），論文仍直接代入同一個式子。
- 調和平均的性質：任一項接近 0，H 就接近 0。例：Table 1 NVILA IconQA zero-shot，$A_{down}=0.1$，Avg 31.2、H 0.2。CIDEr 不是 0–100 的準確率，可超過 100（Table 1 NVILA COCO Dowser 135.5），所以「兩個準確率的調和平均」並不精確。
- 驗算例〔本頁推導，表中未列 $A_{up}$〕：Table 1 NVILA COCO Dowser，$A_{up}=(69.3+48.6+68.8+77.7+50.8+60.7)/6=62.65$，Avg $=99.1$、H $=85.7$，與表相符。

#### 步驟 2：工作假說（§1 第 6 段、§3.3 首句）

作者寫「We hypothesize that preserving generalization can be achieved by minimizing output shifts induced by downstream fine-tuning」。這是**未證明的假說**：論文沒有任何式子把 $\|\Delta f\|$ 連到 $A_{up}$。之後的理論都在處理「輸出位移」這個可計算的代理量。

#### 步驟 3：Theorem 3.1——單一權重擾動（Eq. 1 = Eq. 10；證明 App. B）

$$\|\Delta f\|_2\approx\|J_i^{(l)}\|_2\cdot|\Delta W_{ij}^{(l)}|\cdot|h_j^{(l-1)}|\tag{Eq. 1}$$

1. 只擾動 $W_{ij}$：$z_i'=\sum_k(W_{ik}+\delta_{kj}\Delta W_{ij})h_k=z_i+\Delta W_{ij}h_j$（Eq. 12），其餘 $\Delta z_k=0$。**精確**（線性層）。
2. 把 $f$ 寫成 $F(z)$，一階 Taylor：$F(z+\Delta z)=F(z)+\frac{\partial F}{\partial z}\Delta z+O(\|\Delta z\|^2)$（Eq. 13），丟掉高階項得 $\Delta f\approx\sum_kJ_k\Delta z_k$（Eq. 14）。**全證明唯一的近似**。
3. 和式只剩 $k=i$：$\Delta f\approx J_i(\Delta W_{ij}h_j)$（Eq. 15）。
4. $\Delta W_{ij}h_j$ 是純量，可提出範數：$\|\Delta f\|_2\approx\|J_i\|_2|\Delta W_{ij}||h_j|$（Eq. 16）。**精確**。

- 適用條件（頁面 S3 以一個「理想化條件」小框列出）：(a) 單一 token 位置——實際 Transformer 中同一個 $W_{ij}$ 作用在每個位置，Eq. 12 的「精確」只對單一位置成立；(b) $F$ 在 $z$ 可微（ReLU 的折點除外）；(c) $J_i$、$h_j$ 在預訓練權重與給定輸入下求值；(d) $|\Delta W_{ij}h_j|$ 夠小。
- 定理沒有用到激活函數的形式：GELU/SiLU 的非線性全部在 $J_i$ 裡。這正是作者用它處理非齊次激活的理由（§3.1）。
- 何時不準〔本頁推導，`c1_taylor.py`〕：玩具網路 $f=V\,\mathrm{GELU}(Wx)$，$W=\begin{bmatrix}1&-2\\0.5&3\end{bmatrix}$、$x=(1,0.5)$、$V=\begin{bmatrix}1&2\\0&-1\\2&0\end{bmatrix}$。$z=(0,2)$，$\mathrm{GELU}'(0)=0.5$、$\mathrm{GELU}'(2)=1.08523$，$\|J_1\|=1.11803$、$\|J_2\|=2.42665$。神經元 1 位於 GELU 彎曲最大的區域：$\Delta W_{11}=0.01$ 時預測 0.01118、真實 0.01127；$\Delta W_{11}=-1$（把 $W_{11}$ 歸零）時預測 1.11803、真實 0.35476。神經元 2 位於接近線性的區域：把 $W_{22}$ 歸零（$\Delta W=-3$）時預測 3.63998、真實 3.59732。結論：一階近似的誤差取決於預激活落在激活函數的哪一段，而不只取決於 $|\Delta W|$。

#### 步驟 4：Corollary 3.2——多權重、多層（Eq. 2 = Eq. 17；證明 App. C）

$$\|\Delta f\|_2\lessapprox\sum_{l,i,j}\|J_i^{(l)}\|_2\,|\Delta W_{ij}^{(l)}|\,|h_j^{(l-1)}|\tag{Eq. 2}$$

全微分 $\Delta f\approx\sum\frac{\partial f}{\partial W_{ij}}\Delta W_{ij}$（Eq. 18）→ 鏈鎖律 $\frac{\partial f}{\partial W_{ij}}=J_ih_j$（Eq. 19）→ 代入（Eq. 20）→ 三角不等式（Eq. 21–22）。

- 「⪅」= 先 ≈（Taylor）再 ≤（三角不等式）。它是**一階位移**的上界，不是真實位移的上界。〔本頁推導，`c2_scores.py`〕C2 玩具層 $b_3=-0.75$、ρ = 4/16 時，Dowser 遮罩下 $\sum S=0.053$，真實歸零位移 0.698。
- 三角不等式丟掉方向抵消，上界可很鬆〔本頁推導：$n$ 個互相正交、大小相同的項，真實範數 ∝ $\sqrt n$，上界 ∝ $n$〕。
- App. C 末段說「$S_{ij}$ 的總和是總位移的理論上界」；這只在 $|\Delta W_{ij}|\le|W_{ij}|$ 時成立〔本頁推導〕。剪枝（$\Delta W=-W$）滿足，微調不保證。App. C 的標題就是 "Multi-layer and Multi-weight **Pruning**"。

#### 步驟 5：重要性分數（Eq. 3；§3.1、App. B 末段）

$$S_{ij}^{(l)}=\|J_i^{(l)}\|_2\cdot|W_{ij}^{(l)}|\cdot|h_j^{(l-1)}|\tag{Eq. 3}$$

- 把未知的 $|\Delta W_{ij}|$ **代換**成現有的 $|W_{ij}|$。這是定義上的選擇，不是推導。作者的兩句理由：「maximal local perturbation 的一階代理」（§3.1）與「不是要預測真實位移，而是保守的一階排序替代品」（App. B）。
- 三個因子（§3.1）：輸出敏感度 $\|J_i\|_2$（下游影響）、連接強度 $|W_{ij}|$（擾動的可能大小）、輸入活動量 $|h_j|$（上游訊號）。作者稱它結合了「結構流方法的梯度敏感度」與「局部剪枝的激活感知」。
- 索引：$i$ 是輸出節點（$W$ 的第 $i$ 列），$\|J_i\|$ 對同一列的所有權重都相同；頁面 S4 先寫出這一點，S5 的標題才依賴它。
- 為什麼 magnitude 不夠（**論文的主張**，§2.2、§3.1）：magnitude 準則假設齊次激活，GELU/SiLU/GLU 非齊次，權重大小不再對應輸出影響。頁面只陳述這個主張，**不宣稱玩具示範了非齊次性**。
- 玩具例（放在 S5、緊接 C2 之前，先展示 4×4 自選玩具層再引用其數字）〔本頁推導〕：C2 層中 $W_{22}=2.0$ 是 16 個權重中最大的，但 $z_2=-2.03$，$\mathrm{GELU}'(z_2)=-0.082$，$\|J_2\|=0.109$，所以 $S_{22}=0.044$，由大到小排第 14（即由小到大第 3）。排序規則：由小到大排列，同分時格子編號小者在前；$S_{21}$（格 5）與 $S_{22}$（格 6）精確同分（$0.4\times0.1094$），所以由小到大為格 8（0.013）、格 5、格 6，$S_{22}$ 是由小到大第 3、由大到小第 14。這個例子顯示的是「magnitude 忽略了 $\|J_i\|$ 與 $|h_j|$」；ReLU 的關閉單元（$J=0$）也會產生同樣效果，所以它不是非齊次性的證據。

#### 步驟 6：與剪枝分數的關係（§2.2；〔本頁推導〕為主）

| 準則 | 單一權重的分數 | 需要什麼 |
| --- | --- | --- |
| Magnitude（PackNet 等，§2.2） | $|W_{ij}|$ | 只要權重 |
| Wanda（§2.2、App. K） | $|W_{ij}|\cdot|h_j|$（App. K 正文寫 $\|h_j\|_2$） | 權重 + 輸入激活 |
| SNIP 型（連接敏感度） | $|W_{ij}\cdot\partial L/\partial W_{ij}|$ | 純量損失 $L$（要標籤） |
| OBD 型（二階） | $\tfrac12H_{ij,ij}W_{ij}^2$（損失 Hessian 對角） | 損失與二階資訊 |
| Model-Dowser（Eq. 3） | $\|J_i\|_2|W_{ij}||h_j|$ | 權重 + 激活 + 輸出 Jacobian（不需標籤） |

頁面分配：**內文**只講 magnitude、Wanda、Dowser 三列與下面第 1 點；SNIP、OBD、OBS、MAS 放在 S5 末尾預設收合的 `<details>`「與其他重要性準則的關係〔本頁推導〕」（論文沒有討論 SNIP/OBD），表格也只在收合區出現完整五列。
1. （內文）$S_{ij}=|W_{ij}|\cdot|h_j|\cdot\|J_i\|_2$ = Wanda 分數 × $\|J_i\|_2$。**單一輸入時**，同一輸出節點 $i$ 的權重共用 $\|J_i\|_2$，所以同一列內兩者排序相同；差別在不同列之間。多個輸入時不再成立：Eq. 5 對 $N$ 個探測平均的是乘積 $\|J_{i,n}\||h_{j,n}|$，App. K 的 Wanda 用 $\|h_j\|_2$；而 Eq. 7 把同一層所有列放在一起排序。
2. （收合區）$S_{ij}=\|W_{ij}\,\partial f/\partial W_{ij}\|_2$（由 Eq. 19，單一輸入、單一位置）：把 $W_{ij}$ 歸零時的一階輸出位移。SNIP 型分數 $|W\,\partial L/\partial W|$ 用純量損失（要標籤），這裡用整個輸出向量（不要標籤）。OBD 型在損失極小點用二階項 $\tfrac12H_{ii}W^2$。論文只提到 OBS（二階、成本過高，§2.2）。MAS 以輸出對參數的敏感度累積重要性（筆記 `dowser-catastrophic-forgetting-mllm-fine-tuning`），論文只在 App. A 引用。
3. 原本的 $S^2=\mathbb E_\xi[(W\,\partial(\xi^\top f)/\partial W)^2]$ 移到 S6b 的收合區（它需要 Eq. 4 與 $\xi$，S5 時尚未介紹）。

#### 步驟 7：Hutchinson 式估計（Eq. 4；§3.2）

$$\mathbb E_\xi\Big[\Big(\frac{\partial(\xi^\top f)}{\partial z_i}\Big)^2\Big]=\|J_i\|_2^2\tag{Eq. 4}$$

- 推導〔本頁推導，論文只寫結論〕：令 $a=J_i$，$X=\xi^\top a$，$\mathbb E[X^2]=\sum_ka_k^2\mathbb E[\xi_k^2]+\sum_{k\ne m}a_ka_m\mathbb E[\xi_k\xi_m]=\sum_ka_k^2$，因為 $\xi_k^2=1$、獨立且均值 0。
- 省成本的原因：對純量 $\xi^\top f$ 做**一次**反向傳播，就同時得到所有層、所有節點的 $\xi^\top J_i$（向量–Jacobian 乘積）。顯式建 Jacobian 需要 $d_{final}$ 次反向傳播（§3.2 Computational Complexity）。
- 論文稱為 "Hutchinson Trace Estimator"；這裡估的是 $J^\top J$ 的**對角元素** $\|J_i\|^2$，不是 trace〔本頁推導〕。
- 論文只給 Eq. 4（期望值）與 Eq. 5（對 $N$ 平均），沒有說 $R$ 個樣本如何合成一個 $\|J_{i,n}\|$。本頁與 C3 採「$\sqrt{\text{平方的平均}}$」與「絕對值的平均」兩種，標為**推定**。前者對 $\|J_i\|$ 是偏低的有偏估計（Jensen）〔本頁推導〕。單次樣本 $\mathrm{Var}[X^2]=2(\|a\|^4-\sum a_k^4)$〔本頁推導〕；$a=(3,4)$：$\mathbb E[X^2]=25$、Var = 576。$a$ 只有一個非零分量時變異數為 0。

#### 步驟 8：L1 替代估計量（Theorem D.1，Eq. 23–27；App. D）

$$\sqrt{\tfrac12}\,\|J_i\|_2\le\mathbb E_\xi\big[|\xi^\top J_i|\big]\le\|J_i\|_2\tag{Eq. 26}$$

- 上界：$\sqrt{\cdot}$ 凹，Jensen（Eq. 25）。下界：由論文引用的 Khintchine 型不等式（論文引 Khintchine 1923、Haagerup 1981，不自證；$p=1$ 時最佳常數 $\sqrt{1/2}$ 一般歸於 Szarek 1976，頁面用「由引用的 Khintchine 型不等式」的說法並在註腳提 Szarek）。兩邊乘 $|\Delta W_{ij}||h_j|$ 得 Eq. 23/27。
- 兩界都可取到〔本頁推導，`c3_hutch.py`〕：$a=(1,1)$ 時 $\mathbb E|X|=1=\|a\|/\sqrt2$（下界）；單一非零分量時 $\mathbb E|X|=\|a\|$（上界）；$a=(3,4)$ 比值 0.8；8 個 1 比值 0.7734。
- 動機（App. D "Numerical Stability near Zero"）：FP16/BF16 下平方小梯度會下溢為 0；L1 版保留原尺度。〔本頁推導〕BF16 指數範圍與 FP32 相同，下溢風險主要在 FP16。
- "L1" 指一階絕對動差 $\mathbb E|\xi^\top J_i|$，不是向量 1-範數；App. D 的 "$\|J_i\|_2\le\|J_i\|_1$" 一句與證明無關。
- 夾擠只保證常數倍等價。論文說「不失去相對排序」；〔本頁推導〕兩節點的 $\|J\|_2$ 相差不到 $\sqrt2$ 倍時排序可能相反。例：$a=(1,1)$（$\|a\|=1.414$、$\mathbb E|X|=1$）與 $b=(1.05)$（$\|b\|=1.05$、$\mathbb E|X|=1.05$）：L2 判 $a$ 較重要，L1 判 $b$ 較重要。
- 實作用哪一個沒說定：正文 Eq. 4 為平方（L2）、App. D 稱 "Our L1-based estimator"、Fig. 2 寫 $\mathbb E_\xi[\partial(\xi^\top f)/\partial z_i]$（無平方無絕對值）。

#### 步驟 9：無資料探測（Eq. 5–6；§3.2、App. I–J）

$$\bar S_{ij}=\frac1N\sum_{n=1}^N\|J_{i,n}\|_2\cdot|W_{ij}|\cdot|h_{j,n}|\tag{Eq. 5}$$
$$\hat x_n=f(\epsilon;\theta_{pre})\tag{Eq. 6}$$

- 從 tokenizer 詞表隨機抽 token 當種子 $\epsilon$，讓預訓練模型生成 $N$ 個 prompt，再在這些 prompt 上做前向與反向傳播。作者認為這些合成輸入「誘發反映模型功能結構的多樣激活」。主實驗只用文字探測，不含影像。
- 成本 $O(N\cdot R)$ 次前向＋反向（§3.2），實驗 $N=64,\ R=8$（§4.1）。
- 驗證（Table 5/10，與 64 筆上游真實文字查詢算出的分數比較，5 次執行）：$R=8,N=64$ 時 Hamming 距離 0.050 ± 0.002、Spearman 0.891 ± 0.003；隨機基準 0.226、0.000。$N$ 的影響大於 $R$：$N=64$ 時 $R=2/4/8$ 的 Spearman 為 0.886/0.885/0.891。
- Hamming 的尺度（頁面以收合 aside 呈現，標〔本頁推導〕）：**若** Hamming 是一般的「不一致比例」，兩個密度 $p$ 的遮罩距離 $=2p(1-\text{重疊率})$、上限 $2p$，$p=0.1$ 時 0.050 對應重疊 75%。但論文的隨機基準 0.226 超過這個上限 0.2（獨立隨機期望應為 0.18），所以論文的正規化方式未知，75% 只能當條件式推算，不放進學習目標答案。另外 Table 10/11 比較的是 **top** 10% 遮罩，訓練時更新的是 **bottom** 10%（Eq. 7），驗證的對象不是訓練用的那個集合（R7）。
- 純文字 vs 含影像探測（Table 11，$R=8,N=64$）：Hamming 0.081 ± 0.001、Spearman 0.882 ± 0.002。對照組影像是以 ImageNet 通道均值 [0.485, 0.456, 0.406]、標準差 [0.229, 0.224, 0.225] 取樣的高斯雜訊（App. J）。作者稱「almost identical」。0.081 與 Table 5 的 0.050 不能直接比：兩者的參考對不同（Table 11 是純文字探測 vs 文字＋雜訊影像探測；Table 5 是合成探測 vs 真實資料），Spearman 分別為 0.882 與 0.891。頁面只並列兩組數字並註明參考對不同。
- 論文未寫明：$f$ 是哪個張量、序列位置如何彙總（同一 $W$ 作用於每個 token，真正的 $\partial f/\partial W_{ij}=\sum_tJ_{i,t}h_{j,t}$，一般不等於範數相乘〔本頁推導〕）、$R$ 個樣本如何合成一個 $\|J_{i,n}\|$、prompt 長度與取樣溫度、bias/LayerNorm/embedding 是否評分。

#### 步驟 10：遮罩與稀疏微調（Eq. 7–8；§3.3、Table 3）

$$M_{ij}=\begin{cases}1,&\bar S_{ij}\text{ 落在該層最低的 }\rho\text{ 百分位}\\0,&\text{其餘}\end{cases}\tag{Eq. 7}$$
$$\theta^*=\theta-\lambda\cdot\Big(M\odot\frac{\partial L}{\partial\theta}\Big)\tag{Eq. 8}$$

- $M=1$ 表示**更新**。ρ = 0.1 ⇒ 凍結 90%，只更新分數最低的 10%（Fig. 2 圖說 "update the least important ρ%"）。「凍結少數關鍵參數」是常見誤讀。
- 遮罩在微調前算一次、之後固定。記憶體（Table 3）：Full FT、DARE、Tailor、Dowser 為 O(|P|)；Grafting O(2|P|)；SPIDER O(3|P|)。NVILA/LLaVA 最後 20 層的可更新參數：Dowser 143M / 438M、SPIDER 714M / 2.3B、Full FT 1.4B / 4.5B。Table 3 的 ρ 欄對各方法意義不同，Grafting 寫 100% 卻列 143M / 438M（R7）。
- 作者的 justification（§3.3）：凍結高分參數就是壓住一階模型中輸出位移的主要來源；這依賴步驟 2 的假說。

常見錯誤（頁面 S8 逐條列出）：
1. 把 $M=1$ 讀成凍結：Eq. 7 中 $M=1$ 是更新。
2. Fig. 2 寫 $\theta^*=\theta+\lambda(M\odot\partial L/\partial\theta)$，與 Eq. 8 的減號相反；以 Eq. 8 為準。
3. 逐層或全域排序：§3.3 寫「在每一層內」排序，App. C 末段寫 "global parameter selection"。「層」是指每個權重矩陣還是每個 Transformer block，論文沒寫。
4. 把 $\sum S$ 當成真實位移的上界：只在一階、且 $|\Delta W|\le|W|$ 時成立（步驟 4）。
5. 期待 L1 與 L2 估計量排序完全相同：只保證 $\sqrt2$ 倍內的等價（步驟 8）。
6. 探測階段的記憶體：Table 3 的 O(|P|) 只談微調階段；探測時要為每個參數累積一個分數，論文沒有討論。

#### 步驟 11：實驗（§4、App. F–L）

設定（§4.1）：每個下游資料集隨機取 10k 筆、lr $2\times10^{-5}$、5 epochs、總 batch 128、8 張 A100（40 GB）；Grafting 與 SPIDER 改用 H200（143 GB）。

主結果（Tables 1–2，$L=20$、ρ = 0.1；H-score，Dowser vs 次佳）：

| 模型 | COCO | ImageNet-R | Flickr30k | IconQA |
| --- | --- | --- | --- | --- |
| NVILA-Lite-2B | 85.7 vs SPIDER 78.3 | 71.2 vs SPIDER 60.5 | 75.8 vs SPIDER 68.5 | 76.0 vs Tailor 71.5 |
| LLaVA-1.5-7B | 79.9 vs Tailor 75.8 | 69.7 vs Grafting 63.8 | 69.6 vs Tailor 67.0 | 51.3 vs SPIDER 50.4 |

- 作者說「各方法下游表現相近，差別在上游」（§4.2）。在 captioning 不成立：NVILA COCO Dowser 135.5 vs Full-FT 98.5、Flickr30k 96.0 vs 64.3；LLaVA COCO 123.6 vs 100.0。只更新 10% 的下游分數反而高於全微調，論文沒有解釋。
- Dowser 贏的是平衡，不是每一欄：LLaVA ImageNet-R 的 OKVQA 58.0 → 48.4、GQA 62.0 → 56.5；同格 Grafting 上游保留更好，但下游只有 67.3。
- 深度（Fig. 1(c)(d)、Fig. 3、Fig. 5）：Grafting、DARE、Tailor 在最後 4–16 層穩定，延伸到早期層時崩潰；SPIDER 可撐更多層，但「即使用全部 32 層仍不如 Model-Dowser」（§4.2；與 §4.1「L > 20 無法訓練」矛盾，R7）。論文沒有給曲線數據，頁面只用文字描述。
- 隨機遮罩對照（Table 4 = Table 8 NVILA ImageNet-R；Tables 8/9 為**全層**微調：NVILA 全 28 層、LLaVA 全 32 層，與主表的最後 20 層不同。例：NVILA ImageNet-R 的 Dowser H 在 20 層為 71.2（Table 1）、全 28 層 ρ = 0.1 為 69.7（Table 4）；後者與 LLaVA 主表的 69.7 相同純屬巧合）：ρ = 0.1 時隨機遮罩已拿到大部分效果（H：NVILA ImageNet-R 68.0 ± 0.8 vs 69.7；NVILA COCO 85.2 ± 0.1 vs 85.7；LLaVA ImageNet-R 67.6 ± 0.1 vs 68.5；LLaVA COCO 79.1 ± 0.2 vs 79.3）；ρ = 0.5 時差距明顯（NVILA ImageNet-R 55.0 ± 0.8 vs 62.7；LLaVA ImageNet-R 33.2 ± 0.5 vs 42.6）。§4.2 說差距在 ρ = 0.5 最大；〔本頁推導〕NVILA ImageNet-R 的差距依 ρ = 0.1/0.25/0.5/0.75/0.9 為 1.7/3.4/7.7/11.2/5.8，最大在 0.75（隨機組 ±5.5）。p 值：正文 Avg p = 0.003、H p = 0.004（two-tailed t-test）；App. H 把 0.003 歸給 NVILA COCO、0.004 歸給 NVILA ImageNet-R，LLaVA 為 0.08 與 0.007，並稱 0.08「statistically significant」。表中只有 Random 附標準差（三個 seed）；論文未說明本法的執行次數。
- ρ 的穩健性（Fig. 4、Table 8）：§4.2 說上游到 ρ = 0.25 都穩定。LLaVA ImageNet-R 全層 $A_{up}$：ρ = 0.1 為 55.8、0.25 為 48.2、0.5 為 27.9、0.9 為 10.2（Full FT 9.4；zero-shot 61.0，Table 6）。
- Jacobian 消融（Table 12，LLaVA COCO，最後 20 層）：ρ = 0.1 時去掉 $\|J_i\|_2$（Wanda 式）$A_{up}$ 58.4、CIDEr 116.0、H 77.7；完整分數 59.1、123.6、79.9（+2.2）。ρ = 0.25、0.5 的 H 差 +1.7、+0.6。H 的增益主要來自下游；Wanda 式分數的 77.7 已高於 Table 2 的 SPIDER 75.2 與 Tailor 75.8。
- 學習率（Table 7，LLaVA 最後 20 層）：Dowser ImageNet-R H 在 lr $10^{-4}$/$2\times10^{-5}$/$5\times10^{-6}$ 為 61.4/69.7/71.1；在 $5\times10^{-6}$ 時 SPIDER 69.6、Full-FT 66.2。所有方法都在最小 lr 最好，差距縮小。這與站內 No.136 的結論（遺忘主要由學習率決定）相呼應。
- 視覺編碼器（Table 6，LLaVA 32 層 + 視覺編碼器，視覺端用 ImageNet 統計量的高斯輸入探測）：H COCO 79.9、ImageNet-R 68.9；Full-FT 13.3、18.1。
- 持續學習（Table 13，App. L；S9 在引用數字前先用一句話各定義 $T$、MFT、MFN、BWT，見符號表；LLaVA-1.5-7B，5 個領域任務 RS、Med、AD、Sci、Fin；上游基準換成另一組 9 個：TextVQA、POPE、VQAv2、MM-VET、ScienceQA-Img、MMBench EN/CN、GQA、SEED-Bench，所以 Zeroshot $A_{up}$ 為 63.9，不是主表的 61.0）：Dowser BWT −2.5、MFN 66.6、H 63.8；MoELoRA −7.8、60.0、60.6；Full-FT −15.6、54.7、46.3。〔本頁推導〕由定義，$\text{MFN}-\text{MFT}=\frac1T\sum_{i=1}^{T-1}(A_{T,i}-A_{i,i})$（$i=T$ 項為 0），所以 Eq. 31 的 BWT $=\frac T{T-1}(\text{MFN}-\text{MFT})=1.25\times(\text{MFN}-\text{MFT})$：Full-FT −19.4、Tailor −16.5、MoELoRA −9.9、Dowser −3.1。表中值（−15.6、−13.2、−7.8、−2.5）與 MFN − MFT（−15.5、−13.2、−7.9、−2.5）在四捨五入內一致，即除以 $T$。頁面顯示論文表值，並寫明「表值相當於除以 $T$；依 Eq. 31 除以 $T-1$ 則為上述 1.25 倍；四個方法的排序不變」。ρ、深度、遮罩是否每任務重算皆未寫明。

#### 步驟 12：論文沒有證明的部分（頁面 S10）

1. 定理只說「給定 ΔW 時一階輸出位移多大」，沒說微調時 ΔW 多大（Eq. 3 用 |W| 代替），也沒把 ‖Δf‖ 連到上游準確率。
2. 隨機遮罩在 ρ = 0.1 已接近本法；分數的作用在 ρ 大時才明顯。
3. 內部不一致（照錄並標示）：Table 1 NVILA ImageNet-R Tailor 的 Avg 66.9 由同列數字重算為 56.8〔本頁推導，`c4_tables.py`〕；p 值歸屬；SPIDER 32 層；逐層 vs 全域；Fig. 2 符號；估計量 L1/L2；Table 10 用 top 10% 遮罩而訓練用 bottom 10%；Hamming 0.226 超過上限；BWT 分母；App. L 稱 61.2「超過」Zeroshot（63.9）與 MoELoRA（61.2）。

---

## 2. 學習目標（驗收測試）

每章結尾 `<details>`「檢查理解」照搬題目與參考答案。LO10、LO12 為記憶型題目（審查已接受），其餘要求推理或計算。參考答案必須在指定章節的**內文段落**中寫出（不能只出現在元件、表格、圖或答案框）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明問題設定與深度發現 | 「微調深度 L」指什麼？post-merging 法在哪個範圍穩定？ | 只更新語言解碼器最後 L 層（從輸出端往輸入端數）；Grafting、DARE、Tailor 在最後 4–16 層穩定，延伸到早期層時崩潰（§4.1、§4.2、Fig. 3） | S1 |
| LO2 | 計算並解讀 H-score | 寫出 Avg 與 H-score。90 與 10 的兩種平均各是多少？為什麼 NVILA IconQA zero-shot 的 Avg 是 31.2 而 H 只有 0.2？ | Avg $=(A_{up}+A_{down})/2$，H $=2A_{up}A_{down}/(A_{up}+A_{down})$（Eq. 9）；90 與 10：Avg 50、H 18；$A_{down}=0.1$ 接近 0 時調和平均趨近 0，算術平均不會；$A_{down}$ 可為超過 100 的 CIDEr，論文仍代入同式 | S1 |
| LO3 | 陳述 Theorem 3.1 與其近似 | 單一權重擾動的輸出位移公式？證明中哪一步是近似？在什麼理想化條件下成立？ | $\|\Delta f\|_2\approx\|J_i\|_2|\Delta W_{ij}||h_j|$（Eq. 1）；只有一階 Taylor 丟掉 $O(\|\Delta z\|^2)$（Eq. 13→14）是近似，$\Delta z_i=\Delta W_{ij}h_j$ 與提出純量是精確的；條件：單一 token 位置、$F$ 可微、小 $\Delta W$ | S3 |
| LO4 | 判斷一階近似何時失準 | C1 玩具網路中，把 $W_{11}$ 歸零時預測與真實位移各是多少？為什麼 $W_{22}$ 歸零時預測準確？ | 預測 1.118、真實 0.355；$z_1=0$ 位於 GELU 彎曲最大處，$z_2=2$ 接近線性段（$W_{22}$ 歸零：3.640 vs 3.597）〔本頁推導〕 | S3 |
| LO5 | 說明分數的來源與 Corollary 3.2 的性質 | $S_{ij}$ 如何由 Eq. 1 得到？Eq. 2 的「⪅」是什麼意思？ | 把未知的 $|\Delta W_{ij}|$ 代換成 $|W_{ij}|$（定義上的選擇，非推導）；「⪅」是先一階近似再用三角不等式，是一階位移的上界，不是真實位移的上界（Eq. 2–3、App. C） | S4 |
| LO6 | 說明只看權重大小會漏掉什麼 | 論文為什麼認為 magnitude 準則不可靠？C2 中 $W_{22}=2.0$ 為什麼分數很低？ | 論文主張：magnitude 假設齊次激活（ReLU 滿足 $\phi(\alpha z)=\alpha\phi(z)$，α > 0），GELU/SiLU 不滿足（§2.2、§3.1）；玩具中 $z_2=-2.03$、$\mathrm{GELU}'=-0.082$、$\|J_2\|=0.109$，$S_{22}=0.044$（magnitude 第 1、S 第 14）——它顯示 magnitude 忽略了 $\|J_i\|$，不是非齊次性的證明〔本頁推導〕 | S4（主張）、S5（玩具例） |
| LO7 | 比較 Dowser 與 Wanda 分數 | Dowser 分數與 Wanda 分數差在哪個因子？什麼情況下兩者在同一列內排序相同？ | $S=$ Wanda 分數 $|W||h|$ × $\|J_i\|_2$；單一輸入時同一輸出節點（同一列）共用 $\|J_i\|$，列內排序相同；對 $N$ 個輸入平均（Eq. 5）或跨列排序（Eq. 7）時不同；App. K 的消融就是拿掉 $\|J_i\|$ | S5 |
| LO8 | 推導 Hutchinson 式估計並說明省成本的原因 | 為什麼 $\mathbb E[(\xi^\top a)^2]=\|a\|^2$？一次反向傳播得到什麼？ | 交叉項 $\mathbb E[\xi_k\xi_m]=0$、$\xi_k^2=1$；對純量 $\xi^\top f$ 反向傳播一次就得到所有層所有節點的 $\xi^\top J_i$，顯式 Jacobian 需要 $d_{final}$ 次（Eq. 4、§3.2） | S6a |
| LO9 | 說明 L1 替代估計量的界與限制 | Theorem D.1 的不等式？何時取等？為什麼要用它？ | $\sqrt{1/2}\|J_i\|\le\mathbb E|\xi^\top J_i|\le\|J_i\|$；$a=(1,1)$ 取下界、單一非零分量取上界；避免 FP16/BF16 平方下溢（App. D）；只保證常數倍等價，$\|J\|$ 相差不到 $\sqrt2$ 倍時排序可能相反〔本頁推導〕 | S6b |
| LO10 | 描述無資料探測與其驗證（記憶型） | 合成探測輸入怎麼來？$N,R$ 與成本？和真實資料分數一致到什麼程度？ | 從詞表隨機抽 token 當種子，讓預訓練模型生成 $N$ 個 prompt（Eq. 6）；$N=64,R=8$，成本 $O(N\cdot R)$ 次前向＋反向；Spearman 0.891 ± 0.003、Hamming 0.050 ± 0.002（Table 5；隨機基準 0.000、0.226） | S7 |
| LO11 | 說明遮罩、更新式與記憶體 | ρ = 0.1 時哪些參數被更新？更新式？記憶體與 SPIDER、Grafting 比較？ | 每層內分數最低的 10% 更新（$M=1$），其餘 90% 凍結；$\theta^*=\theta-\lambda(M\odot\partial L/\partial\theta)$（Eq. 7–8）；Dowser O(|P|)，SPIDER O(3|P|)，Grafting O(2|P|)（Table 3） | S8 |
| LO12 | 讀出實驗結果與其限制（記憶型） | 主表 H-score？全層微調時 ρ = 0.1 與 0.5 的隨機遮罩差多少？拿掉 Jacobian 項的影響？ | 最後 20 層：NVILA COCO 85.7、ImageNet-R 71.2；LLaVA COCO 79.9、ImageNet-R 69.7（Tables 1–2）；NVILA ImageNet-R **全 28 層**：隨機 68.0 vs 69.7（ρ = 0.1）、55.0 vs 62.7（ρ = 0.5）（Table 4）；LLaVA COCO 最後 20 層 ρ = 0.1 去掉 $\|J\|$ H 77.7 vs 79.9，增益主要來自 CIDEr 116.0 → 123.6（Table 12） | S9 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），圖版加寬到 1080px。頁面章節編號連續顯示為 01–14（JetBrains Mono，`01 · …`）；下表的 S2a/S2b、S6a/S6b 是設計內部代號。標題一律陳述句。**視覺記號**：三格公式條「‖J_i‖ × |W_ij| × |h_j|」（三格標「輸出敏感度」「連接強度」「輸入活動量」），在 S4、S5、C2 重複出現；C2 中各分數用到的格子為實心並附文字「用」，未用到的為空心並附「未用」。Hero 不放公式條（術語尚未定義）。

| # | id | 標題 | 內容（依序；每個符號在第一次使用前定義） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Model-Dowser：微調前先找出不能動的權重〉 | 副標（只用白話）：「這篇論文在微調之前，估計每個權重被改動時模型輸出會變多少；變化大的權重固定不動，只更新變化最小的一小部分。估計不需要訓練資料。」三張事實卡（見下）。連結：arXiv、`../index.html#paper-586`（在關聯圖中查看）、`dowser-lineage-hub.html` | 1 螢幕 |
| S1 | `problem` | 01 · 作者觀察到：微調較早的解碼器層時，MLLM 遺忘得更多 | MLLM 三元件；上游／下游、zero-shot、災難性遺忘定義（§2.1）；微調最後 L 層；資料集；**先**定義調和平均與 90/10 例（Avg 50、H 18），再寫 Eq. 9、IconQA 0.2 例、$A_{down}$ 可超過 100（CIDEr）；post-merging 與 sparse FT 兩類方法及其失效（深度、記憶體），寫出 LO1、LO2 答案。延伸：`dowser-catastrophic-forgetting-mllm-fine-tuning.html`、`dowser-llava-nvila-setting.html`、`dowser-post-merging-methods.html`、`dowser-sparse-finetuning-hft-spider.html`；站內 `odd-on-policy-forgetting-sparsity-claims.html` | 550–650 字 |
| S2a | `linear` | 02 · 一個線性層把輸入加權相加成預激活 | $z=Wh$，$z_i=\sum_jW_{ij}h_j$；$i$ 是輸出節點（列）、$j$ 是輸入分量（欄）；2-2-3 小網路靜態 SVG（$h_j\to W_{ij}\to z_i\to a_i\to f$）；玩具網路 $f=V\,\mathrm{GELU}(Wh+b)$ 與 GELU 定義 $z\Phi(z)$、小圖（GELU 與 GELU′ 兩條曲線，標出 GELU′ 在 z ≈ −0.75 為 0、z ≈ −1.5 為負）、齊次的定義（ReLU 滿足、GELU 不滿足，例 GELU(2)=1.954 但 2·GELU(1)=1.683〔本頁推導〕）；純量版鏈鎖律：$y=g(wx)$ 時 $dy/dw=g'(wx)\cdot x$，對應「輸出敏感度 × 輸入」 | 300–350 字 + 2 SVG |
| S2b | `jacobian` | 03 · 輸出是向量時，敏感度是一個欄向量 | $f\in\mathbb R^{d_{final}}$；$F$；$J_i=\partial f/\partial z_i$ 是 $d_{final}$ 個偏導數組成的「第 $i$ 個欄向量」；向量版鏈鎖律 Eq. 19 $\partial f/\partial W_{ij}=J_ih_j$；一階 Taylor（$F(z+\Delta z)\approx F(z)+\sum_kJ_k\Delta z_k$，誤差為 $\Delta z$ 的二次方等級）；工作假說（標「未證明」，§1）。末尾可收合的符號表（1.4） | 300–350 字 |
| S3 | `theorem` | 04 · 在一階近似下，單一權重的輸出位移等於三個量相乘 | 「理想化條件」小框（單一位置、可微、預訓練點、小 ΔW）；Eq. 12→13→14→15→16，四步各標「精確／近似」；C1（玩具的 $V$、$x$ 標「本頁自選」）；C1 之後一段寫出 LO4 答案 | 400–500 字 + C1 |
| S4 | `score` | 05 · 以權重大小代替擾動，得到重要性分數 | Eq. 3 與代換說明（不是推導）；三因子與三格公式條；寫明「$\|J_i\|$ 對同一列相同」；Eq. 2 與「⪅」三點（一階、三角不等式、$|\Delta W|\le|W|$）；論文的齊次性主張（只陳述，不用玩具數字） | 400–500 字 |
| S5 | `pruning` | 06 · 單一輸入時，Dowser 分數是 Wanda 分數乘上輸出敏感度 | magnitude、Wanda、Dowser 三列小表；Wanda 關係與「單一輸入」限定，並以前向指引說明多輸入與跨列排序時不同（「S7 會介紹對 $N$ 個探測輸入取平均的 Eq. 5；S8 會介紹在整層內排序的 Eq. 7」，此處不展開公式）；先展示 4×4 自選玩具層（$h$、$W$、$V$ 三個矩陣以小表列出）→ $W_{22}$ 例（並說明它顯示的是忽略 $\|J_i\|$，不是非齊次性）→ C2 → 一段寫出 $b_3=-0.75$ 時一階排序失準（Σ S 0.053 vs 真實 0.698）；末尾收合區「與其他重要性準則的關係〔本頁推導〕」（SNIP、OBD、OBS、MAS，1.4 步驟 6 第 2 點） | 500–600 字 + C2 |
| S6a | `hutchinson` | 07 · 一次反向傳播就能估計所有節點的輸出敏感度 | Eq. 4 先用 $d=2$、$a=(3,4)$ 列出 4 個符號向量手算（$X=7,1,-1,-7$，$\mathbb E[X^2]=25$）再一般化；VJP 成本；trace 借名；R 樣本合成方式為推定；Jensen 偏差；Var；C3 | 450–550 字 + C3 |
| S6b | `l1` | 08 · 絕對值版估計量與 ‖J‖ 只差常數倍 | Theorem D.1、取等情形（C3 預設組可驗證）、排序反轉例 (1,1) vs (1.05)、FP16 下溢、三處寫法不一致；收合區：$S^2=\mathbb E_\xi[(W\,\partial(\xi^\top f)/\partial W)^2]$〔本頁推導〕 | 300–400 字 |
| S7 | `probe` | 09 · 讓模型自己產生探測輸入 | Eq. 5–6、$N,R$、成本；Table 5/10 數字；Table 11 並列並註明參考對不同；收合 aside：Hamming 尺度（條件式、0.226 問題、top vs bottom 10%）；未寫明事項清單 | 400–500 字 |
| S8 | `sparse` | 10 · 凍結高分權重，只更新每層分數最低的 ρ | **先**放 3×3 靜態遮罩例（9 個分數，ρ = 2/9 → 最低 2 格 $M=1$ 標「更新」，其餘 7 格 $M=0$ 標「凍結」，見 4.6）；再 Eq. 7–8；Table 3 記憶體（表）；遮罩固定的意義；6 條常見錯誤 | 400–500 字 + 小圖 + 表 |
| S9 | `results` | 11 · 在 LLaVA 與 NVILA 的 8 個主設定中，H-score 都最高 | 設定；主結果表（**最後 20 層**，1.4 步驟 11）；下游高於 Full FT、上游個別欄退步；深度（文字）；C4（圖說寫「全層微調：NVILA 28 層、LLaVA 32 層」）；隨機遮罩、ρ、Jacobian 消融、lr、視覺編碼器、持續學習（註明上游基準不同）各一段。延伸：`dowser-experiments-depth-and-memory.html`；站內 `on-policy-vs-off-policy-distillation.html`（lr 與遺忘）、`depth-nonuniform-lora-rank.html`（深度相依敏感度） | 700–850 字 + 表 + C4 |
| S10 | `limits` | 12 · 論文沒有證明的部分與內部不一致 | 1.4 步驟 12；每條附出處 | 300–400 字 |
| S11 | `history` | 13 · 研究脈絡（由 notes/timeline.json 提供） | 時間線（6.2）＋一段把它接到三格公式條：magnitude（PackNet、SynFlow）→ ×激活（Wanda）→ ×輸出敏感度（本文）；遺忘線（EWC、MAS）；post-merging；sparse FT | 時間線 + 120 字 |
| S12 | `next` | 14 · 延伸閱讀 | 筆記依總覽順序；站內相關論文：`../index.html#paper-247`（Self-Distillation Enables Continual Learning）、`#paper-141`（Local Support Learning）、`#paper-279`（Maximal Brain Damage Without Data：無資料找出關鍵參數的另一路線）、`#paper-136` 與教學 `on-policy-vs-off-policy-distillation.html`；`context-distillation-self-distillation.html`、`odd-sft-vs-rl-generalization-claims.html`、`lottery-ticket-redundancy-scaffold.html` | 連結清單 |
| — | — | 頁尾 | 引用格式、arXiv（v7）；`#pm-bar` | — |

Hero 事實卡（只用白話，不用「分數」「神經元」「敏感度」等未定義詞）：
1. 「作者觀察到：微調多模態模型時，更新的層越靠近輸入端，模型越容易忘記原本會的任務。」
2. 「方法在微調前估計：改動某個權重時，模型輸出會變多少。這個估計由三個量相乘：輸出對該處的反應大小、權重本身的大小、流進該權重的輸入大小。估計值大的權重在微調時固定不動。」
3. 「估計只用模型自己生成的文字，不需要訓練資料。實驗中每層只更新估計值最低的 10% 權重。」

桌機（≥ 1024px）左側 sticky 章節目錄；< 1024px 頂部橫向捲動章節列（沿用參考頁）。

---

## 4. 互動元件規格

### 4.0 共用數學（頁面與 `verify/common.py` 逐行對應）

- `erf(x)`：Numerical Recipes 的 erfc Chebyshev 擬合：$z=|x|$，$t=1/(1+0.5z)$，$r=t\exp(-z^2-1.26551223+t(1.00002368+t(0.37409196+t(0.09678418+t(-0.18628806+t(0.27886807+t(-1.13520398+t(1.48851587+t(-0.82215223+t\cdot0.17087277)))))))))$，$\mathrm{erf}(x)=1-r$（$x\ge0$）或 $r-1$（$x<0$），在 [−6, 6] 對 `math.erf` 最大絕對誤差 8.3e−8。$\Phi(x)=\tfrac12(1+\mathrm{erf}(x/\sqrt2))$、$\varphi(x)=e^{-x^2/2}/\sqrt{2\pi}$。
- `gelu(x)=x·Φ(x)`（精確 GELU，非 tanh 近似）；`dgelu(x)=Φ(x)+x·φ(x)`（解析導數）。頁面註明「GELU′ 在 z ≈ −0.75 附近為 0，在 z ≈ −1.5 附近為負」。
- 小網路前向：$z=Wh+b$，$a=\mathrm{GELU}(z)$，$f=Va$；$J_i=V_{:,i}\cdot\mathrm{GELU}'(z_i)$（閉式，不做數值微分）。
- PRNG：`mulberry32(seed)`（與參考頁相同字串）；已以 node 驗證與 Python 版前三個輸出一致（seed 7：0.0117048, 0.0619583, 0.9769076）。
- 數字格式：讀數一律 JetBrains Mono，固定小數位（C1 五位、C2 三位、C3 四位、C4 一位）。

### C1 · 一階近似與它失準的地方（S3）

- **概念**：Theorem 3.1；一階預測 $\|J_i\|\,|\Delta W|\,|h_j|$ 在 $\Delta W$ 小時準確；誤差大小取決於 $z_i$ 在 GELU 的哪一段；$|\Delta W|=|W|$（歸零）時可能差很多。
- **固定網路**（頁面寫出，並標「本頁自選數值（推定），非論文設定」）：$W=\begin{bmatrix}1&-2\\0.5&3\end{bmatrix}$、$x=h=(1,0.5)$、$V=\begin{bmatrix}1&2\\0&-1\\2&0\end{bmatrix}$、$b=0$。
- **操作**：四個 chip 選權重 $(i,j)\in\{(1,1),(1,2),(2,1),(2,2)\}$，預設 (1,1)；滑桿 $\Delta W\in[-3,3]$，步長 0.01，預設 0.10；按鈕「ΔW = −W（歸零）」（chip 與按鈕標籤用純文字＋`<sub>`，不放 TeX，見 6.3）；換 chip 時 $\Delta W$ 保留但夾在範圍內。
- **計算**（即時）：$f_0$ 與擾動後 $f_1$ 各做一次前向；真實 $\|f_1-f_0\|$；預測 $\|J_i\|\,|\Delta W|\,|h_j|$；相對誤差 $(\text{預測}-\text{真實})/\text{真實}$，以帶正負號的百分比顯示一位小數（例「+215.1%」「−7.4%」；正值表示一階預測偏大），讀數旁註明這個符號慣例。曲線：$\Delta W$ 在 [−3, 3] 取 601 點各算真實值，預測為 V 形折線。
- **數值防護**：真實值 < 1e−12（含 ΔW = 0）時相對誤差顯示「—」；縱軸上限 = 該權重 601 點中真實與預測兩條曲線的最大值 × 1.08，因此預測的 V 形在 ΔW = ±3 兩端一定在圖內；換 chip 時重算一次（拖滑桿時不變）。四個 chip 的上限（`expected/c1_out.txt`）：(1,1) 7.2351、(1,2) 3.3804、(2,1) 7.8624、(2,2) 3.9312。例：(1,2) 在 ΔW = ±3 的預測為 1.6771，低於上限，兩端可見。
- **顯示**：左側 2-2-3 網路小圖（與 S2a 同圖），選中的邊以 `--accent` 粗線＋文字標籤；主圖橫軸 ΔW、縱軸 ‖Δf‖：真實（`--c-1` 實線，直接標「真實」）、一階預測（`--c-2` 虛線，標「一階預測」），目前 ΔW 的兩個點（實心圓／空心方形）與垂直細線；$\Delta W=-W_{ij}$ 處畫小刻度標「歸零」。讀數表：$z_i$、GELU′($z_i$)、‖J_i‖、|h_j|、$S_{ij}$、預測、真實、相對誤差。live region（500ms debounce）宣告「W₁₁，ΔW 0.10：預測 0.11180，真實 0.12071，誤差 −7.4%」。下方 2×2 小表列四個 $S_{ij}$（1.11803、1.11803、1.21333、3.63998）。
- **應觀察到**：(1) ΔW → 0 時兩線相切；(2) (1,1)：ΔW = +0.5 已差 28%，歸零時預測 1.118、真實 0.355；(3) (2,2)：歸零（ΔW = −3）時 3.640 vs 3.597；(4) $|W_{12}|=2|W_{11}|$ 但 $|x_2|=0.5$，兩者 $S$ 相同。

### C2 · 三種分數、三種遮罩（S5）

- **概念**：magnitude、Wanda、Dowser 三種排序產生不同遮罩；$\|J_i\|$ 讓整列（同一輸出節點）一起升降；「歸零更新集合」時的真實輸出位移比較；一階排序在 GELU′ ≈ 0 附近會失準。
- **固定層**（頁面寫出，單一輸入、單一位置；論文對 $N$ 個輸入平均，這裡省略）：
  $h=(1.0,-0.2,1.5,0.6)$；
  $W=\begin{bmatrix}0.8&-1.5&0.3&0.6\\-0.4&2.0&-0.9&0.2\\1.2&0.5&-0.3&-1.0\\0.1&-0.7&0.6&1.4\end{bmatrix}$；
  $V=\begin{bmatrix}1.0&-0.5&0.8&0.3\\0.4&1.2&-0.6&0.9\\-0.7&0.3&1.1&-0.4\end{bmatrix}$；$b=(0,0,b_3,0)$。Theorem 3.1 不受 bias 影響（$\Delta z$ 不含 $b$），頁面一句說明。三個矩陣與 $h$ 的數值為本頁自選（推定），頁面在 S5 正文先以小表展示。
- **操作**：滑桿 $b_3\in[-4,2]$，步長 0.05，預設 0；三個 chip「b₃ = 0」「b₃ = −0.75（GELU′ ≈ 0）」「b₃ = −4（神經元 3 幾乎不輸出）」；滑桿 $m\in\{1,\dots,15\}$（可更新的權重數），預設 4，標示 ρ = m/16。
- **計算**：
  - 三組分數（格子編號 $k=4(i-1)+j$，1–16）：mag $=|W_{ij}|$；wanda $=|W_{ij}||h_j|$；full $=\|J_i\||W_{ij}||h_j|$。
  - 遮罩：排序鍵 = `Math.round(score / ε)`，ε = 1e−9，同鍵依 $k$ 小者在前；前 $m$ 個 $M=1$（更新）。這處理 $S_{21}=S_{22}$ 的精確同分與 $b_3=-4$ 時近零分數的浮點雜訊；格內顯示的三位小數由同一個捨入後的值格式化。$m=15$ 時只剩 1 格凍結（各分數的最高格），照常計算（驗收值見 5.2）。
  - 每個遮罩的兩個數：真實位移（把 $M=1$ 的權重同時設為 0 再前向，$\|f_1-f_0\|$）與一階總和 $\sum_{M=1}S^{full}_{ij}$（Cor. 3.2，$|\Delta W|=|W|$）。
  - 隨機遮罩參考：以 Gray code 列舉全部 $2^{16}$ 個子集——第 $s$ 步翻轉第 `ctz(s)` 個權重，只更新該列的 $z_i\mathrel{+}=\mp W_{ij}h_j$，再做 4 次 GELU 與 $V$ 乘法——依集合大小累積真實位移並取平均（一次走完得到所有 $m$）。`c2_scores.py` 的 `random_mean_gray` 與逐一前向的結果差 < 1e−9（已斷言）。**預設以分段計算**：每段 4096 步，用 `setTimeout(…,0)` 接續；開始時讀數的隨機列顯示「計算中」；每次 $b_3$ 改變遞增 `runToken`，舊工作在下一段開始時檢查 token 不符即放棄。完成後存入 `Map`（key 為 `b3.toFixed(2)`）。$b_3$ 拖動中只更新熱圖與三個遮罩，隨機列在放開或停止 150ms 後才啟動計算。
  - Spearman（以同一捨入鍵、平均名次處理同分）：full 對 mag、full 對 wanda。任一方名次變異數為 0 時顯示「—」（本層的 mag／wanda 不會發生，但 full 在極端 $b$ 下可能大量同分，仍需防護）。
- **顯示**：三張 4×4 熱圖並排（< 768px 直向堆疊）。格子明度 ∝ 分數 / 該圖最大分數（最大值為 0 時全部最淺）；格內印三位小數，填色比例 > 50% 時文字改用 `--surface` 色，淺／深色模式下皆 ≥ 4.5:1（建置時以對比工具確認並回寫）；$M=1$ 的格子加 45° 斜線紋樣（線色 token `--hatch`：淺 #1D1C19 35% 不透明、深 #EEEBE4 45%）＋粗外框。每張圖上方是三格公式條，用到的因子實心並寫「用」、未用到的空心並寫「未用」（mag：只有 |W|；wanda：|W|、|h|；full：三格）。full 圖左側標每列 ‖J_i‖ 與 GELU′($z_i$)，上方標 $h_j$。圖下讀數表：列 = mag／wanda／full／隨機平均；欄 = 可更新格子編號、真實位移、Σ S。390px 時三張熱圖直向堆疊（每張寬 = 內容寬，格子 ≥ 44px），讀數表改為三欄（名稱、真實位移、Σ S），「可更新格子」移到每張熱圖標題下方一行。
- **應觀察到**：(1) $b_3=0$：$W_{22}=2.0$ 在 mag 排第 1，在 full 只有 0.044（神經元 2 的 $z_2=-2.03$）；(2) $b_3$ 往 −4 拉，第 3 列 full 分數趨近 0，mag 與 wanda 不變；(3) $b_3=-0.75$、$m=4$：full 遮罩選了第 3 列的 3 格與 $W_{24}$，Σ S 只有 0.053，但同時歸零的真實位移 0.698，大於 wanda 遮罩的 0.268——$z_3=-0.70$ 處 GELU′ ≈ 0.023，一階項幾乎為 0，二階以上的變化才是主項；(4) 多數設定下 full 的真實位移最小，但不是保證（頁面明寫）。
- 頁面在圖版底部註明兩行：「這是單一輸入的自選玩具層，比較的是把可更新權重歸零時的輸出位移，不是論文的微調實驗。」「玩具沒有測試『輸出位移小 ⇒ 上游能力保留』這個假說（§1，論文也未證明）；位移較小不代表遮罩較好。」

### C3 · Hutchinson 估計與 L1 替代量（S6a；S6b 引用其預設組）

- **概念**：Eq. 4 的期望值精確等於 ‖a‖²；有限 R 的估計有變異、開根號後偏低（R 個樣本的合成方式為推定，見 1.4 步驟 7）；Theorem D.1 的比值 $\mathbb E|X|/\|a\|$ 永遠在 [0.7071, 1]。
- **操作**：維度 $d\in\{1,\dots,8\}$ 分段按鈕（預設 2）；每個分量一支滑桿 [−3, 3]，步長 0.05；預設組 chip：「(3, 4)」（預設）、「(1, 1)：下界」「(3)：上界」「八個 1」「一大七小 (3, 0.5×7)」；$R\in\{1,2,4,8,16,32,64\}$ 分段按鈕（預設 8）；「重新抽樣」（seed 由 1 起，每按 +1）。換預設組時同時設定 $d$。
- **計算**：
  - 精確：列舉 $2^d$ 個符號向量（$\xi_k=-1$ 若 bit $k$ 為 1），得 $X$ 的分布、$\mathbb E[X^2]$、$\mathbb E|X|$；$\mathrm{Var}[X^2]=2(\|a\|^4-\sum a_k^4)$。
  - MC：`r = mulberry32(seed)`，依樣本、再依分量順序取 $\xi_k=(r()<0.5)?-1:+1$；L2 估計 $\sqrt{\frac1R\sum X_r^2}$、L1 估計 $\frac1R\sum|X_r|$。
  - 重複：seed 1–200 各跑一次，報告兩個估計量的平均 ± 樣本標準差（與目前 seed 無關，只隨 a、R 改變）。
- **數值防護**：$\|a\|=0$ 時比值顯示「—」、數線只畫 0；$X$ 分布中相同值（差 < 1e−9）合併；長條圖縱軸為機率 [0, max]。
- **顯示**：(a) $X=\xi^\top a$ 的精確分布長條圖（x 軸標值、長條上印機率），本次 R 個樣本以小三角形標在長條下方；(b) 一條數線：$\sqrt{1/2}\|a\|$ 與 $\|a\|$ 兩個端點（方括號）、$\mathbb E|X|$（實心菱形，標「E|X|」）、L2 估計（空心圓）、L1 估計（空心方形），每個標記都有直接文字標籤；兩個標記相距 < 標籤寬度時，標籤上下錯開並以引線連到標記（例：a = (3,4)、R = 64 時 L2 4.99 與端點 5.00 幾乎重合）；(c) 讀數表。Tab 順序：維度 → 預設組 → 各分量滑桿（依序）→ R → 重新抽樣 → 讀數。live region 宣告「E|X|/‖a‖ = 0.8000；L2 估計 5.5678；L1 估計 4.7500」。
- **應觀察到**：(1) $\mathbb E[X^2]$ 永遠等於 ‖a‖²；(2) (1,1) 比值 0.7071、單一分量 1.0000；(3) R = 1 時 L2 估計就是 |X|，200 次平均 4.000（a = (3,4)），小於 ‖a‖ = 5；R = 64 時 4.986 ± 0.328。

### C4 · 隨機遮罩 vs 重要性遮罩，以及 H-score 計算器（S9）

- **概念**：分數相對於隨機遮罩的增益隨 ρ 改變；ρ = 0.1 時差距小；調和平均對低分項敏感。
- **資料**（圖說寫明「全層微調：NVILA 28 層、LLaVA 32 層；與主表的最後 20 層不同」）：Tables 8/9 原樣。頁面常數名 `C4_TABLE`（key `"NVILA|ImageNet-R"` 等；每項 `{full:[A_up,A_down,Avg,H], rand:[[A_up,A_down,Avg,H,H_sd]×5], ours:[[A_up,A_down,Avg,H]×5]}`）與 `C4_AUP_SD`（隨機組 $A_{up}$ 標準差，同 ρ 順序），逐字取自 `verify/c4_tables.py` 的 `TABLE`、`AUP_SD`（ρ = 0.1, 0.25, 0.5, 0.75, 0.9 與 Full FT ρ = 1.0）。這些是論文數字，不是本頁模擬，頁面明寫出處。
- **操作**：分段按鈕「模型 × 任務」：NVILA·ImageNet-R（預設）／LLaVA·ImageNet-R／NVILA·COCO／LLaVA·COCO；分段按鈕「指標」：H-score（預設）／$A_{up}$。計算器：兩個 `<input type="number">`，$A_{up}\in[0,100]$、$A_{down}\in[0,200]$，步長 0.01；預設組 chip：「Table 1 NVILA COCO Dowser（62.65, 135.5）」（預設）「IconQA zero-shot（62.33, 0.1）」「Table 1 Tailor ImageNet-R（33.38, 80.3）」。
- **計算**：圖上畫表值；每個 ρ 的差距 = Ours − Random 平均，與「差距 / 隨機標準差」。計算器：Avg、H（Eq. 9）；$A_{up}+A_{down}=0$ 時 H = 0；輸入超出範圍時夾回並在欄位下方顯示範圍說明。Tailor 預設組旁註明「表中 Avg 66.9，由同列數字算得 56.8」。
- **顯示**：橫軸 ρ（類別位置等距：0.1, 0.25, 0.5, 0.75, 0.9, 1.0），縱軸 [0, 100]。Ours：`--c-1` 實線＋實心圓，直接標「重要性遮罩」；Random：`--c-2` 虛線＋空心方形＋±1 sd 誤差線，標「隨機遮罩」；Full FT：ρ = 1.0 的星形標記，標「全微調」。每個 ρ 欄是一個可聚焦的透明 `<rect>`（roving tabindex：整張圖一個 tab stop，左右方向鍵移動欄，Enter／Space 選取），點擊／點按亦可；固定讀數框顯示該欄的兩組數字與差距（觸控不用 hover），live region 宣告同樣內容。
- **應觀察到**：NVILA·ImageNet-R 的 H 差距 1.7、3.4、7.7、11.2、5.8；LLaVA·COCO ρ = 0.1 只差 0.2（79.1 ± 0.2 vs 79.3）；計算器預設組得 Avg 99.08、H 85.68。

### 4.5 無障礙（四個元件共同）

- 每個滑桿有 `<label>` 與 `aria-valuetext`（例「ΔW = 0.10」）；分段按鈕用 `aria-pressed`；觸控區 ≥ 44px。
- 每個 SVG `role="img"`＋`aria-label` 描述目前狀態；讀數區 `aria-live="polite"`，只在 `change` 或輸入停止 500ms 後更新。
- 曲線與系列一律以線型＋標記形狀＋直接文字標籤區分；C2 遮罩用斜線紋樣與外框，不只用顏色。
- 沒有任何資訊只能靠 hover 取得（C4 用點選欄位）。
- 每個圖版只有一個 `aria-live="polite"` 區（500ms debounce），宣告 2–3 個關鍵數字（C1–C4 的宣告內容見各元件）。
- C2 熱圖：每張圖一個 tab stop（roving tabindex），方向鍵在 16 格間移動；每格 `aria-label`「第 i 列第 j 欄，分數 0.044，可更新／凍結」。
- 動態：元件狀態更新一律即時重畫，不用 transition；`prefers-reduced-motion: reduce` 時關閉章節目錄與時間線的 smooth scroll。

### 4.6 S8 的 3×3 遮罩靜態例

9 個分數（列優先）：0.90、0.12、0.55、0.30、0.05、0.71、0.48、0.66、0.20；ρ = 2/9 → 最低兩個（0.05、0.12）$M=1$，格內寫「更新」並加斜線紋樣；其餘 7 格 $M=0$ 寫「凍結」。圖說：「M = 1 表示更新，不是凍結（Eq. 7）。論文 ρ = 0.1 時是每層 10% 更新、90% 凍結。」

---

## 5. 驗收數值表（由 `verify/` 腳本產生，與頁面算法相同）

### 5.1 C1（`c1_taylor.py` → `expected/c1_out.txt`，已去除重複列；數值容許 ±1e−5，相對誤差百分比容許 ±0.1 個百分點）

| (i,j) | ΔW | 一階預測 | 真實 ‖Δf‖ | 相對誤差（(預測−真實)/真實） |
| --- | --- | --- | --- | --- |
| (1,1) | +0.01 | 0.01118 | 0.01127 | −0.8% |
| (1,1) | +0.10 | 0.11180 | 0.12071 | −7.4% |
| (1,1) | +0.50 | 0.55902 | 0.77308 | −27.7% |
| (1,1) | −1.00（歸零） | 1.11803 | 0.35476 | +215.1% |
| (1,2) | +2.00 | 1.11803 | 1.88130 | −40.6% |
| (2,1) | −0.50（歸零） | 1.21333 | 1.24037 | −2.2% |
| (2,2) | +0.10 | 0.12133 | 0.12103 | +0.2% |
| (2,2) | −3.00（歸零） | 3.63998 | 3.59732 | +1.2% |

另：$z=(0,2)$；GELU′ = (0.5, 1.08523)；‖J‖ = (1.11803, 2.42665)；$S$ = [[1.11803, 1.11803], [1.21333, 3.63998]]。縱軸上限：(1,1) 7.2351、(1,2) 3.3804、(2,1) 7.8624、(2,2) 3.9312。S2a 齊次例：GELU(2) = 1.9545、2·GELU(1) = 1.6827。

### 5.2 C2（`c2_scores.py` → `expected/c2_out.txt`；真實位移與 Σ S 容許 ±5e−4，遮罩集合必須完全相同）

| b₃ | m | mag：格子／真實／ΣS | wanda：格子／真實／ΣS | full：格子／真實／ΣS | 隨機平均真實 |
| --- | --- | --- | --- | --- | --- |
| 0 | 2 | 8,13／0.1058／0.1251 | 10,13／0.1716／0.1922 | 5,8／0.0362／0.0569 | 0.7588 |
| 0 | 4 | 3,8,11,13／0.9339／1.1191 | 8,10,13,14／0.3152／0.3621 | 5,6,8,10／0.1576／0.1809 | 1.3588 |
| 0 | 8 | 3,4,5,8,10,11,13,15／2.4122／2.7570 | 2,4,5,6,8,10,13,14／1.1884／1.3778 | 5,6,7,8,10,11,13,14／0.7493／0.9584 | 2.4369 |
| −0.75 | 4 | 3,8,11,13／0.7530／0.7736 | 8,10,13,14／0.2683／0.2853 | 8,10,11,12／0.6977／0.0531 | 1.2573 |
| −2 | 4 | 3,8,11,13／0.7206／0.8186 | 8,10,13,14／0.2583／0.2953 | 5,6,8,10／0.0973／0.1141 | 1.2060 |
| −4 | 4 | 3,8,11,13／0.7290／0.7584 | 8,10,13,14／0.2648／0.2819 | 9,10,11,12／0.0000／0.0021 | 1.2075 |
| 0 | 15 | 凍結 6／4.0653／6.9935 | 凍結 7／4.0595／6.8896 | 凍結 1／3.2714／5.9122 | 3.8780 |

排序鍵為 `round(score/1e−9)`；改用此鍵後所有遮罩與舊輸出相同。Gray code 列舉的隨機平均與逐一前向相差 < 1e−9（腳本內斷言）。另：$b_3=0$ 時 $z=(1.91,-2.03,0.05,1.98)$、‖J‖ = (1.4064, 0.1094, 0.8026, 1.1195)；Spearman(full, mag) = 0.296、(full, wanda) = 0.631。$b_3=-4$ 時 Spearman 0.169、0.296。$b_3=-0.75$ 時 $z_3=-0.70$、GELU′ = 0.0234、‖J_3‖ = 0.0348。

### 5.3 C3（`c3_hutch.py` → `expected/c3_out.txt`；精確值容許 ±1e−4，MC 值必須逐位相同）

| a | E[X²] = ‖a‖² | E\|X\| | ‖a‖ | 比值 | Var[X²] |
| --- | --- | --- | --- | --- | --- |
| (3) | 9.0000 | 3.0000 | 3.0000 | 1.0000 | 0 |
| (1,1) | 2.0000 | 1.0000 | 1.4142 | 0.7071 | 4 |
| (3,4) | 25.0000 | 4.0000 | 5.0000 | 0.8000 | 576 |
| 八個 1 | 8.0000 | 2.1875 | 2.8284 | 0.7734 | 112 |
| (3, 0.5×7) | 10.7500 | 3.0078 | 3.2787 | 0.9174 | 68.25 |

MC（seed 1）：a = (3,4)，R = 1/2/8/64 → L2 1.0000/5.0000/5.5678/5.3619，L1 1.0000/4.0000/4.7500/4.4688。200 次重複（seed 1–200）：a = (3,4)，R = 1：L2 4.0000 ± 3.0075；R = 8：L2 4.9809 ± 0.9247、L1 4.0825 ± 1.0681；R = 64：L2 4.9855 ± 0.3282、L1 3.9953 ± 0.4110。八個 1，R = 8：L2 2.6940 ± 0.7018、L1 2.1338 ± 0.6904。

### 5.4 C4 與內文數字（`c4_tables.py` → `expected/c4_out.txt`）

- 由 $A_{up},A_{down}$ 重算的 Avg/H 與 Tables 8/9 的 40 列＋4 列 Full FT 全部標 `ok`（0 列 MISMATCH）。判準 `rounding_ok`：存在未捨入的 $U\in[A_{up}\pm0.05]$、$D\in[A_{down}\pm0.05]$，使 Avg、H 都四捨五入到表值（±0.05）。三列直接重算差 > 0.05（NVILA·ImageNet-R ρ = 0.1：H 69.64 vs 69.7；LLaVA·COCO ρ = 0.5：56.45 vs 56.5；ρ = 0.9：12.57 vs 12.5）都屬這種捨入。對照：Table 1 Tailor（33.38, 80.3）印 Avg 66.9，`rounding_ok` 為 False，是真正的不一致（R7）。頁面一律顯示論文表值。
- H 差距（Ours − Random）：NVILA·ImageNet-R 1.7/3.4/7.7/11.2/5.8；LLaVA·ImageNet-R 0.9/4.4/9.4/2.4/0.7；NVILA·COCO 0.5/1.3/9.8/11.0/4.4；LLaVA·COCO 0.2/1.0/6.7/3.8/0.6。
- 計算器：(62.65, 135.5) → Avg 99.08、H 85.68；(62.33, 0.1) → Avg 31.22、H 0.20；(33.38, 80.3) → Avg 56.84、H 47.16。
- Hamming：0.050 → 重疊 0.750；0.081 → 0.595；上限 0.2、獨立期望 0.18。BWT：1.25×(MFN − MFT) = −19.38、−16.50、−9.88、−3.12。

### 5.5 selfTest（`?test`）必須通過的斷言

`erf(0.5)` 對 0.5204999 誤差 < 2e−7；C1 (1,1) 歸零真實 0.35476、預測 1.11803（±1e−5）；C2 $b_3=-0.75,m=4$ full 遮罩 = {8,10,11,12}、真實 0.6977（±5e−4）；C2 $b_3=0$ 隨機平均 $m=4$ = 1.3588（Gray code 分段計算完成後檢查）；C2 $b_3=0$ 時 $S_{21}$ 與 $S_{22}$ 同鍵且 $m=2$ 的 full 遮罩 = {5,8}；C3 (3,4) seed 1、R = 8 的 L2 = 5.5678；C3 八個 1 的 E|X| = 2.1875；C4 計算器 (62.65, 135.5) → H 85.68；`scrollWidth <= innerWidth`。輸出一行 `selfTest: N passed, M failed`。

---

## 6. 實作計畫

### 6.1 沿用參考頁（`tutorials/prediction-targets-k-diff.html`）的部分，不重寫

CSS tokens 與深色模式、字型與 Google Fonts／KaTeX 0.16.9 載入（cdnjs、`defer`、auto-render、`$…$`／`$$…$$`、載入失敗保留原文）、appbar 與主題切換（`pm-theme`）、`#pm-bar`、左側 sticky 章節目錄／窄螢幕橫列、`.plate`／`.plate-h`／`.plate-f`／`.readout`／`.seg`／`.rng`／`.legend`／`.check` details／`.eq`＋`.eq-n` 方程式編號／`.tscroll` 表格橫捲、`el()`、`scaleLin`、`mulberry32`、`renderTimeline`（卡片列、篩選 chip、左右捲動、JSON 解析失敗的退路）、`selfTest` 框架、reduced-motion 處理。頁面結構與參考頁 5.3 相同。

### 6.2 只列新的部分

- 檔案：`tut/586/build/index.html`，單檔、全部內嵌，不 `fetch`。slug `model-dowser-importance-probing`；論文連結 `../index.html#paper-586`；arXiv `https://arxiv.org/abs/2602.04509`；筆記 `<slug>.html`。
- 圖表色：沿用參考頁兩個資料色的數值，改名為 `--c-1`（#3F6E8C／深色 #8DB4D3：真實值、重要性遮罩）與 `--c-2`（#6B7A3A／深色 #A9BC7C：一階預測、隨機遮罩）。C2 熱圖用 `--accent` 的明度階（`color-mix(in oklab, var(--accent) p%, var(--surface))`，p = 8–80）。
- 新函式：`erf`、`Phi`、`phi`、`gelu`、`dgelu`、`fwd(W,b,V,h)→{z,f}`、`jcols(V,z)`、`norm`、`stableLowest(scores,m)`、`spearman(a,b)`、`randomMeans(b3)`（快取）、`exactX(a)`、`mcX(a,R,seed)`、`hs(u,d)`、`avg(u,d)`。
- 元件 init：`initC1()`、`initC2()`、`initC3()`、`initC4()`，各自只碰自己的容器；狀態物件各一個；`render*()` 純函式式重畫整張 SVG。
- 靜態 SVG：S2a 的 2-2-3 網路圖（與 C1 左側共用一個繪圖函式 `drawNet(svg, sel)`）；S4／S5／C2 的三格公式條為 HTML（`.factor-bar`，三個 `<span>`，`data-on` 控制實心／空心，並各附文字「用」／「未用」）；S2a 的 GELU／GELU′ 小圖；S8 的 3×3 遮罩例（4.6）。新 token：`--hatch`（見 C2）。
- S8 Table 3、S9 主結果表為靜態 HTML 表格（`<caption>`、`.tscroll`）。
- 時間線：`notes/timeline.json` 原樣嵌入 `<script type="application/json" id="timeline-data">`。`TL_GROUP`（以 arXiv ID 為 key）：

  | 類別標籤 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 遺忘與持續學習 | 1312.6211、1612.00796、1711.09601 | `dowser-catastrophic-forgetting-mllm-fine-tuning.html` |
  | 重要性與剪枝 | 1711.05769、2006.05467、2306.11695、2402.17762 | `dowser-parameter-importance-pruning-scores.html` |
  | Post-merging | 2302.06600、2311.03099、2402.12048 | `dowser-post-merging-methods.html` |
  | Sparse FT | 2404.18466、2411.10928 | `dowser-sparse-finetuning-hft-spider.html` |
  | 模型 | 2412.04468 | `dowser-llava-nvila-setting.html` |
  | 本文 | 2602.04509 | `dowser-functional-importance-theory-and-probing.html` |

  其餘 ID 歸「其他」，連到 `dowser-lineage-hub.html`。本文卡片 `--accent` 外框。
- S12 筆記順序（照總覽）：`dowser-lineage-hub`、`dowser-catastrophic-forgetting-mllm-fine-tuning`、`dowser-llava-nvila-setting`、`dowser-post-merging-methods`、`dowser-parameter-importance-pruning-scores`、`dowser-sparse-finetuning-hft-spider`、`dowser-functional-importance-theory-and-probing`、`dowser-experiments-depth-and-memory`，各附 frontmatter `title`。

### 6.3 數學排版與引用

- 公式右側等寬字標論文編號（「Eq. 3」）；**每個** `.eq` 包在 `overflow-x:auto` 容器並加右側漸層捲動提示（Eq. 2、5、7、8、26 等）；Eq. 2 與 Eq. 26 在 < 600px 時改為 `aligned` 兩行。
- SVG 內、讀數、chip、按鈕與 `aria-label` 一律不放 `$…$`（這些節點會被重畫，不跑 KaTeX）；下標用 SVG `<tspan baseline-shift="sub" font-size="75%">` 或 HTML `<sub>`，不用 Unicode 下標字元 ᵢⱼ（JetBrains Mono 缺字）；數字下標 ₁₂ 可用（字型有）。只有 `<details>` 展開後的靜態答案文字含 TeX，展開時呼叫一次 `renderMathInElement`。
- 所有 `.tscroll` 表格（S8 Table 3、S9 主結果表、5.x 不上頁）第一欄 `position:sticky; left:0; background:var(--surface)`，390px 時橫捲不丟失列名。
- 內文每個論文數字附出處；元件圖版底部一行「讀數由本頁依 Eq. ○ 即時計算」＋必要的「本頁推導」標籤。〔本頁推導〕用小 pill 標籤（`--ink-3` 邊框）＋一行依據；「作者認為」與論文結果分開寫。
- 文案：繁體中文、台灣用語、字面學術語氣；不用比喻與反問。用詞固定：「更新」（M = 1）／「凍結」（M = 0）、「輸出敏感度」、「一階近似」、「常見錯誤」。專有名詞首次出現附英文。

### 6.4 建置後自我檢查

截圖 390／768／1024／1440＋深色、鍵盤 focus、reduced-motion；依字體、留白、視覺層級、色彩、動態、微互動、響應式、原創性逐項評分並修正；`?test` 全數通過；逐條核對 12 個學習目標的答案出現在指定章節內文；跑 `npx impeccable detect`。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具 ≠ 論文實驗 | C1–C2 是作者沒有做的玩具計算；玩具 MLP 沒有「預訓練泛化」可遺忘。 | 圖版底部明寫；不用玩具重現「早期層更脆弱」或「隨機遮罩崩潰」。C4 只畫論文表格數字。 |
| R2 | 玩具數值是自選的 | C1 網路取自顧問筆記；C2 的 $W,V,h$ 是本設計選來讓三種排序互不相同（推定：無論文依據）。 | C1、C2 圖版都標「本頁自選數值（推定），非論文設定」，C1 的 $V$、$x$ 亦同。 |
| R3 | 論文未寫明的實作細節 | $f$ 的定義、序列位置彙總、R 樣本如何合成、prompt 長度／溫度、哪些參數被評分、「層」的粒度、探測記憶體。 | S7 列為「論文未寫明」，不自行補成事實。 |
| R4 | L1 vs L2 估計量 | 正文、App. D、Fig. 2 三種寫法。 | S6b 照錄三處；C3 同時顯示兩者。 |
| R5 | 剪枝譜系是本頁推導 | 論文未提 SNIP／OBD；筆記 `dowser-parameter-importance-pruning-scores` 也未收錄。 | 依協調者決定：不新增 SNIP 筆記。SNIP／OBD／OBS／MAS 放在 S5 末尾預設收合的 aside，標〔本頁推導〕，只寫公式形式，不引用其實驗數字。 |
| R6 | 主張與證據的距離 | 「不損失預訓練知識」「各方法下游相近」「對 ρ 不敏感」「對 lr 穩定」「text-only almost identical」均有反例數字。 | S9／S10 以表中數字呈現，用「作者認為」區分解釋。 |
| R7 | 內部不一致 | Tailor Avg 66.9；p 值歸屬；SPIDER 32 層；逐層 vs 全域；Fig. 2 加號；top vs bottom 10%；Hamming 0.226；Table 3 Grafting 欄；BWT 分母；App. L 61.2 敘述。 | S10 逐條列出，引用時避開 Tailor 66.9 這格（C4 計算器以它示範重算）。 |
| R8 | 深度曲線沒有數據 | Fig. 1、3、5 只有圖。 | 只用 §4.2 文字（4–16 層），不重繪曲線。 |
| R9 | C2 隨機平均的運算量 | 2¹⁶ 子集 × 前向；低階手機可能 > 50ms。 | 預設即分段（Gray code 增量更新、每段 4096 步、`runToken` 取消舊工作、「計算中」狀態、結果快取）。不採用 Web Worker：單檔頁面需以 Blob 建立 Worker，增加程式而分段已足以避免阻塞；若建置時單段實測 > 16ms，再把每段降到 1024 步。 |
| R10 | 筆記與時間線更新 | 建置前 `notes/` 若有增減需同步 `TL_GROUP` 與 S12。 | 建置時重新讀取；未列 ID 歸「其他」並在建置回報中列出。 |
| R11 | 初學者門檻 | Jacobian、Taylor、Rademacher 對初學者是新的。 | S2a／S2b 用小網路圖逐一定義；S6a 先用 d = 2 手算再一般化。 |
| R12 | 版本 | 本文件依 v7（2026-06-01）。表格數字以 v7 為準。 | 頁尾註明版本。 |
| R13 | 與審查意見的差異 | (a) 第 3 項：保留 $W_{22}$ 玩具例，但改寫為「magnitude 忽略 $\|J_i\|$」，齊次性只作為論文主張陳述，沒有刪除例子（它仍是 LO6 的可計算部分）。(b) 第 11 項：選分段計算而非 Worker（理由見 R9）。(c) 第 12 項：排序鍵以 1e−9 捨入，但畫面顯示三位小數（由同一捨入值格式化）；顯示到 1e−9 對讀者無意義。(d) 第 4 項：75% 只留在條件式 aside，不進 LO10 答案。 | 已在對應章節落實。 |

---

## 8. 修訂紀錄（round 1）

| 審查項 | 變更位置 |
| --- | --- |
| 1 H-score | 1.4 步驟 1 新增調和平均定義與 90/10 例、CIDEr 說明；S1 大綱；LO2 |
| 2 S2 過載 | S2 拆為 S2a（線性層、玩具網路、GELU、齊次、純量鏈鎖律）與 S2b（欄向量 $J_i$、向量鏈鎖律、Taylor、假說、符號表）；全頁統一「第 i 個欄向量」 |
| 3 齊次例 | 1.4 步驟 5、LO6、S4／S5 大綱、R13(a) |
| 4 Hamming 75% | 1.4 步驟 9 改為條件式 aside 並加 top/bottom 10%；LO10 刪除 75% |
| 5 同列排序 | 1.4 步驟 6 第 1 點加「單一輸入」與 Eq. 5／Eq. 7／App. K 的例外；S4 先寫 $i$＝列、$\|J_i\|$ 列內相同；LO7 |
| 6 設定混用 | 1.4 步驟 11 隨機遮罩段（71.2 vs 69.7）；LO12；C4 圖說；S9 大綱 |
| 7 §1.1 | 改為「由 Theorem 3.1 把 |ΔW| 代換為 |W| 定義」 |
| 8 玩具架構 | 符號表加 $a,V,b,\mathrm{GELU},\Phi,\varphi$；S2a 先定義 $f=V\,\mathrm{GELU}(Wh+b)$ 與 GELU 小圖 |
| 9 前向引用 | $W_{22}$ 例移到 S5、C2 之前，先展示玩具層 |
| 10 S5 密度 | SNIP/OBD/OBS/MAS 移入收合 aside；$S^2$ 等式移到 S6b 收合區 |
| 11 C2 主執行緒 | Gray code ＋分段＋runToken＋計算中；`c2_scores.py` 新增 `random_mean_gray` 並斷言一致；R9 |
| 12 同分 | ε = 1e−9 捨入排序鍵；Spearman 變異數 0；m = 15 驗收列 |
| 13 C1 格式 | 百分比與符號慣例；縱軸上限規則與四個值（`expected/c1_out.txt`） |
| 14 390px | 6.3：所有 `.eq` 橫捲＋提示、`.tscroll` sticky 首欄；C2 390px 版面 |
| 15 定理條件 | 1.4 步驟 3 理想化條件；S3 小框；LO3 |
| 16 R 合成 | 1.4 步驟 7 標推定；C3 概念 |
| 17 C2 頁尾 | 加「玩具沒有測試假說」一行 |
| 18 單次執行 | 改為「表中只有 Random 附標準差；論文未說明本法次數」 |
| 19 Table 11 vs 5 | 1.4 步驟 9 註明參考對不同 |
| 20 Table 13 | 1.4 步驟 11 註明另一組 9 個上游基準、Zeroshot 63.9 |
| 21 Szarek | 1.4 步驟 8 |
| 22 C1 推定 | C1 固定網路標籤；R2 |
| 23 動態 KaTeX | 6.3 |
| 24 無障礙 | 4.5 新增 live region 內容、C2 roving tabindex、C3 tab 順序、C4 方向鍵／Enter |
| 25 對比 | C2 格內文字切色、`--hatch`、公式條「用／未用」、C3 標籤錯開引線 |
| 26 動態 | 4.5 即時重畫、reduced-motion 關 smooth scroll |
| 27 文風 | Hero 副標與卡片改白話；S1 標題「作者觀察到」；S6 拆 6a／6b 並調長度；S8 先放 3×3 遮罩例（4.6）；LO10、LO12 標記憶型 |

## 9. 修訂紀錄（readiness 後）

- 符號表加入 $|P|$、$T$、$A_{t,i}$、MFT、MFN、BWT；BWT 採 Eq. 31 定義並說明表值相當於除以 $T$（步驟 11）。
- `c4_tables.py` 改用 `rounding_ok` 判準，0 列 MISMATCH；5.4 改寫為與腳本輸出一致；Tailor 66.9 以同判準確認為真正不一致。
- S5 對 Eq. 5／Eq. 7／$N$ 改為前向指引；C2 排序方向與同分規則寫明；C4 常數命名 `C4_TABLE`、`C4_AUP_SD`；4.0 寫出 erf 係數；`c1_out` 去重；相對誤差容許 ±0.1 個百分點；驗收來源移到 `verify/expected/`，以 `check.sh` 比對。

## 10. 建置紀錄（build round）

- 組裝：`build/src/{body1,body2,body3}.html`、`extra.css`、`app.js` 由 `build/src/assemble.py` 與參考頁的 head／CSS 合成 `build/index.html`（重跑即重建）；〔本頁推導〕在組裝時轉為 `.tag` pill。
- `?test`：79 項斷言全過（1440 與 390 寬）；涵蓋 5.1–5.5 全部驗收表、無水平捲動。
- 與設計的差異：(a) C2 熱圖改以「格內底部長條長度 ∝ 分數」編碼，而非填色明度——底色固定，文字對比不受分數影響，也不依賴顏色；(b) 非檢查題的收合區用 `details.aside`（驗收腳本只剝除 `details.check`）；(c) 頁面內文以頁面章節號（01–14）互相指引，取代設計代號 S2a 等；(d) Eq. 7、8 拆成兩個方程式塊（390px 不需橫捲）；(e) 390px 時 C2 讀數表隱藏「可更新格子」欄（已顯示在各熱圖標題下）；(f) C3 長條圖只在相鄰長條間距 ≥ 44px 時標示數值。
- impeccable：0 anti-patterns；advisory 為 numbered-section-labels（DESIGN.md 的目錄式編號）與 repeating-stripes（C2／遮罩例的斜線紋樣是「可更新」的非顏色編碼，保留）。
- 頁首 hash 錨點截圖在 headless 下不可靠，視覺檢查改用整頁截圖裁切（`build/caps/`）。
