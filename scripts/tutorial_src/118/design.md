# 設計文件：〈Generative Verifiers: Reward Modeling as Next-Token Prediction〉互動教學

- 論文：Lunjun Zhang、Arian Hosseini、Hritik Bansal、Mehran Kazemi、Aviral Kumar、Rishabh Agarwal，arXiv 2408.15240（v3，ICLR 2025）。資料庫編號 No. 118。
- 原文：`tut/118/src/paper.txt`（PDF 同目錄）。本文件的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」皆指論文原文編號（Table 1 指 MMLU 表；附錄表格寫作 Table A.1、E.12 等）。
- 產出：`tut/118/build/index.html`（單一 HTML 檔）。上線位置 `tutorials/generative-verifiers.html`（page slug `generative-verifiers`），站內相對連結以上線位置為準。
- 驗算腳本：`tut/118/verify/check.py`（純 Python，與頁面相同的演算法與 erfc 近似；輸出存在 `verify/check.out`）。第 5 節驗收表全部由它產生。
- 研究筆記：`tut/118/notes/`（總覽 `genrm-lineage-hub` + 7 篇主題筆記 + `timeline.json`，14 筆）。
- 標記慣例：本頁自行推導、論文沒寫出的結論標「〔本頁推導〕」；本頁為示範而自訂、與論文無關的輸入值標「〔示範值〕」；用論文數字推估、論文沒明寫的事實標「推定」。作者的解釋寫成「作者認為／作者推測」，與量測結果分開。
- 前一批教學的教訓已併入：每個學習目標的參考答案寫成指定章節內文裡的一段完整敘述（5.6）；符號與術語（含 hero 卡片裡的 benchmark 名稱）在首次出現處定義；玩具模型第一行寫出內建假設、不得寫死論文結論；百分比與百分點分開；390px 版面、每元件一個 debounced live region、動態文字不用組合字元、重計算不阻塞主執行緒。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

大型語言模型（LLM）解數學題時常寫出「看似合理但錯」的解答。常見補救是 **Best-of-N**：讓生成模型寫 N 份候選解答，由一個學過的**驗證器**（verifier，也稱獎勵模型 reward model，RM）替每份打分，選分數最高者（Sec. 2）。過去的 LLM 驗證器多訓練成**判別式分類器**：在某個特殊 token 的 logit 上接 sigmoid 輸出正確機率（Eq. 3），完全不使用 LLM 生成文字的能力。本文提出 **GenRM**：把驗證寫成下一個 token 的預測——問模型「Is the answer correct (Yes/No)?」，以「Yes」token 的機率當分數（Eq. 4），用一般的 SFT 損失訓練（Eq. 2）。因為驗證只是「再生成一個 token」，GenRM 可以 (1) 與解題資料混合訓練（Eq. 5）；(2) 先生成一段驗證用的思維鏈（chain-of-thought，CoT）再回答 Yes/No（GenRM-CoT，Eq. 6）；(3) 抽 K 條驗證 CoT 平均 Yes 機率（majority voting，Eq. 7），用更多推論運算換更準的驗證。在固定生成器的 Best-of-N 評估中，論文報告演算法任務 5% → 45.3%、GSM8K 73% → 93.4%、只用 GSM8K 訓練的驗證器轉到 MATH 28% → 44.6%、MMLU abstract algebra 37.9% → 53.5%（Abstract、Fig. 1、Table 1）。箭頭起點：Table 1 明寫 37.9% 是生成器 Pass@1；Fig. 1 與 Abstract 的三個起點未寫明，本頁推定同為 Pass@1（推定，見 R1）。所以這些箭頭是「加上驗證器 vs 不用驗證器」，不是驗證器之間的比較；同一表中與判別式 RM 的同條件比較是 abstract algebra 50% → 53.5%（Table 1，表中未註明 N）。GSM8K 的訓練用驗證 CoT 由 Gemini 1.0 Pro 在「看得到參考解答」的條件下合成（Sec. 3.3、Table A.2）。

### 1.2 研究動機（Sec. 1）

1. LLM 會自信地犯邏輯與事實錯誤；推理題中一個錯誤就讓整份解答失效。
2. Best-of-N 的效果取決於驗證器多準，所以要找更好的驗證器訓練法。
3. 判別式 RM 只輸出一個數字，錯過 LLM 的三項能力：統一的指令微調、CoT 推理、用額外推論運算提升表現。
4. LLM-as-a-Judge（直接提示現成 LLM 當評審）有這三項能力，但在推理題上通常不如訓練過的驗證器（作者在 Fig. 1 也觀察到）。

### 1.3 貢獻

1. GenRM：以 next-token prediction 訓練驗證器，分數＝$p_\theta(\text{Yes}\mid x,y,I)$（Sec. 3.1，Eq. 4）。
2. 統一訓練：驗證損失加上 $\lambda$ 倍的解題 SFT 損失（Sec. 3.2，Eq. 5），兩個任務互相幫助（Sec. 4.2）。
3. GenRM-CoT 與 majority voting：先生成驗證理由再判斷，並平均 K 條理由（Sec. 3.3，Eq. 6–7）。
4. 參考解答引導的合成驗證理由，不需要人工或更強的模型（Sec. 3.3、4.4）。
5. 實驗：演算法字串任務、GSM8K、MATH、MMLU 數學子集；與判別式 RM、DPO 驗證器、LLM-as-a-Judge、self-consistency 比較；模型大小與推論運算的 scaling（Sec. 4）。

### 1.4 從作者的角度逐步拆解

#### 符號表（頁面不放單一總表：每章末尾一張可收合小表，只列該章新定義的符號；下表的「章」欄即其位置）

| 符號 | 意義 | 章 |
| --- | --- | --- |
| $\mathbf x$ | 輸入題目（context），例如一道數學題 | S1 |
| $\mathbf y=(y_1,\dots,y_T)$ | 一份解答（token 序列）；$\mathbf y^+$ 正確解、$\mathbf y^-$ 錯誤解 | S1 |
| $\theta$ | LLM 參數 | S2 |
| $p_\theta(\mathbf y\mid\mathbf x)$ | 自迴歸生成機率（Eq. 1） | S2 |
| $M$ | 詞彙表大小（只在 Sec. 2 出現；頁面避免與其他量混用） | —（頁面不用） |
| $z_t$、$\gamma$ | token 的 logit、softmax 溫度（Sec. 2；原文誤植一處為 $\tau=0$） | S2 |
| $\mathcal D_{\text{correct}}$、$\mathcal D_{\text{incorrect}}$ | 正確／錯誤的「題目–解答」配對集合 | S2 |
| $r_\theta(\mathbf x,\mathbf y)\in[0,1]$ | 驗證器分數 | S2 |
| $I$ | 固定提示「Is the answer correct (Yes/No)?」 | S3 |
| $I_{\text{CoT}}$ | 固定提示「Let's verify step by step.」 | S5 |
| $\mathbf v_{\text{CoT}}$ | 驗證理由（驗證用 CoT） | S5 |
| $\lambda$ | 解題資料的混合係數（Eq. 5） | S4 |
| $K$ | majority voting 抽的驗證理由條數（Eq. 7） | S6 |
| $N$ | Best-of-N 的候選解答數 | S1 |
| Pass@1、Pass@N | 單次取樣的答對率；N 份中至少一份正確的題目比率（＝完美驗證器下的 Best-of-N），皆為對題目的平均 | S1 |
| μ | C1 玩具模型中正確解分數比錯誤解高出的平均量（以標準差為單位） | S1 |
| softmax、logit、sigmoid、BCE | 見 S2 的前置知識段 | S2 |
| $K_{\text{top}}$ | weighted SC 每個答案只加總的最高分份數（App. C） | S9 |

#### 步驟 1：問題設定——Best-of-N 與兩種評估指標（Sec. 2、Sec. 4 Evaluation protocol）

- Best-of-N：對一道題從生成器抽 N 份解答，驗證器打分，選最高分者為最終答案。驗證器越好，選到正確解的機會越大。
- 兩種指標：**Best-of-N 解題率**（固定生成器，用驗證器選解後的測試集答對率）與 **RM accuracy**（驗證器把單份解答分類為對／錯的準確率）。論文寫兩者相關，但 RM accuracy 只評估逐點分類，Best-of-N 評估的是「排序後選出正確解」的能力。
- 兩個邊界〔本頁推導，頁面 C1 呈現〕：驗證器分數與正確性無關時，Best-of-N＝隨機挑一份＝Pass@1；完美驗證器時 Best-of-N＝Pass@N。對**單一題目**，若每份解答獨立地以機率 $p$ 正確，Pass@N $=1-(1-p)^N$。實際資料集的 Pass@N 是這個量對各題（各有自己的 $p$）的平均；各題 $p$ 不同時，它不等於把平均正確率代入公式的值（例如一半題目 $p=0$、一半 $p=1$，平均 $p=0.5$，Pass@N 恆為 0.5）〔本頁推導〕。C1、C3 的玩具模型讓每題都有相同的 $p$，頁面明寫這一點。

#### 步驟 2：判別式 RM 與其他基線（Sec. 2、Sec. 4 Baselines）

（以下兩條是 S2 前置知識段的內容；頁面在 S2 才引入 Eq. 1–2、softmax 與溫度 $\gamma$，S1 不出現。）

- 自迴歸生成：$p_\theta(\mathbf y\mid\mathbf x)=\prod_{t=1}^{T}p_\theta(y_t\mid\mathbf x,y_{<t})$（Eq. 1），每個 token 由 softmax（溫度 $\gamma$）在 logit 上算出；溫度 0 即貪婪解碼。
- SFT 損失：$\mathcal L_{\text{SFT}}(\theta,\mathcal D)=-\mathbb E_{(\mathbf x,\mathbf y)\sim\mathcal D}\big[\sum_{t}\log p_\theta(y_t\mid\mathbf x,y_{<t})\big]$（Eq. 2）。

$$\mathcal L(\theta,\mathcal D_{RM})=-\mathbb E_{(\mathbf x,\mathbf y^+)\sim\mathcal D_{\text{correct}}}\log r_\theta(\mathbf x,\mathbf y^+)-\mathbb E_{(\mathbf x,\mathbf y^-)\sim\mathcal D_{\text{incorrect}}}\log\big(1-r_\theta(\mathbf x,\mathbf y^-)\big),\quad r_\theta=\mathrm{sigmoid}(z_{cls})\tag{Eq. 3}$$

- $z_{cls}$ 是一個特殊詞彙 token $cls$ 的 logit。資料一律正確／錯誤各半（balanced）。論文稱它為 Discriminative RM 或 ORM（outcome reward model），是主要基線（Cobbe et al., 2021）。
- LLM-as-a-Judge：不微調，直接提示 LLM 驗證（Table A.3 的 zero-shot 提示）；實驗中用 CoT 提示產生 32 條驗證理由，取多數決。
- DPO 驗證器：依 Hosseini et al. (2024)，在「正確解 vs 錯誤解」配對上以 DPO 訓練。App. B：分數直接用 $\log\pi_{\text{DPO}}(\text{solution}\mid\text{question})$，不減參考政策的 log 機率，表現較好（Fig. D.5）。
- Self-consistency：不用驗證器，抽多份解答取最常見的最終答案。作者指出它與驗證器互補，可結合成 weighted self-consistency（步驟 8）。

#### 步驟 3：GenRM 直接驗證器（Sec. 3.1，Eq. 4）

- 訓練資料：$\mathcal D_{\text{Direct}}=\{((\mathbf x,\mathbf y^+,I),\text{'Yes'})\}\cup\{((\mathbf x,\mathbf y^-,I),\text{'No'})\}$，在其上最小化 Eq. 2 的 SFT 損失，等於最大化正確解的 $\log p_\theta(\text{Yes}\mid\cdot)$ 與錯誤解的 $\log p_\theta(\text{No}\mid\cdot)$。
- 推論：$r_{\text{Direct}}(\mathbf x,\mathbf y)=p_\theta(\text{Yes}\mid\mathbf x,\mathbf y,I)$（Eq. 4）。作者認為使用機率（而不是只看二元 Yes/No）納入了驗證器的信心，降低測試時判錯的機會。
- 〔本頁推導，C2 呈現〕$p_\theta(\text{Yes})$ 是對整個詞彙表做 softmax：$p(\text{Yes})=e^{z_Y/\gamma}/\big(e^{z_Y/\gamma}+e^{z_N/\gamma}+\sum_{\text{其他}}e^{z_j/\gamma}\big)$。若其他 token 的機率可忽略，$p(\text{Yes})\approx\mathrm{sigmoid}\big((z_Y-z_N)/\gamma\big)$，正確解上的 SFT 損失 $-\log p(\text{Yes})$ 就和 Eq. 3 的 BCE 形式相同。所以兩者的**損失形式相同**；實際分數是否近似，取決於其他 token 分到的機率，論文沒有量測。兩者的參數也不同：判別式 RM 用特殊 token $cls$ 的 logit，GenRM 用語言模型頭上既有的 Yes／No token 並接在提示 $I$ 之後。其餘差別在於分數由語言模型頭的既有 token 產生、可以和其他 SFT 資料混合、可以接在 CoT 之後。論文的 Fig. D.1 也顯示直接 GenRM（不用 CoT）只「略好或相當」於判別式 RM，主要增益來自 CoT 與 majority voting（Sec. 4.1）。
- 溫度：$\gamma$ 是 Sec. 2 為生成定義的；Eq. 4 沒有說評分時用 $\gamma\ne1$，本頁推定評分取 $\gamma=1$（推定）。C2 的 $\gamma$ 滑桿只為示範 softmax 的性質，標〔示範值〕。
- 〔本頁推導〕$-\log p(\text{Yes})$ 對 logit 的梯度：$\partial/\partial z_j=(p_j-\mathbb 1[j=\text{Yes}])/\gamma$。

#### 步驟 4：統一生成與驗證（Sec. 3.2，Eq. 5，Sec. 4.2）

$$\mathcal L_{\text{GenRM}}(\theta,\mathcal D_{\text{verify}})=\mathcal L_{\text{SFT}}(\theta,\mathcal D_{\text{verify}})+\lambda\,\mathcal L_{\text{SFT}}(\theta,\mathcal D_{\text{correct}})\tag{Eq. 5}$$

- $\mathcal D_{\text{verify}}$ 是驗證資料：S4 只用已定義的 $\mathcal D_{\text{Direct}}$ 寫出 Eq. 5，並加一句「S5 會介紹另一種驗證資料 $\mathcal D_{\text{CoT}}$（每份解答附一段逐步驗證理由），它同樣可以放進 Eq. 5」；$\mathcal D_{\text{correct}}$ 是「題目→正確解」的生成資料；$\lambda>0$ 控制混合比例。論文預設所有 GenRM 都用 Eq. 5。
- 預設值（App. D）：演算法任務 $\lambda=1/3$，GSM8K $\lambda=1/4$。
- 結果：加入解題資料在所有任務上都提升驗證（Fig. 9；「Verification Only」即 $\lambda=0$；Last Letter 與 Word Sorting 用 2B，GSM8K 用 7B）。但解題資料過多會降低驗證表現，中間值最好（Fig. D.3，Gemma-7B，GSM8K，$\lambda\in\{0,1/4,1/3,1/2,1\}$，$1/4$ 是好值）。
- 反方向：把 CoT 驗證資料加進生成器訓練，生成器本身的 Pass@N（Best-of-N with oracle verifier）也較高；演算法任務（驗證資料為真值）的增益大於 GSM8K（合成理由可能有錯）（Fig. 10）。作者推測教模型驗證能加深它對生成過程的理解。
- 作者對比 DPO（Sec. 5）的內容不放在 S4：DPO 在 S2 只有一句定義，這段比較移到 S11 研究脈絡（DPO／V-STaR 卡片之後一段：DPO 以政策 logits 隱式表示獎勵，作者引用的文獻指出它在推理上有錯誤外推與表徵退化；作者寫 GenRM 損失較簡單、不需參考政策、驗證表現顯著較好）。

#### 步驟 5：GenRM-CoT 與 majority voting（Sec. 3.3，Eq. 6–7）

- 訓練資料：$\mathcal D_{\text{CoT}}=\{((\mathbf x,\mathbf y^+,I_{\text{CoT}}),(\mathbf v_{\text{CoT}},I,\text{'Yes'}))\}\cup\{((\mathbf x,\mathbf y^-,I_{\text{CoT}}),(\mathbf v_{\text{CoT}},I,\text{'No'}))\}$，$I_{\text{CoT}}$＝「Let's verify step by step.」。理由可以是人寫或 LLM 生成。
- 推論：

$$r_{\text{CoT}}(\mathbf x,\mathbf y)=p_\theta(\text{Yes}\mid\mathbf x,\mathbf y,I_{\text{CoT}},\mathbf v_{\text{CoT}},I),\qquad \mathbf v_{\text{CoT}}\sim p_\theta(\cdot\mid\mathbf x,\mathbf y,I_{\text{CoT}})\tag{Eq. 6}$$

$$r_{\text{MajV@}K}(\mathbf x,\mathbf y)=\frac1K\sum_{i=1}^{K}p_\theta\big(\text{Yes}\mid\mathbf x,\mathbf y,I_{\text{CoT}},\mathbf v^{(i)}_{\text{CoT}},I\big),\qquad \mathbf v^{(i)}_{\text{CoT}}\sim p_\theta(\cdot\mid\mathbf x,\mathbf y,I_{\text{CoT}})\tag{Eq. 7}$$

- 名稱雖叫 majority voting，實際算的是 K 條理由的 Yes 機率**平均**。作者認為單條理由可能推理出錯，平均可減輕其影響；判別式驗證器沒有這種用推論運算換準確度的途徑。除非另註，GenRM-CoT 結果皆為 $K=32$。
- 「greedy」指以貪婪解碼產生一條理由（Eq. 6）。Fig. 11：三種 Gemma 大小下，$K=2$ 以內就超過 greedy。

#### 步驟 6：合成驗證理由（Sec. 3.3、Sec. 4.4、App. C、Table A.1–A.2）

- 演算法任務：以程式產生真值驗證 CoT（Table A.1），代表「訓練理由沒有雜訊」的理想情形。
- GSM8K：只依最終 Yes/No 過濾的做法是用「Let's verify step by step」提示並只保留最終 Yes/No 判對的理由；作者指出這樣的理由品質仍差，因為隨機猜也有 50% 會判對。
- 改進：提示裡額外給一份**參考解答**（任何得到正確最終答案的模型解答），讓 LLM 比較容易指出待驗解答的錯誤（reference-guided grading，Table A.2，Gemini 1.0 Pro）。參考解答只用於產生訓練資料，微調驗證器時不放入，避免訓練／測試不一致。
- 過濾（App. C）：答案檢查器（字串比對或 SymPy）會有偽陰性；錯誤推理也可能碰巧得到正確答案（偽陽性）。作者只保留「超過 50% 的驗證理由與答案檢查器判定一致」之解答的理由。
- 品質與數量都重要：有參考引導 91.7% vs 無引導 87.8%（Gemma-7B，GSM8K，Fig. 13 圖說數字；圖說未寫指標，圖的橫軸到 N = 16，推定為 Best-of-16）；每份解答的理由數 1→2→4 提升 RM accuracy 與 Best-of-N（Fig. 14，greedy）；每題解答數 8→16→32 也提升（Fig. D.2，每份 4 條理由）。作者推測多條理由有「集成」效果，避免過擬合到單條理由的錯誤。
- 生成解答與合成理由用同一個模型（Gemini 1.0 Pro），不需要更強的模型或人工（Sec. 4.4）。

#### 步驟 7：實驗結果（Sec. 4）

任務（Sec. 4、App. A）：
- Last Letter Concatenation（取每個字的最後一個字母並串接）與 BBH Word Sorting（依字母排序）：訓練長度 {2,3,4}；Sec. 4 寫評估長度 {5,6}，App. A 寫 Last Letter 評估長度 6、Word Sorting 長度 5（兩處照錄，見 R4）。驗證器 Gemma-2B，生成器 Gemma-2B。
- GSM8K（小學數學應用題；約 7.2K 訓練題、128 驗證題、1.3K 測試題）：每題生成 50 份解答，訓練時每題最多取 16 份正確＋16 份錯誤；測試每題 16 份解答（所以 GSM8K 是 Best-of-16）。生成器 Gemini 1.0 Pro；驗證器 Gemma 2B、7B、Gemma-2 9B。
- MATH500（高中競賽題 MATH 的 500 題保留集，Lightman et al. 2023）與 MMLU 數學子集：只用 GSM8K 訓練的驗證器做 easy-to-hard 泛化評估。

主要數字：
| 位置 | 設定 | 結果 |
| --- | --- | --- |
| Fig. 1 左 | 演算法 2 任務平均，Gemma-2B，Best-of-32 | 5.0% → 45.3%；GenRM-CoT 接近 oracle verifier（Sec. 4.1） |
| Fig. 1 中 | GSM8K，Gemma2-9B 驗證器，Best-of-16 | 73.0% → 93.4%；作者寫超過 GPT-4 與 Gemini 1.5 Pro（Sec. 1，未列數字） |
| Fig. 1 右 | 轉移到 MATH，Best-of-32 | 28.0% → 44.6%（＝16.6 個百分點；Sec. 1 寫「17% more」是百分點〔本頁推導〕） |
| Table 1 | MMLU，Pass@1 / Disc-RM / GenRM-CoT | elementary 80.1 / 90.6 / 91.1（+0.5）；high_school 52.2 / 74.8 / 76.1（+1.3）；college 47.6 / 53 / 56.1（+3.1）；abstract_algebra 37.9 / 50 / 53.50（+3.5）。「+」為相對 Disc-RM 的百分點 |
| Fig. 7 | MMLU college，Best-of-32 | Disc RM 53.0%，GenRM-CoT（32 票）56.1%；只用 1 條理由也超過 Disc RM |
| Fig. 5 | 樣本效率 | MATH：GenRM-CoT 以少 6.4 倍的解答數達到判別式 RM 的 Best-of-32；圖上另標演算法 1.5×、GSM8K 1.2×（比較基準論文未明寫） |
| Fig. 6 | MATH 分科與難度 | Pre-Algebra、Algebra、Pre-Calculus 特別好；五個難度級都較高 |
| Fig. 12 | 模型大小 2B / 7B / 9B（MATH） | GenRM、GenRM-CoT、Disc RM 皆隨大小上升；生成式在各大小都較好 |
| Fig. 11 | 推論運算（GSM8K，Best-of-16） | K 增加分數上升；微調的 Gemma GenRM-CoT 勝過用同樣 CoT 與票數、但模型更強的 Gemini 1.0 Pro LLM-as-a-Judge |
| Sec. 4.1、Fig. D.1 | 直接 GenRM（無 CoT） | 勝過 LLM-as-a-Judge 與 DPO；與判別式 RM 相當或略好 |

- Fig. 1 箭頭起點：論文寫「Δ 表示用 GenRM-CoT 做 Best-of-N 增加的解題數」，未寫明起點是哪一個方法。Table 1 與 Fig. 7 的同類寫法（37.9%、47.6%）起點是生成器 Pass@1，Fig. 9 的 y 軸也從 Pass@1 起算；本頁推定 Fig. 1 起點亦為 Pass@1，頁面寫成「論文以 Pass@1 為起點（推定，Fig. 1 未明寫）」，見 R1。
- 質性例子：Fig. 2（GSM8K，忽略「each」）判別式 0.999、GenRM-CoT 多數決 0.0015；Fig. 15（未把 90 分鐘換成 1.5 小時）0.8414 vs 0.2498；Fig. 4（MATH，化簡錯誤）0.827 vs 0.438。反例：Table E.3 是一份正確解，判別式 0.0399、GenRM-CoT 0.5131。
- 模型大小的解釋屬作者推測：「較大的模型文字生成能力較強，讓 GenRM-CoT 更能利用 CoT 推理」（Sec. 4.3）。

#### 步驟 8：與 self-consistency 結合（Sec. 4.1、App. C、Fig. 8、Fig. D.4）

- Weighted self-consistency：把每份解答依驗證器分數加權，對每個最終答案加總分數，選總和最大的答案。
- App. C 的修改：每個答案只加總**分數最高的 K 份**解答（top-K），而非全部；GSM8K 用 K = 6，MATH 用 K = 4。作者寫這「略為」提升表現。注意這裡的 K 與 Eq. 7 的 K 是不同的量，頁面寫成 $K_{\text{top}}$。
- 結果：MATH 上以 GenRM-CoT 加權的 SC 達到以判別式 RM 加權的 SC 同樣表現，所需解答數少 2.5 倍（Fig. 8）；GSM8K 幾乎沒有增益，16 份解答時 93.4% → 93.5%，作者認為是改進空間已飽和（Fig. D.4）。

#### 步驟 9：實作細節（App. A–C）

| 項目 | 值 |
| --- | --- |
| GenRM LR | 掃描後 {2e-6, 1e-6, 5e-7} 表現良好，2e-6 通常最好（掃描範圍論文未給） |
| 最佳化 | Adam + decoupled weight decay 1e-2；不用 dropout；梯度範數截斷 1.0 |
| 排程 | 線性 warmup 1000 步；cosine 衰減到峰值的 10% |
| 訓練長度 / batch | 300K 步 / 64；資料混合用 seqio |
| checkpoint | 依保留題目上的驗證準確率挑選 |
| 資料平衡 | 正確 50%、錯誤 50% |
| Disc RM | 以特殊 token 的 logit 分類；LR、weight decay、dropout 大範圍掃描；Z-loss $=10^{-4}\cdot\log^2 Z$；最終 LR 1e-7、dropout 0 |
| DPO | 先 SFT 得 $\pi_{\text{ref}}$；LR 1e-6、$\beta=0.1$；分數用 $\log\pi_{\text{DPO}}$（不減 $\log\pi_{\text{ref}}$） |
| 資料量 | Last Letter：每長度 350 題 × 128 次嘗試，去重後約 50K；Word Sorting：每長度 4096 題 × 64 次，約 100K |
| $\lambda$ / $K$ / $K_{\text{top}}$ | 1/3（演算法）、1/4（GSM8K）／32／6（GSM8K）、4（MATH） |

常見錯誤（頁面 S10 逐條列出）：
1. 把 $r_{\text{Direct}}$ 算成 $\{Yes,No\}$ 兩者的正規化機率：論文 Eq. 4 是整個詞彙表上的 $p_\theta(\text{Yes})$（Fig. 3 畫出「Other tokens」）。
2. 推論時把參考解答放進驗證器輸入：參考解答只用在產生訓練理由（Sec. 3.3）。
3. 只依最終 Yes/No 過濾合成理由、不用參考引導：隨機猜也有 50% 判對（Sec. 3.3）。
4. 把 Eq. 7 當成「票數」：它平均的是 Yes 機率。
5. 解題資料比例越大越好：Fig. D.3 顯示 $\lambda$ 過大會降低驗證表現。
6. 混淆兩個 K：Eq. 7 的投票數（32）與 weighted SC 的 top-K（6／4）。
7. 混淆指標：Best-of-N 解題率與 RM accuracy 衡量不同能力（Sec. 4）。

#### 步驟 10：限制與未來方向（Sec. 6）

論文本身沒有獨立的限制段；從實驗設定可讀出〔本頁整理，非論文明寫〕：只測數學與演算法字串任務；GSM8K 只評 16 份解答；GenRM-CoT 的增益依賴驗證理由的品質，K = 32 時每份解答需生成 32 條 CoT〔本頁推導：由預設 K 推得，論文未報告成本〕。作者列的未來方向：程式碼、對齊、文字到影像、開放式生成；過程級監督；以 RL 訓練 CoT 驗證器；RAG、many-shot、多階段提示、工具使用；把 GenRM 放進 LLM 的 RL 流程。

---

## 2. 學習目標（驗收測試）

每個章節結尾以 `<details>`（summary「檢查理解」）列出該章題目與參考答案。**參考答案的內容必須以一段完整敘述出現在指定章節的內文裡**（不只在 details、表格或圖中）；5.6 列出各段落的位置。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明 Best-of-N 與它的上下界 | Best-of-N 怎麼運作？驗證器完全無用與完美時，結果各是多少？ | 抽 N 份解答、驗證器打分、選最高分者。驗證器分數與正確性無關時等於 Pass@1（隨機挑一份）；完美驗證器時等於 Pass@N。單一題目、每份獨立以機率 $p$ 正確時 Pass@N $=1-(1-p)^N$；資料集的 Pass@N 是各題此值的平均，C1 讓每題 $p$ 相同〔本頁推導〕 | S1 |
| LO2 | 區分兩種指標 | Best-of-N 解題率與 RM accuracy 差在哪裡？ | RM accuracy 只看單份解答分類對錯（逐點）；Best-of-N 看驗證器能否在多份中排出並選到正確解（排序）。兩者相關但不相同（Sec. 4） | S1 |
| LO3 | 寫出判別式 RM | 判別式 RM 的分數與損失是什麼？為什麼說它沒用到生成能力？ | $r_\theta=\mathrm{sigmoid}(z_{cls})$，以 BCE 訓練（Eq. 3），正負資料各半；只在一個專用輸出（$cls$ 的 logit）上給一個數字、不產生文字，所以不能先寫 CoT，也無法和解題 SFT 在同一個輸出頭上統一（論文 Sec. 5：Yang et al. 2024 雖加了輔助的生成式 SFT 損失，但放在另一個頭上作正則化、訓練後丟棄，查詢 RM 時不產生文字） | S2 |
| LO4 | 寫出 GenRM 直接驗證器 | GenRM 的分數怎麼定義、怎麼訓練？它與判別式 RM 在數學上的關係？ | $r_{\text{Direct}}=p_\theta(\text{Yes}\mid\mathbf x,\mathbf y,I)$，$I$＝「Is the answer correct (Yes/No)?」，以 SFT 損失訓練正確解→Yes、錯誤解→No（Eq. 2、4）。若其他 token 機率可忽略，$p(\text{Yes})\approx\mathrm{sigmoid}(z_Y-z_N)$，損失與 BCE 形式相同；是否真的近似取決於其他 token 的機率，論文未量測〔本頁推導〕；直接 GenRM 只與判別式 RM 相當或略好（Fig. D.1） | S3 |
| LO5 | 說明統一訓練 | Eq. 5 是什麼？$\lambda$ 的預設值與過大的後果？ | $\mathcal L_{\text{GenRM}}=\mathcal L_{\text{SFT}}(\mathcal D_{\text{verify}})+\lambda\mathcal L_{\text{SFT}}(\mathcal D_{\text{correct}})$；演算法 1/3、GSM8K 1/4；加入解題資料提升驗證（Fig. 9），但過多會降低驗證表現（Fig. D.3）；反過來驗證資料也提升生成的 Pass@N（Fig. 10） | S4 |
| LO6 | 寫出 GenRM-CoT | GenRM-CoT 推論時依序做什麼？ | 先接「Let's verify step by step.」生成驗證理由 $\mathbf v_{\text{CoT}}$，再接 $I$，取 Yes 機率為分數（Eq. 6） | S5 |
| LO7 | 說明 majority voting | Eq. 7 平均的是什麼？預設 K？判別式 RM 為何不能這樣做？ | 平均 K 條抽樣理由下的 Yes 機率（不是票數）；預設 K = 32；論文寫判別式驗證器做不到這件事（Sec. 3.3）；〔本頁說明〕原因是它對同一輸入只有一個確定分數，沒有可抽樣的推理路徑。Fig. 11：K = 2 以內就勝過 greedy | S6 |
| LO8 | 說明合成理由的做法 | GSM8K 的訓練理由怎麼產生、怎麼過濾？參考解答在推論時用嗎？ | Gemini 1.0 Pro 在提示中看參考解答（得到正確答案的解）來逐步批改（Table A.2）；只保留判定與答案檢查器一致的理由，且只取超過 50% 理由一致的解答（App. C）；參考解答只用於產生資料，推論不用。有引導 91.7% vs 無引導 87.8%（Fig. 13） | S7 |
| LO9 | 讀出主要結果並正確使用單位 | 三個 Best-of-N 主結果與 MMLU abstract algebra 結果？「MATH 多解 17%」的單位？ | 演算法 5.0% → 45.3%、GSM8K 73.0% → 93.4%、MATH 28.0% → 44.6%、MMLU abstract algebra 37.9% → 53.5%；MATH 增加 16.6 個百分點，論文的「17%」是百分點〔本頁推導〕 | S8 |
| LO10 | 說明 scaling 結論 | 增加模型大小與推論運算時，GenRM 有何表現？與 LLM-as-a-Judge 比較？ | 2B→7B→9B 時 GenRM、GenRM-CoT 與 Disc RM 都上升，生成式在各大小都較好（Fig. 12）；K 增加分數上升，微調的 Gemma GenRM-CoT 勝過更強的 Gemini 1.0 Pro LLM-as-a-Judge（Fig. 11）；MATH 上以少 6.4 倍的解答數達到 Disc RM 的 Best-of-32（Fig. 5） | S8 |
| LO11 | 說明 weighted self-consistency | weighted SC 怎麼選答案？App. C 的修改與兩個資料集的結果？ | 依驗證分數對每個最終答案加總、取最大；App. C 只加總每個答案分數最高的 $K_{\text{top}}$ 份（GSM8K 6、MATH 4）；MATH 上以 GenRM-CoT 加權比以 Disc RM 加權少用 2.5 倍解答（Fig. 8），GSM8K 只從 93.4% 到 93.5%（Fig. D.4） | S9 |
| LO12 | 說出 GenRM 的資料需求與它和判別式 RM 的設定差異 | 訓練 GenRM-CoT 需要哪些資料？哪三項設定與判別式 RM 不同？ | 資料：正確／錯誤各半的「題目–解答」配對與其 Yes/No 標記、每份解答的驗證理由（GSM8K 為參考引導合成並過濾），以及題目→正確解的生成資料（Eq. 5）。三項差異：(1) 分數來自語言模型頭的 Yes token，而不是專用 $cls$ 的 logit；(2) 損失是 next-token SFT，並與 $\lambda$ 倍的解題資料混合；(3) 推論可先生成理由並平均 K 條（App. A–C、Eq. 3–7）。超參數表（LR 掃描後 {2e-6,1e-6,5e-7} 表現良好等）留作參考，不列入驗收 | S10 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），互動元件放在加寬圖版（最大 1080px）。章節編號以 JetBrains Mono 顯示（`01 · …`）。視覺記號：一條「分數軸」0 到 1（驗證器分數），在 hero、C1、C3、C4 重複出現，正確解以實心圓、錯誤解以空心叉標示（不靠顏色）。章節標題用陳述句。

| # | id | 標題 | 內容（依序；每個符號先定義再使用） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈生成式驗證器：把驗證寫成下一個 token 的預測〉 | 副標：「這篇論文用『Yes』這個 token 的機率當作解答正確性的分數，讓驗證器能先寫出檢查步驟再下判斷。」三張事實卡（下方）。連結：arXiv、`../index.html#paper-118`、`genrm-lineage-hub.html` | 1 螢幕 |
| S1 | `bon` | 01 · Best-of-N 的效果取決於驗證器 | LLM 會犯錯 → 生成器、驗證器、Best-of-N 定義 → Pass@1、Pass@N（含「公式是單題的；資料集值是各題平均」一句）→ 兩種指標 → **μ 的白話段**（「驗證器替正確解打的分數，平均比錯誤解高出 μ 個標準差；μ = 0 表示分數與對錯無關，μ 越大越容易分開。μ 越大，單份解答分類準確率（RM accuracy）越高，但 Best-of-N 還取決於 p 與 N」，不出現積分）→ **C1**（積分公式放在圖版下方 `<details>`「計算方式」）→ LO1、LO2 段落。末尾本章符號小表。延伸：`genrm-best-of-n-discriminative-verifiers.html` | 450–550 字 + C1 |
| S2 | `disc` | 02 · 判別式驗證器只輸出一個分數 | **前置知識段**：logit（模型對每個 token 輸出的未正規化分數）→ softmax（$p_j=e^{z_j/\gamma}/\sum_i e^{z_i/\gamma}$，把 logit 轉成機率，溫度 $\gamma$）→ sigmoid（$\sigma(z)=1/(1+e^{-z})$，兩類時的 softmax）→ BCE（二元交叉熵，$-[y\log r+(1-y)\log(1-r)]$）→ Eq. 1–2（自迴歸與 SFT）→ Eq. 3（定義 $z_{cls}$）→ 三種其他基線（LLM-as-a-Judge、DPO、self-consistency）各一句定義 → LO3 段落。延伸：`genrm-step-level-supervision-verifiers.html`、`genrm-llm-as-a-judge.html` | 400–500 字 |
| S3 | `direct` | 03 · GenRM 以 Yes token 的機率作為分數 | Fig. 3 上半的靜態 SVG（題目＋解答＋$I$ → LLM → 機率長條 Yes / No / 其他）→ $\mathcal D_{\text{Direct}}$ → Eq. 4 → softmax 溫度 → **C2** → 〔本頁推導〕與 BCE 的關係 → LO4 段落。延伸：`genrm-yes-token-method.html` | 400–500 字 + C2 |
| S4 | `unified` | 04 · 同一個模型同時學解題與驗證 | Eq. 5 → $\lambda$ 預設 → Fig. 9、D.3、10 的結論（文字＋小表，不重繪）→ LO5 段落（DPO 對比移到 S11）。延伸：`genrm-dpo-implicit-verifiers-unification.html` | 300–400 字 |
| S5 | `cot` | 05 · GenRM-CoT 先寫驗證理由再回答 Yes/No | Fig. 3 下半左側（一條理由）的靜態 SVG → $\mathcal D_{\text{CoT}}$ → Eq. 6 → greedy 的定義 → Fig. 2 例子（判別式 0.999 vs 0.0015，原文節錄並翻譯 3 行）→ LO6 段落。延伸：`genrm-critiques-and-cot-reward-models.html` | 300–400 字 |
| S6 | `vote` | 06 · 平均多條驗證理由的 Yes 機率 | Fig. 3 下半右側（K 條理由 → 平均）→ Eq. 7（強調平均機率、不是票數）→ $K=32$、Fig. 11 → **C3** → LO7 段落。延伸：`genrm-inference-time-compute-majority-voting.html` | 300–400 字 + C3 |
| S7 | `synth` | 07 · 訓練用的驗證理由由同一個 LLM 參考正確解答合成 | Table A.1（演算法真值 CoT，節錄一例）→ 只依最終 Yes/No 過濾的問題 → Table A.2 提示（原文英文照錄於 `<pre>`，旁附中譯）→ App. C 過濾 → Fig. 13、14、D.2 → LO8 段落。延伸：`genrm-critiques-and-cot-reward-models.html` | 400–500 字 |
| S8 | `results` | 08 · GenRM-CoT 在 Best-of-N 上勝過判別式驗證器 | 任務與模型（定義 GSM8K、MATH500、MMLU、Gemma、Gemini）→ 步驟 7 主表 → Table 1 → Fig. 5、6、7、11、12 結論 → 質性例子與反例（Table E.3）→ LO9、LO10 段落 | 500–600 字 + 2 表 |
| S9 | `wsc` | 09 · 驗證分數也能加權 self-consistency | weighted SC 定義 → App. C top-$K_{\text{top}}$ → **C4** → Fig. 8、D.4 → LO11 段落 | 300–400 字 + C4 |
| S10 | `impl` | 10 · 實作設定與常見錯誤 | 步驟 9 表、7 條常見錯誤 → LO12 段落 | 300–400 字 + 表 |
| S11 | `history` | 11 · 研究脈絡 | 時間線（6.7）+ DPO 對比段（自 S4 移入，見步驟 4）+ 一段文字：判別式 RM（Stiennon 2020、Cobbe 2021）→ 過程監督（Uesato 2022、Lightman 2023）→ CoT 與 self-consistency（2022）→ LLM-as-a-Judge（2023）→ DPO / V-STaR（2023–2024）→ 本文（2024） | 時間線 + 120 字 |
| S12 | `next` | 12 · 限制與延伸閱讀 | 步驟 10（明標哪些是本頁整理）；筆記清單（依 hub 閱讀順序）＋ No. 550 的 r1-* 相關筆記（6.7）；arXiv | 200 字 + 清單 |
| — | — | 頁尾 | 引用格式、arXiv；`#pm-bar` | — |

Hero 三張事實卡（每個術語卡內定義）：
1. 「Best-of-N：讓語言模型對同一題寫 N 份解答，由驗證器（替解答打分的模型）選分數最高的一份。」
2. 「GenRM 問模型『答案正確嗎（Yes/No）？』，把下一個 token 是 Yes 的機率當分數；還能先寫出逐步檢查（思維鏈，CoT）再回答。」
3. 「在 GSM8K（小學數學應用題資料集）上，用 GenRM-CoT 選 16 份解答中的一份，解題率由 73.0% 提高到 93.4%（Fig. 1；起點推定為單次取樣的解題率）。」
- 每張卡片末行附章節指標：卡 1「詳見 01」、卡 2「詳見 03、05」、卡 3「詳見 08」。

Hero 圖：分數軸上兩個標記，取自 Fig. 2 的同一份錯誤解：「判別式 RM 0.999（詳見 02）」與「GenRM-CoT 0.0015（詳見 05、06）」，標記為空心叉（錯誤解），軸上方一行小字「同一份錯誤解答在兩種驗證器下的分數（Fig. 2）」。兩標籤位在軸的兩端，不會互相重疊；處理的是端點裁切：靠 0 的標籤以標記為左緣向右展開，靠 1 的標籤以標記為右緣向左展開（`text-anchor` start／end），358px 內容寬下兩標籤合計寬度 < 300px。

桌機（≥ 1024px）左側 sticky 章節目錄；< 1024px 頂部橫向捲動章節列（沿用 238 頁 `initToc`）。

---

## 4. 互動元件規格

共用：`erfc`（Numerical Recipes erfcc，分數誤差 < 1.2e-7）、$\Phi(s)=\tfrac12\mathrm{erfc}(-s/\sqrt2)$、$\varphi(s)$；`binomPmf(N,p)` 以遞迴 $q_c=q_{c-1}\cdot\frac{N-c+1}{c}\cdot\frac{p}{1-p}$ 計算，滑桿範圍已排除 $p=0,1$，不另外夾限。函式與 `verify/check.py` 逐行對應。所有讀數旁的說明列寫「讀數由本頁依 ○○ 即時計算」，並依 5.6 加標籤。

### C1 · Best-of-N 模擬器：驗證器的分離度決定能選到多少正確解（S1）

- **圖版第一行（內建假設）**：「假設：每道題目都有相同的單份正確率 p，每份解答獨立地以機率 p 正確（真實資料各題 p 不同，Pass@N 是各題的平均）；驗證器給正確解的分數服從 N(μ, 1)、給錯誤解的分數服從 N(0, 1)，彼此獨立；RM accuracy 以正確、錯誤各半的測試集與閾值 μ/2 計算。這是本頁自訂的玩具模型，μ 與 p 都是〔示範值〕，不對應論文中任何驗證器。」
- **教的概念**：Best-of-N 的定義；下界 Pass@1、上界 Pass@N；驗證器越能分開正確與錯誤解（μ 越大），Best-of-N 越接近上界；RM accuracy 與 Best-of-N 是不同量。
- **輸入**：$p\in[0.05,0.95]$ 步長 0.05，預設 0.30；$\mu\in[0,4]$ 步長 0.1，預設 1.5（兩個滑桿的 DOM 值用整數：`p20`∈[1,19]、`mu10`∈[0,40]，計算時 p = p20/20、μ = mu10/10，避免 0.30000000000000004 類累加誤差）；$N$ 分段按鈕 {1,2,4,8,16,32,64}，預設 16。
- **計算**：
  - $W(c,m,\mu)=P(\max\text{ of }c\text{ 正確分數}>\max\text{ of }m\text{ 錯誤分數})=\int c\,\varphi(s-\mu)\,\Phi(s-\mu)^{c-1}\Phi(s)^m\,ds$；$c=0\Rightarrow0$，$m=0\Rightarrow1$。
  - 積分：梯形法，$s\in[\min(0,\mu)-8,\max(0,\mu)+8]$，步長 0.01。對每個 $\mu$ 先算好陣列 $\varphi(s-\mu)$、$\ln\Phi(s-\mu)$、$\ln\Phi(s)$（$\Phi$ 夾在 ≥ 1e-300 後取 log），被積函數用 $\exp((c-1)\ln\Phi(s-\mu)+m\ln\Phi(s))$，避免大次方下溢成 0 後再相乘的問題。
  - $\text{BoN}(p,N,\mu)=\sum_{c=0}^{N}\binom Nc p^c(1-p)^{N-c}\,W(c,N-c,\mu)$。
  - Pass@N $=1-(1-p)^N$；RM accuracy（正負各半、閾值 μ/2）$=\Phi(\mu/2)$〔本頁推導〕。
  - 運算量：每次更新算 7 個 N 值，約 $\sum_N(N+1)\times1700\approx2.3\times10^5$ 次 exp，< 10 ms，主執行緒同步即可；`input` 事件以 `requestAnimationFrame` 節流。
- **顯示**：
  - 主圖：橫軸 N（log₂，1–64，7 個刻度），縱軸解題率 [0,1]。三條線：Best-of-N（`--accent` 實線＋實心圓）、Pass@N（`--ink-3` 虛線，標「完美驗證器 Pass@N」）、Pass@1（`--ink-3` 點線水平，標「隨機挑選 Pass@1」）。目前的 N 以垂直細線與三個數值標籤標出。
  - 副圖（主圖右側，< 768px 移到下方）：分數軸上兩條密度曲線，正確解 N(μ,1)（實線，標「正確解」）、錯誤解 N(0,1)（虛線，標「錯誤解」），閾值 μ/2 細線。
  - 讀數框（固定位置）：Pass@1、Best-of-N、Pass@N、RM accuracy，四位小數改為百分比一位小數（例「84.7%」）；另一列「同一 μ 下 Best-of-1 ＝ 30.0%」恆顯示，讓讀者看到 Best-of-N 隨 N 改變、RM accuracy 不隨 N 改變。
  - 讀數框下方固定兩行定義：「RM accuracy：一份解答的分數高於閾值就判為正確，判對的比率（逐點）。」「Best-of-N：N 份中分數最高的一份是否正確（排序）。」
  - 390px：主圖寬＝容器寬（358px），高 220px；刻度只標 1、4、16、64；控制列堆疊在圖下方，滑桿觸控區 ≥ 44px。
- **讀者應觀察到**：(1) μ = 0 時三條線中 Best-of-N 與 Pass@1 重合（30.0%），加大 N 沒有用；(2) μ 增加時曲線往 Pass@N 靠近；(3) μ = 1.5 時 RM accuracy 為 77.3%，與 N 無關；Best-of-N 在 N = 1 時是 30.0%、N = 16 時是 84.7%。RM accuracy 是逐點閾值判斷，Best-of-N 是排序後選最高分，兩者定義不同，數值不能直接比大小。

### C2 · Yes token 的分數與損失（S3）

- **圖版第一行**：「這是單一位置的 softmax 計算，logit 為〔示範值〕；實際 LLM 的詞彙表有數十萬個 token，這裡把 Yes、No 以外的 token 合併成一個『其他』，其 logit 代表它們的 log-sum-exp。」
- **教的概念**：Eq. 4 的分數是整個詞彙表上的機率；溫度的作用；〔本頁推導〕忽略其他 token 時等於 sigmoid$((z_Y-z_N)/\gamma)$；SFT 損失對 logit 的梯度。
- **輸入**：$z_Y\in[-6,6]$（預設 2）、$z_N\in[-6,6]$（預設 0）、$z_O\in[-30,6]$（預設 −4）、步長 0.1；$\gamma\in[0.25,2]$ 步長 0.05（預設 1；滑桿標〔示範值〕，旁註「論文評分推定用 γ = 1」）；切換鈕「此解答其實是：正確（目標 Yes）／錯誤（目標 No）」。
- **計算**（log-sum-exp，先減最大值）：$a_j=z_j/\gamma$，$m=\max a_j$，$p_j=e^{a_j-m}/\sum e^{a_i-m}$；兩類版本 $\sigma((z_Y-z_N)/\gamma)$ 以 $1/(1+e^{-x})$ 計算，$x<-700$ 時直接取 0；損失 $=-(a_{\text{target}}-m-\ln\sum e^{a_i-m})$；梯度 $(p_j-\mathbb 1[j=\text{target}])/\gamma$。
- **顯示**：三根水平長條（Yes / No / 其他）以長度與數字表示機率，長條內再以斜線圖樣標示目標 token；下方兩行讀數：「GenRM 分數 $r=p(\text{Yes})$」與「只看 Yes 與 No 的 sigmoid」，兩者差值以文字顯示；損失與三個梯度值：數值欄標明「∂損失/∂z」，另一欄以文字寫「梯度下降後：z_Y 上升／z_N 下降／其他 下降」（方向＝負梯度），不用箭頭符號以免與數值正負混淆。
- **讀者應觀察到**：(1) 其他 token 的 logit 從 −4 提到 2，分數由 0.879 降到 0.468，而 sigmoid 版本固定 0.881；(2) $z_O$ 很小時兩者幾乎相等；(3) 降低溫度把分數推向 0 或 1（$\gamma=0.5$ 時 0.982）。

### C3 · 平均多條驗證理由（MajV@K）（S6）

- **圖版第一行（內建假設）**：「假設（皆為本頁簡化）：(1) 每條理由的 Yes 機率以 0 或 1 近似（只看判斷），所以平均值就是 Yes 票的比例；論文 Eq. 7 平均的是連續機率（例如 Fig. 2 的 0.0015）。(2) 驗證器對正確解與錯誤解對稱：判對機率相同。(3) 理由彼此獨立；實際上同一模型抽出的理由錯誤會相關。(4) 開啟『系統性判錯的解答』時，比例 h 對正確解與錯誤解相同，這些解答上判對機率為 q_h。假設 (3) 本身就保證判對機率 > 0.5 的解答在 K 增加時被判得更準；這不是論文的結論，論文的 Fig. 11 是實際量測。所有參數皆為〔示範值〕。」
- **教的概念**：Eq. 7 平均 K 條理由；平均能抵消個別理由的隨機錯誤，但不能修正系統性錯誤（系統性判錯的解答上判對機率 < 0.5 時，K 越大越穩定地判錯）；對 Best-of-N 的影響。
- **輸入**：$K$ 分段按鈕 {1,2,4,8,16,32}（預設 1，「逐步增加 K」按鈕每按一次進一格）；$q_e\in[0.5,1]$ 步長 0.05（預設 0.80）；勾選框「加入系統性判錯的解答」（預設不勾，此時 $h=0$、$h,q_h$ 滑桿隱藏）；勾選後出現 $h\in[0,0.5]$ 步長 0.05（預設 0.20）與 $q_h\in[0,0.5]$ 步長 0.05（預設 0.30）；滑桿 DOM 值皆為整數（×20）；$p$ 固定 0.30、$N$ 固定 16（圖版內寫明）。
- **計算**（全部是有限和，精確值）：
  - 判對理由數 $Y$ 的分布：$P(Y=y)=(1-h)\binom Ky q_e^y(1-q_e)^{K-y}+h\binom Ky q_h^y(1-q_h)^{K-y}$。
  - 正確解的分數層級 $\ell=Y$（分數 $\ell/K$）；錯誤解的 Yes 票數為 $K-Y$，所以其層級分布是同一個陣列反轉。
  - RM accuracy（閾值 0.5，分數恰為 0.5 計半對）：$\sum_{2\ell>K}a_\ell+\tfrac12 a_{K/2}$。
  - Best-of-N（同分時均勻隨機選一）：對每個正確解數 $C$（$M=N-C$）、每個最高層級 $\ell$，$\sum_{i\ge1}\sum_{j\ge0}\binom Ci a_\ell^iA_\ell^{C-i}\binom Mj b_\ell^jB_\ell^{M-j}\cdot\frac{i}{i+j}$，其中 $A_\ell=P(\text{層級}<\ell)$；再以 $\binom NC p^C(1-p)^{N-C}$ 加權。$K=32,N=16$ 時約 $33\times\sum_C C\cdot M\approx2.4\times10^4$ 項，同步計算。二項係數用遞迴乘法，不用階乘。
  - K 從 1 到 32 的 6 個點一次算完畫成曲線。
- **顯示**：
  - 左圖：分數軸上的兩個直方圖（正確解向上、錯誤解向下，鏡射），柱寬為 1/(K+1)；正確解柱為實心、錯誤解柱為斜線圖樣。
  - 右圖：橫軸 K（log₂），兩條線：RM accuracy（實線＋圓點）、Best-of-16（虛線＋方點），另一條 Pass@16＝99.7% 的參考虛線。
  - 讀數框：K、RM accuracy、Best-of-16。
  - 390px：兩圖上下堆疊，各高 200px。
- **讀者應觀察到**：(1) $h=0$ 時 K 增加 RM accuracy 往 1 走（80.0% → 99.99%），這是獨立性假設的直接結果；(2) 勾選後 $h=0.2$：RM accuracy 趨近 80%（$1-h$），系統性判錯的解答被越來越穩定地判錯。K = 16 的 80.66% 高於 K = 32 的 80.18%，因為這些解答在有限 K 時仍有一點機會被判對，這個機會隨 K 遞減，曲線由上方逼近 80%〔本頁推導〕；圖下以一句話說明，避免被讀成程式錯誤；(3) K = 1 與 K = 2 的 RM accuracy 相同（K = 2 會出現 1:1 平手，計半對）；(4) K = 1 的分數只有 0 與 1 兩層，大量同分讓 Best-of-N 只能隨機選，K 增加後分數層級變多，Best-of-N 提升幅度大於 RM accuracy。

### C4 · Weighted self-consistency 與 top-K（S9）

- **圖版第一行**：「8 份候選解答的最終答案與驗證分數都是〔示範值〕，用來比較四種選法的規則，不代表論文的資料。」
- **教的概念**：self-consistency、Best-of-N、weighted SC（全部加總）、weighted SC top-$K_{\text{top}}$（App. C）四種規則的差異。
- **輸入**：預設組 3 個（分段按鈕）＋「自訂」；每份解答一列：答案（A–D 的 chip 群組，`role="radiogroup"`）與分數滑桿 [0,1] 步長 0.05；$K_{\text{top}}\in\{1,\dots,8\}$（預設 4，與 MATH 相同）；正確答案（chip，預設組固定）。任何編輯（答案、分數、正確答案）都會把分段按鈕切到「自訂」並保留目前值。每列答案是一個 `role="radiogroup"`：Tab 在列之間移動、方向鍵在 A–D 間切換（roving tabindex）。某答案沒有任何候選時，小表該列顯示 0 票、總分 0，不參與勝出（四種方法都只在出現過的答案中選）。
  - P1「分數最高的一份是錯的」：正確 B；A .40、A .35、A .30、A .45、B .90、B .85、C .20、D .95。
  - P2「驗證器與多數一致」：正確 A；A .90、A .92、A .88、A .95、A .90、B .30、C .20、A .85。
  - P3「很多低分的錯誤答案」：正確 B；A .30 ×6、B .85、B .80。
- **計算**：SC＝票數最多的答案；Best-of-N＝分數最高的那份（同分取編號小者）；WSC＝每個答案分數加總取最大；WSC-top＝每個答案只加總最高的 $K_{\text{top}}$ 個分數取最大。答案間同分時取字母序較前者（頁面寫明此規則）。加總比較前先四捨五入到 1e-9，避免 0.1+0.2 類浮點誤差影響同分判定。
- **顯示**：左側 8 列候選（答案 chip＋分數長條＋數字），右側四張結果卡（方法名、選出的答案、✓／✗ 文字「正確」「錯誤」，不只靠顏色），下方每個答案的「票數／總分／top-K 和」小表（390px 時為 4 欄窄表，欄名縮寫並以 `<abbr title>` 提供全名）。
- **讀者應觀察到**：P1：SC 選 A、Best-of-N 選 D（都錯），WSC 選 B；$K_{\text{top}}=1$ 時 WSC-top 等於 Best-of-N（選 D），$K_{\text{top}}\ge2$ 選 B。P3：WSC 被 6 個低分 A 拉走（1.80 > 1.65），$K_{\text{top}}\le5$ 時 top-K 選 B，$K_{\text{top}}\ge6$ 又選 A。P2：四種方法一致，對應 GSM8K 上 weighted SC 幾乎沒有增益（93.4% → 93.5%）的情形——頁面寫成「類似情形」，不宣稱這解釋了論文結果。

---

## 5. 驗收數值表（`verify/check.py` → `check.out`，同演算法）

另由 `check.py` 斷言：Best-of-1（μ = 1.5）= 0.3000、Best-of-16 = 0.8470；C3 在 h = 0.2 時 RM acc(K=16) > RM acc(K=32) > 0.8；「一半題目 p = 0、一半 p = 1」的資料集 Pass@16 = 0.5。

### C1（p, μ；N = 1, 2, 4, 8, 16, 32, 64；容許 ±0.0005）

| p | μ | 1 | 2 | 4 | 8 | 16 | 32 | 64 | Pass@16 | RM acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 0.0 | 0.3000 | 0.3000 | 0.3000 | 0.3000 | 0.3000 | 0.3000 | 0.3000 | 0.9967 | 0.5000 |
| 0.3 | 1.0 | 0.3000 | 0.4093 | 0.5187 | 0.6176 | 0.7002 | 0.7657 | 0.8163 | 0.9967 | 0.6915 |
| 0.3 | 1.5 | 0.3000 | 0.4493 | 0.6088 | 0.7477 | 0.8470 | 0.9084 | 0.9438 | 0.9967 | 0.7734 |
| 0.3 | 3.0 | 0.3000 | 0.5029 | 0.7405 | 0.9204 | 0.9882 | 0.9986 | 0.9997 | 0.9967 | 0.9332 |
| 0.1 | 1.5 | 0.1000 | 0.1640 | 0.2520 | 0.3614 | 0.4830 | 0.6034 | 0.7095 | 0.8147 | 0.7734 |
| 0.6 | 1.5 | 0.6000 | 0.7707 | 0.8844 | 0.9428 | 0.9697 | 0.9826 | 0.9893 | 1.0000 | 0.7734 |

交叉檢查：蒙地卡羅 200,000 次（p = 0.3, N = 16, μ = 1.5）0.8465，解析 0.8470。μ = 0 時所有 N 皆為 p（assert）。

### C2（容許 ±0.0005）

| $z_Y,z_N,z_O,\gamma$ | p(Yes) | p(No) | p(其他) | sigmoid 兩類 | 損失（目標 Yes） | 梯度 Yes, No, 其他 |
| --- | --- | --- | --- | --- | --- | --- |
| 2, 0, −4, 1 | 0.8789 | 0.1189 | 0.0022 | 0.8808 | 0.1291 | −0.1211, 0.1189, 0.0022 |
| 2, 0, 0, 1 | 0.7870 | 0.1065 | 0.1065 | 0.8808 | 0.2395 | −0.2130, 0.1065, 0.1065 |
| 2, 0, 2, 1 | 0.4683 | 0.0634 | 0.4683 | 0.8808 | 0.7586 | −0.5317, 0.0634, 0.4683 |
| 2, 0, −4, 0.5 | 0.9820 | 0.0180 | 0.0000 | 0.9820 | 0.0182 | −0.0360, 0.0360, 0.0000 |
| 0, 0, −30, 1 | 0.5000 | 0.5000 | 0.0000 | 0.5000 | 0.6931 | −0.5000, 0.5000, 0.0000 |

### C3（p = 0.3, N = 16；K = 1, 2, 4, 8, 16, 32；容許 ±0.0005）

| $q_e$ | $q_h$ | $h$ | RM accuracy | Best-of-16 |
| --- | --- | --- | --- | --- |
| 0.8 | 0.3 | 0.0（預設：系統性判錯關閉） | 0.8000 0.8000 0.8960 0.9667 0.9958 0.9999 | 0.6313 0.8620 0.9725 0.9949 0.9966 0.9967 |
| 0.8 | 0.3 | 0.2（勾選後的預設） | 0.7000 0.7000 0.7600 0.7985 0.8066 0.8018 | 0.4999 0.6326 0.7248 0.7840 0.8412 0.9005 |
| 0.7 | 0.3 | 0.2 | 0.6200 0.6200 0.6704 0.7244 0.7700 0.7943 | 0.4115 0.5058 0.5841 0.6210 0.6305 0.6313 |
| 0.6 | 0.3 | 0.0 | 0.6000 0.6000 0.6480 0.7102 0.7869 0.8716 | 0.3913 0.4873 0.6073 0.7372 0.8672 0.9579 |
| 0.8 | 0.3 | 0.5 | 0.5500 0.5500 0.5560 0.5463 0.5229 0.5047 | 0.3437 0.3709 0.4174 0.4850 0.5734 0.6833 |

Pass@16 = 0.9967。交叉檢查：蒙地卡羅 100,000 次（K = 4，預設參數）0.7271，解析 0.7248（差 0.0023，在 MC 標準誤 ≈ 0.0014 的 2σ 內）。
註：預設參數下 Best-of-16 在 K = 32 仍上升（0.9005）而 RM accuracy 已持平，因為排序只需正確解的分數層級高於錯誤解，不需越過 0.5 閾值〔本頁推導〕。頁面在 C3 下方以一句話說明。

### C4（精確，無容許誤差）

| 預設組 | 正確 | SC | Best-of-N | WSC（總和） | WSC-top，$K_{\text{top}}$=1 / 2 / 4 / 8 |
| --- | --- | --- | --- | --- | --- |
| P1 | B | A | D | B（A 1.50、B 1.75、C 0.20、D 0.95） | D / B / B / B |
| P2 | A | A | A | A（A 5.40） | A / A / A / A |
| P3 | B | A | B | A（A 1.80、B 1.65） | B / B / B / A（$K_{\text{top}}$=5 → B，6 → A） |

### 論文數字的單位核對（`check.py` 末段）

MATH 44.6 − 28.0 = 16.6 個百分點；GSM8K 93.4 − 73.0 = 20.4；演算法 45.3 − 5.0 = 40.3；MMLU abstract algebra 53.5 − 37.9 = 15.6。頁面一律寫「百分點」，不寫成相對百分比。

---

## 6. 實作計畫

只列與 238 頁（`tutorials/prediction-targets-k-diff.html`）不同或新增之處；其餘完全沿用：CSS tokens（含深色模式與 `pm-theme`）、Google Fonts 與 KaTeX 0.16.9 載入方式、appbar、`#pm-bar`、`initToc`、`initTheme`、`.plate`／`.plate-h`／`.plate-f`／`.readout`／`.ctl`／`.seg`／`.tscroll`／`.check`、`announce(id,s)`（每元件一個 `#<id>-live`，500ms debounce）、`renderMath`、`?test` 的 `selfTest()` 外框（含 `scrollWidth <= innerWidth` 檢查）、`renderTimeline`。

### 6.1 檔案與連結

- `tut/118/build/index.html`，單檔，CSS／JS／timeline 全內嵌，不 `fetch`。上線為 `tutorials/generative-verifiers.html`。
- `<title>`：「生成式驗證器 GenRM」。Hero 連結：`https://arxiv.org/abs/2408.15240`、`../index.html#paper-118`、`genrm-lineage-hub.html`。
- 圖表色：`--c-pos`（正確解）淺色 #3F6E8C／深色 #8DB4D3，`--c-neg`（錯誤解）淺色 #6B7A3A／深色 #A9BC7C，在 238 頁的三處 tokens 區塊（`:root`、`prefers-color-scheme: dark` 內的 `:root:not([data-theme="light"])`、`:root[data-theme="dark"]`）都要定義；只用於圖中資料系列，且每個系列另有線型或圖樣。SVG 斜線圖樣（`<pattern>`）的 stroke 用 `var(--ink-2)`，密度曲線與直方圖的 fill／stroke 一律引用 token，不寫死色碼。

### 6.2 新增的 JS（IIFE 內，函式名與 `check.py` 對應）

- `erfc(x)`、`Phi(s)`、`phi(s)`、`binomPmf(N,p)`。
- C1：`c1Arrays(mu)`（快取 φ、lnΦ 陣列，key 為整數 `mu10`）、`win(c,m,arr)`、`bestOfN(p,N,mu)`、`renderC1()`、`initC1()`。
- C2：`c2(zy,zn,zo,g,target)` → `{p:[Y,N,O], two, loss, grad}`；`renderC2()`、`initC2()`。
- C3：`votesPmf(K,qe,qh,h)`、`c3(K,qe,qh,h,p,N)` → `{acc,bon,a,b}`；`renderC3()`、`initC3()`；6 個 K 的結果以 `${qe}|${qh}|${h}` 快取。
- C4：`PRESETS`、`c4(cands,Ktop)` → `{sc,bon,wsc,wtk,votes,sum,topk}`；`renderC4()`、`initC4()`。
- 靜態 SVG：`figGenRM()`（Fig. 3 上下兩半的重繪：輸入框 → 模型 → 機率長條／K 條理由 → 平均），S3 用上半、S5 用下半左側（一條理由）、S6 用下半右側（K 條→平均）。SVG 內文字一律用 `<text>`，不放 KaTeX。
- `heroAxis()`：分數軸＋兩個端點標記（端點避讓，見 S0 規格；不用分層演算法）。
- 效能：C1 單次 < 10 ms、C3 < 5 ms，同步執行；若 `performance.now()` 量到單次 > 16 ms，C1 改成只在 `change` 時重算曲線、`input` 時只更新目前 N 的值。不需要 Web Worker（說明寫在程式註解）。
- 動態文字（讀數、aria-label、live region）只用 Unicode 基本字元：「μ」「Δ」「≈」，不用組合字元（例如上橫線）；數學符號在動態文字中寫成「K_top」「p(Yes)」。

### 6.3 `selfTest()` 斷言（`?test`）

- `bestOfN(0.3,16,1.5)` = 0.8470 ±0.0005；`bestOfN(0.3,N,0)` = 0.3 ±1e-4（N = 1…64）；`Phi(0.75)` = 0.7734 ±1e-4。
- `c2(2,0,0,1,'Y').p[0]` = 0.7870、`.two` = 0.8808、`.loss` = 0.2395（±5e-4）；`c2(0,0,-30,1)` 的 p(Yes) 與 sigmoid 差 < 1e-9。
- `c3(4,.8,.3,.2,.3,16)`：acc 0.7600、bon 0.7248；`c3(32,.8,.3,0,.3,16).acc` = 0.9999（±5e-4）；兩類 pmf 各自總和為 1（±1e-12）。
- `c4(P1,1)` = (A, D, B, D)；`c4(P3,4)` = (A, B, A, B)；`c4(P3,6).wtk` = A。
- 同時沿用 `scrollWidth <= innerWidth` 檢查，並在 390px 寬下逐一確認每個 `.katex-display` 的 `scrollWidth` 不撐開 `main`（公式本身可在自己的容器內橫捲）。
- 輸出一行 `selfTest: N passed, M failed`。

### 6.4 版面與無障礙（新增部分）

- 390px：所有圖寬＝容器寬，圖上文字 12–13px 不縮放；C1、C3 的雙圖改上下堆疊；步驟 7 主表與 Table 1、步驟 9 表包 `overflow-x:auto`＋左右漸層提示，< 480px 時 Table 1 改為每個 MMLU 子集一張卡片（四行：Pass@1、Disc-RM、GenRM-CoT、差值），步驟 7 主表改為每列一張卡片。
- 公式：`.katex-display{overflow-x:auto;overflow-y:hidden;max-width:100%}`；Eq. 3、5、6、7 在 < 600px 時以 `\\`（aligned）拆成兩行，拆點寫在建置時的 TeX 原始碼中。
- Table A.2 提示以 `<pre>` 原文呈現，`white-space:pre-wrap`，390px 不水平捲動。
- 每個元件一個 live region，只在 `change` 或輸入停止 500ms 後宣告一次（例：「μ = 1.5，Best-of-16 84.7%，RM accuracy 77.3%」）；SVG `role="img"`＋同步更新的 `aria-label`（同一個 debounce）。
- 不依賴 hover：C1、C3 的數值都在固定讀數框；C4 的分數用滑桿，不用拖曳長條。
- 正確／錯誤一律以「實心圓／空心叉」或「實心／斜線圖樣」＋文字區分。

### 6.5 引用與標籤

- 每個論文數字附出處（Eq./Fig./Table/Sec./App.）；元件讀數以圖版底部一行說明代替逐數字出處。
- 三種小標籤（`--ink-3` 邊框 pill）：「本頁推導」「示範值」「推定」，每個附一行依據。
- 作者解釋一律以「作者認為／作者推測」開頭（Fig. D.4 飽和、Fig. 14 集成效果、Fig. 12 生成能力、Eq. 4 信心、Fig. 10 理解生成過程）。

### 6.6 學習目標參考答案在內文的位置

每個 LO 的答案以**一段**內文寫完（可含公式），位置如下；Goldfish 驗收時逐段比對：LO1、LO2 → S1 C1 之後的兩段；LO3 → S2 末段；LO4 → S3 C2 之後一段；LO5 → S4 末段；LO6 → S5 末段；LO7 → S6 Eq. 7 之後、C3 之前一段；LO8 → S7 末段；LO9、LO10 → S8 主表之後兩段；LO11 → S9 C4 之後一段；LO12 → S10 表格之後一段。`<details>` 只重述同一答案。

### 6.7 研究脈絡（S11，來自 `notes/timeline.json`）

- 建置時把 `timeline.json`（目前 14 筆陣列，欄位 `year,title,authors_short,arxiv,role,source`）原樣貼進 `<script type="application/json" id="timeline-data">`，沿用 238 頁 `renderTimeline` 的卡片列、篩選 chip、左右捲動與錯誤處理。
- `TL_GROUP`（以 arXiv ID 為 key，依 `role` 人工對應；不在表中歸「其他」）：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 判別式 RM | 2009.01325、2110.14168 | `genrm-best-of-n-discriminative-verifiers.html` |
  | 過程／結果監督 | 2211.14275、2305.20050 | `genrm-step-level-supervision-verifiers.html` |
  | CoT 與推論運算 | 2201.11903、2203.11171、2407.21787 | `genrm-inference-time-compute-majority-voting.html` |
  | 批評模型 | 2206.05802 | `genrm-critiques-and-cot-reward-models.html` |
  | LLM 評審 | 2306.05685、2401.10020 | `genrm-llm-as-a-judge.html` |
  | 生成與驗證統一 | 2203.14465、2305.18290、2402.06457 | `genrm-dpo-implicit-verifiers-unification.html` |
  | 本文 | 2408.15240 | `genrm-yes-token-method.html` |

- S12 延伸閱讀依 hub 的閱讀順序：`genrm-lineage-hub`（總覽）、`genrm-best-of-n-discriminative-verifiers`、`genrm-step-level-supervision-verifiers`、`genrm-llm-as-a-judge`、`genrm-critiques-and-cot-reward-models`、`genrm-dpo-implicit-verifiers-unification`、`genrm-inference-time-compute-majority-voting`、`genrm-yes-token-method`，連結寫成 `<slug>.html`，文字取 frontmatter `title`。
- 其後另列「相關筆記（No. 550）」：`r1-rlhf-ppo-background.html`、`r1-process-supervision-prm.html`、`r1-cot-and-test-time-compute.html`、`r1-star-self-generated-reasoning.html`，連結文字取各自 frontmatter `title`（No. 550 先於本頁上線，正常連結）。

### 6.8 文案規則

繁體中文、台灣用語、字面學術語氣，不用比喻與宣傳形容詞；章節標題用陳述句。用詞：驗證器（verifier）、獎勵模型（reward model）、驗證理由（verification rationale）、思維鏈（CoT）、多數決（majority voting，並註明實為平均機率）、百分點。專有名詞首次出現附英文。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | Fig. 1 箭頭起點 | 論文只說 Δ 為 GenRM-CoT 的增量，未寫起點；Table 1、Fig. 7、Fig. 9 的同類寫法以 Pass@1 為起點。 | 頁面寫「推定為 Pass@1」並加「推定」標籤；不寫成「比判別式 RM 提高 X」。 |
| R2 | 「17% more」 | Sec. 1 的「17%」實為 16.6 個百分點。 | 頁面寫百分點並加「本頁推導」。 |
| R3 | 「超過 GPT-4 與 Gemini 1.5 Pro」 | Sec. 1 有此宣稱但論文未列這兩者的數字。 | 照錄並註明「論文未列數字」，不放進任何表格。 |
| R4 | 評估長度不一致 | Sec. 4 寫 {5,6}；App. A 寫 Last Letter 6、Word Sorting 5。 | 兩處照錄。 |
| R5 | 玩具模型可能被誤讀為論文結果 | C1、C3 的曲線形狀由假設決定（C3 的獨立性假設必然讓 K 增加變準）。 | 圖版第一行寫出假設；C3 可開啟「系統性判錯的解答」（$h,q_h$，預設隱藏）顯示平均無法修正系統性錯誤，避免只呈現「K 越大越好」；所有參數標「示範值」。 |
| R6 | 直接 GenRM 與判別式 RM 的關係 | 〔本頁推導〕兩者在其他 token 機率可忽略時同形；可能被讀成「GenRM 沒有優勢」。 | 頁面同時寫論文量測：直接 GenRM 只相當或略好（Fig. D.1），主要增益來自 CoT 與多數決（Fig. 1、5）。 |
| R7 | 圖表缺少原始數據 | Fig. 5、8、9、10、11、12、13、14 只有曲線，文字只給部分數字。 | 只引用文字或圖上明示的數字（6.4×、2.5×、1.5×、1.2×、91.7/87.8、93.4/93.5），不重繪曲線、不讀圖估值。 |
| R8 | r1-* 筆記 | 多篇 genrm 筆記的 `related` 指向 No. 550 的 `r1-*` 筆記；協調者確認 No. 550 先於本頁上線。 | 正常連結 `r1-rlhf-ppo-background.html`、`r1-process-supervision-prm.html`、`r1-cot-and-test-time-compute.html`、`r1-star-self-generated-reasoning.html`（6.7）。 |
| R9 | 符號衝突 | 論文 $M$ 同時可能被讀成詞彙表大小；Eq. 7 的 K 與 App. C 的 top-K 同名；溫度原文誤植 $\tau$。 | 頁面詞彙表大小不寫成 $M$（只說「詞彙表」）；App. C 寫 $K_{\text{top}}$；溫度一律 $\gamma$，並在符號表註明誤植。 |
| R10 | 初學者門檻 | 讀者可能不熟 SFT 與 softmax。 | S2 的前置知識段定義 logit、softmax、sigmoid、BCE；C1 前有 μ 的白話段，積分收進 `<details>`；符號表改為每章一張小表。 |

---

## 8. 修訂紀錄（round 1，依 `reviews/round1.md`）

| # | 變更（對應 round1.md 的項目見該檔「修訂」欄） |
| --- | --- |
| 1 | 步驟 9 表與 LO12：LR 改為「掃描後 {2e-6,1e-6,5e-7} 表現良好，2e-6 通常最好；掃描範圍論文未給」。 |
| 2 | C1 觀察 (3) 改為依定義對比（逐點閾值 vs 排序），讀數框恆顯示同一 μ 的 Best-of-1（30.0%）與兩行定義。 |
| 3 | 步驟 1、LO1、C1 首行、符號表：$1-(1-p)^N$ 標明為單題公式；玩具模型每題 p 相同；資料集 Pass@N 是各題平均（附 p=0/1 反例，`check.py` 斷言）。 |
| 4 | 1.1：箭頭起點標「推定」（Table 1 的 37.9% 為明寫 Pass@1）；補同條件比較 Disc-RM 50% → 53.5%，註明 Table 1 未寫 N。 |
| 5 | 步驟 3、LO4：「數學上很接近」改為「損失形式相同；是否近似取決於其他 token 機率，論文未量測」，並寫出 $cls$ 頭 vs LM 頭的參數差異。 |
| 6 | 步驟 3、C2：γ 滑桿標〔示範值〕，評分推定 γ = 1；LO4 公式去掉 γ。 |
| 7 | LO3：改為「無法在同一輸出頭上與 SFT 統一、不產生文字」，並提 Yang et al. 2024 的分離生成頭。 |
| 8 | LO7：判別式 RM 不能做多數決的理由標〔本頁說明〕，論文原話只說做不到。 |
| 9 | C3 首行改為四條假設：0/1 近似（Eq. 7 實為連續機率）、對稱驗證器、獨立性（實際相關）、h 對兩類相同。 |
| 10 | 步驟 6：Fig. 13 數字標「圖說未寫指標，推定 Best-of-16」。 |
| 11 | 步驟 7：Fig. 5 的 1.5×、1.2× 標「比較基準論文未明寫」。 |
| 12 | C1 首行：RM accuracy 以正負各半、閾值 μ/2 計算。 |
| 13 | 步驟 10：「32 條 CoT」標〔本頁推導〕，論文未報告成本。 |
| 14 | S0：hero 標籤加章節指標（02／05、06），卡片加「詳見」；改為端點避讓，不再用分層演算法。 |
| 15 | 符號表加「章」欄，頁面改為每章一張小表；新增 μ、$K_{\text{top}}$、前置知識條目。 |
| 16 | S2 新增前置知識段（logit、softmax、sigmoid、BCE、$z_{cls}$）。 |
| 17 | S1 在 C1 前加 μ 的白話段；積分公式移入 `<details>`「計算方式」。 |
| 18 | C3：h、q_h 預設隱藏（勾選框），「困難解答」改名「系統性判錯的解答」；驗收表標出預設列。 |
| 19 | S5 拆為 S5（Eq. 6，LO6）與 S6（Eq. 7＋C3，LO7）；其後章節順延為 S7–S12；6.6 位置同步。 |
| 20 | S4 的 DPO 對比移到 S11 研究脈絡。 |
| 21 | 6.4：`.katex-display` 橫捲與 max-width、長式窄螢幕拆行；selfTest 加公式寬度檢查；`figGenRM` 文字用 `<text>`。 |
| 22 | 6.1：`--c-pos`／`--c-neg` 在三處 token 區塊定義深色值；pattern stroke 用 `--ink-2`。 |
| 23 | C1／C3 滑桿用整數 DOM 值（`p20`、`mu10`、×20），快取 key 為 `mu10`；刪除 p 夾限（`check.py` 改為 assert）。 |
| 24 | C2：梯度欄標「∂損失/∂z」，方向改用文字（上升／下降）。 |
| 25 | C4：任何編輯切到「自訂」；radiogroup 的 roving tabindex；空答案顯示 0。 |
| 26 | C3 觀察 (2)：說明 K = 16 > K = 32 的 RM accuracy 是由上方逼近 1−h〔本頁推導〕，`check.py` 斷言。 |
| 27 | 步驟 6：「天真做法」改為「只依最終 Yes/No 過濾的做法」。 |
| 28 | LO12 改為資料需求與三項設定差異；超參數表留作參考。 |
| 29 | R8：r1-* 筆記正常連結（No. 550 先上線）；6.7 加相關筆記清單。 |

## 9. 修訂紀錄（readiness 後）

- 阻斷 1：S4 的 $\mathcal D_{\text{verify}}$ 只寫 $\mathcal D_{\text{Direct}}$，$\mathcal D_{\text{CoT}}$ 以一句前向說明帶過，正式定義仍在 S5。
- 阻斷 2：Eq. 1–2 與溫度 $\gamma$ 的條目由步驟 1 移到步驟 2（S2 前置知識段），與第 3 節大綱及符號表一致。
- 非阻斷：C2 γ = 0.5 的 Yes 梯度以 `check.py` 確認為 −0.0360（$(p_Y-1)/\gamma=(0.98201-1)/0.5=-0.03597$），驗收表不變。LO3 的 Yang et al. 2024 一句改用論文 Sec. 5 原意：「在另一個頭上加輔助生成式 SFT 損失作正則化，訓練後丟棄；查詢 RM 時不產生文字」。


## 10. 建置紀錄（build round）

- 原始碼在 `build/src/`（head.html、css_base.css 照抄 131、css_add.css、body1–3.html、app.js、assemble.py、lo.py），`python3 build/src/assemble.py` 產生 `build/index.html`（約 119 KB）。
- 實作與設計的差異：
  - `--c-pos`／`--c-neg` 在 css_add 以 `var(--c-par)`／`var(--c-perp)` 定義，繼承 css_base 三處 token 區塊中已有的深色值。
  - C2 目標 token 以外框＋「（目標）」文字標示（impeccable 會把 repeating-gradient 斜線標為 advisory）；C3 錯誤解的斜線改用 SVG `<pattern>` 與 CSS mask。
  - Fig. 3 的三張流程圖（`flowFig`）在寬度 ≥ 600px 時橫排，較窄時直排，每個方框依自己的行數決定高度。
  - Hero 分數軸：390px 時兩個標籤分兩列（「判別式 RM」那一列在上方，並以引線連到標記）。
  - Eq. 3、5、6、7 在 390px 時於 `.katex-display` 容器內橫向捲動，沒有拆成兩行；頁面本身不產生水平捲動（selfTest 有檢查）。
  - C4 結果卡的名稱縮短為「SC（票數）」「Best-of-N」「加權（全部）」「加權 top-K」。
- 驗收：`?test` 在 1440 與 390 皆為 31 passed、0 failed；`lo.py` 12／12；impeccable 0 項 anti-pattern，只剩 12 則 numbered-section-labels advisory。

## 11. 驗收後修正（accept/grade.md）

- 表格儲存格之間加入視覺隱藏的「；」（`.sr`），讀數 `dl` 的 dt／dd 間加「：」「；」；主要結果表的設定欄以「；」收尾（例「Gemma-2B，Best-of-32；5.0% → 45.3%」），避免 N 與百分比黏在一起。
- S10 新增 K／K_top／λ 三列對照表。
- C1–C3 讀數框的主要數字旁加「非論文結果」，C4 結果卡下方加一行同義說明。
- 重測：`?test` 1440／390 皆 31 passed、0 failed；`lo.py` 12／12。
