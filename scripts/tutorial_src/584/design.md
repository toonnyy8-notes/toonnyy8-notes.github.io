# 設計文件：〈Attention Sink Forges Native MoE in Attention Layers〉互動教學

- 論文：Zizhuo Fu、Wenxuan Zeng、Runsheng Wang、Meng Li，arXiv 2602.01203（v3，ICML 2026）。資料庫編號 No. 584。
- 原文：`584/src/paper.txt`（PDF 同目錄）。本文件中的「Eq. n」「Sec. n」「App. X」「Fig. n」「Table n」皆指論文原文編號（正文 Eq. 1–8；App. A 為 Eq. 9；App. B 為 Eq. 10–23）。
- 產出：`584/build/index.html`（單一 HTML 檔）。上線位置：`tutorials/attention-sink-native-moe.html`（slug `attention-sink-native-moe`）。站內相對連結以上線位置為準：論文 `../index.html#paper-584`、筆記 `<slug>.html`。
- 驗算腳本：`584/verify/check.py`（輸出存於 `verify/check_out.txt`），與頁面用同一演算法；第 5 節驗收表全部出自它。
- 研究筆記：`584/notes/`（總覽 `asink-lineage-hub` + 7 篇 `asink-*` 主題筆記 + `timeline.json`，15 筆），用法見 6.6。
- 文件慣例：**論文沒有寫出、由本設計自行推導**的結論標「〔本頁推導〕」；**為了示範而自訂、論文沒有給的數值**標「示意」；用論文數字校準的值標「推定」（本文目前沒有推定值）。頁面用同樣的標籤。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

大型語言模型常把大量注意力權重放在第一個 token 上，稱為注意力匯聚（attention sink）。GPT-OSS 的 Sink Attention 在 softmax 分母加一個可學習的 `sink` 項（Eq. 1），Qwen3-Next 的 Gated Attention 為每個 head 乘上一個 sigmoid 門控 $G=\sigma(xW_\theta)$（Eq. 2），兩者都能消除 attention sink。本文指出三種機制的共同點：Vanilla 與 Sink Attention 中，分給 sink 成分的注意力權重 $A_{\text{sink}}$ 讓輸出寫成 $(1-A_{\text{sink}})\cdot\sum\tilde A v$（Eq. 3），所以 $G=1-A_{\text{sink}}$（Eq. 4）是一個**隱式門控**，並且可以寫成 sigmoid 形式 $\sigma(\mathrm{LSE}-\mathrm{sink})$，與 Gated Attention 的**顯式門控**對應。多個 head 是專家，門控是路由器，所以注意力層本身就是一個 Mixture-of-Experts（MoE）。從這個角度看，先前觀察到「只有少數 head 真正有貢獻」就是 MoE 的專家崩塌，作者稱為 **head collapse**。作者以每個 head 的平均門控值定義重要度 $\mathrm{Imp}$（Eq. 5），用變異係數（CV）量化不平衡（Eq. 6），提出輔助負載平衡損失 $\mathcal L_{\text{aux}}=\lambda\sum_l N_h[\mathrm{CV}]^2$（Eq. 7）；微調已訓練模型時，保留 top-$m$ 個 head 為 shared heads，只對其餘 routed heads 做平衡（Eq. 8）。門控值可由 Flash Attention 回傳的 LSE 算出，訓練延遲增加不到 2%（Sec. 5.1）。從頭訓練 0.6B／1B／2B 模型與微調 Qwen3-4B／8B、LLaMA3.1-8B，加上輔助損失後平均分數都上升（Table 1、Table 2）。

### 1.2 研究動機（Sec. 1）

1. Sink Attention 與 Gated Attention 實務上有效，但「消除 attention sink 為什麼有幫助」沒有解釋。
2. 三種注意力機制之間缺少統一的觀點。
3. 先前研究（DuoAttention；Sandoval-Segura et al., 2025；Sok et al., 2026）發現只有固定一小部分 head 有貢獻，把超過 25% 的 head 歸零仍保持準確率（Sec. 4.1），但沒有把它與 MoE 的專家崩塌連起來，也沒有訓練時的對策。

### 1.3 貢獻

1. 隱式門控：Vanilla 與 Sink Attention 的 $1-A_{\text{sink}}$ 對應 Gated Attention 的顯式門控（Eq. 3–4，App. B 推導），因此注意力層是原生 MoE（Sec. 3.1）。
2. 解釋消除 sink 的好處：用第一個 token 當 sink 會讓 query／key 幾何呈兩極結構，壓縮點積差異（Sec. 3.2，Fig. 3）。
3. 以門控定義 head 重要度與不平衡指標，指出 head collapse 普遍存在且隨訓練加劇（Sec. 4.1，Fig. 4–5，App. C）。
4. Sink-aware 輔助負載平衡損失：從頭訓練用 Eq. 7，微調用 shared／routed 版 Eq. 8（Sec. 4.2，App. D）；以 LSE 與 Flash Attention 相容。
5. 實驗：三種機制 × 三種規模 × 有無輔助損失；三個預訓練模型的 LoRA 微調（Sec. 5）。

### 1.4 符號表（頁面 S1 末尾以可收合表格提供）

| 符號 | 意義 |
| --- | --- |
| $l,h$ | 層、head 的索引；$N_L$ 層數，$N_h$ 每層 head 數 |
| $t$ | 目前 token（query）的位置；causal mask 下可看到 $j=0,\dots,t$ |
| $q_t,k_j,v_j\in\mathbb R^{d_h}$ | query、key、value；$d_h$ 為 head 維度 |
| $z_{t,j}=q_t^\top k_j/\sqrt{d_h}$ | softmax 前的 logit（App. B.1） |
| $A_{t,j}$ | 注意力權重；$A_{t,\text{sink}}$ 為 sink 成分的權重（Vanilla 為 $A_{t,0}$，Sink 為合成 sink 的權重） |
| $\tilde A_{t,j}$ | 只在非 sink token 上重新正規化的權重（和為 1） |
| $\mathrm{sink}^{l,h}$ | Sink Attention 每個 head 一個可學習純量（App. B.3） |
| $G_t^{l,h}$ | 門控值：Gated 為 $\sigma(x_tW_\theta)$；Vanilla／Sink 為 $1-A_{t,\text{sink}}$ |
| $\sigma(u)$ | sigmoid $1/(1+e^{-u})$ |
| $\mathrm{LSE}(\{a_i\})$ | $\log\sum_i e^{a_i}$；$\mathrm{LSE}_{\neg0}$ 不含第一個 token，$\mathrm{LSE}_{\text{tok}}$ 只含真實 token |
| $\mathrm{Imp}^{l,h}$ | head 重要度，所有 token 上 $G$ 的平均（Eq. 5） |
| $\mathrm{CV}$ | 變異係數 std/mean（Eq. 6） |
| $\lambda$、$m$、$\mathcal T_m^l$ | 輔助損失強度；shared head 數；第 $l$ 層 top-$m$ head 集合 |

### 1.5 從作者的角度逐步拆解

#### 步驟 1：Vanilla Attention 與 attention sink（Sec. 2.1）

$$A^{l,h}_{t,j}=\mathrm{Softmax}\Big(\frac{q_t^{l,h\top}k_j^{l,h}}{\sqrt{d_h}}\Big),\qquad O_t^{l,h}=\sum_{j=0}^{t}A^{l,h}_{t,j}v_j^{l,h}$$

- StreamingLLM（Xiao et al., 2023）發現 LLM 常把高注意力權重放在第一個 token，與語意無關，稱為 attention sink。已被用在 KV cache 最佳化、長脈絡生成、head 剪枝、量化（Sec. 1）。
- 成因（Sec. 2.1）：softmax 強迫權重和為 1。head 只需要部分或完全不需要分配注意力時，剩下的權重分配到第一個 token（論文用 absorb 一詞）；第一個 token 語意內容少，其 value 向量接近 0（Gu et al., 2024；Guo et al., 2024），所以這部分權重滿足正規化但不影響輸出。
- 頁面 S1 的注意力入門（給不熟 Transformer 的讀者，論文未寫、屬背景）：每個 token 的表示經三個線性映射得到 query、key、value；logit $z_{t,j}=q_t^\top k_j/\sqrt{d_h}$ 衡量 token $t$ 對 token $j$ 的匹配；causal mask 讓 token $t$ 只看 $j\le t$；softmax 把 logit 轉成和為 1 的權重；一層有 $N_h$ 個 head，各有自己的映射，輸出相加（經 $W_O$）。LSE 定義為 $\mathrm{LSE}(\{a_i\})=\log\sum_ie^{a_i}$（App. B.1），softmax 可寫成 $A_j=e^{z_j-\mathrm{LSE}}$。
- 下一步的理由：既然問題出在「和為 1」，兩類解法是改 softmax 或加門控。

#### 步驟 2：兩種消除 sink 的機制（Sec. 2.2，Fig. 1）

$$\mathrm{Softmax}_{\text{sink}}(x)_i=\frac{\exp(x_i)}{\sum_j\exp(x_j)+\exp(\mathrm{sink})}\tag{Eq. 1}$$

- Sink Attention（GPT-OSS；Fig. 1 另列 DeepSeek-V4）：放寬正規化，以 value 為 0 的合成 sink 取代第一個 token。

$$G_t^{l,h}=\sigma\big(x_t^lW_\theta^{l,h}\big),\quad W_\theta^{l,h}\in\mathbb R^{d_{\text{model}}\times1};\qquad O_t^l=\sum_{h=1}^{H}G_t^{l,h}\cdot O_t^{l,h}W_O^{l,h}\tag{Eq. 2}$$

- Gated Attention（Qwen3-Next；Fig. 1 另列 Qwen3.5、3.6）：head 專屬門控 + sigmoid。Sec. 2.2 寫：這個機制在注意力層引入 MoE 結構，使 head 稀疏使用並消除 sink。
- Fig. 1 把 Vanilla（Qwen2、3，LLaMA2、3）、Sink 歸為「隱式門控」，Gated 歸為「顯式門控」。

#### 步驟 3：value drain——第一個 token 的 value 接近 0（Sec. 3.1，Fig. 2，App. A）

- Fig. 2：Qwen2.5-7B、Qwen3-8B、Qwen3-14B、LLaMA-2-7B、LLaMA-3.1-8B、LLaMA-2-13B 中，第一個 token value 的 $\ell_2$ 範數都遠小於位置 1–16 的 token。
- 作者的解釋：第一個 token 收到大量權重，把它的 value 設成接近 0 才不影響輸出。Sink Attention 的 sink 參數沒有 value，貢獻恰為 0。所以兩者都是「把多出來的權重分到 value 為 0（或接近 0）的成分」。
- 頁面 S3 在 Eq. 9 之前先定義：sink 成分＝收到這些權重、value 為 0 或接近 0 的成分；Vanilla 中是第一個 token，所以 $A_{t,\text{sink}}=A_{t,0}$（Sink 中的合成 sink 在 S4 介紹）。
- App. A 介入實驗：KV cache 生成後把第一個 token 的 value 歸零。兩種設定：全部 head 歸零（All）；只對 head 級 sink 比例 $\alpha_{\text{sink}}^{l,h}=\frac1T\sum_{i=0}^{T-1}A^{l,h}_{i,\text{sink}}$（Eq. 9）大於 $\tau=0.75$ 的 head 歸零。Table 3 平均：

  | 模型 | None | All | τ = 0.75 |
  | --- | --- | --- | --- |
  | Qwen3-8B | 70.36 | 64.10 | 70.38 |
  | Qwen3-14B | 66.91 | 65.90 | 67.09 |

  論文文字寫「全部歸零只造成可忽略的下降」，但 Qwen3-8B 的 All 平均下降 6.26（MMLU 75.42 → 50.27、HellaSwag 74.97 → 61.02）。頁面照錄數字，並寫明「選擇性歸零（τ = 0.75）與不介入幾乎相同，全部歸零在 Qwen3-8B 下降 6.26」（見 R3）。

#### 步驟 4：隱式門控（Sec. 3.1，Eq. 3–4，App. B）

把 sink 成分的 value 視為 0：$A_{t,\text{sink}}v_{\text{sink}}\approx0$。則

$$O_t^{l,h}=\sum_{j\ne\text{sink}}A_{t,j}v_j=(1-A_{t,\text{sink}})\cdot\sum_{j\ne\text{sink}}\tilde A_{t,j}v_j,\qquad \tilde A_{t,j}=\frac{A_{t,j}}{1-A_{t,\text{sink}}}\tag{Eq. 3}$$

$$G_t^{l,h}=1-A^{l,h}_{t,\text{sink}}\tag{Eq. 4}$$

- App. B.2（Vanilla）：$D_t=\sum_{m=0}^t e^{z_{t,m}}$、$D_{t,\neg0}=\sum_{m=1}^t e^{z_{t,m}}$；$A_{t,j}=\tilde A_{t,j}\cdot D_{t,\neg0}/D_t$（Eq. 12–13），$D_{t,\neg0}/D_t=1-A_{t,0}$，得 Eq. 14：$O_t\approx(1-A_{t,0})\sum_{j=1}^t\tilde A_{t,j}v_j$。**Vanilla 的等式是近似**：誤差恰為 $A_{t,0}v_0$（Eq. 10 的 ≈）。
- Sigmoid 形式（Eq. 15–16）：$1-A_{t,0}=\dfrac{D_{t,\neg0}}{D_{t,\neg0}+e^{z_{t,0}}}=\sigma\big(\mathrm{LSE}_{t,\neg0}-z_{t,0}\big)$，$z_{t,0}=q_t^\top k_0/\sqrt{d_h}$。
- App. B.3（Sink）：$D_{t,\text{tok}}=\sum_{m=0}^te^{z_{t,m}}$，$D_{t,\text{all}}=D_{t,\text{tok}}+e^{\mathrm{sink}}$；Eq. 19–21 得 $O_t=(1-A_{t,\text{sink}})\sum_{j=0}^t\tilde A_{t,j}v_j$，**恰好成立**（合成 sink 沒有 value）。Eq. 22–23：$1-A_{t,\text{sink}}=\sigma(\mathrm{LSE}_{t,\text{tok}}-\mathrm{sink})$。
- 作者指出隱式門控與 Gated Attention 的兩個核心設計一致（Sec. 3.1）：(1) head 專屬，每個 head 有自己的 sink 權重；(2) 有等價的 sigmoid 形式，而 Qiu et al. (2025b) 實驗發現 sigmoid 門控最好。
- 〔本頁推導〕Vanilla 要讓門控等於 $G$，第一個 token 的 logit 必須比其他 token 的 LSE 高出 $z_{t,0}-\mathrm{LSE}_{t,\neg0}=\ln\frac{1-G}{G}$（由 Eq. 16 取反函數）。例：$G=0.1$ 需要高出 $\ln9\approx2.197$。這是與步驟 6 幾何描述相容的必要條件（Sec. 3.2 的幾何是定性的 PCA 觀察，不是由此式推出）。
- 頁面拆成兩小節（降低密度）：S4a「Sink Attention：恰好成立」（Eq. 1 → Eq. 19–21 → Eq. 22–23）；S4b「Vanilla Attention：在 $v_0\approx0$ 時近似成立」（Eq. 12–16、誤差 $A_{t,0}v_0$、$\ln\frac{1-G}{G}$）。App. B.2 的逐行推導放在 S4b 的 `<details>`。C1 放在 S4b 末尾，只有 Vanilla／Sink 兩種模式。

#### 步驟 5：注意力層是原生 MoE（Sec. 3.1 末，Sec. 6.2）

- MoE 兩要素：獨立專家模組 + 決定貢獻的路由機制（Shazeer et al., 2017；Fedus et al., 2021）。多個 head 是獨立專家，sink 形成的隱式門控是路由，所以 attention sink 在注意力層形成原生 MoE。
- 限制（Sec. 6.2）：標點等 token 也可能成為局部 sink（Yu et al., 2024），但它們常匯集前文資訊，不一定只收權重而不貢獻輸出（論文用語 pure attention dumps；Chen et al., 2024）；第一個 token 在 causal mask 下語意有限，較可靠（Barbero et al., 2025）。更一般的隱式門控：$G_t^{l,h}=1-\sum_{j\in S^{l,h}}A^{l,h}_{t,j}$，$S^{l,h}$ 為 sink 類成分集合，Eq. 4 是 $S$ 只含一個成分的特例。DeepSeek-V4 中 sink 參數沒有收下全部多出來的權重，第一個 token 仍有大量權重，此時 $A_{\text{sink}}$ 應同時包含兩者；實務上可用「高注意力權重 + 低 value 範數」找出這類成分。Gated Attention 也可能另外形成隱式門控（若某些 token 收到冗餘權重且 value 小），所以 head 啟用可能由多個門控共同決定。頁面在 S5 以文字說明這一點，C1 不提供 Gated 模式（見 R14）。

#### 步驟 6：消除 sink 的好處——query／key 幾何（Sec. 3.2，Fig. 3）

- 方法：以 LongBench 的 2k token 樣本，將 query 與 key 做 PCA 投影到二維。頁面附一句說明：PCA（主成分分析）找出資料變異最大的兩個正交方向，把高維向量投影到這兩個方向上畫成平面圖。
- Vanilla（Fig. 3(a)(b)(c)）：為維持第一個 token 的高權重，後續 token 的 query 聚在 $k_0$ 附近，key 被推離 $k_0$，形成兩極結構。query 只能在窄角度範圍變化，key 的有效維度也受限，點積的相對差異被壓縮，softmax 區分候選 token 的精度下降；長脈絡中造成注意力雜訊（Liu et al., 2023）。
- Sink／Gated（Fig. 3(d)(e)(f)）：$k_0$ 不再占特殊位置，query 與 key 呈分散分布，沒有明顯兩極化。
- Fig. 3 的 (a)(b)(d)(e) 是實際模型（Qwen3-8B、LLaMA-3.1-8B、GPT-OSS-20B、Qwen3-Next-80B-A3B），(c)(f) 是示意圖（Q-allowed／K-allowed 區域與 All-allowed）。(a)(b)(d)(e) 的座標軸標注如「PC1 44.7%, λ = 0.303」，論文正文沒有解釋這些數字。頁面不引用、不詮釋這些百分比，只描述正文的定性結論（見 R10）。
- 先前研究已知的負面效應：極端激活離群值使量化困難；降低表達能力；消除 sink 提升長脈絡表現（Sec. 3.2 引用）。

#### 步驟 7：head collapse 與重要度指標（Sec. 4.1，Eq. 5–6，Fig. 4–5，App. C）

$$\mathrm{Imp}^{l,h}=\frac{1}{|\mathcal T|}\sum_{(i,t)\in\mathcal T}G^{l,h}_{i,t}\tag{Eq. 5}$$

$$\text{Head Imbalance}=\frac1{N_L}\sum_{l=1}^{N_L}\mathrm{CV}\big(\{\mathrm{Imp}^{l,h}\}_{h=1}^{N_h}\big),\qquad \mathrm{CV}(\cdot)=\frac{\mathrm{std}(\cdot)}{\mathrm{mean}(\cdot)}\tag{Eq. 6}$$

- $\mathcal T$ 為 batch 中所有 token（$i$ 樣本、$t$ 位置）。$G$ 對三種機制用統一定義：Gated 用 Eq. 2，Vanilla／Sink 用 Eq. 4。$\mathrm{Imp}$ 越高，head 對層輸出的貢獻越大。
- Fig. 4（LLaMA-3.1-8B、GPT-OSS-20B、Qwen3-Next-80B-A3B）與 App. C（GSM8K 500 個樣本；Fig. 10 六個模型：GPT-OSS-20B／120B、LLaMA-2-7B／13B／70B、LLaMA-3.3-70B；Fig. 11 十個 Qwen2.5／Qwen3 模型；合計 16 個）：三種機制都有少數高重要度 head、多數低重要度 head。App. C：模型越大，崩塌越嚴重。Qwen3-Next 使用 element-wise 門控，先在 head 內對維度平均再套 Eq. 5。
- Fig. 5：從頭訓練時，Sec. 4.1 概括為「不平衡在早期上升，最後停在較高水準」；Sec. 5.2 細分為 Gated 早期急升後持平、Vanilla 與 Sink 逐步上升。作者認為某些 head 早期占優勢後得到較多的更新（原文 attract disproportionate updates）。
- 崩塌的三個後果（Sec. 4.1）：有效參數量低於名目值；不平衡妨礙學習多樣表示（類比 MoE）；崩塌的 head 仍消耗運算。
- 〔本頁推導〕每個 $G\in[0,1]$，所以 $\mathrm{Imp}\in[0,1]$。Vanilla 在 $t=0$ 時只有一個 token，$A_{0,0}=1$、$G=0$（$\mathrm{LSE}_{\neg0}$ 為空集合）；Eq. 5 平均所有 $(i,t)$，論文沒有說明 $t=0$ 是否排除。只要有任一 $t\ge1$ 的 token 使 $G>0$，$\mathrm{Imp}>0$，CV 的分母為正。
- 頁面用的手算例（LO7）：$\mathrm{Imp}=(0.9,0.1,0.1,0.1)$：mean 0.3，母體 std $\sqrt{0.12}=0.3464$，CV = 1.1547。

#### 步驟 8：從頭訓練的輔助損失（Sec. 4.2，Eq. 7）

$$\mathcal L_{\text{aux}}=\lambda\sum_{l=1}^{N_L}N_h\Big[\mathrm{CV}\big(\{\mathrm{Imp}^{l,h}\}_{h=1}^{N_h}\big)\Big]^2,\qquad \mathcal L=\mathcal L_{\text{base}}+\mathcal L_{\text{aux}}\tag{Eq. 7}$$

- 形式來自 MoE 負載平衡（Shazeer et al., 2017；Fedus et al., 2021）。從頭訓練 $\lambda=1\times10^{-4}$（Sec. 5.1）。
- 〔本頁推導〕用母體 std（$\mathrm{std}^2=\frac1N\sum(x_h-\mu)^2$）時，$N\cdot\mathrm{CV}^2=\sum_{h}\big(\tfrac{x_h}{\mu}-1\big)^2$，$x_h=\mathrm{Imp}^{l,h}$、$\mu$ 為平均。所以 Eq. 7 的 $N_h$ 把「每個 head 相對平均的偏差平方」加總，而非平均。手算例：$(0.9,0.1,0.1,0.1)$ 得 $2^2+3\cdot(2/3)^2=5.333$。
- 〔本頁推導〕對單層：$\dfrac{\partial(N\mathrm{CV}^2)}{\partial x_h}=\dfrac{2(x_h-\mu)}{\mu^2}-\dfrac{2S}{N\mu^3}$，$S=\sum_i(x_i-\mu)^2$。第一項把高於平均的 head 往下推、低於平均的往上推；第二項對所有 head 相同且為負，在梯度下降中使所有 head 一起增加。CV 對整體縮放不變：$\sum_hx_h\,\partial(N\mathrm{CV}^2)/\partial x_h=0$，所以這個損失不限制平均值，只限制相對差異。
- 頁面處理（降低密度）：S8 正文只寫兩句——Eq. 7 的定義，以及「CV 不隨所有 head 同乘一個倍數而改變，所以損失只要求各 head 相對一致，不規定它們的平均水準」。恆等式、梯度式與手算 5.333 放在 S8 的 `<details>`「推導」，LO8 的答案句另寫在正文中。
- 論文未說明 std 用母體（除以 $N$）或樣本（除以 $N-1$）版本；PyTorch `torch.std` 預設為樣本版本。頁面採母體版本並註明（R4）。

#### 步驟 9：微調已訓練模型——head pinning 與 shared／routed（Sec. 4.2，Eq. 8，App. D）

- head pinning effect（App. D，Fig. 12 中圖）：直接用 Eq. 7 微調 Qwen3-8B 一個 epoch 後，幾乎所有 head 都有一致的高重要度。作者的解釋：預訓練時崩塌出的主導 head 對生成已不可或缺，降低它們會傷害表現，所以它們的門控在微調中幾乎不變（原文 resist change）；損失要求一致時，其餘 head 的門控升高到與主導 head 相近。表面上 CV 變小，但不是期望的平衡。
- 對策（借鑑 DeepSeekMoE 的 shared／routed experts）：每層 top-$m$ 個最重要 head 為 shared heads，保留已學到的行為；其餘為 routed heads，只對它們做平衡。$\mathcal T_m^l=\arg\mathrm{top}_m\{\mathrm{Imp}^{l,h}\}_{h=1}^{N_h}$：

$$\mathcal L_{\text{aux}}=\lambda\sum_{l=1}^{N_L}(N_h-m)\Big[\mathrm{CV}\big(\{\mathrm{Imp}^{l,h}:h\notin\mathcal T_m^l\}\big)\Big]^2\tag{Eq. 8}$$

- Fig. 12 右圖：用 Eq. 8 微調後，routed heads 間更平均，shared heads 維持高啟用。微調 $\lambda=1\times10^{-2}$（Sec. 5.1）。
- 論文**沒有給 $m$ 的值**，也沒有說明 $\mathcal T_m^l$ 是固定（微調前選定）還是每個 batch 重選（R5）。
- 頁面只陳述：C3 的玩具模型顯示「固定幾個高重要度 head 時，最小化 CV 的梯度下降會讓其餘 head 一起升高」，方向與 App. D 描述相同。主導 head 以「凍結」表示「幾乎不變」，這是本頁的設定，pinning 的結果直接來自這個設定；玩具模型不證明論文的因果解釋（R9）。

#### 步驟 10：與 Flash Attention 相容的實作（Sec. 4.2 末）

- 問題：$G=1-A_{\text{sink}}$ 需要注意力權重，Flash Attention 等融合 kernel 不輸出權重；改用 eager 模式會犧牲效率。
- Vanilla：$G=\sigma(\mathrm{LSE}_{\neg0}-q_t^\top k_0/\sqrt{d_h})$。呼叫 `out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v)`，由回傳的 LSE 與第一個 token 的 query-key 乘積算出 $G$，不需實體化完整權重。
- Sink：現有做法（Agarwal et al., 2025）用非融合運算，顯著降低效率。作者改為對所有 token 跑標準 Flash Attention，再乘上 $G=\sigma(\mathrm{LSE}-\mathrm{sink})$，$\mathrm{LSE}$ 由 kernel 取得；同一個 $G$ 直接供輔助損失使用。
- 〔本頁推導〕標準 Flash Attention 對真實 token 正規化，其輸出就是 $\sum_j\tilde A_{t,j}v_j$，由 Eq. 21，$O_t=\sigma(\mathrm{LSE}_{\text{tok}}-\mathrm{sink})\cdot\text{FlashOut}$ 是 Sink Attention 的**精確**輸出。Vanilla 的 Flash 輸出本身就是完整輸出，$G$ 只用於 $\mathrm{Imp}$ 與 $\mathcal L_{\text{aux}}$，**不可**再乘到輸出上。
- 〔本頁推導〕Flash Attention 回傳的 LSE 含第一個 token。由它可得 $G=1-e^{z_{t,0}-\mathrm{LSE}}$，應寫成 `-expm1(z0 - lse)`；或先算 $\mathrm{LSE}_{\neg0}=\mathrm{LSE}+\log(-\mathrm{expm1}(z_{t,0}-\mathrm{LSE}))$ 再代入 sigmoid，兩者數學相同、精度相同。$A_{t,0}\to1$ 時 LSE 本身已四捨五入，相對精度流失（C4：logit 放大 200 倍時得到 0；若另外對 $j\ge1$ 計算 $\mathrm{LSE}_{\neg0}$，得 $1.9\times10^{-174}$）。論文只寫「由回傳的 LSE 與首 token 的 query-key 乘積計算」，沒有說明 $\mathrm{LSE}_{\neg0}$ 如何排除首 token（R6）；頁面把「另外計算 $\mathrm{LSE}_{\neg0}$」標為「需另行取得 LSE¬0（論文未說明）」，不宣稱是論文做法。對 $\mathrm{Imp}$ 而言絕對誤差 $<10^{-16}$，可忽略。
- LSE 本身是 Flash Attention 分塊計算時維護的量：逐塊更新最大值 $m$ 與縮放後的和 $\ell$，最後 $\mathrm{LSE}=m+\ln\ell$（C4）。這是 FlashAttention 的標準做法，不是本論文內容，頁面標「背景」。

#### 步驟 11：實驗設定（Sec. 5.1）

| 項目 | 從頭訓練 | 微調 |
| --- | --- | --- |
| 模型 | 0.6B：20 層、每層 10 head、hidden 1280；1B：26 層、13 head、1664；2B：32 層、16 head、2048 | Qwen3-4B、Qwen3-8B、LLaMA3.1-8B |
| 機制 | Vanilla、Sink、Gated，各有無 $\mathcal L_{\text{aux}}$（六種組合） | 模型原有機制 |
| 資料 | FineWeb-Edu；token 數 = 20 × 參數量（Chinchilla） | AceReason-Nemotron（長序列推理） |
| 最佳化 | lr 0.02，最後 20% 線性衰減到 0；其他依 AdamW 預設 | LoRA rank 16 |
| $\lambda$ | $1\times10^{-4}$ | $1\times10^{-2}$ |
| 評測 | MMLU、GSM8K、HumanEval、ARC-E/C、HellaSwag、PIQA、OpenBookQA、BoolQ、Winogrande；validation BPB | 推理：ARC、MuSR、GSM8K、MATH-500、Comp-Math、ProcessBench；LongBench 六項：HotpotQA、Qasper、TriviaQA、NarrativeQA、2WikiMQA、MultiFieldQA |
| 硬體／成本 | 額外訓練延遲 < 2% | NVIDIA A100 |

#### 步驟 12：實驗結果（Sec. 5.2–5.3，Table 1–2，Fig. 5–8）

Table 1 平均分數（base → base+aux）與 validation BPB：

| 規模 | Vanilla | Sink | Gated |
| --- | --- | --- | --- |
| 0.6B 平均 | 39.54 → 40.82（+1.28） | 40.31 → 41.11（+0.80） | 40.43 → 41.35（+0.92） |
| 1B 平均 | 42.59 → 43.35（+0.76） | 43.27 → 43.75（+0.48） | 42.69 → 43.98（+1.29） |
| 2B 平均 | 45.46 → 46.83（+1.37） | 46.35 → 46.50（+0.15） | 46.30 → 47.70（+1.40） |
| 2B BPB | 0.7243 → 0.7198 | 0.7226 → 0.7202 | 0.7232 → 0.7197 |

- 作者的結論一：Sink 與 Gated 在三個規模都優於 Vanilla（base 欄成立）。作者寫 Gated「在各規模最好」；照數字只在 base+aux 欄成立（41.35、43.98、47.70 皆最高），base 欄 1B、2B 是 Sink 略高（43.27 vs 42.69、46.35 vs 46.30）。頁面照數字寫（R7）。
- 結論二：加上 $\mathcal L_{\text{aux}}$ 後平均分數與 BPB 在所有機制、規模都改善；個別基準不一定（例：2B Sink BoolQ 59.80 → 57.40）。
- Fig. 5：加上輔助損失後 CV 維持低值。不加時，Gated 早期急升後持平；Vanilla、Sink 逐步上升（Sec. 5.2 原文 gradual increase）；Vanilla 震盪較大。作者解釋：Gated 的顯式門控讓 head 立即專門化；Vanilla／Sink 要先建立 sink 分配才有隱式門控；Vanilla 還要同時學會 value drain。
- Fig. 6（2B）：無平衡時啟用集中在少數 head 且跨資料集一致；有平衡時較平均、隨資料集變化。Fig. 7：Sink、Gated 準確率全程高於 Vanilla。Fig. 8：早期輔助損失影響小，後期 loss 與 BPB 較低。
- Table 2（平均，base → base+aux）：

  | 模型 | 推理 | LongBench |
  | --- | --- | --- |
  | Qwen3-4B | 74.73 → 75.24 | 51.46 → 52.75 |
  | Qwen3-8B | 74.31 → 76.01 | 50.67 → 55.38 |
  | LLaMA3.1-8B | 56.50 → 58.97 | 40.51 → 43.98 |

  作者觀察：較大的模型受益較多（本頁對照：Qwen3-8B 的增幅 +1.70／+4.71 大於 Qwen3-4B 的 +0.51／+1.29；論文未逐一列出增幅），與 MoE 研究中專家數越多越需要負載平衡一致。個別任務有下降（例：LLaMA3.1-8B TriviaQA 46.34 → 41.82）。

#### 步驟 13：head collapse 的機制假說（Sec. 6.1，Fig. 9）

$$\text{Value Drain}\to\text{Attention Sink}\to\text{Implicit Gating}\to\text{Head Collapse}$$

- Fig. 9 追蹤三個診斷：value drain（第一個 token value 的 $\ell_2$ 範數 ÷ 其他 value 平均範數）、attention sink（sink 成分平均權重）、Head Load Imbalance CV。
- Gated：門控從一開始就是顯式的，崩塌可直接發生（早且嚴重）。Sink：sink 沒有 value，不需先壓低真實 token 的 value；sink 一取得權重，隱式門控就形成，介於兩者之間。Vanilla：要走完整條鏈。三者不必嚴格分階段、可共同演化；重點是「有效的門控是崩塌的前提」。論文稱為假說（hypothesis）。

#### 步驟 14：常見錯誤（分散到相關章節，避免集中在 S10）

S4b 末尾：
1. 把 Vanilla 的 Eq. 3 當成恆等式：它依賴 $v_0\approx0$，誤差是 $A_{t,0}v_0$；Sink Attention 才是恰好成立。

S8 末尾：
2. std 的版本：論文未說明；`torch.std` 預設除以 $N-1$。損失中的 $\mathrm{CV}^2$ 兩版相差 $N/(N-1)$ 倍（std 與 CV 本身相差 $\sqrt{N/(N-1)}$ 倍），會改變 $\lambda$ 的有效大小。
3. $\lambda$ 混用：從頭訓練 $10^{-4}$、微調 $10^{-2}$。
4. 跨機制比較 CV 絕對值：Fig. 5 的 Vanilla 縱軸刻度為 1e-3（0.6B）與 1e-4（2B），Sink、Gated 為 1e-1（R8）。頁面只比較同一機制有無輔助損失。

S9 末尾：
5. 微調時直接用 Eq. 7：造成 head pinning；改用 Eq. 8。$m$ 值論文未給。

S10 末尾：
6. Vanilla 用 Flash 時把 $G$ 乘到輸出：Vanilla 的輸出不變，$G$ 只用於重要度與損失〔本頁推導〕；Sink Attention 才是 FlashOut × $G$。
7. 用 $1-A_{t,0}$ 直接相減：$A_{t,0}\to1$ 時相消；至少用 `-expm1`，若能另外取得 $\mathrm{LSE}_{\neg0}$ 則用 $\sigma(\mathrm{LSE}_{\neg0}-z_{t,0})$〔本頁推導〕。
8. 直接加總 $e^{z}$：logit 大於約 709.78 時 float64 溢位；用減最大值或分塊的 $m+\ln\ell$。

---

## 2. 學習目標（驗收測試）

每個參考答案必須寫在該章節的正文中（不只在元件、表格、圖或答案框）。章節末以 `<details>` 呈現檢查題（summary「檢查理解」）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明 attention sink 是什麼與成因 | attention sink 指什麼？為什麼會出現？ | LLM 把大量注意力權重放在第一個 token，與語意無關；softmax 強迫權重和為 1，head 不需要完整分配時，剩下的權重分到 value 接近 0 的第一個 token（Sec. 1、2.1） | S1 |
| LO2 | 寫出兩種消除 sink 的機制 | Sink Attention 與 Gated Attention 各改了什麼？ | Sink：softmax 分母加 $\exp(\mathrm{sink})$，sink 為可學習參數、沒有 value（Eq. 1，GPT-OSS）；Gated：每個 head 輸出乘上 $G=\sigma(xW_\theta)$，$W_\theta\in\mathbb R^{d_{\text{model}}\times1}$（Eq. 2，Qwen3-Next） | S2 |
| LO3 | 說明 value drain 與其證據 | 什麼是 value drain？App. A 歸零實驗的結果？ | 第一個 token 的 value 範數接近 0（Fig. 2）；只對 sink 比例 > 0.75 的 head 歸零時 Qwen3-8B 平均 70.38（不介入 70.36），全部 head 歸零為 64.10（Table 3） | S3 |
| LO4 | 推導隱式門控與 sigmoid 形式 | Vanilla 與 Sink Attention 的輸出如何寫成門控形式？$G$ 的 sigmoid 形式？ | $O=(1-A_{\text{sink}})\sum\tilde A v$，$\tilde A=A/(1-A_{\text{sink}})$，$G=1-A_{\text{sink}}$（Eq. 3–4）；Sink：$G=\sigma(\mathrm{LSE}_{\text{tok}}-\mathrm{sink})$，Vanilla：$G=\sigma(\mathrm{LSE}_{\neg0}-q^\top k_0/\sqrt{d_h})$（Eq. 16、23）；Vanilla 依賴 $v_0\approx0$，誤差為 $A_0v_0$；Sink 恰好成立 | S4a、S4b |
| LO5 | 說明「原生 MoE」的對應 | 注意力層中誰是專家、誰是路由器？隱式門控與 Gated Attention 共有哪兩個設計？ | head 是專家，sink 形成的門控是路由；head 專屬、有 sigmoid 形式（Sec. 3.1） | S5 |
| LO6 | 說明消除 sink 的幾何好處 | Vanilla 模型的 query／key 分布有什麼限制？為何影響長脈絡？ | 兩極結構：query 聚在 $k_0$ 附近、key 被推離 $k_0$；點積相對差異被壓縮，softmax 區分 token 的精度下降，長脈絡中產生注意力雜訊（Sec. 3.2、Fig. 3） | S6 |
| LO7 | 計算 head 重要度與不平衡指標 | $\mathrm{Imp}$ 與 Head Imbalance 如何定義？$\mathrm{Imp}=(0.9,0.1,0.1,0.1)$ 的 CV？ | $\mathrm{Imp}$＝所有 token 上 $G$ 的平均（Eq. 5）；Head Imbalance＝各層 CV 的平均，CV＝std/mean（Eq. 6）；例：mean 0.3、std 0.3464、CV 1.1547（母體 std） | S7 |
| LO8 | 寫出從頭訓練的輔助損失 | Eq. 7 是什麼？$\lambda$ 取多少？上例的 $N\cdot\mathrm{CV}^2$？ | $\mathcal L_{\text{aux}}=\lambda\sum_lN_h[\mathrm{CV}]^2$，$\lambda=10^{-4}$；$N\mathrm{CV}^2=\sum_h(x_h/\mu-1)^2=5.333$〔本頁推導〕 | S8 |
| LO9 | 說明 head pinning 與 Eq. 8 | 直接用 Eq. 7 微調會怎樣？Eq. 8 怎麼改？ | head pinning：主導 head 的門控幾乎不變（原文 resist change），其餘 head 的門控升高到與之相近，幾乎所有 head 都變高（App. D、Fig. 12）；top-$m$ 為 shared heads 不進損失，只對 routed heads 算 $(N_h-m)[\mathrm{CV}]^2$，$\lambda=10^{-2}$（Eq. 8） | S9 |
| LO10 | 說明 Flash Attention 相容的實作與數值注意事項 | 為何不能直接取注意力權重？作者如何得到 $G$？成本？Vanilla 由 kernel 的 LSE 算 $G$ 有什麼數值問題？ | Flash 等融合 kernel 不輸出權重，eager 模式太慢；用 kernel 回傳的 LSE：Sink 為 $\sigma(\mathrm{LSE}-\mathrm{sink})$ 並乘到標準 Flash 輸出，Vanilla 為 $\sigma(\mathrm{LSE}_{\neg0}-q^\top k_0/\sqrt{d_h})$；額外訓練延遲 < 2%（Sec. 4.2、5.1）。kernel 的 LSE 含首 token，論文未說明 $\mathrm{LSE}_{\neg0}$ 如何取得；由含首 token 的 LSE 相減在 $A_0\to1$ 時相消〔本頁推導〕 | S10 |
| LO11 | 讀出實驗結果 | 2B 從頭訓練三種機制加上輔助損失後的平均分數？微調 LLaMA3.1-8B 的結果？ | Vanilla 45.46→46.83、Sink 46.35→46.50、Gated 46.30→47.70（Table 1）；LLaMA3.1-8B 推理 56.50→58.97、LongBench 40.51→43.98（Table 2） | S11 |
| LO12 | 說明崩塌的機制假說 | Sec. 6.1 的假說鏈為何？為何 Gated Attention 最早崩塌？ | Value Drain → Attention Sink → Implicit Gating → Head Collapse；Gated 的門控一開始就是顯式的，不需先建立 sink，所以崩塌直接發生（Sec. 6.1、Fig. 5） | S12 |

---

## 3. 頁面大綱

沿用 benchmark 頁面的單欄（680–720px）+ 加寬圖版（≤1080px）、章節編號（`01 · …`）、左側 sticky 目錄。視覺記號：一條「head 列」——一排 $N_h$ 個方塊，高度表示 $\mathrm{Imp}$，shared head（S8 定義）以斜線紋路標示；出現在 hero（無紋路，因為此時尚未定義 shared）、C2、C3。章節標題用陳述句。

| # | id | 標題 | 內容（依序；每個符號在使用前定義） | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈注意力匯聚在注意力層裡形成原生 MoE〉 | 副標（只用初學者已知的詞）：「語言模型的注意力常集中在第一個 token。這篇論文說明，這個現象等於替每個注意力 head 的輸出乘上一個 0 到 1 的係數，並提出讓各 head 係數較平均的訓練損失。以下術語從第 01 節開始定義。」三張事實卡（下方）；靜態 head 列（資料取 C2 預設輪廓，但 hero 不提 β 與 C2；圖說只寫「每根長條是一個 head 的平均係數，示意資料」）；連結：arXiv、`../index.html#paper-584`、`asink-lineage-hub.html` | 1 螢幕 |
| S1 | `sink` | 01 · Softmax 的總和限制產生 attention sink | 注意力入門（1.5 步驟 1 第三點：Q/K/V、logit、causal mask、softmax、multi-head、LSE）；Sec. 2.1 公式、$z$、$A$、$O$；attention sink 定義與成因（LO1）；StreamingLLM；用途清單。符號表（1.4，可收合）。延伸閱讀：`asink-softmax-normalization-and-first-token.html`、`asink-sink-applications-kv-cache-and-streaming.html` | 450–550 字 |
| S2 | `fixes` | 02 · 兩種機制消除 attention sink | Eq. 1、Eq. 2（LO2）；Fig. 1 重繪為三欄靜態 SVG（規格見 3.1）。延伸閱讀：`asink-eliminating-sink-softmax-modification.html`、`asink-gated-attention-explicit-gating.html` | 300–400 字 + SVG |
| S3 | `drain` | 03 · 第一個 token 的 value 接近 0 | value drain、Fig. 2 文字描述；先定義 sink 成分與 $A_{t,\text{sink}}=A_{t,0}$（Vanilla），再寫 Eq. 9、τ 與 Table 3（LO3，含 6.26 的下降）。延伸閱讀：`asink-mechanism-studies-value-drain-over-mixing.html` | 300–400 字 + 表 |
| S4a | `gate` | 04 · Sink 的權重是一個隱式門控 | 04a「Sink Attention：恰好成立」：合成 sink 的權重、Eq. 3–4、Eq. 21、Eq. 23 | 250–300 字 |
| S4b | `gate-vanilla` | （同章）04b「Vanilla Attention：$v_0\approx0$ 時近似成立」 | Eq. 14、16；誤差 $A_0v_0$；〔本頁推導〕$\ln\frac{1-G}{G}$；App. B.2 推導於 `<details>`；常見錯誤 1；C1（LO4） | 300–350 字 + C1 |
| S5 | `moe` | 05 · 多個 head 與門控構成原生 MoE | MoE 兩要素、對應、兩個共同設計（LO5）；Sec. 6.2 的一般化 $S^{l,h}$、DeepSeek-V4 例外、Gated Attention 也可能另有隱式門控。延伸閱讀：`asink-native-moe-and-head-collapse.html`、`moe-conditional-computation-lineage.html`（站內既有） | 300–400 字 |
| S6 | `geometry` | 06 · 第一個 token 當 sink 會限制 query 與 key 的分布 | Sec. 3.2、Fig. 3 定性描述（LO6，不引用 PC 百分比）；靜態示意 SVG（規格見 3.1） | 350–450 字 + SVG |
| S7 | `collapse` | 07 · 只有少數 head 有貢獻：head collapse | 先前研究（DuoAttention、>25% head 可歸零）；類比專家崩塌；Eq. 5、Eq. 6 與手算例（LO7）；Fig. 4、App. C 描述；崩塌三後果 | 450–550 字 |
| S8 | `aux` | 08 · 輔助損失懲罰各層重要度的變異係數 | Eq. 7、$\lambda=10^{-4}$；正文兩句縮放不變；C2 之前一段定義 shared heads（每層 Imp 最高的 $m$ 個 head）與 routed heads（其餘），並寫明「為什麼微調時要這樣分在 S9 說明」，Eq. 8 的式子也在此先列出；`<details>` 內 $N\mathrm{CV}^2$ 恆等式與梯度（LO8，答案句 5.333 寫在正文）；C2；常見錯誤 2–4。延伸閱讀：`asink-expert-collapse-and-load-balancing.html` | 350–450 字 + C2 |
| S9 | `pinning` | 09 · 微調時保留 shared heads，只平衡 routed heads | head pinning（App. D、Fig. 12）；Eq. 8、$\lambda=10^{-2}$、$m$ 未給；C3 與其範圍說明（LO9）；常見錯誤 5 | 400–500 字 + C3 |
| S10 | `flash` | 10 · 門控值可由 Flash Attention 的 LSE 算出 | 問題、Vanilla 與 Sink 的做法、PyTorch 呼叫、< 2%（LO10）；〔本頁推導〕FlashOut × G、Vanilla 不乘、相消；背景：分塊 LSE；C4；常見錯誤 6–8 | 450–550 字 + C4 |
| S11 | `results` | 11 · 輔助損失在三種機制與兩種情境都提高平均分數 | 設定表（步驟 11）；Table 1 摘要（步驟 12 表）、Table 2 摘要；Fig. 5–8 文字描述（不重繪，論文未給數據）；Gated 最佳的限定（LO11） | 400–500 字 + 3 表 |
| S12 | `mechanism` | 12 · 有效的門控是 head collapse 的前提 | Sec. 6.1 假說鏈（以四個方框 + 箭頭的 HTML 流程列呈現）、Fig. 9 三個診斷、三種機制的差異（LO12） | 250–350 字 |
| S13 | `history` | 13 · 研究脈絡 | 時間線（6.6，由 notes/timeline.json） + 100 字串接：Transformer → StreamingLLM → 成因研究 → 改 softmax／門控 → head 重要度與 MoE 負載平衡 → 本文 | 時間線 + 100 字 |
| S14 | `next` | 14 · 限制與延伸閱讀 | Sec. 6.2 未決問題（隱藏 sink 集合與顯式門控的交互）；研究筆記清單（6.6 順序）；arXiv | 150–200 字 |

Hero 事實卡（只用已知詞；術語在 S1 起定義）：
1. 「在 Vanilla 注意力的語言模型中（Fig. 2），大部分注意力常放在第一個 token 上，而這個 token 送出的向量長度接近 0，所以這些注意力幾乎不改變輸出。」
2. 「沒放到第一個 token 的比例是一個 0 到 1 的係數，乘在每個注意力 head 的輸出上，作用像開關；多個 head 各配一個係數，結構與專家混合模型（Mixture-of-Experts）相同。」
3. 「多數 head 的平均係數偏低，只有少數 head 偏高。作者加入一個懲罰各 head 平均係數差異的損失，在 0.6B 到 8B 的模型上提高平均分數。」

### 3.1 靜態圖規格

- **S2 Fig. 1 三欄 SVG**（viewBox 0 0 720 300；< 600px 改為三張直排、每張 viewBox 0 0 240 300）：每欄由下往上四個圓角方框（`--surface-2` 底、`--line` 框、13px 字）：「q、k、v 線性映射」→「MatMul q·k」→ 中間方框（Vanilla：`softmax`；Sink：`softmax，分母 + exp(sink)`；Gated：`softmax`）→「MatMul × v」。Gated 欄在最上方多一個方框「× σ(x W_θ)」以 `--accent` 外框標示。欄頂標題（Vanilla／Sink／Gated Attention）與副標（「隱式門控」「隱式門控」「顯式門控」）；欄底小字列代表模型（Qwen2、3、LLaMA2、3／GPT-OSS、DeepSeek-V4／Qwen3-Next、Qwen3.5、3.6）。箭頭用 `--ink-2` 1.5px。
- **S6 幾何示意 SVG**（兩個面板，各 viewBox 0 0 300 300，圓心 (150,150)、半徑 120 的淺色圓）：以 `mulberry32(584)` 產生點。左「Vanilla」：$k_0$ 為星形標記在角度 0°、半徑 100；40 個 query（圓點，`--c-par`）角度均勻於 [−20°, 20°]、半徑 [60,110]；40 個 key（方形，`--c-perp`）角度均勻於 [150°, 210°]、半徑 [60,110]。右「Sink／Gated」：$k_0$ 在角度 0°、半徑 60；40 個 query 與 40 個 key 角度均勻於 [0°, 360°)、半徑 [30,110]。兩面板下方標「示意，非論文數據」。形狀 + 文字圖例區分 query、key，不只靠顏色。

---

## 4. 互動元件規格

共用：`sig(u)` 以穩定式（$u\ge0$：$1/(1+e^{-u})$；否則 $e^u/(1+e^u)$）；`lse(arr)` 先減最大值。所有讀數用 JetBrains Mono。圖上與讀數的下標不用 Unicode 下標字元（JetBrains Mono 可能缺字）：SVG 用 `<tspan baseline-shift="sub" font-size="75%">`，HTML 用 `<sub>`；讀數表的列名（含 TeX）在初始化時以 KaTeX 渲染一次，之後只更新數字欄的 `textContent`。
共用動態規則：每個元件初始化時讀一次 `matchMedia('(prefers-reduced-motion: reduce)')` 並監聽 `change`；reduced-motion 時 C1 的長條高度與箭頭不做 transition，C3 改為分段計算後一次重畫，C4「全部」直接顯示結果、不逐塊動畫，hero 不做進場動畫。

### C1 · 一個 head 的隱式門控（S4b）

- **教的概念**：Eq. 3–4；Sink 的恆等式（Eq. 21、23）與 Vanilla 的近似（Eq. 14、16）；Vanilla 的等式依賴 $v_0\approx0$。
- **固定資料**（示意）：6 個真實 token（$j=0..5$）。內容 logit $z_{1..5}=(1.0,0.5,2.0,-0.5,0.0)+s$；二維 value $v_{1..5}=(1,0),(0,1),(-1,0.5),(0.5,-1),(0.8,0.8)$；$v_0=\rho\cdot(1/\sqrt2,1/\sqrt2)$。
- **輸入**：
  - 分段按鈕「機制」：Vanilla／Sink（預設 Vanilla）。切換時套用各自預設：Vanilla $z_0=4$、$\rho=0$；Sink $z_0=0.5$、sink = 4、$\rho=0$。
  - 滑桿 $z_0\in[-4,12]$，步長 0.1（第一個 token 的 logit）。
  - 滑桿 $s\in[-4,4]$，步長 0.1，預設 0（內容 logit 平移，代表 query 與內容 key 的整體匹配程度）。
  - 滑桿 sink $\in[-4,12]$，步長 0.1（只在 Sink 顯示）。
  - 滑桿 $\rho=\lVert v_0\rVert\in[0,1]$，步長 0.05，預設 0（兩種模式都顯示）。Vanilla 中 $\rho$ 決定近似誤差；Sink 中 $j=0$ 是一般 token，$v_0$ 照 $\rho$ 進入輸出，Eq. 21 對任何 $\rho$ 都恰好成立——讀者可藉此對照兩種機制。
- **計算**（每次輸入即時）：
  - Vanilla：$L=\mathrm{lse}(z_{0..5})$，$A_j=e^{z_j-L}$；$L_{\neg0}=\mathrm{lse}(z_{1..5})$；$G_{\text{sig}}=\sigma(L_{\neg0}-z_0)$；$G_{\text{sub}}=1-A_0$；$\tilde A_j=e^{z_j-L_{\neg0}}$；$O_{\text{direct}}=\sum_{j=0}^5A_jv_j$；$O_{\text{gate}}=G_{\text{sig}}\sum_{j=1}^5\tilde A_jv_j$；誤差 $\lVert O_{\text{direct}}-O_{\text{gate}}\rVert=A_0\rho$。
  - Sink：$L_{\text{tok}}=\mathrm{lse}(z_{0..5})$，$L_{\text{all}}=\mathrm{lse}(z_{0..5},\mathrm{sink})$；$A_j=e^{z_j-L_{\text{all}}}$，$A_{\text{sink}}=e^{\mathrm{sink}-L_{\text{all}}}$；$G=\sigma(L_{\text{tok}}-\mathrm{sink})$；$\tilde A_j=e^{z_j-L_{\text{tok}}}$；$O_{\text{direct}}=\sum_{j=0}^5A_jv_j$，$O_{\text{gate}}=G\sum_{j=0}^5\tilde A_jv_j$（兩者都含 $v_0$），誤差恆為 0（浮點內）。誤差讀數在兩種模式都顯示。
  - Sink 模式另顯示 $A_0$；當 $A_0>0.2$ 時在讀數下方加一行：「首 token 也收到 A₀ 的權重（Sec. 6.2 的 DeepSeek-V4 情形）。若首 token 的 value 小（ρ 接近 0），論文建議 A_sink 應同時計入它；這裡的 G 只計入合成 sink。」（文字中的下標依上方規則排版）
- **數值保護**：所有 softmax 減最大值；門控一律用 $\sigma(\cdot)$ 計算、$G_{\text{sub}}$ 只作對照顯示。讀數 $<10^{-4}$ 時用科學記號（例 `1.50e-6`）。
- **顯示**：
  - 左：長條圖，橫軸 token（「0（首）」「1」…「5」，Sink 模式多一根「sink」，以斜線紋路 + `--c-perp`；內容 token 用 `--c-par` 實心），縱軸 $A\in[0,1]$；每根長條上方標數值（兩位小數）。sink 成分的長條另有括號標「A_sink → G = 1 − A_sink」。
  - 中：小圖「G 對 $z_0$（Vanilla）／對 sink（Sink）」：sigmoid 曲線 + 目前點，橫軸 [−4,12]，標出 $L_{\neg0}$ 或 $L_{\text{tok}}$ 處（$G=0.5$）的垂直虛線。
  - 右：二維向量圖（座標 [−1.2,1.2]²），畫 $O_{\text{direct}}$（`--ink` 實線箭頭）與 $O_{\text{gate}}$（`--accent` 虛線箭頭），兩者重合時圖例顯示「兩種算法相同」；$\rho>0$ 的 Vanilla 顯示差向量 $A_0v_0$（細點線）。
  - 讀數表：$A_{\text{sink}}$、$1-A_{\text{sink}}$、$\sigma(\mathrm{LSE}-\cdot)$、LSE、$O_{\text{direct}}$、$O_{\text{gate}}$、誤差。
- **讀者應觀察到**：1) 兩種門控算法數值相同；2) $z_0$（或 sink）越大，$G$ 越小，head 的輸出向量越短；3) Vanilla 在 $\rho>0$ 時兩種輸出分開，差恰為 $A_0\rho$——所以 value drain 是隱式門控成立的條件；4) Sink 對任何 $\rho$ 都恆等成立（誤差 0）；Vanilla 只有 $\rho=0$ 時成立。
- **無障礙**：長條除顏色外以紋路區分 sink；滑桿 `aria-valuetext`（「z0 = 4.0」）；讀數區 `aria-live="polite"`，在 `change` 或停止輸入 500ms 後更新；圖 `role="img"` + `aria-label` 描述「A_sink = 0.80，G = 0.20」。

### C2 · CV、N·CV² 與兩種輔助損失（S8）

- **教的概念**：Eq. 5–8；CV 縮放不變；$N\mathrm{CV}^2=\sum(x/\mu-1)^2$〔本頁推導〕；shared／routed 的計算差異。
- **輸入**：
  - 分段按鈕 $N_h$：10／13／16（對應 0.6B／1B／2B，Sec. 5.1），預設 16。改變 $N_h$ 時 $m\leftarrow\min(m,N_h-2)$，並更新 $m$ 滑桿的 max。
  - 滑桿「崩塌程度」$\beta\in[0,8]$，步長 0.1，預設 4（β 在圖版第一行定義：「β 是本頁示意輪廓的參數，β = 0 時所有 head 相同，越大越集中在少數 head」）。重要度輪廓（示意）：$\mathrm{Imp}_h=0.05+0.9\,e^{-\beta r_h/(N_h-1)}$，$r_h=(7h+3)\bmod N_h$（$h=0..N_h-1$；7 與 10、13、16 互質，所以 $r$ 是排列，最高的 head 不在最左邊，接近 Fig. 4 的分散樣子）。
  - 滑桿 $m\in[0,N_h-2]$，整數，預設 3（論文未給 $m$，標「示意」）。
  - 勾選框「顯示梯度下降方向」。
- **計算**：
  - 全部 head：$\mu$、母體 std、CV、$N\mathrm{CV}^2=S/\mu^2$；$\mathcal L_{\text{aux}}$（Eq. 7，$\lambda=10^{-4}$，單層）$=\lambda N\mathrm{CV}^2$。
  - top-$m$（依 Imp 由大到小，同值取索引小者）為 shared；routed 的 $\mu_r$、$\mathrm{CV}_r$、$(N-m)\mathrm{CV}_r^2$；$\mathcal L_{\text{aux}}$（Eq. 8，$\lambda=10^{-2}$，單層）。
  - 梯度：$g_h=2(x_h-\mu)/\mu^2-2S/(N\mu^3)$；顯示 $\sum_hx_hg_h$（應為 0，浮點內）。
  - 保護：routed 數 ≤ 1 時 CV 定為 0；$\mu$ 恆 ≥ 0.05，不會除以 0；$\max_h|g_h|<10^{-12}$（例：β = 0）時不畫箭頭，改顯示「所有 head 相同，梯度為 0」。
- **顯示**：head 列長條圖（縱軸 [0,1]）；shared head 用斜線紋路 + 頂端「S」字樣；平均 $\mu$ 與 $\mu_r$ 以兩條水平虛線（不同線型）並直接標字。勾選梯度時，每根長條下方畫箭頭：方向為 $-g_h$，長度 ∝ $|g_h|/\max|g|$。讀數表：μ、std、CV、N·CV²、Σ(x/μ−1)²（與上一列相同，示範恆等式）、Eq. 7 損失、shared 索引、μ_r、CV_r、(N−m)·CV_r²、Eq. 8 損失。另有可展開的「各 head 數值表」（head、Imp、shared/routed、梯度），作為長條圖的無障礙替代。
- **讀者應觀察到**：1) β = 0 時所有 head 0.95，CV = 0；2) β 增大，CV 增大；3) N·CV² 與 Σ(x/μ−1)² 相同；4) 梯度箭頭中，除了最高的幾個 head 向下，其餘都向上；5) 移除 top-$m$ 後 routed 的 CV 較小。
- **無障礙**：長條圖整體是一個 `role="group"`（`aria-label`「各 head 重要度」），只佔一個 Tab 停駐點；以 roving tabindex 讓左右方向鍵在長條間移動，聚焦長條時 live region 讀出「head 7，Imp 0.36，routed」。不依賴 hover 或點按；觸控使用者讀上述數值表。live region 規則同 C1。

### C3 · head pinning 的玩具動態（S9）

- **教的概念**：固定幾個高重要度 head 時，以 Eq. 7 最小化 CV 會讓其餘 head 一起升高（方向與 App. D 中圖相同）；Eq. 8 讓 routed heads 彼此相同、shared heads 不變（方向與 App. D 右圖相同）。
- **簡化**（頁面寫出，標「本頁玩具模型，非論文結果」）：只有一層、$N_h=16$；每個 head 一個門控 logit $\theta_h$，$\mathrm{Imp}_h=\sigma(\theta_h)$（不模擬 token 與注意力）；沒有 $\mathcal L_{\text{base}}$；主導 head「幾乎不變」以凍結 top-$m$ 的 $\theta$ 表示，所以 pinning 的結果直接來自這個設定，不證明論文的因果解釋；$\lambda$ 併入學習率。
- **初始**：C2 的 $\beta=4$ 輪廓，$\theta_h=\ln\frac{x_h}{1-x_h}$。shared／凍結集合 $\mathcal T_m$ 由初始 Imp 決定，訓練中不重選（R5）。
- **輸入**：分段按鈕「損失」：Eq. 7／Eq. 8（預設 Eq. 7）；勾選框「凍結 top-m」預設勾選，只在 Eq. 7 可操作（Eq. 8 中 shared head 不在損失內，梯度本來是 0，勾選框顯示為已勾選且停用，旁註原因）；滑桿 $m\in\{1,\dots,6\}$，預設 3；按鈕「執行／暫停」「重設」；改變任何輸入時自動重設。
- **計算**（解析梯度，純梯度下降）：
  - Eq. 7：$x=\sigma(\theta)$，$g_h$ 同 C2（全部 16 個 head）；Eq. 8：只對 routed 子集算 $g$（子集大小 $n_r=N-m$，公式中的 $N$ 換成 $n_r$），shared 的梯度為 0。
  - 更新：$\theta_h\leftarrow\theta_h-\eta\,g_h\,x_h(1-x_h)$，$\eta=0.05$；被凍結者不更新。
  - 3000 步，每影格 30 步；每 10 步記錄一次 routed 的最小／平均／最大、3 個 shared 值、損失（共 301 筆）。
  - 保護：$x(1-x)$ 在飽和時趨近 0，$\theta$ 不會發散；初始 $x\in[0.066,0.95]$，logit 有限。
- **顯示**：上：head 列（同 C2 樣式），顯示目前 Imp，另以淺灰描邊畫初始高度。下左：Imp 對步數——routed 以最小–最大的淺色帶 + 平均實線（`--c-par`）表示（Eq. 8 時 13 條曲線重合，帶寬為 0），$m$ 條 shared 以 `--c-perp` 虛線各畫一條並直接標值；縱軸 [0,1]。下右：損失對步數，縱軸對數 $[10^{-6},\ \max(L_0,1)]$，損失 $<10^{-6}$ 時畫在下緣。讀數：步數、損失、routed 平均、routed 最小／最大、全部 CV。
- **讀者應觀察到**（皆為本頁玩具模型的結果）：1) Eq. 7 + 凍結：routed heads 一起上升到約 0.78，高於 shared 中的 0.739 與 0.578、低於 0.950；2) Eq. 7 不凍結：所有 head 收斂到同一個中間值 0.442；3) Eq. 8：routed heads 收斂到同一個值約 0.161，shared 維持 0.950／0.739／0.578。
- **附註**（圖下一行，〔本頁推導〕）：Eq. 7 + 凍結（m = 3）時，使 16 個 head 的 CV 最小的 routed 共同值為 0.786（`check.py` 以步長 1e-5 格點搜尋），3000 步時為 0.7805。
- **reduced-motion**：按「執行」以 `setTimeout` 分段（每段 500 步）後一次重畫。live region 只在完成或暫停時宣告「第 N 步，routed 平均 0.78」。

### C4 · 分塊 LSE 與門控（S10）

- **教的概念**：Flash Attention 分塊時維護 $m$（目前最大值）與 $\ell$（以 $m$ 縮放的指數和），最後 $\mathrm{LSE}=m+\ln\ell$（背景）；由 LSE 得到 Sink 的 $G=\sigma(\mathrm{LSE}-\mathrm{sink})$ 與 Vanilla 的 $G$；直接加總 $e^{z}$ 會溢位；由含首 token 的 LSE 相減會失去精度〔本頁推導〕（LO10）。
- **固定資料**（示意）：12 個 logit $z=(5.0,1.2,-0.3,2.1,0.7,-1.5,3.0,0.2,1.8,-0.8,2.6,0.9)\times c$，第一個為首 token；sink $=3.0\times c$。
- **輸入**：分段按鈕「區塊大小 B」：1／2／3／4／6／12（預設 4）；分段按鈕「logit 倍率 c」：1／10／200（預設 1）；分段按鈕「機制」：Sink／Vanilla；按鈕「下一塊」「全部」「重設」。
- **計算**：每塊：$m'=\max(m,\max z_{\text{blk}})$；$\ell\leftarrow\ell e^{m-m'}+\sum_{\text{blk}}e^{z-m'}$（第一塊 $\ell=\sum e^{z-m'}$）；$m\leftarrow m'$；顯示 $m+\ln\ell$。完成後：直接 LSE（一次減最大值）、直接加總 $\ln\sum e^{z}$（$\max z>709.78$ 時顯示「溢位（Infinity）」）；Sink：$G=\sigma(\mathrm{LSE}-\mathrm{sink})$；Vanilla 三欄：
  1. `1 - exp(z0 - lse)`（只用 kernel 的 LSE）；
  2. `-expm1(z0 - lse)`（只用 kernel 的 LSE；等同 $\sigma(\mathrm{LSE}_{\neg0}-z_0)$ 且 $\mathrm{LSE}_{\neg0}=\mathrm{LSE}+\log(-\mathrm{expm1}(z_0-\mathrm{LSE}))$）；
  3. $\sigma(\mathrm{LSE}_{\neg0}-z_0)$，$\mathrm{LSE}_{\neg0}$ 直接對 $z_{1..11}$ 計算，欄名下標註「需另行取得 LSE¬0（論文未說明）」。
- **顯示**：12 個方塊排成一列，已處理的區塊塗色並加勾號字元、目前區塊加粗外框；下方表格每列一塊：$m$、$\ell$、$m+\ln\ell$。結果卡片列出上述各值（科學記號 4 位有效數字）。
- **讀者應觀察到**：1) 不論 B，最終 LSE 相同（c = 1 時 5.3308）；2) c = 200 時直接加總溢位，分塊法仍得 1000；3) Vanilla、c = 200：只用 kernel LSE 的兩欄得 0；另行計算 $\mathrm{LSE}_{\neg0}$ 的第三欄得 $1.915\times10^{-174}$。
- **無障礙**：處理狀態以填色 + 勾號表示；結果 live region 只在按鈕動作後更新。

---

## 5. 驗收數值表（`verify/check.py`；頁面算法相同）

容許誤差（頁面與 check.py 同算法，差異只來自浮點加總順序；以下規則依量級選擇）：

| 量 | 規則 |
| --- | --- |
| 量級 ≥ 0.01 的讀數（A、G、LSE、O、μ、std、CV、N·CV²、C3 平均） | 絕對 ±1e-4（表中四位小數；C3 另註 ±0.0005） |
| 量級 < 0.01 的讀數（C2 的 Eq. 7／Eq. 8 損失、C1 的 1.504e-6、C4 的 2.0992e-9） | 相對誤差 ≤ 1e-3 |
| 極小值（C4 的 1.9152e-174） | $|\log_{10}(\text{頁面})-\log_{10}(\text{表})|\le10^{-3}$ |
| 應為 0 的量（C1 Sink 誤差、Σx·g） | 絕對值 < 1e-12；C2 的 CV = 0（β = 0）須精確為 0 |
| 應為 0 的失敗路徑（C4 c = 200 的 `1 - exp`、`-expm1`） | 精確為 0，對照欄須 > 0 |

**C1**（$s=0$、$\rho=0$ 除非另註）

| 設定 | $A_{\text{sink}}$ | $G$（兩種算法） | LSE | $O_{\text{direct}}$ | 誤差 |
| --- | --- | --- | --- | --- | --- |
| Vanilla $z_0=4$ | 0.8034 | 0.1966 | $L_{\neg0}$ 2.5925 | (−0.0525, 0.0815) | 0 |
| Vanilla $z_0=4$、$\rho=1$ | 0.8034 | 0.1966 | 2.5925 | (0.5156, 0.6495)；$O_{\text{gate}}$ (−0.0525, 0.0815) | 0.8034 |
| Vanilla $z_0=8$ | 0.9955 | 0.0045 | 2.5925 | (−0.0012, 0.0018) | 0 |
| Vanilla $z_0=-2$ | 0.0100 | 0.9900 | 2.5925 | (−0.2643, 0.4102) | 0 |
| Vanilla $z_0=4$、$s=2$ | 0.3561 | 0.6439 | 4.5925 | (−0.1719, 0.2668) | 0 |
| Sink $z_0=0.5$、sink 4 | 0.7843 | 0.2157 | $L_{\text{tok}}$ 2.7088 | (−0.0513, 0.0795) | 0 |
| Sink $z_0=0.5$、sink 4、$\rho=1$ | 0.7843 | 0.2157 | 2.7088 | (−0.0345, 0.0963)；$O_{\text{gate}}$ 相同 | 0 |
| Sink $z_0=0.5$、sink −4 | 0.0012 | 0.9988 | 2.7088 | (−0.2374, 0.3684) | 0 |
| Sink $z_0=4$、sink 4（首 token 也收到 $A_0=0.4455$；顯示 Sec. 6.2 附註行。$1-A_{\text{sink}}-A_0=0.1090$，若首 token value 小，G 0.5545 高估了實際的輸出比例） | 0.4455 | 0.5545 | 4.2189 | (−0.0291, 0.0452) | 0 |

Vanilla $z_0=12$、$s=-4$：$G=1.504\times10^{-6}$（兩種算法相同，科學記號顯示）。

**C2**（$\lambda$ 欄為單層損失）

| $N_h$ | β | m | μ | std | CV | N·CV² | shared 索引 | μ_r | CV_r | (N−m)CV_r² | Eq. 7（λ=1e-4） | Eq. 8（λ=1e-2） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 4 | 3 | 0.2869 | 0.2575 | 0.8974 | 12.8858 | 2, 9, 11 | 0.1787 | 0.6579 | 5.6261 | 0.001289 | 0.056261 |
| 16 | 0 | 3 | 0.9500 | 0 | 0 | 0 | 0, 1, 2 | 0.9500 | 0 | 0 | 0 | 0 |
| 16 | 8 | 3 | 0.1861 | 0.2422 | 1.3020 | 27.1220 | 2, 9, 11 | 0.0838 | 0.6238 | 5.0585 | 0.002712 | 0.050585 |
| 10 | 4 | 2 | 0.2979 | 0.2758 | 0.9260 | 8.5749 | 1, 4 | 0.1752 | 0.6596 | 3.4807 | 0.000857 | 0.034807 |
| 13 | 4 | 3 | 0.2910 | 0.2645 | 0.9087 | 10.7350 | 7, 9, 11 | 0.1626 | 0.6091 | 3.7104 | 0.001074 | 0.037104 |

$N_h=16$、β = 4 的 Imp（三位）：0.454, 0.113, 0.739, 0.157, 0.066, 0.232, 0.078, 0.360, 0.098, 0.578, 0.132, 0.950, 0.189, 0.072, 0.287, 0.087；梯度（三位）：−1.546, −9.850, 5.376, −8.780, −10.969, −6.955, −10.686, −3.845, −10.206, 1.456, −9.386, 10.493, −7.988, −10.846, −5.606, −10.478（只有 head 2、9、11 為正，即梯度下降時向下）；$\sum x_hg_h$ 量級 1e-15。解析梯度與有限差分最大差 3e-9。

**C3**（$\eta=0.05$；容許 ±0.0005）

| 情形 | 步 100 routed 平均 | 步 300 | 步 1000 | 步 3000 | 步 3000 損失 | 最終 shared |
| --- | --- | --- | --- | --- | --- | --- |
| Eq. 7 + 凍結，m = 3 | 0.5033 | 0.6460 | 0.7424 | 0.7805 | 0.1181 | 0.739、0.578、0.950（不變） |
| Eq. 7 不凍結，m = 3 | 0.4287 | 0.4415 | 0.4416 | 0.4416 | 0.0000 | 全部 0.442 |
| Eq. 8，m = 3 | 0.1614 | 0.1614 | 0.1614 | 0.1614 | 0.0000 | 不變 |
| Eq. 8，m = 1 | 0.2337 | 0.2337 | 0.2337 | 0.2337 | 0.0000 | 0.950 |
| Eq. 7 + 凍結，m = 1 | 0.4835 | 0.6233 | 0.7520 | 0.8366 | 0.0169 | 0.950 |

Eq. 7 + 凍結（m = 3）routed 共同值的極限（格點 1e-5）：0.78647，此時 N·CV² = 0.1180。

**C4**

| c | B | 各塊後 $m+\ln\ell$ | 最終 LSE | 直接加總 |
| --- | --- | --- | --- | --- |
| 1 | 4 | 5.0792, 5.2159, 5.3308 | 5.330816 | 5.330816 |
| 1 | 1 | 5.0, 5.0221, 5.0270, 5.0792, 5.0916, 5.0930, 5.2093, 5.2159, 5.2483, 5.2506, 5.3188, 5.3308 | 5.330816 | 同 |
| 200 | 任意 | 每塊 1000 | 1000 | 溢位 |

門控：c = 1：Sink $G=0.911397$；Vanilla 三種算法皆 $2.8166\times10^{-1}$。c = 10：Vanilla $2.0992\times10^{-9}$（三者在 4 位有效數字一致）。c = 200：Vanilla `1 - exp` 與 `-expm1` 為 0（只用 kernel LSE 推回的 $\mathrm{LSE}_{\neg0}$ 為 −∞，c = 10 時為 30.018276、直接計算 30.018277），$\sigma(\mathrm{LSE}_{\neg0}-z_0)=1.9152\times10^{-174}$；Sink（sink = 600）$G=1.000000$。

---

## 6. 實作計畫

### 6.1 檔案

單一檔案 `584/build/index.html`，複製 `tutorials/prediction-targets-k-diff.html` 的外殼後替換內容：保留 CSS tokens（含 `--c-par`、`--c-perp`，本頁語意改為「內容 token／routed head」與「sink／shared head」）、`#pm-bar`、appbar 與主題切換、`initToc`、`.plate`／`.callout`／`.check`（`<details>`）／`.tscroll` 表格、`.seg` 分段按鈕、`.rng` 滑桿、`announce()` live region、`el()`／`tx()`／`lp()`／`frame()`／`arrow()` SVG 輔助、`renderMath()` KaTeX 設定、`?test` → `selfTest()`、`renderTimeline()`。移除 k 軸（`kaxis`、`layoutKAxis`）、C1–C4 舊元件與其數學（`weights`、`modeParts`、`kStarCurve` 等）。

### 6.2 新增內容

- 共用數學：`sig(u)`、`lse(arr)`、`cvStats(x)` → `{mu, sd, cv, ncv2, S}`（長度 ≤ 1 時全 0）、`gradNcv2(x)`、`topM(x, m)`、`profile(N, beta)`。
- 元件：`initC1()/renderC1()`（state `{mode, z0, s, sink, rho}`）；`initC2()/renderC2()`（`{N, beta, m, showGrad}`）；`initC3()/stepC3()/runC3()`（`{mode, pin, m, theta[], step, hist[], running}`，`requestAnimationFrame` 每影格 30 步）；`initC4()/renderC4()`（`{B, c, mode, done}`）。
- 新視覺：head 列元件 `headRow(svg, imp[], shared:Set, opts)`，C2、C3、hero 共用；斜線紋路以 SVG `<pattern id="hatch">` 定義一次。
- S2 Fig. 1 三欄 SVG、S6 幾何示意 SVG、S12 假說鏈流程列（HTML flex，四個方框 + 箭頭字元，< 480px 改直排）。
- `selfTest()` 檢查（`console.assert`，最後輸出 `selfTest: N passed, M failed`）：
  - C1 Vanilla $z_0=4$：$G=0.19663\pm10^{-5}$，兩種門控差 < 1e-12；$\rho=1$ 誤差 $=A_0$（±1e-9）；Sink 預設：$G=0.21570\pm10^{-5}$、輸出誤差 < 1e-12；Sink $\rho=1$：輸出誤差 < 1e-12。
  - `cvStats` 於 (0.9,0.1,0.1,0.1)：CV 1.154700、ncv2 5.333333（±1e-6）；C2 預設 N·CV² 12.8858（±1e-4）、(N−m)CV_r² 5.6261；`sum(x*grad)` < 1e-12。
  - C3 Eq. 7 + 凍結 3000 步 routed 平均 0.7805 ± 0.0005；Eq. 8 m = 3 為 0.1614 ± 0.0005。
  - C4 c = 1 LSE 5.330816 ± 1e-6（B = 1 與 B = 12 皆同）；c = 200 時 $\sigma$ 路徑 > 0 且 `expm1` 路徑 = 0。
  - 頁面無水平捲動（`scrollWidth <= innerWidth`）。
- 連結：hero「在關聯圖中查看」→ `../index.html#paper-584`；arXiv `https://arxiv.org/abs/2602.01203`；筆記 `<slug>.html`。
- 頁面標題 `<title>`：「Attention Sink 與原生 MoE」。

### 6.3 文案規則（沿用 benchmark，補充用詞）

繁體中文、台灣用語、字面學術語氣；不用比喻、反問句、宣傳形容詞（hero 卡 2 的「作用像開關」是唯一允許的類比，因為論文本身用 gating 一詞；其他地方寫「係數」「門控值」）。不寫擬人化動詞：用「收到權重」「分到」「幾乎不變」，不用「抗拒」「搶」「吸收」「傾倒」「吸引」；引用論文原詞時加「原文 …」。專有名詞第一次出現附英文：注意力匯聚（attention sink）、value 歸零現象（value drain）、head 崩塌（head collapse）、head 釘住效應（head pinning effect）、共享／路由 head（shared／routed heads）、變異係數（coefficient of variation, CV）。區分「論文結果」「作者認為／作者解釋」「本頁推導」「示意」。

### 6.4 引用與標籤

內文每個論文數字附出處；元件圖版底部一行「讀數由本頁依 Eq. ○ 即時計算」，用到〔本頁推導〕或示意資料時加對應標籤（`--ink-3` 邊框 pill）。

### 6.5 無障礙、響應式、色彩

- 沿用 benchmark 規則（滑桿 label 與 `aria-valuetext`、live region 500ms debounce、圖 `role="img"`、focus 2px `--accent`、觸控區 ≥ 44px、390px 無水平捲動、四欄以上表格放 `.tscroll`）。
- 長公式（Eq. 2、7、8、$O_t$ 的求和、Eq. 3）：`.katex-display{overflow-x:auto;overflow-y:hidden;max-width:100%}`，並給該容器 `tabindex="0"` 與 `aria-label="公式，可橫向捲動"`，讓鍵盤使用者能捲動。
- C2 長條圖：一個 Tab 停駐點 + roving tabindex（見 C2）；16 根長條在 390px 寬時每根寬度 ≥ 16px（只是視覺尺寸，不作觸控目標），head 索引標籤只標偶數；觸控與螢幕閱讀器使用可展開的數值表。
- 色彩 token（沿用 benchmark 頁面實際值）：`--c-par` light #3F6E8C／dark #8DB4D3；`--c-perp` light #6B7A3A／dark #A9BC7C。斜線紋路 `<pattern id="hatch">` 的線條用 `--ink-2`（light #57534B／dark #B5B0A5，在 `--surface` 上對比皆 ≥ 3:1），底色用 `--c-perp` 淡化 30%。建置時以對比檢查確認 ≥ 3:1，不足則加粗線條。

### 6.6 研究脈絡與筆記

- 時間線：把 `notes/timeline.json`（15 筆）原樣嵌入 `<script type="application/json" id="timeline-data">`。每筆欄位 `year`、`title`、`authors_short`、`arxiv`、`role`、`source`；依 `year` 排序（同年保持檔案順序）。本文 `2602.01203` 已在檔案中，不需補卡。
- `TL_GROUP`（以 arXiv ID 為 key；不在表中的歸「其他」）與卡片的筆記連結：

  | 類別 | arXiv ID | 卡片連到的筆記 |
  | --- | --- | --- |
  | 基礎：softmax 注意力 | 1706.03762 | `asink-softmax-normalization-and-first-token.html` |
  | sink 現象與應用 | 2309.17453、2410.10819 | `asink-sink-applications-kv-cache-and-streaming.html` |
  | sink 成因 | 2410.10781、2410.13835、2504.02732、2510.06477 | `asink-mechanism-studies-value-drain-over-mixing.html` |
  | 改 softmax | 2504.20966、2508.10925 | `asink-eliminating-sink-softmax-modification.html` |
  | 門控 | 2306.12929、2505.06708 | `asink-gated-attention-explicit-gating.html` |
  | MoE 負載平衡 | 1701.06538、2101.03961、2401.06066 | `asink-expert-collapse-and-load-balancing.html` |
  | 本文 | 2602.01203 | `asink-native-moe-and-head-collapse.html` |

  卡片呈現、篩選 chip、捲動按鈕、缺欄位處理沿用 benchmark 的 `renderTimeline()`。2508.10925（gpt-oss model card）在論文參考文獻中無 arXiv ID，`source` 欄註明已另外核對；照檔案顯示。
- S13 串接文字（100 字）依上表類別順序寫，不新增論文數字。
- S14 延伸閱讀依 hub 的閱讀順序：`asink-lineage-hub.html`（總覽）、`asink-softmax-normalization-and-first-token.html`、`asink-sink-applications-kv-cache-and-streaming.html`、`asink-mechanism-studies-value-drain-over-mixing.html`、`asink-eliminating-sink-softmax-modification.html`、`asink-gated-attention-explicit-gating.html`、`asink-expert-collapse-and-load-balancing.html`、`asink-native-moe-and-head-collapse.html`；以及站內既有 `moe-conditional-computation-lineage.html`、`llmmem-moe-as-conditional-memory.html`。每項附 frontmatter 的 `title`。
- 各章延伸閱讀已寫在第 3 節大綱（S1、S2、S3、S5、S8）。
- JSON 解析失敗時整節顯示「時間線資料尚未提供」並隱藏控制鈕。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | Eq. 7 的括號位置 | paper.txt 排版模糊，已以 PDF 第 5–6 頁影像確認：$\lambda\sum_lN_h[\mathrm{CV}]^2$、$\lambda\sum_l(N_h-m)[\mathrm{CV}]^2$。 | 已解決。 |
| R2 | 示意資料 | C1 logit／value、C2 輪廓、C4 logit 都是自訂。 | 每個元件標「示意」，不與論文數據並列。 |
| R3 | App. A 文字與數字不一致 | 「全部歸零只造成可忽略的下降」，但 Qwen3-8B 平均降 6.26。 | 頁面照錄數字並明寫差異；LO3 用 τ = 0.75 的結果。 |
| R4 | CV 的 std 版本 | 論文未說明母體或樣本 std。 | 頁面用母體版並註明；常見錯誤第 5 條。 |
| R5 | $m$ 與 $\mathcal T_m^l$ | 論文沒給 $m$，也沒說 top-$m$ 是否每 batch 重選。 | C2、C3 的 $m$ 標「示意」；C3 固定初始集合並寫明。 |
| R6 | Vanilla 的 $\mathrm{LSE}_{\neg0}$ 取得方式 | 論文只說「由回傳的 LSE 與首 token 的 query-key 乘積」計算；kernel 的 LSE 含首 token，由它推回 $\mathrm{LSE}_{\neg0}$ 與 `-expm1` 路徑精度相同。 | C4 第三欄標「需另行取得 LSE¬0（論文未說明）」；LO10 答案寫明來源未說明；精度問題標〔本頁推導〕。 |
| R7 | 「Gated 在各規模最好」 | 只在 base+aux 欄成立。 | 頁面照數字寫限定條件。 |
| R8 | Fig. 5 縱軸刻度 | Vanilla 為 1e-3／1e-4，Sink、Gated 為 1e-1；論文未討論。 | 不跨機制比較 CV 絕對值；常見錯誤第 4 條。 |
| R9 | C3 玩具模型的範圍 | 凍結代替「幾乎不變」（原文 resist change）、pinning 因此是設定的直接結果、無 base loss、無 token；結果只示範方向，不重現 Fig. 12 數值。 | 標「本頁玩具模型」；觀察敘述用「一起上升／彼此相同」，不給論文對照數字。 |
| R10 | Fig. 3 PCA 比例 | 只出現在圖中標注，正文未引用。 | 頁面不引用、不詮釋（round 1 第 12 項）。 |
| R11 | 本頁推導的數量 | 共 6 項：$\ln\frac{1-G}{G}$、$N\mathrm{CV}^2$ 恆等式、梯度與縮放不變、$\mathrm{Imp}\in[0,1]$ 與 $t=0$、FlashOut × G 與 Vanilla 不乘、`expm1` 相消；另 C3 的 0.786 極限。原先的「pinning 的梯度解釋」已刪除（round 1 第 4 項）。 | 一律加標籤與一行依據。 |
| R12 | 學習率 0.02 與 AdamW | Sec. 5.1 寫 lr 0.02 且「其他依 AdamW 預設」，對 AdamW 偏大但原文如此。 | 照錄，不評論。 |
| R13 | 筆記更新 | 6.6 已依 round 1 時的 notes 填入。 | 建置時重讀 `notes/`，有增減就同步 `TL_GROUP` 與延伸閱讀並回報。 |
| R14 | C1 移除 Gated 模式 | round 1 第 2 項：$\rho>0$ 時 $1-A_0$ 不是門控；第 6 項：C1 太密。 | C1 只留 Vanilla／Sink；Gated 的顯式門控與可能並存的隱式門控在 S2、S5 以文字說明。 |
| R15 | Vanilla 在 $t=0$ 的 $G$ | $G=0$ 且 $\mathrm{LSE}_{\neg0}$ 為空；論文未說明 Eq. 5 是否排除 $t=0$。 | 步驟 7 寫明；元件不涉及。 |

---

## 8. 修訂紀錄（round 1）

全部 23 項都已處理，沒有駁回。

| 項 | 處理 |
| --- | --- |
| 1 | 步驟 10、LO10（併入數值穩定，仍 12 個 LO）、C4 第三欄標「需另行取得 LSE¬0（論文未說明）」，並說明由 kernel LSE 推回 LSE¬0 與 `-expm1` 精度相同；check.py 加印推回路徑（c = 200 為 −∞）。 |
| 2 | C1 移除 Gated 模式（R14）。 |
| 3 | 步驟 7 改為 Imp∈[0,1]，寫出 t = 0 的情形（R15）。 |
| 4 | 步驟 8、9、C3 刪除因果宣稱與「最便宜」；改為「固定 head 時最小化 CV 會讓其餘升高」；0.442、0.161、0.78 標「本頁玩具模型，非論文結果」；0.78 寫成「高於 0.739、0.578，低於 0.950」。 |
| 5 | Hero 副標與事實卡改用已知詞並註明術語從 S1 起定義；S1 加注意力入門與 LSE 定義；S3 在 Eq. 9 前定義 sink 成分。 |
| 6 | S4 拆成 S4a（Sink 恰好成立）／S4b（Vanilla 近似，含 C1）；S8 的恆等式與梯度移到 `<details>`；常見錯誤分散到 S4b、S8、S9、S10。 |
| 7 | 刪除擬人化用詞；「開關」只留在 hero 卡 2；引用 App. D 時寫「原文 resist change」；6.3 加用詞規則。 |
| 8 | C2：max|g| < 1e-12 不畫箭頭；改 N_h 時 m 夾到 N−2。 |
| 9 | 6.5：`.katex-display` 可橫捲、`tabindex="0"`。 |
| 10 | C2 改為單一 group + roving tabindex，加數值表，刪除「點按 = 聚焦」。 |
| 11 | App. C 模型數改為 6 + 10 = 16。 |
| 12 | Fig. 3 百分比不引用、不詮釋；註明 (c)(f) 為示意圖（R10）。 |
| 13 | 改為「與幾何描述相容的必要條件」。 |
| 14 | C1 Sink 模式在 A₀ > 0.2 時顯示 Sec. 6.2 附註；§5 該列標 1 − A_sink − A₀ = 0.1090。 |
| 15 | Fig. 5 描述改為只有 Gated 持平，Vanilla、Sink 逐步上升（步驟 7、12）。 |
| 16 | 常見錯誤 2：CV² 差 N/(N−1)，std／CV 差 √(N/(N−1))。 |
| 17 | 增幅比較標「本頁對照」並列出 +1.70／+4.71 對 +0.51／+1.29。 |
| 18 | 事實卡 1 加「Vanilla 注意力的語言模型（Fig. 2）」。 |
| 19 | check.py 新增格點搜尋（步長 1e-5）：0.78647，N·CV² = 0.1180。 |
| 20 | 第 4 節開頭：下標用 `<tspan baseline-shift>`／`<sub>`；KaTeX 列名只渲染一次。 |
| 21 | C3：routed 以最小–最大帶 + 平均線，shared 各一條；每 10 步記錄；損失對數軸 [1e-6, max(L₀,1)]。 |
| 22 | 第 4 節開頭加 matchMedia 規則；6.5 列出 `--c-par`、`--c-perp` light/dark 值與紋路對比。 |
| 23 | 3.1 新增 S2 Fig. 1 與 S6 示意圖的完整規格（方框、角度範圍、點數、seed 584）。 |
| 筆記 | 6.6 依 notes 填入時間線分類、卡片連結、S14 順序；大綱各章加延伸閱讀。 |

## 9. 修訂紀錄（readiness 後）

- §5 容許誤差改為依量級：絕對 1e-4／相對 1e-3／log10 1e-3／精確 0。
- β 只在 C2 圖版定義，hero 不提 β、head 列無紋路；shared／routed 與 Eq. 8 式子在 S8 的 C2 之前先定義，S9 說明原因。
- C1 Sink 模式：$v_0$ 照 $\rho$ 進入輸出（兩種模式都顯示 ρ 與誤差），Eq. 21 對任何 ρ 恰好成立；§5 新增 Sink ρ = 1 列；check.py 同步。
- 事實卡 3 改為「多數 head 偏低、少數偏高」；S6 加一句 PCA 說明。

## 10. 建置紀錄（build round）

- 產出：`build/index.html`（約 113 KB），由 `build/src/{body.html,app.js,extra.css,build.py}` 組合；`build.py` 從 benchmark 頁複製 CSS（去掉 k 軸）與 JS 輔助函式，並嵌入 `notes/timeline.json`。改內容時改 `src/` 再跑 `python3 build/src/build.py`。
- 驗證：`?test` selfTest 32/32（C1 九列、C2 五列、C3 五種情形、C4 c = 1／10／200、390px 無水平捲動，iframe 內亦通過）；KaTeX 0 錯誤；無 console 錯誤；`impeccable detect` 只有 numbered-section-labels advisory；`extract_page_text.py` 後 12 個 LO 的關鍵詞與數字都在正文中，各 LO 答案集中在同一章的一段。
- 與設計的差異：
  - selfTest 在 `load` 後執行（先確保 KaTeX 已渲染，否則未渲染的 TeX 字串會讓 390px 的 scrollWidth 誤判）；計算仍為同步。
  - `cvStats`／`gradNcv2` 在所有值相同時直接回傳 0，使 β = 0 的 CV 精確為 0（§5 規則）。
  - C1 向量圖自動縮放（±max(0.15, 1.25·|O|)），因預設輸出長度約 0.1。
  - Eq. 2 改為兩行（gathered），避免 390px 橫捲；其他長公式可橫捲並自動加 `tabindex="0"`。
  - 符號表與推導用 `<details class="more">`；`class="check"` 只用於檢查題。
  - C2 長條圖以 `role="group"` + 方向鍵切換，選取框只在聚焦時顯示。
  - 圖例斜線樣本改用非重複 linear-gradient（impeccable 檢出 repeating-gradient）。
- 截圖：`build/caps/`（1440 各元件、390 iframe、深色模式、`--force-dark-mode`）。
