# 設計文件：系列教學〈Transformer 的 token 縮減與解析度還原：從 ViT token merging 到 RTI 與 HiT〉

- 類型：系列（lineage）教學，不屬於單篇論文。頁面 slug `token-compression-lineage`；建置檔 `tutorial_src/series-token-compression/build/index.html`，上線位置 `tutorials/token-compression-lineage.html`，站內相對連結一律以上線位置為準。
- 起點論文（資料庫內）：No.217 RTI（arXiv 2608.29281，〈Elastic Token Compression for Pixel-Space Diffusion Transformers〉）、No.585 HiT（arXiv 2611.22962，〈Hierarchical Diffusion Transformer for End-to-End Image Generation〉）。兩篇各有單篇教學 `rti-elastic-token-compression.html`、`hit-hierarchical-diffusion-transformer.html`（設計文件 `tutorial_src/217/design.md`、`585/design.md`，平行撰寫中）。**分工**：本頁負責系譜、設計軸與跨方法比較；RTI、HiT 只講到「它在每個設計軸上選了什麼」的深度，機制細節（READ/WRITE 的 RoPE 平均、Hilbert 順序性質、軟權重、壓縮損失、群組引導）一律連到單篇教學，不重教。
- 主要來源：`research/lineage.md`（研究備忘，26 張論文卡）；全文 `research/papers/<arXiv>.txt`（13 篇）；RTI、HiT 全文為 `tutorial_src/217/src/paper.txt`、`585/src/paper.txt`。本文件每個方程式與數字都回到原文核對過，標明 Sec./Eq./Tab.；只讀摘要的論文其數字標「僅摘要」。
- 標記慣例（頁面沿用）：〔本頁推導〕＝本文件自行推導、論文沒有寫出；「推定」＝為了讓元件可算而選定、論文沒有給出的值；「僅摘要」＝數字只出自摘要。
- 驗算：`verify/common.py`（場景、PRNG、Hilbert、五種分組、指標，與頁面相同演算法）、`c1_tome.py`、`c2_cost.py`、`c3_group.py`；輸出 `verify/out_*.txt`。第 5 節全部出自這些輸出（`python3 -I verify/c3_group.py` 等，可在任何目錄執行，約 1 秒）。
- 研究筆記：`notes/`：hub `tokc-lineage-hub` 與 7 篇主題筆記（`tokc-vit-pruning-and-merging`、`tokc-training-free-diffusion-merging`、`tokc-learned-routing-dynamic-dit`、`tokc-hourglass-and-latent-interfaces`、`tokc-dynamic-chunking-bytes-to-pixels`、`tokc-content-adaptive-tokenization`、`tokc-position-for-merged-tokens`）、`timeline.json`（23 筆）。單篇論文的 hub：`rti-lineage-hub`（217/notes）、`hit-lineage-hub`（585/notes）。
- 修訂：round 1 審查（`reviews/round1.md`）的處理見第 8 節。

---

## 1. 介紹（步驟 6-1）

### 1.1 一段話總結

Transformer 每個 block 的成本由 token 數決定；擴散 Transformer（DiT）在每一層、每一個取樣步都對每個 patch 付相同成本。從 2020 年起，研究者沿八個設計軸（縮減什麼、如何分組、在哪裡縮減、如何還原、如何訓練、預算、位置、擴散特有問題）反覆取捨：語言模型的沙漏結構先證明「中段短、兩端全長」可行；ViT 分類的剪枝與合併（ToMe）帶來內容相依的縮減，但分類不需要還原；把合併搬到擴散（ToMeSD、ToDo）時必須處理稠密輸出、CFG 與時間步；學得的路由（MoD、DiffCR）讓比例可學，卻是「刪除」而非「總結」；pixel-space 擴散（HDiT、JiT）用固定網格把 token 數壓到可負擔；動態切塊（H-Net）讓切分可學但只控制平均比例。2026 年的 RTI 與 HiT 是兩個端點：RTI 以無參數的 Hilbert 最大落差切割得到**空間連續、每張精確 R 個**的區域，改造凍結骨幹以加速；HiT 以學得的親和度與平均連結法得到**可非局部、批次平均預算**的群組，從頭訓練以提升品質。兩者各自的消融看起來對「分組是否必須空間連續」指向不同方向，但 HiT 的對照組（固定的水平再垂直配對）本身是連續分組，HiT 並未直接測試「非局部」本身；兩個比較的條件不同，這是本頁結尾的開放問題（1.7）。

### 1.2 核心問題：token 數與成本（備忘 1.1；〔本頁推導〕）

- 一個標準 block 在序列長度 $L$、寬度 $d$、MLP 擴張比 $m$ 時的前向 FLOPs（乘加計 2）：QKV 與輸出投影 $8Ld^2$；MLP $4mLd^2$；注意力分數 $QK^\top$ 與 $AV$ 共 $4L^2d$。
  $$C(L)=(8+4m)\,L\,d^2+4L^2d\;\overset{m=4}{=}\;24Ld^2+4L^2d$$
- 二次項占總成本比例 $=\dfrac{4L^2d}{24Ld^2+4L^2d}=\dfrac{L}{6d+L}$。$d=768$：$L=256$ 為 5.26%，$L=1024$ 為 18.18%，$L=4096$ 為 47.06%，$L=16384$ 為 78.05%（`out_c2_cost.txt`）。
- 結論一：DiT 常見長度（256–1024）下成本約與 token 數成正比，token 減半約使 block 成本減半；注意力的 $O(L^2)$ 只在數千 token 以上才成為主項。〔本頁推導〕這與 ToMeSD、ToDo 在大圖加速最多一致（ToDo 摘要：一般尺寸至多 2×，2048×2048 至多 4.5× 以上）；ToMeSD 作者對小圖加速較少的解釋則是「擴散模型不是瓶頸」（Sec. 5），兩者不衝突。
- 結論二：牆鐘時間可能比 FLOPs 下降得更快。RTI App. A（Fig. 10）：RTX 4090、batch 4 下，量測時間在每個預算都低於解析 FLOPs 曲線（作者歸因於這些長度的注意力受記憶體頻寬限制）。
- pixel-space 的 token 數：JiT（No.349）以 patch 16 讓 256² 影像為 16×16 = 256 個 token；RTI 所改造的 MiniT2I-B/16 在 512²、patch 16 為 32×32 = 1024 個 token（RTI Sec. 3.1）。RTI Sec. 1：latent 模型在 VAE 中「一次付清」低階空間冗餘，pixel-space 模型「在每個 block 都付一次」。HiT Sec. 1 的動機不同：一 patch 一 token 貫穿骨幹，不鼓勵模型在較粗層級總結內容，所以分組是品質與訓練效率的手段，不只是加速。

### 1.3 設計軸（備忘 1.3；頁面 S3 的定義表，後續各節以代號引用）

| 軸 | 問題 | 選項 |
| --- | --- | --- |
| A 縮減 | 被縮減的 token 怎麼處理 | 丟棄（剪枝、跳過）；合併（平均）；匯聚到學得 latent；切塊選代表；只縮 K/V |
| B 分組 | 哪些 token 歸成一組 | 固定（步幅、網格）；啟發式相似度；重要性分數；學得；無參數但內容自適應 |
| C 位置層 | 在網路哪裡縮減 | 每 block 合併後還原；逐層累進；只縮中段（兩端全長）；多層巢狀 |
| D 還原 | 怎麼回到每個 patch 一個輸出 | 不需要（分類）；複製回來源（unmerge／broadcast）；殘差寫回；skip connection；注意力上採樣 |
| E 訓練 | 是否需要訓練 | 免訓練；在預訓練模型上微調或適配；從頭訓練 |
| F 預算 | token 數由誰決定 | 固定比例；逐層／逐時間步不同；批次平均、單張可變；彈性（一個 checkpoint 支援多種預算） |
| G 位置 | 合併 token 的位置怎麼表示 | 不處理；ToMe 的對數大小項（S5 定義）；原始 2D 座標；平均 RoPE 相量；不加或學得 |
| H 擴散 | 擴散特有的限制 | 輸出稠密；冗餘隨時間步變化；CFG 兩分支要一致或可刻意不對稱；每步重新分組 |

### 1.4 系譜：各期的問題、解法與新增能力

每期依「面臨的問題 → 解法 → 前期做不到、本期補上的 → 遺留限制」整理。

**第 0 期（2020–2021）序列縮短的兩個原型**
- 問題：長序列成本高；語言建模需要逐 token 輸出，縮短後要還原。
- Funnel-Transformer（arXiv 2006.03236，僅摘要）：逐步壓縮隱藏序列，省下的 FLOPs 用於加深加寬，需要逐 token 輸出時以解碼器還原。
- Hourglass（arXiv 2110.13711，Sec. 2）：shorten（係數 $k$）→ 中段 Transformer → upsample＋殘差 → 全長 Transformer；為避免自回歸洩漏，縮短前右移 $k-1$ 個 token。Sec. 2.3 比較上採樣：重複（無法區分組內位置）、線性上採樣、注意力上採樣 $x=U(x,x')+\mathrm{Attention}(Q=U(x,x'),K=V=x')$，$U(x,x')=x+\mathrm{LinearUpsampling}(x')$。ImageNet32 生成 3.741 bits/dim（Sec. 3.4）。
- Perceiver IO（arXiv 2107.14795，僅摘要）：固定數量的 latent 以 cross-attention 讀入任意長度輸入，以輸出查詢讀出，成本對輸入輸出長度線性。
- 新增：證明「中段在短序列上運算、兩端保持全長」可行。限制：分組完全固定（步幅），與內容無關；latent 瓶頸不做空間分組（RTI Sec. 4.2 指出，其 Latent Array 基線在凍結骨幹上同時失去分組、連續與位置）。

**第 1 期（2021–2022）ViT 分類中的 token 縮減**
- 問題：ViT 對每個 patch 等量計算，背景 patch 浪費。
- DynamicViT（arXiv 2106.02034，僅摘要）：預測模組估計重要性、以注意力遮罩可微地剪枝；剪 66% token，FLOPs 降 31%–37%，吞吐量升 40% 以上，精度降 0.5% 以內。EViT（2202.07800）以 CLS 注意力保留重要 token、融合其餘。
- ToMe（arXiv 2210.09461，Sec. 3）**二部軟配對（bipartite soft matching）**，每個 block：(1) 把 token 分為大小相近的 A、B 兩集合（App. D 程式碼：A 取偶數索引、B 取奇數索引）；(2) 每個 A token 連一條邊到 B 中最相似者（key 的餘弦相似度）；(3) 保留最相似的 $r$ 條邊；(4) 合併相連的 token（以 token 大小 $s$ 加權平均，Tab. 1d）；(5) 串回。每層減少 $r$ 個 token，不需迭代。**proportional attention**（Eq. 1）：$A=\mathrm{softmax}\big(QK^\top/\sqrt d+\log s\big)$，$s$ 為 token 代表的 patch 數。〔本頁推導〕$e^{\,q\cdot k/\sqrt d+\log s}=s\,e^{\,q\cdot k/\sqrt d}$，所以加 $\log s$ 恰好等於把該 key（與 value）複製 $s$ 份。數字（摘要）：現成 ViT-L@512、ViT-H@518 吞吐量 2×、精度降 0.2–0.3%。
- 新增：內容相依的縮減；ToMe 做到「現成模型、免訓練」。限制：分類只用 CLS，被合併的 token **不需要還原**；分組只看外觀，不顧空間位置。

**第 2 期（2023–2024）免訓練的擴散 token 合併**
- 問題：擴散要對每個 token 預測（稠密輸出）；CFG 跑條件與無條件兩個分支；多個時間步重複計算。
- ToMeSD（arXiv 2303.17604）：在 self-attn、cross-attn、MLP 之前合併、之後 **unmerge**：若 $x^*_{1,2}=(x_1+x_2)/2$，則 $x'_1=x'_2=x^*_{1,2}$（Sec. 3.1，Eq. 1）。以比例 $r\%$ 取代固定個數。分割（Sec. 4.1）：ToMe 的交替分割使 dst 形成規則欄位；改為**每個 2×2 區域隨機挑一個 dst**，並且**在整個 batch 固定隨機性**，讓 CFG 的條件與無條件樣本以相同方式配置 dst。Tab. 3：只對 self-attn 合併最划算。Tab. 4（SD v1.5、512²）：基線 FID 33.12、3.09 s/im；$r=50\%$ FID 33.02、1.65 s/im；$r=60\%$ FID 33.37、1.52 s/im。
- ToDo（arXiv 2402.13573，Sec. 2.2）：假設空間相鄰 token 相似，以最近鄰下採樣 $D(\cdot)$ 取代相似度配對（$O(n)$）；只對 key、value 下採樣、保留全部 query：$\mathrm{Attention}=\mathrm{softmax}\big(Q\,D(K)^\top/\sqrt{d_k}\big)D(V)$，輸出仍為全長，不需 unmerge。Sec. 3 的合併比例 0.75、0.89 是**被移除**的比例，等於 2×、3× 下採樣。
- AT-EDM（2405.05252，僅摘要）：注意力圖上以 G-WPR 排序剪枝，並依去噪步調整剪枝預算（DSAP）。Importance-Based Token Merging（2411.16720，僅摘要）：以 CFG 等現成訊號作為重要性分數保護高資訊 token。
- 新增：第一次把 token 縮減用在稠密生成而不需訓練；開始顧及 CFG 與時間步。限制：合併群組由外觀相似度決定，可能把畫面兩端的 patch 併在一起（RTI Sec. 1 稱為失去連續性與位置）；unmerge 是純複製，同組 token 完全相同；ToDo 只省注意力項。

**第 3 期（2022–2025）學得的路由與架構層級瓶頸**
- 問題：免訓練方法的比例要手調、與層和時間步無關；架構本身仍對每個 token 等量計算。
- 在既有模型上學路由：Mixture-of-Depths（2404.02258，摘要）以 top-$k$ router 讓部分 token 沿殘差流跳過 block；DyDiT（2410.03456，摘要）時間步相依寬度＋空間動態 token，額外微調不到 3% 迭代，DiT-XL FLOPs 降 51%、加速 1.73×、FID 2.07；DiffCR（2412.16822）每層 router＋可微的逐層、逐時間步壓縮比，PixArt-Σ 上品質相當並省 20.68% 延遲與 8.33% 記憶體（Sec. 4）；FlexiDiT（2502.20126，僅摘要）把預訓練 DiT 轉為多預算模型；Sparse Guidance（2601.01608，摘要）以 token 層級稀疏度作為引導訊號，ImageNet-256 FID 1.58、FLOPs 少 25%。
- 架構瓶頸：RIN（2212.11972，僅摘要）以 latent 承擔全域注意力並以 cross-attention 讀寫；U-ViT（2209.12152，僅摘要）以長 skip 連接淺層與深層；HDiT（2401.11605）見第 4 期。
- 新增：比例可學、可隨層與時間步變化。限制：跳過類方法是「刪除」而非「總結」——RTI 在凍結 MiniT2I 上重實作的 Token Skip GenEval 82.3、只有 1.19× 加速（Tab. 1），比 RTI 低 2.4 點（Sec. 4.2；Token Skip 的 576 個 token 不是同一種預算，Tab. 1 註記，作者說 2.4 低估了差距，因為 Token Skip 只有 1.19× 而 RTI 2.15×）。

**第 4 期（2023–2025）pixel-space 擴散的回歸**
- 問題：latent 擴散依賴另外訓練的 VAE；直接在像素上建模 token 太多。
- simple diffusion（2301.11093，僅摘要）：下採樣是避免高解析度特徵圖的有效策略。HDiT（2401.11605）：Hourglass 結構，編碼端以 Pixel-UnShuffle 把 2×2 token 合併為一，解碼端 Pixel-Shuffle 分裂，skip 以可學係數線性插值（lerp）合併；高解析度層用鄰域注意力、最內層全域注意力，整體 $O(n)$；FFHQ-1024² FID 5.23（Tab. 3）。JiT（2511.13720，No.349，僅摘要）：x-prediction 搭配大 patch（16、32）的單純 Transformer。
- 新增：去除 tokenizer 的端對端生成。遺留：大 patch 與 2×2 合併都是**固定、與內容無關**的壓縮；512² T2I 仍有 1024 個 token，每層每步成本相同——RTI 與 HiT 的共同起點。

**第 5 期（2023–2025）內容自適應的切分與動態切塊**
- 問題：固定網格對平坦區與細節區分配相同 token。
- Quadformer（2304.00287，摘要）：quadtree＋顯著性評分器產生大小不一的 patch。BLT（2412.09871，僅摘要）：依下一個 byte 的熵切出動態長度 patch。
- H-Net（arXiv 2507.07955，Sec. 2.2）：routing（Eq. 4）$q_t=W_q\hat x_t$、$k_t=W_k\hat x_t$，$p_t=\tfrac12\big(1-\frac{q_t^\top k_{t-1}}{\lVert q_t\rVert\lVert k_{t-1}\rVert}\big)\in[0,1]$，$b_t=\mathbb 1\{p_t\ge0.5\}$，$p_1=1$；downsampler 只保留邊界位置的向量；dechunking（Sec. 2.2.2）先以平滑模組（EMA）處理，再把每個壓縮向量重複到下一個邊界，並乘上經 straight-through 的信心值。ratio loss（Sec. 2.2.3，Eq. 10）：$\mathcal L_{\text{ratio}}=\frac{N_{\text H}}{N_{\text H}-1}\big((N_{\text H}-1)FG+(1-F)(1-G)\big)$，$N_{\text H}$ 為目標壓縮比（論文記為 $N$），$F$ 為實際選取比例（不可微）、$G$ 為平均邊界機率（可微），$F=G=1/N_{\text H}$ 時為最小值 1。
- 新增：token 數隨內容變化；切塊可與主任務聯合訓練。限制：H-Net 為一維序列；ratio loss 只控制**平均**比例；tokenizer 類方法改變輸入表示，不能用在已釋出的 checkpoint（RTI Sec. 2）。

**第 6 期（2026）擴散 Transformer 中的彈性與自適應 token**
- ELIT（arXiv 2603.12245）：短空間頭 → latent 核心 → 空間尾；Read $l_{CA}=l+\mathrm{CA}(Q=l;K,V=s)$，Write 對稱；為降成本把空間 token 分為 $G$ 個不重疊**固定網格**群組做 grouped cross-attention；訓練時隨機丟棄尾端 latent，使 latent 依重要性排序，推論時可任意選數量（彈性）。摘要：ImageNet-1K 512px FID 平均改善 35.3%、FDD 39.6%。
- DC-DiT（arXiv 2603.06351）：把 H-Net 式動態切塊帶到 2D latent DiT，router 以局部可預測性給出邊界機率，tail dropping 實現彈性推論。摘要：推論 FLOPs 至多降 36.8%、FID 至多改善 37.8%。
- RTI（No.217）與 HiT（No.585）見 1.5。
- 本期的兩組不同選擇：「分組空間連續且帶位置」（RTI）對「分組由模型學出、允許非局部」（HiT）；「改造既有模型以加速」對「從頭訓練以提升品質」。哪一種分組較好尚無同條件的比較（1.7）。

### 1.5 兩個端點的設計選擇（概要；細節在單篇教學）

**RTI（arXiv 2608.29281）**
- 四項需求（Sec. 1，由凍結 MiniT2I-B/16 的探測 Fig. 2 導出）：分組要**連續**、大小**隨內容**、帶**位置**、**總結而非刪除**。
- 只壓縮中段（Sec. 3.2）：B/16 共 17 個 block，核心為 [3, 13]，前後各 3 個全解析度 block；文字流始終全長。
- 分組（Sec. 3.3，Eq. 1–2）：沿 Hilbert 曲線排列 patch，$d_j=\lVert h_{\pi(j+1)}-h_{\pi(j)}\rVert_2$（頁面記為 $u_j$，避免與寬度 $d$ 混淆），在最大的 $R-1$ 個 $d_j$ 處切開，得到 $R$ 段連續區間；無參數、$O(N)$、每步重切。Hilbert 曲線上相鄰序號必為格點鄰居，所以每一段都是連通區域。
- READ 為區域內 softmax 加權池化＋大小嵌入，位置取區域內 RoPE 相量的平均（Eq. 3、Sec. 3.4）；WRITE（Eq. 4）$h_j^{\text{out}}=h_j^{\text{in}}+g\big([h_j^{\text{in}}\Vert z'_{c(j)}-z_{c(j)}]\big)$（$h^{\text{in}}$ 為核心前、$h^{\text{out}}$ 為核心後的特徵；論文以層索引 $s_0$、$s_1+1$ 表示，頁面改寫以免與 ToMe 的 $s$ 混淆）：每個 patch 用自己的前置特徵與區域變化量產生自己的更新。
- 訓練：骨幹凍結，只訓練 LoRA 與介面；每步自 {64, 128, 256, 512} 抽 $R$，一個 checkpoint 對應任意預算。
- 數字（Tab. 1，512²，GenEval）：B/16 稠密 87.2；RTI $R=512$ 87.8（1.84×）、$R=256$ 84.7（2.15×）；同預算 Feat Sim（ToMe 式相似度分組）84.7／80.5；Latent Array 45.7；Token Skip 82.3（1.19×）。L/16 $R=256$ 86.3（2.59×）。Sec. 4.2：相對同預算 RTI，Token Skip（刪除而非總結）低 2.4、Feat Sim（打散群組）低 4.2、Latent Array（完全不分組，同時失去連續與位置）低 39.0 GenEval 點；加速約在 2.5× 飽和。Tab. 4（$R=256$）：Feat Sim 群組散布 5.16 格、RTI 0.49 格。Sec. 4.3：均勻切割在 $R=128$ 代價 2.7；以平均池化＋廣播取代整個 READ/WRITE（池化與還原同時改變）代價 7.2。

**HiT（arXiv 2611.22962）**
- 巢狀骨幹（Sec. 3.1，Eq. 1）$N\to K_1\to\cdots\to K_J\to\cdots\to K_1\to N$；預設兩個 2× 階段（ImageNet 256、p16：256 → 128 → 64），18 個 block 配置 3/3/6/3/3。
- 分組（Sec. 3.2，Eq. 2、Eq. 8）：$S_{ij}=\tfrac12(q_i^\top k_j+q_j^\top k_i)$，$q,k$ 為學得投影；自單一 token 起反覆合併平均親和度 $s(U,V)=\frac1{|U||V|}\sum_{i\in U}\sum_{j\in V}S_{ij}$ 最高的兩群（average linkage），保留分數 $\ge\tau$ 的合併；群內 token 可空間相距很遠。池化與解組用對**所有** token 的軟權重（Eq. 3、9），再加投影 skip。
- 預算（Sec. 3.3，Eq. 4）：壓縮損失只要求整個批次的保留比例等於 $\rho=1/r$，每張影像、每個時間步的群組數可以不同（Sec. 3.3「allows group counts to vary with image content and diffusion timestep」）。門檻 $\tau$ 在訓練前校準後凍結。
- 數字：ImageNet-256 HiT-XL/16 FID 1.48（600 epochs，Tab. 1）；FID ≈ 2.16 時訓練計算少 2.4×（Fig. 1d）；T2I 512² 五項平均 65.93 對 MiniT2I-B/16 64.64，名目 GFLOPs 139.4 對 201.6（乘加計 1，以目標群組數計；Tab. 2、Sec. 4.1）。文字–影像共同分組只用在 T2I（App. A.5）。消融（HiT-B/16，80 epochs）：學得分組 6.61 對固定空間配對 7.11（Tab. 7，Sec. 4.5；固定對照組在第一階段水平配對、第二階段垂直配對）；只在硬群內池化 10.16–10.32、硬查表讀回 7.97、預設 6.61（Tab. 3）；2/2/10/2/2 → 3/3/6/3/3 使 FID 8.48 → 6.61（Tab. 5）；學得 RoPE＋per-group MLP 6.61 → 6.25（Tab. 9）。
- 兩者都把全解析度 block 放在兩端：RTI Fig. 2c 顯示首末 block 對每個 patch 敏感；HiT Tab. 5 把 block 從核心移到外層（2/2/10/2/2 → 3/3/6/3/3）使 FID 8.48 → 6.61，但 GFLOPs 也由 18.3 升到 22.2，計算量與位置同時改變。

### 1.6 RTI 與 HiT 的逐軸差異（備忘 5.2；頁面 S13 的表）

| 軸 | RTI | HiT |
| --- | --- | --- |
| 目的 | 推論加速（凍結骨幹，2×–2.6× 牆鐘） | 品質與訓練效率（從頭訓練） |
| B 分組 | 無參數，Hilbert 最大特徵落差，保證空間連續 | 學得親和度＋平均連結＋門檻，允許非局部 |
| F 預算 | 每張精確 $R$，訓練隨機抽 $R$，推論可設未見過的 $R$ | 批次平均 $\rho$，單張、單步群組數由內容決定 |
| A 池化 | 只在區域內做單查詢 softmax | 每群對所有 token 軟加權（限制在硬群內 FID 6.61 → 10.16–10.32） |
| D 還原 | 寫回區域變化量 $\Delta z=z'-z$，以每個 patch 的前置特徵個別化（$g$） | 以同一組軟權重混回 $A\,M(Z)$，再加投影 skip |
| G 位置 | 平均 RoPE 相量（換成無位置的 latent array 低 39.0，但該基線同時失去分組與連續） | 預設 compact 層不加 RoPE（學得 RoPE 只 6.61 → 6.25） |
| C 層級 | 一層（1024 → R → 1024） | 兩層（256 → 128 → 64 → 128 → 256） |
| E 訓練 | 凍結骨幹，只訓練 rank-32 LoRA 與 READ/WRITE（B/16 可訓練 16.3M，5.9%） | 從頭訓練（壓縮損失 λ = 0.01 加在擴散損失上） |
| H 擴散 | 每個取樣步依當前特徵重切；文字流不縮 | 每步重新分組；T2I 時文字與影像共同分組；group guidance 以不同群組數的兩次預測做引導（App. A.6） |

位置差異的一個可能解釋（備忘 5.2 第 5 點的推論，未經實驗驗證，頁面標「作者未驗證的解釋」）：凍結骨幹無法學新的位置機制，必須提供它讀得懂的位置；從頭訓練的模型可以從 skip 與全解析度層取得位置。

### 1.7 開放問題（備忘 5.3–5.4）

1. **分組必須空間連續嗎？** RTI：免學習的相似度分組（Feat Sim）散布 5.16 格、$R=256$ 時 GenEval 比 RTI 低 4.2 點。HiT：學得（可非局部）分組優於固定空間配對（6.61 對 7.11）。兩個比較條件不同：RTI 的對照是**免學習**的相似度分組、骨幹**凍結**且依賴 RoPE；HiT 的對照是**形狀固定**（水平再垂直配對）而非內容自適應的連續分組，且 HiT 的軟池化讓每個 token 都能貢獻給每個群。尚無研究在同一設定下比較「內容自適應但連續」與「學得且非局部」。
2. 精確預算與內容自適應預算的結合（RTI 的 $R$ 由使用者決定；H-Net、DC-DiT、HiT 只控制平均）。
3. 文字流壓縮：RTI 的加速上限部分來自文字流全長。
4. 合併 token 的位置表示仍無定論（HiT 各變體差距小）。
5. 與步數縮減的關係：RTI App. B.3（Fig. 7，MiniT2I-L/16、RTI $R=256$）：每張影像低於約 3 秒時 token 壓縮優於減少步數（1.8 s：少步稠密 GenEval 65.9、RTI 82.7），高於約 5 秒時相反（11.2 s：稠密 87.6、RTI 86.3）。
6. HiT 的 average linkage 在 $N$ 增大時的實際成本未報告；GFLOPs 是以目標群組數計的名目值，實際成本隨每張影像與時間步的群組數改變（HiT Sec. 4.1）。

---

## 2. 學習目標（驗收測試）

每個參考答案都必須寫在指定章節的正文（不能只在元件、表格或答案框）。每章結尾以 `<details>` 呈現檢查題（題目與答案照搬）。章節編號見第 3 節。LO2、LO3、LO7、LO8、LO11 沒有對應元件，由正文與表格教；S14 對照表、S15 時間線是整理用的章節，刻意不設 LO（審查 31）。

| # | 學習目標 | 檢查題 | 參考答案 | 章節 |
| --- | --- | --- | --- | --- |
| LO1 | 說明 token 數與成本的關係 | 一個 block 的 FLOPs 怎麼寫？$d=768$、$L=1024$ 時注意力二次項占多少？token 減半時成本約如何變化？ | $C(L)=24Ld^2+4L^2d$（$m=4$）；二次項比例 $L/(6d+L)=18.2\%$；在 256–1024 個 token 下成本約與 token 數成正比，減半約減半，二次項到數千 token 才成為主項 | S2 |
| LO2 | 說出設計軸 | 每個 token 縮減方法都要在哪些軸上做選擇？ | 縮減什麼、如何分組、在哪裡縮減、如何還原、如何訓練、預算、位置表示、擴散特有問題（A–H） | S3 |
| LO3 | 說明第 0 期的沙漏結構與限制 | Hourglass 的結構是什麼？它的分組有什麼限制？ | shorten → 中段在短序列上運算 → upsample＋殘差 → 全長層，兩端全長；分組為固定步幅，與內容無關 | S4 |
| LO4 | 描述 ToMe 與 proportional attention | ToMe 每層如何合併？為什麼要加 $\log s$？為什麼分類不需要還原？ | 交替分成 A、B，每個 A 連到最相似的 B（key 餘弦），保留最相似的 $r$ 條邊，依大小加權平均合併；加 $\log s$ 等於把該 key 複製 $s$ 份；分類只用 CLS token，被合併的 token 不必回到原位置 | S5 |
| LO5 | 說明把合併搬到擴散時新增的處理 | ToMeSD 如何還原？為什麼改用隨機 2×2 dst，且要固定隨機性？ToDo 為何不需要 unmerge？ | unmerge 把合併結果複製回所有來源位置；交替分割使 dst 形成規則欄位，改為每個 2×2 隨機一個 dst，並在整個 batch 固定隨機性讓 CFG 兩分支一致；ToDo 只對 K、V 下採樣、保留全部 query，輸出仍是全長；ToMeSD $r=50\%$ FID 33.02（基線 33.12）、1.65 對 3.09 s/im（Tab. 4） | S6 |
| LO6 | 用成本模型比較「在哪裡縮減」 | RTI B/16 的 3＋11＋3 個 block 只縮中段時，加速上限是多少？RTI 實測多少？ToDo 式在 1024 token 時為何效益小？ | 不計文字時上限 $17/6=2.83\times$（兩端 6 個 block 不縮）；RTI B/16 在 $R=256$ 實測 2.15×、加速約在 2.5× 飽和；ToDo 只縮短注意力分數的 K/V 長度，1024 token 時二次項只占 18%；在本頁「K、V 投影仍全長」的假設下 $\kappa=0.25$ 只有 1.16× | S7 |
| LO7 | 區分「刪除」與「總結」 | 學得路由（MoD、DiffCR）屬於哪一種縮減？RTI 重實作的 Token Skip 結果如何？ | 跳過／刪除：未選中的 token 沿殘差流繞過 block，不被總結；Token Skip GenEval 82.3、只有 1.19× 加速，比 RTI（2.15×）低 2.4 點，作者說這低估了差距（Tab. 1、Sec. 4.2） | S8 |
| LO8 | 說明固定壓縮與動態切塊 | JiT 與 HDiT 用什麼方式控制 token 數？H-Net 如何決定邊界、如何控制壓縮比例？ | JiT 用大 patch（16、32），HDiT 用固定 2×2 Pixel-UnShuffle 合併的沙漏結構，都與內容無關；H-Net 以相鄰向量餘弦 $p_t=\tfrac12(1-\cos(q_t,k_{t-1}))$，$p_t\ge0.5$ 為邊界，ratio loss 只控制平均比例 | S9 |
| LO9 | 比較分組規則 | Hilbert 最大落差切割為什麼保證連通？門檻法與 RTI 的預算有何不同？相似度合併在 RTI 中的代價？ | Hilbert 上相鄰序號必為格點鄰居，所以每一段連續區間都連通；RTI 每張精確 $R$ 個，門檻法（HiT）的群組數由內容決定、只有批次平均受壓縮損失約束；Feat Sim 散布 5.16 格對 RTI 0.49 格，GenEval 低 4.2 點 | S11 |
| LO10 | 說明還原方式的差異 | 廣播（unmerge）的最小誤差是什麼？殘差寫回多了什麼？哪個消融只改還原？ | 廣播只能讓同組共用一個更新，最小誤差是目標在組內的變異；殘差寫回保留每個 patch 的核心前特徵並據此個別化更新；HiT 只改讀回方式的消融：硬查表 FID 7.97 對軟混合 6.61（RTI 的 −7.2 同時改了池化與還原） | S12 |
| LO11 | 比較 RTI 與 HiT | 兩者在目的、分組、預算、位置上各選了什麼？ | RTI：凍結骨幹＋LoRA 加速；無參數 Hilbert 連續區域；每張精確 $R$、彈性；平均 RoPE 相量。HiT：從頭訓練求品質；學得親和度＋平均連結、可非局部；批次平均、單張可變；compact 層預設不加 RoPE | S13 |
| LO12 | 指出開放問題 | 為什麼 RTI 與 HiT 關於「分組要不要空間連續」的結論不能直接比較？ | RTI 的對照是凍結骨幹上的免學習相似度分組；HiT 的對照是形狀固定的空間配對（本身連續），且軟池化讓每個 token 貢獻每個群；兩者都沒有在同一設定下比較「內容自適應且連續」與「學得且非局部」 | S16 |

---

## 3. 頁面大綱

單欄閱讀（680–720px），元件圖版加寬到 1080px，沿用參考頁的 hero、章節目錄、plate、callout、`<details>` 檢查題。章節標題用陳述句；章節編號 JetBrains Mono（`04 · 第 0 期`）。全頁視覺記號：**期別色帶**——一條水平「第 0–6 期」刻度條，在 hero、每個期別章節開頭（標出目前期別）、時間線上重複出現（期別以文字＋位置區分，不只靠顏色）。

| # | id | 標題 | 內容 | 長度 |
| --- | --- | --- | --- | --- |
| S0 | `top` | Hero：〈Transformer 的 token 縮減與解析度還原〉 | 副標：「從 ViT 的 token 合併到像素擴散的 RTI 與 HiT，七期研究在八個設計軸上的選擇。」三張事實卡（下）。期別色帶。連結：關聯圖 `../index.html#paper-217`、`#paper-585`、`#paper-349`；兩篇教學；總覽筆記 `tokc-lineage-hub.html` | 1 螢幕 |
| S1 | `basics` | 01 · 先備：擴散 Transformer 與評估指標 | 只定義後文用到的詞，每個 1–3 句：token、patch、attention 回顧；**DiT**（以 Transformer 為去噪網路的擴散模型）；前向過程 $x_t=t\,x+(1-t)\,\varepsilon$（本頁採 RTI／JiT 方向：$t=1$ 乾淨、$t=0$ 純噪聲，$\varepsilon$ 為標準高斯噪聲）；**x-prediction**（網路直接預測乾淨影像 $x$）；取樣步；**CFG**（每步跑有條件與無條件兩個分支再外插）；**latent／pixel-space**（先經 VAE 壓縮或直接在像素上）；**FID**（越低越好）、**GenEval**（文字到影像的物件、數量、顏色、位置正確率，越高越好）；**牆鐘時間**（實際秒數）對 FLOPs；**LoRA**（凍結原權重、只訓練低秩增量）；**RoPE**（以旋轉相量編碼位置）。延伸：`pixel-space-diffusion.html`、`x-prediction-manifold-view.html`、`generative-eval-metrics.html` | 350–450 字 |
| S2 | `cost` | 02 · token 數決定 Transformer 的成本 | 定義 $L,d,m$、$C(L)$、二次項比例（LO1）；兩個結論（結論一標〔本頁推導〕）；pixel-space 的 token 數（JiT 256、MiniT2I 1024）；RTI 與 HiT 的不同動機（1.2）。延伸：`aniso-pixel-patchified-setting.html` | 400–500 字 |
| S3 | `axes` | 03 · 每個方法都在八個設計軸上做選擇 | 只放 1.3 的軸定義表與一段正文（LO2）；說明後文以 A–H 代號引用。各軸裡的術語（dst、unmerge、殘差寫回…）在首次使用的章節才定義 | 250–350 字＋表 |
| S4 | `hourglass` | 04 · 第 0 期：沙漏結構讓中段在短序列上運算 | Funnel、Hourglass（含上採樣三種）為主；Perceiver IO 一句；新增與限制（LO3）。延伸：`tokc-hourglass-and-latent-interfaces.html` | 350–450 字 |
| S5 | `vit` | 05 · 第 1 期：ViT 分類以剪枝與合併減少 token | DynamicViT 一句；ToMe 五步、定義 A、B、$r$、$s$；Eq. 1 與〔本頁推導〕複製等價（LO4）；「分類不需還原」。預告 C1 在 S6 末。延伸：`tokc-vit-pruning-and-merging.html`、`tokc-position-for-merged-tokens.html` | 400–500 字 |
| S6 | `diffmerge` | 06 · 第 2 期：擴散的稠密輸出需要把合併還原 | 先定義 src（被併入者）、dst（接收者）、unmerge；ToMeSD unmerge（Eq. 1）、交替分割的問題、隨機 2×2 dst、固定 batch 隨機性與 CFG、Tab. 3–4 數字；ToDo 公式與比例定義（LO5）；AT-EDM、IBTM 各一句；**C1**。延伸：`tokc-training-free-diffusion-merging.html` | 450–550 字＋C1 |
| S7 | `where` | 07 · 縮減的位置決定加速上限 | 先在正文介紹兩種配置：RTI B/16 有 17 個 block，前 3、後 3 保持全解析度、中間 11 個縮短（3＋11＋3）；HiT 有 18 個 block，分成 3/3/6/3/3 五段，外兩段全解析度、中兩段縮一次、核心縮兩次（細節在 S10、S13）。每種策略與「同 block 數的稠密模型」比較。Amdahl 定律一句定義（只加速一部分時，總加速受不加速部分限制），上限 $17/6$（LO6）；RTI 實測 2.15× 與 2.5× 飽和、牆鐘快於 FLOPs（App. A）；**C2** | 400–500 字＋C2 |
| S8 | `routing` | 08 · 第 3 期：學得的路由讓比例可學，但刪除而非總結 | 代表方法只教兩個：MoD（top-$k$ router 跳過）與 DiffCR（逐層、逐時間步可微比例，數字 20.68%／8.33%）；Token Skip 82.3／1.19×／−2.4 與作者的低估說明（LO7）；DyDiT、FlexiDiT、Sparse Guidance、RIN、U-ViT 只列一行並連到對照表與筆記。延伸：`tokc-learned-routing-dynamic-dit.html`、`depth-redundancy-adjacent-methods.html`、`moe-conditional-computation-lineage.html` | 350–450 字 |
| S9 | `chunk` | 09 · 第 4–5 期：像素擴散用固定網格，動態切塊讓切分可學 | 代表方法：HDiT（Pixel-UnShuffle 2×2、lerp skip、$O(n)$、FID 5.23；註明 RTI、HiT 都未引用、依主題補入）、JiT（大 patch）；H-Net Eq. 4、dechunking 一句、Eq. 10（定義 $p_t,b_t,F,G$）（LO8）；simple diffusion、Quadformer、BLT 各一句。延伸：`tokc-dynamic-chunking-bytes-to-pixels.html`、`tokc-content-adaptive-tokenization.html`；資料庫相鄰論文 `#paper-232` | 450–550 字 |
| S10 | `rules` | 10 · 第 6 期：三種分組規則——固定網格、Hilbert 切割、平均連結 | 每條規則一段：(a) 固定網格（HDiT、ELIT 的群組）：本頁以 Hilbert 序列等長切段實作，$R=16$、64 為正方形塊，$R=32$、128 為 2:1 長方形；(b) **Hilbert 曲線**（一條走遍 $2^k\times2^k$ 格子、相鄰序號必為格點鄰居的路徑，配小圖），$u_j$ 定義（論文記為 $d_j$），RTI 在最大的 $R-1$ 個 $u_j$ 處切，連續段必連通（LO9 前半）；(c) **平均連結**（average linkage：從每個 token 一群開始，反覆合併平均親和度最高的兩群，記錄每次合併的分數，得到一棵合併樹 dendrogram），HiT 以門檻 $\tau$ 截斷，同群可不相鄰（**非局部**）。ELIT、DC-DiT 各一句。延伸：兩篇教學、`rti-lineage-hub.html`、`hit-lineage-hub.html` | 450–550 字 |
| S11 | `budget` | 11 · 預算型態與兩個量測：EV 與連通性 | 定義 EV（Eq. 5）、spread、連通組比例；兩種預算：每張精確 $R$（RTI、固定網格、ToMe 式）對門檻決定、只有批次平均受約束（HiT 壓縮損失 Eq. 4；門檻在訓練前以 1,024 個雜訊樣本校準後凍結，Sec. 4.1）；Feat Sim 5.16 對 0.49、−4.2（LO9 後半）；時間依賴以文字說明（RTI Fig. 2b、DiffCR）；**C3** | 400–500 字＋C3 |
| S12 | `restore` | 12 · 還原方式決定同一組內能否各自更新 | 先定義「核心前特徵」（進入縮短段之前每個 patch 的特徵 $h_j$）、廣播、殘差寫回；軸 D 的五種選項回顧；廣播的最小誤差＝組內變異〔本頁推導〕；RTI Eq. 4 與 HiT 軟混合；HiT 7.97／6.61（只改讀回）與 RTI 7.2（同時改池化與還原）（LO10）；**C4**（明寫純玩具） | 400–500 字＋C4 |
| S13 | `rtihit` | 13 · RTI 與 HiT 在八個設計軸上的選擇對照 | 1.6 的逐軸表（LO11 答案寫成兩段正文）；兩端全解析度與其計算量混淆；位置差異的未驗證解釋；兩張「前往單篇教學」卡片。< 768px 表格改為每軸一張卡片（同 S14） | 400–500 字＋表 |
| S14 | `table` | 14 · 十六種方法的設計軸對照 | 全方法對照表（3.1）；軸篩選 chip | 表＋100 字 |
| S15 | `history` | 15 · 研究時間線 | 互動時間線（6.4） | 時間線＋100 字 |
| S16 | `open` | 16 · 開放問題與延伸閱讀 | 1.7 的六點，第 1 點以 callout 呈現並寫出 LO12 答案；延伸閱讀清單（6.4）；資料庫相鄰論文 `#paper-232`、`#paper-50` | 350–450 字＋清單 |

Hero 事實卡（不使用未定義術語）：
1. 「Transformer 每層的計算量大約與 token 數成正比；擴散模型每一步、每一層都對每個 patch 付相同的計算。」
2. 「七期研究的差別在於：哪些 token 合成一組、在網路哪裡合、合完之後怎麼回到每個 patch。」
3. 「RTI 把相鄰且相似的 patch 合成連續區域，用來加速現成模型；HiT 讓模型自己學分組、允許相隔很遠的 patch 同組，從頭訓練以提升品質。」

### 3.0 符號（首次使用處定義；S3 末尾附可收合符號表，含「易混淆」欄）

| 符號 | 意義 | 首次 | 易混淆 |
| --- | --- | --- | --- |
| $x,\ \varepsilon,\ x_t,\ t$ | 乾淨影像、標準高斯噪聲、取樣狀態 $t\,x+(1-t)\varepsilon$、擴散時間（$t=1$ 乾淨） | S1 | 與文字 token 數 $n_{\text{txt}}$ 無關 |
| $L,\ d,\ m,\ C(L)$ | 序列長度、模型寬度、MLP 擴張比、block FLOPs | S2 | $d$ 只指寬度；相鄰特徵距離另記 $u_j$ |
| A、B、$r$、$s$ | ToMe 的兩個集合、每層合併數、token 代表的 patch 數 | S5 | $r$（合併數）≠ $R$（區域數）；$s$（大小）≠ $S_{ij}$（親和度） |
| src、dst | 被併入的 token、接收的 token | S6 | — |
| $N,\ n_{\text{txt}},\ \kappa$ | 影像 token 數、文字 token 數（不縮）、保留比例 | S7 | — |
| $p_t,\ b_t,\ F,\ G$ | H-Net 邊界機率、邊界指示、實際與平均邊界比例 | S9 | H-Net 的 $N$ 是目標壓縮比，全頁寫成 $N_{\text{H}}$（Eq. 10 亦同） |
| $R,\ \pi,\ u_j,\ S_{ij},\ \tau,\ \rho$ | 區域數、Hilbert 順序、相鄰特徵距離（RTI 記 $d_j$）、HiT 親和度、合併門檻、目標保留比例 | S10–S11 | — |
| $c(j)$ | patch $j$ 所屬的組 | S11 | — |
| $y_j,\ h_j,\ \beta$ | 目標（乾淨 patch）、核心前特徵、還原係數 | S12 | — |

### 3.1 全方法對照表（S14）

列（16）：Hourglass、Perceiver IO／RIN、DynamicViT、ToMe、ToMeSD、ToDo、AT-EDM、MoD／DiffCR、DyDiT、HDiT、JiT、H-Net、ELIT、DC-DiT、**RTI**、**HiT**。欄：期別、A 縮減、B 分組、C 位置層、D 還原、E 訓練、F 預算、G 位置、空間連續。內容照備忘第 4 節表格，修正兩處：Perceiver IO／RIN 的 G 欄寫「latent（RTI 的 Latent Array 基線無位置）」；ELIT 的 G 欄寫「每群學得位置嵌入（未逐行核對）」。FlexiDiT 與 Sparse Guidance 機制未核對，不列入。每列 `id="ax-<key>"`（key：`hourglass`、`perceiver`、`dynamicvit`、`tome`、`tomesd`、`todo`、`atedm`、`mod`、`dydit`、`hdit`、`jit`、`hnet`、`elit`、`dcdit`、`rti`、`hit`），供時間線卡片連結。互動：軸 chip（A–H、連續）點選後該欄加 `--accent-soft` 底與粗體表頭（`aria-pressed`），同一時間一欄；RTI、HiT 列以粗體與左側標記區分。≥ 768px 為橫捲表格（首欄 sticky，左右漸層提示）；< 768px 改為每方法一張卡片，只列被選中的軸＋期別（未選軸時列 B、D、F 三軸）。

---

## 4. 互動元件規格

### 4.0 共用

- PRNG：`mulberry32(seed)`、Box–Muller 只取 cos 分支（同參考頁與 217）。
- 場景（C3、C4）：完全沿用 `217/verify/common.py` 的 `scene('landscape')` 與 `scene('checker')`（128×128 px，`mulberry32(1)`，每像素 row-major 恰好抽兩次）；噪聲場 `mulberry32(2)`，每像素 3 次 `gauss`。**唯一差異**：patch 8×8 px，格子 $G=16$、$N=256$ 個 token，特徵 $h_j$ 為 192 維原始像素（順序 py、px、ch）。理由：與 RTI 教學用同一張圖；256 個 token 讓平均連結可在瀏覽器完成。頁面寫明「特徵用原始像素，代替模型內部特徵」（推定的簡化）。
- Hilbert：Wikipedia `d2xy`（與 217 相同），$n=16$；`HIL_PID[j] = y*16 + x`。
- 連通：4 鄰接 BFS。spread（本頁定義，推定）：每個 patch 到其組中心（組內格座標平均）的歐氏距離，對全部 256 個 patch 平均，單位為格（RTI Tab. 4 對每組平均，頁面註明差異）。EV（RTI Eq. 5）：$1-\sum_j\lVert h_j-\bar h_{c(j)}\rVert^2/\sum_j\lVert h_j-\bar h\rVert^2$，分母 < 1e-18 顯示「—」。
- 單位化：`unit_centered(f, mu)`：$f-\mu$ 的範數 ≤ 1e-12 時回傳 0 向量（餘弦為 0）。
- canvas 一律依 `devicePixelRatio` 放大內部解析度；組邊界的外框（halo）用 `--bg`（深色模式自動反轉），斜線圖樣用 `--ink` 50% 透明度並確認在淺、深兩種底色上與底圖對比 ≥ 3:1。
- 所有讀數 JetBrains Mono；live region 只在 `change` 或停止輸入 500ms 後更新；滑桿 `aria-valuetext`；圖 `role="img"`＋動態 `aria-label`。
- reduced-motion：本頁元件沒有連續動畫；C1 邊的淡入、C3 計算完成的淡入、時間線捲動在 `prefers-reduced-motion: reduce` 時一律改為立即顯示／`scroll-behavior:auto`。
- 窄螢幕（< 768px）：C1 三個面板縱向堆疊、各占 100% 寬；C2 的直接曲線標籤改為圖下一列圖例（線型樣本＋文字）；C3、C4 的 canvas 在讀數表上方。

### C1 · ToMe 一層合併、ToMeSD 分割與 unmerge（S6）

- **教的概念**：二部軟配對五步；$r$ 的意義；交替分割的 dst 形成規則欄位；隨機 2×2 dst；unmerge 是複製；proportional attention 等於複製 key（ToMe Sec. 3、Eq. 1；ToMeSD Sec. 3.1、4.1）。
- **玩具 token**（推定）：4×4 格 16 個 token，特徵 = RGB 三維，索引 raster（`i = row*4 + col`）：
  ```
  row0: (.58,.76,.94) (.60,.77,.95) (.62,.79,.95) (.98,.86,.45)   天空×3、太陽
  row1: (.64,.80,.96) (.66,.82,.97) (.22,.45,.20) (.68,.83,.97)   天空、天空、樹、天空
  row2: (.45,.58,.32) (.44,.57,.31) (.40,.28,.18) (.43,.56,.30)   草、草、樹幹、草
  row3: (.42,.55,.30) (.41,.54,.29) (.40,.53,.28) (.39,.52,.27)   草×4
  ```
  key $k_i=x_i-\bar x$（16 個 token 的平均），相似度 = key 餘弦。
- **輸入**：分段按鈕「分割」交替（ToMe，預設）／每個 2×2 隨機一個 dst（ToMeSD）；按鈕「重新抽樣」（只在 ToMeSD 時啟用，seed 由 1 起每次 +1）；滑桿 $r\in[0,|A|]$（交替 $|A|=8$、ToMeSD $|A|=12$），步長 1，預設 4；點選（或方向鍵＋Enter）任一 token 作為注意力的 query，預設 token 0。
- **計算**（`c1_tome.py` 同算法）：
  1. 分割。交替：A = 偶數索引、B = 奇數索引。ToMeSD：對 2×2 區塊依 raster（左上、右上、左下、右下）各抽 `k = min(3, floor(r()*4))`，dst = 區塊內第 k 格（k 的 raster 順序），B = dst 排序，A = 其餘排序。
  2. 每個 A 取餘弦最大的 B（同分取 B 中較前者）；依（−分數，A 中位置）排序取前 $r$ 個為 src。
  3. 合併：每個 dst 與指向它的 src 取平均（初始大小皆 1，等於大小加權平均）；$s$ = 組內 token 數。輸出順序依 ToMe App. D：[未合併的 A（原順序）, B]。
  4. unmerge：每個原位置填回所屬合併 token 的特徵；讀數 unmerge MSE $=\frac1{16}\sum_i\lVert x_i-\hat x_i\rVert^2$。
  5. 注意力（推定尺度）：$q=4k_{q}$、$k'=4(\text{merged}-\bar x)$，logit $=q\cdot k'/\sqrt3$；value = 合併特徵。算 dense（原 16 個 token）、plain（合併後、不加 $\log s$）、prop（加 $\log s$）、dup（合併 token 依 $s$ 複製後、不加偏置）。softmax 先減最大值。讀數：$\lVert o_{\text{plain}}-o_{\text{dense}}\rVert$、$\lVert o_{\text{prop}}-o_{\text{dense}}\rVert$、$\lVert o_{\text{prop}}-o_{\text{dup}}\rVert$（顯示為「< 1e-12」）。
- **顯示**：左「合併前」：4×4 色塊格，每格標索引；A 以空心圓角外框、B（dst）以實心方形標記（形狀區分，不只顏色）；每個 A 到其最佳 B 畫細虛線，保留的 $r$ 條邊改為實線箭頭並標分數（兩位小數）。中「合併後」：合併 token 列成一排色塊，下標 $s$。右「unmerge 後」：4×4 格，每格填所屬合併 token 的顏色；同組格以相同字母標記（a、b…）。下方注意力讀數表＋一行「加上 log s 與把 key 複製 s 份的結果相同」。
- **預期觀察**（第 5 節數值）：
  1. 交替分割時 B 為第 1、3 欄（索引 1、3、5、7、9、11、13、15）：dst 排成規則的直欄，對應 ToMeSD 的批評。
  2. 交替、$r=4$：合併 {4,5}、{8,9}、{12,13}、{14,15}，unmerge MSE 0.00006；query 0 的 plain 誤差 0.0061、prop 0.0002。
  3. 交替、$r=8$（全部 A 都必須合併）：樹 6、樹幹 10 被併入草 {6,10,14,15}，unmerge MSE 升到 0.00457——$r$ 太大時不相似的 token 也被合併。
  4. prop 與 dup 的差恆 < 1e-12（〔本頁推導〕的恆等式）；但群組內差異大時（ToMeSD、$r=12$，query 0：plain 0.0367、prop 0.0552）prop 不一定較接近 dense，頁面照實顯示。
- **防護**：$r$ 上限隨分割改變時夾住；餘弦分母 ≤ 1e-12 → 0。
- **無障礙**：格子是可聚焦按鈕（`aria-label`「token 6，樹，A 集合，合併到 15」）；邊以實線／虛線區分保留與否；讀數 live region 去抖。

### C2 · 縮減位置與加速上限（S7）

- **教的概念**：同一個 $C(L)$ 下，縮減的位置決定加速；Amdahl 式上限；二次項比例隨 $N$ 上升，所以只縮注意力的方法在高解析度才有效（1.2；〔本頁推導〕的成本模型，不是任何論文的數字）。
- **符號**：$N$ 影像 token 數；$n_{\text{txt}}$ 文字 token 數（不縮減）；$\kappa$ 保留比例（縮減後影像 token $=\kappa N$）；$C(L)=24Ld^2+4L^2d$；$L=N+n_{\text{txt}}$、$L_r=\kappa N+n_{\text{txt}}$。
- **四種策略**（加速 = 稠密成本／策略成本，各自與**同 block 數**的稠密模型比；S7 正文已介紹兩種配置）：
  - **ToMeSD 式（只對 self-attention 合併）**，17 block：每 block $8L_rd^2+4L_r^2d+16Ld^2$（只在 self-attention 前合併、MLP 全長，對應 ToMeSD Tab. 3「只對 self-attn 合併最划算」；不計合併與 unmerge 成本；圖版文字寫明此假設）。
  - **ToDo 式（只縮 K/V）**，17 block：每 block $24Ld^2+4L\,L_rd$（假設 K、V 投影仍在全長上計算，保守；ToDo 原文未說明投影在下採樣前或後，R4）。
  - **只縮中段（Hourglass／RTI）**：$6\,C(L)+11\,C(L_r)$，對 $17\,C(L)$（RTI B/16 的 3＋11＋3）。
  - **兩層巢狀（HiT）**：$6\,C(L)+6\,C(\sqrt\kappa N+n_{\text{txt}})+6\,C(\kappa N+n_{\text{txt}})$，對 $18\,C(L)$（3/3/6/3/3，每階段縮 $\sqrt\kappa$）。
- **輸入**：$N$ 分段按鈕 256／1024（預設）／4096／16384（附註：256 = JiT 256²/p16、1024 = MiniT2I 512²/p16）；$d$ 分段 384／768（預設）／1152／1536；$n_{\text{txt}}$ 滑桿 0–256、步長 16，預設 0；$\kappa$ 對數滑桿 1/16–1，預設 0.25；勾選「顯示 RTI 量測點」（預設開）。
- **顯示**：主圖橫軸 $\kappa$（對數，右端 1），縱軸加速：$N=16384$ 時 1×–7×，其他 1×–3.5×；超出上界的曲線段在上緣截斷並以箭頭標「超出」（目前規格內四策略最大 6.495×，不會觸發）。四條曲線以線型區分並直接標籤（窄螢幕改圖例列）：實線（只縮中段）、長虛線（巢狀）、點線（ToMeSD 式）、點劃線（ToDo 式）；只縮中段的上限 $17C(L)/(6C(L)+11C(n_{\text{txt}}))$ 以水平細線標「上限」。RTI 量測點（B/16，Tab. 1：$\kappa=0.5$ 1.84×、$\kappa=0.25$ 2.15×）以空心菱形＋數值標籤，只在 $N=1024$ 時顯示，並註明「牆鐘時間（RTX 4090），模型為 FLOPs；文字流未計入模型」。副圖：一條堆疊長條顯示目前 $L$ 下單一 block 的線性項與二次項比例（帶文字百分比）。讀數表：四策略的加速、上限、二次項比例。
- **預期觀察**：
  1. $N=1024,d=768,n_{\text{txt}}=0,\kappa=0.25$：只縮中段 2.030×、巢狀 1.796×、ToMeSD 式 1.600×、ToDo 式 1.158×；上限 2.833×。RTI 實測 2.15× 高於模型（App. A：注意力受記憶體頻寬限制）。
  2. $N=16384$：ToMeSD 式 4.686×、ToDo 式 2.412×，超過只縮中段的 2.381×——二次項比例 78%。
  3. $N=256$：ToDo 式只有 1.041×（二次項 5.3%）。
  4. $n_{\text{txt}}=256$：只縮中段的上限由 2.833× 降為 2.175×——文字流全長是 RTI 加速上限的一部分（Sec. 5）。
- **防護**：$\kappa\in[1/16,1]$，所有分母 $>0$；$\kappa=1$ 時四者皆 1.000。
- **無障礙**：曲線線型＋直接標籤；量測點有文字；副圖長條附百分比文字。

### C3 · 分組規則在同一批影像上的差異（S11）

- **教的概念**：分組規則（軸 B）決定連通性、保留的變異（EV）與預算型態（軸 F：每張精確 $R$ 對批次平均）。
- **方法**：預設顯示三種（S10 已教）；「進階」展開後再加兩種：
  1. 固定網格（HDiT／ELIT 群組）：Hilbert 序列等長切成 $R$ 段，長 $N/R$；$R=64$ 恰為 2×2 方塊、$R=16$ 為 4×4 方塊，$R=32$、128 為 2:1 長方形（`verify` 已確認每段都是長方形）。
  2. 相似度合併（ToMe 式）：C1 的二部軟配對反覆套用直到剩 $R$ 個 token。每輪 A = 目前序列偶數位、B = 奇數位，$r=\min(|A|,n-R)$，相似度為大小加權平均特徵（減去 256 個 patch 的全域平均）的餘弦；每輪後序列 = [未合併的 A, B]。
  3. Hilbert 最大落差（RTI，Eq. 1–2）：依 $(-u_j,j)$ 排序取前 $R-1$ 個切點。
  4. （進階）Hilbert 切割＋批次預算：同樣的 $u_j$，但兩張圖共用切點配額（本頁為了隔離「預算型態」這一個變數而構造的對照，**不是任何論文的方法**；H-Net／DC-DiT 用的是學得餘弦上的 $p_t\ge0.5$）。
  5. （進階）平均連結＋批次預算（HiT 式）：$S_{ij}=\cos(h_i-\mu,h_j-\mu)$（簡化：$q=k$＝單位化的置中特徵；HiT 的 $q,k$ 是兩個學得投影）。
- **批次預算的理想化**（頁面寫明）：風景與棋盤兩張圖構成一個批次。方法 4：兩圖的 $u_j$ 合併後依 $(-u_j,\text{影像},j)$ 排序，**恰好取前 $2R-2$ 個**作為切點（不用浮點門檻比較，平手不會多切）。方法 5：兩圖完整 dendrogram 的合併分數合併後依 $(-\text{分數},\text{影像},\text{合併序號})$ 排序，**恰好取前 $2N-2R$ 次合併**，每張圖重放自己的前 $m_b$ 次合併。兩者都使兩張圖群組數總和 $=2R$。HiT 實際是在訓練前以 1,024 個雜訊樣本校準 $\tau$ 後凍結（Sec. 4.1、App. A.3），批次平均只由壓縮損失（Eq. 4）鼓勵、不精確——頁面註明「本頁讓批次平均恰好等於 R，是理想化」。讀數另列對應的等效門檻（最後一個保留的 $u_j$、最後一次保留合併的分數）。
- **計算細節**（與 `common.py` 相同）：
  - 平均連結 dendrogram：維護群組對的 $S$ 總和矩陣（Lance–Williams：合併 $j$ 到 $i$ 時列與行相加，群組存在較小索引）與大小；每步掃描 $i$ 遞增、$j>i$ 遞增，取嚴格較大者；記錄 $N-1$ 次合併。平均連結沒有反轉（合併分數單調不增，`c3_group.py` 斷言檢查），所以依合併次數截斷與依門檻停止結果相同。**兩張圖的 dendrogram 各算一次並快取**，之後改 $R$、改方法都只做截斷重放（$O(N^2)$），不重算。
  - 執行方式：dendrogram 在 Web Worker 中計算（Worker 程式碼以 Blob 內嵌，單檔不外連）；Worker 不可用時改為主執行緒分段（每段 8 次合併，`setTimeout(0)`）。計算中圖版顯示「計算中」，方法 4、5 的按鈕停用。`?test` 同步計算。觸發：C3 或 C4 任一首次捲入視窗（`IntersectionObserver`），或讀者直接點方法 4、5。方法 1–3 不需等待。
  - 指標：$K$（實際群組數）、EV、spread（4.0）、連通組比例、最大組大小；另顯示批次平均 $(K_{\text{風景}}+K_{\text{棋盤}})/2$。
- **輸入**：方法分段按鈕（3＋進階 2）；預算 $R$ 分段 16／32（預設）／64／128；影像分頁「風景」（預設）／「棋盤」；選取 patch：滑鼠／觸控點擊，或 canvas `tabindex=0` 後以方向鍵移動 16×16 游標（可見的 2px 外框）、Enter 選取。
- **顯示**：左 256×256 CSS px canvas 畫目前影像，組邊界以 1px `--ink` 線（`--bg` halo）畫在相鄰而不同組的 patch 間；突顯組以斜線圖樣覆蓋，不連通的組額外以數字標示各分量（1、2…）。右讀數表：目前顯示的方法列 × 欄（K 風景／棋盤、EV、spread、連通比例、最大組），目前方法列加粗、目前影像欄加底；表下一行「方法 4、5 兩張圖共用配額；1–3 每張圖各自 R 個」。
- **預期觀察**（$R=32$，第 5 節）：
  1. 固定網格 EV 0.723（風景）；RTI 0.901、連通比例 1.000；ToMe 式 EV 0.837 但連通比例 0.219、spread 3.947 格（RTI 1.913）——保留了資訊卻散落全圖（呼應 RTI Tab. 4）。
  2. 進階：批次預算把群組從簡單圖移給複雜圖：方法 4 風景 48／棋盤 16；方法 5 風景 62／棋盤 2——棋盤兩種顏色各成一群，同色的 8 個方格雖不相鄰仍同組（連通比例 0）。這是 HiT 允許的非局部分組，也是「單張可變」的意思。
  3. 方法 5 的風景 EV 0.997 最高，但它用了 62 個群組，與 RTI 的 32 個不是同一預算——頁面在 $K$ 欄旁標註。
- **防護**：配額取法無浮點門檻；EV 分母保護；計算中停用按鈕。
- **無障礙**：canvas 附 `aria-label`（方法、$R$、影像、$K$、EV、連通比例）與等價的隱藏表格；選取時 live region（去抖）宣告「patch（列 r，行 c）屬於第 k 組，共 n 個 patch」；突顯以圖樣而非顏色；觸控點擊等同 hover。

### C4 · 還原：廣播與殘差寫回（S12）

- **教的概念**：以稠密目標為例，廣播只能讓同組共用一個輸出，誤差下界是組內變異；殘差寫回保留每個 patch 的核心前特徵，可以把組內差異帶回來。**這是純玩具（〔本頁推導〕）**：RTI 的 $h_j$ 是骨幹深層特徵、$g$ 是學得的映射，論文沒有報告「噪聲大時應少用前置特徵」這類行為；頁面圖版標題下一行寫明「這不是 RTI 或 HiT 的行為，只示範廣播的誤差下界與殘差路徑的作用」。
- **設定**（頁面寫出）：目標 $y_j$ = 乾淨風景 patch；核心前特徵 $h_j$ = $x_t$ 的 patch（S1 的前向過程）；分組沿用 C3 目前的方法與 $R$（風景圖）；「理想核心」＝假設縮短段對每組恰好輸出目標的組平均 $\bar y_c$。還原：
  $$\hat y_j(\beta)=\bar y_c+\beta\,(h_j-\bar h_c)$$
  $\beta=0$ 為廣播／unmerge（ToMeSD）；$\beta=1$ 為殘差加回（每個 patch 保留自己的前置特徵、加上組的變化量），與 RTI Eq. 4「每個 patch 以自身特徵與區域變化量形成更新」**類比**，但 Eq. 4 的 $g$ 作用在串接 $[h_j\Vert z'_c-z_c]$ 上、為學得映射，形式不同；$\beta^*$ 為最小平方最佳。
- **計算**：$A_{yy}=\sum_c\sum_{j\in c}\lVert y_j-\bar y_c\rVert^2$、$A_{yh}=\sum\langle y_j-\bar y_c,h_j-\bar h_c\rangle$、$A_{hh}=\sum\lVert h_j-\bar h_c\rVert^2$、$D_y=\sum_j\lVert y_j-\bar y\rVert^2$；相對誤差 $E(\beta)=(A_{yy}-2A_{yh}\beta+A_{hh}\beta^2)/D_y$；$\beta^*=A_{yh}/A_{hh}$（$A_{hh}\le10^{-12}$ 時 $\beta^*=0$）。〔本頁推導〕$E(0)=1-\mathrm{EV}_y$（以目標計算的 EV），且 $E(\beta^*)\le\min(E(0),E(1))$。
- **輸入**：滑桿 $t\in[0,1]$、步長 0.01，預設 0.95；滑桿 $\beta\in[-1,2]$、步長 0.01，預設 0；chip「廣播 β=0」「殘差 β=1」「最佳 β*」——$\beta^*$ 超出滑桿範圍時滑桿夾到端點，讀數仍顯示實際 $\beta^*$ 並註「超出範圍」。方法與 $R$ 由 C3 控制（圖版頂端顯示目前值與「在 C3 更改」連結）。
- **獨立啟動**：C4 自己觀察可見性；若讀者經目錄直接跳到 S12、C3 尚未計算，C4 以預設（RTI、$R=32$）立即計算（方法 1–3 不需 dendrogram），並觸發 C3 的背景 dendrogram 計算。
- **顯示**：左右兩個 128×128 canvas（2× 放大）：$x_t$ 與 $\hat y(\beta)$（截到 [0,1]）。右側小圖：$E(\beta)$ 拋物線（橫軸 $[-1,2]$，縱軸 0 到 $\max(E(0),E(1))\times1.1$ 且至少 0.05），標出 β=0（方形）、β=1（三角）、β*（圓形）三點並直接標籤；目前 β 以垂直細線。讀數：$K$、$E(0)$、$E(1)$、$\beta^*$、$E(\beta^*)$、目前 $E(\beta)$（四位小數）。
- **預期觀察**（RTI、$R=32$）：
  1. $t=1$：$E(0)=0.0985$（廣播留下 9.85% 的組內變異，影像呈區塊狀）、$E(1)=0$、$\beta^*=1$。
  2. $t=0.95$：$E(0)=0.0985$、$E(1)=0.0347$、$\beta^*=0.760$、$E(\beta^*)=0.0277$。
  3. $t=0.8$：$E(1)=0.5559$ 遠大於廣播，$\beta^*=0.127$（在這個原始像素玩具中，前置特徵帶有噪聲）。
  4. 同一 $t=0.95$ 換分組：固定網格 $E(0)=0.2768$、ToMe 式 0.1632、RTI 0.0985——廣播的誤差直接由分組決定；$\beta=1$ 的 $E(1)$ 在這三者間只差 0.0347–0.0353，分組的影響變小。正文連到 HiT Tab. 3（只改讀回：硬查表 7.97 對軟混合 6.61）；RTI 的 −7.2 註明同時改變池化與還原。
- **防護**：$A_{hh}$ 下限；$D_y>0$（風景圖非常數）；canvas 截值；$\beta$ 滑桿夾值。
- **無障礙**：三點以形狀＋文字區分；滑桿 `aria-valuetext`（「t = 0.95」「β = 0.76」）；live region 去抖。

---

## 5. 驗收數值表（全部出自 `verify/out_*.txt`；`python3 -I verify/<script>.py` 重新產生）

容許誤差依數值大小設定（頁面與腳本同算法，只差浮點加總順序，實際差異應 < 1e-9）：
- 整數（$K$、組大小、組成員、dst 索引）：完全相同。
- 約 0.01–10 的量（加速、EV、spread、連通比例、$E\ge0.01$）：絕對 ±0.001。
- 小量（MSE、注意力誤差、$E<0.01$）：相對 ±1% 或絕對 ±1e-7，取較大者（例：1.875e-05 容許 ±1.9e-07）。
- 恆等式（prop−dup、$E(\beta^*)\le\min(E(0),E(1))$）：< 1e-12。

### 5.1 C1（`out_c1_tome.txt`）

| 分割 | seed | $r$ | 合併組 | unmerge MSE | q=0 plain | q=0 prop | q=6 plain | q=6 prop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 交替 | — | 2 | {12,13},{14,15} | 1.8750e-05 | 1.1601e-02 | 3.6674e-06 | 1.2935e-02 | 1.0223e-04 |
| 交替 | — | 4 | {4,5},{8,9},{12,13},{14,15} | 5.6250e-05 | 6.1202e-03 | 1.8171e-04 | 2.7758e-02 | 1.3127e-04 |
| 交替 | — | 6 | {0,1,2},{4,5},{8,9},{12,13},{14,15} | 1.3958e-04 | 2.5462e-02 | 5.0009e-04 | 3.5379e-02 | 1.2480e-04 |
| 交替 | — | 8 | {0,1,2},{4,5},{8,9},{12,13},{6,10,14,15} | 4.5708e-03 | 2.6662e-02 | 6.8920e-04 | 6.6895e-02 | 2.5423e-02 |
| ToMeSD | 1 | 4 | {4,5},{11,12,13},{14,15} | 6.6667e-05 | 5.7128e-03 | 1.8332e-04 | 2.6544e-02 | 1.7470e-04 |
| ToMeSD | 1 | 8 | {0,1,2},{4,5,7},{9,11,12,13},{14,15} | 2.5104e-04 | 4.7824e-02 | 7.4574e-04 | 5.2807e-02 | 3.5345e-04 |
| ToMeSD | 1 | 12 | {0,1,2},{3,4,5,7},{8,9,11,12,13},{6,10,14,15} | 2.2166e-02 | 3.6665e-02 | 5.5233e-02 | 2.2589e-02 | 2.5621e-02 |

分割：ToMeSD seed 1 → dst {2,4,12,15}；seed 2 → {3,4,9,14}。所有列 $\lVert o_{\text{prop}}-o_{\text{dup}}\rVert<10^{-15}$。頁面讀數以三位有效數字顯示（例「6.12×10⁻³」）。

### 5.2 C2（`out_c2_cost.txt`；±0.001）

二次項比例（$d=768$）：$L=256$ 0.0526、1024 0.1818、4096 0.4706、16384 0.7805。只縮中段上限（$T=0$）2.8333；$T=128$ 2.4269；$T=256$ 2.1746。

| $N$ | $d$ | $T$ | $\kappa$ | ToMeSD 式 | ToDo 式 | 只縮中段 | 巢狀 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 768 | 0 | 0.25 | 1.401 | 1.041 | 1.967 | 1.737 |
| 1024 | 768 | 0 | 0.5 | 1.375 | 1.100 | 1.545 | 1.412 |
| 1024 | 768 | 0 | 0.25 | 1.600 | 1.158 | 2.030 | 1.796 |
| 1024 | 768 | 0 | 0.0625 | 1.776 | 1.205 | 2.587 | 2.366 |
| 4096 | 768 | 0 | 0.25 | 2.345 | 1.545 | 2.185 | 1.943 |
| 16384 | 768 | 0 | 0.25 | 4.686 | 2.412 | 2.381 | 2.130 |
| 1024 | 768 | 256 | 0.25 | 1.513 | 1.150 | 1.730 | 1.583 |
| 1024 | 384 | 0 | 0.25 | 1.857 | 1.300 | 2.095 | 1.857 |
| 1024 | 1536 | 0 | 0.25 | 1.468 | 1.081 | 1.990 | 1.758 |

### 5.3 C3（`out_c3_group.txt` 前半）

方法 4、5 依配額取法得到的等效門檻（最後保留的 $u_j$／合併分數，只作讀數，不參與計算）：$R=16$ 5.570990／0.751434；$R=32$ 2.491506／0.993647；$R=64$ 0.373483／0.996588；$R=128$ 0.061100／0.999815（±1e-6）。兩張圖的 dendrogram 無反轉（腳本斷言）。

| 方法 | $R$ | 影像 | K | EV | spread | 連通 | 最大組 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 固定網格 | 32 | 風景 | 32 | 0.723 | 1.144 | 1.000 | 8 |
| ToMe 式 | 32 | 風景 | 32 | 0.837 | 3.947 | 0.219 | 21 |
| ToMe 式 | 32 | 棋盤 | 32 | 1.000 | 5.460 | 0.438 | 106 |
| RTI | 32 | 風景 | 32 | 0.901 | 1.913 | 1.000 | 48 |
| RTI | 32 | 棋盤 | 32 | 1.000 | 1.244 | 1.000 | 16 |
| 方法 4（Hilbert＋批次配額） | 32 | 風景 | 48 | 0.962 | 1.598 | 1.000 | 48 |
| 方法 4（Hilbert＋批次配額） | 32 | 棋盤 | 16 | 1.000 | 1.498 | 1.000 | 16 |
| 方法 5（平均連結＋批次配額） | 32 | 風景 | 62 | 0.997 | 3.227 | 0.887 | 44 |
| 方法 5（平均連結＋批次配額） | 32 | 棋盤 | 2 | 1.000 | 6.113 | 0.000 | 128 |
| 固定網格 | 64 | 風景 | 64 | 0.824 | 0.707 | 1.000 | 4 |
| RTI | 64 | 風景 | 64 | 0.983 | 1.381 | 1.000 | 48 |
| 方法 4（Hilbert＋批次配額） | 64 | 風景／棋盤 | 112／16 | 0.996／1.000 | 0.778／1.498 | 1.000／1.000 | 20／16 |
| 方法 5（平均連結＋批次配額） | 64 | 風景／棋盤 | 126／2 | 0.999／1.000 | 2.197／6.113 | 0.857／0.000 | 24／128 |
| RTI | 16 | 風景 | 16 | 0.427 | 3.155 | 1.000 | 98 |
| 方法 5（平均連結＋批次配額） | 16 | 風景／棋盤 | 30／2 | 0.946／1.000 | 4.282／6.113 | 0.933／0.000 | 108／128 |

其餘組合見 `out_c3_group.txt`（selfTest 只檢查本表）。

### 5.4 C4（`out_c3_group.txt` 後半）

| 方法 | $R$ | $t$ | $E(0)$ | $E(1)$ | $\beta^*$ | $E(\beta^*)$ |
| --- | --- | --- | --- | --- | --- | --- |
| RTI | 32 | 1.00 | 0.098539 | 0.000000 | 1.0000 | 0.000000 |
| RTI | 32 | 0.95 | 0.098539 | 0.034741 | 0.7598 | 0.027658 |
| RTI | 32 | 0.90 | 0.098539 | 0.138963 | 0.4066 | 0.062739 |
| RTI | 32 | 0.80 | 0.098539 | 0.555852 | 0.1266 | 0.088722 |
| 固定網格 | 32 | 0.95 | 0.276807 | 0.035253 | 0.9270 | 0.033746 |
| ToMe 式 | 32 | 0.95 | 0.163212 | 0.034930 | 0.8545 | 0.031096 |
| 平均連結（方法 5） | 32 | 0.95 | 0.002524 | 0.029874 | 0.0730 | 0.002354 |
| RTI | 64 | 0.95 | 0.017027 | 0.029646 | 0.3582 | 0.011320 |

$\beta^*$ 容許 ±0.001。

### 5.5 selfTest（`?test`）

`console.assert` 比對：C1 交替 $r=4$ 的組成員與 MSE；ToMeSD seed 1 的 dst；prop−dup < 1e-12。C2 表中第 3、6、7 列與上限 2.8333。C3 $R=32$ 九列與 $R=16$ 方法 5 列（含等效門檻 ±1e-6；dendrogram 單調）。C4 RTI $R=32$ 四列與恆等式。另檢查 `scrollWidth <= innerWidth`、時間線卡片數 = 23（`timeline.json` 筆數）。輸出 `selfTest: N passed, M failed`。

---

## 6. 實作計畫

### 6.1 沿用與新增

- **沿用**（不重寫）：`tutorials/prediction-targets-k-diff.html` 的外殼——CSS tokens（含深色模式、`--c-par`、`--c-perp`）、appbar、主題切換、章節目錄（桌機 sticky 左欄／窄螢幕頂部橫列）、`#pm-bar`、hero 版型、plate、callout、`<details>` 檢查題、KaTeX 0.16.9 設定、時間線元件骨架、`?test` selfTest 架構、`el`／`scaleLin`／`scaleLog`／`linePath`、`mulberry32`／`gauss`。
- **新增**：
  - 期別色帶元件 `renderStageBar(container, current)`：7 段，段名「第 0 期」…「第 6 期」與年份；目前期別以 `--accent` 外框＋粗體文字（非只靠色）。
  - S10 的 Hilbert 小圖：8×8 格的靜態 SVG，畫曲線折線並標 0、1、2… 前 16 個序號，與一段被切出的連續區間（斜線填滿）。
  - 圖表色：沿用 `--c-par`（藍）、`--c-perp`（綠），C2 的四條曲線另以線型區分，不新增顏色。
  - 共用運算（逐行移植 `verify/common.py`）：`scene(name)`、`noiseField()`、`features(img)`（G=16、PX=8）、`HIL_PID`、`gaps`、`runsToLabels`、`mGrid`、`mTome`、`mRti`、`threshBatch(R)`、`dendrogram(h)`、`cut(merges, m)`、`linkageBatch(R)`、`stats(h, lab)`、`restore(y, h, lab)`。`dendrogram` 另以 Blob Worker 版本執行（同一份函式字串），4.0／C3 的「執行方式」。
  - 元件：`initC1()`、`initC2()`、`initC3()`、`initC4()`；C3、C4 共用 `GROUPING = {method, R, labLandscape, labChecker}`、`DEN = [null, null]`（dendrogram 快取）、`ensureDendrograms()`（回傳 Promise，重複呼叫共用同一個）與 `onGroupingChange` 回呼。
  - S13 逐軸表（< 768px 卡片）、S14 對照表與軸 chip、兩張教學卡片。
- 效能：dendrogram 每張約 5.6M 次比較，用 `Float64Array` 的 $N\times N$ 矩陣，兩張圖各一次；截斷重放 $O(N^2)$。C1、C2 即時。

### 6.2 連結

- 關聯圖：`../index.html#paper-217`（RTI）、`#paper-585`（HiT）、`#paper-349`（JiT）；S9 與 S16 另連 `#paper-232`（〈Rethinking the Design Space of Vanilla Pixel Diffusion Transformers〉，相鄰、內容未讀，只列標題）與 S16 連 `#paper-50`（〈V-CoLA: Vision Token Compression with Linear Attention〉，VLM 視覺 token 壓縮，相鄰、內容未讀）。其餘論文不在資料庫，連 arXiv（`https://arxiv.org/abs/<id>`）。
- 單篇教學：`rti-elastic-token-compression.html`、`hit-hierarchical-diffusion-transformer.html`（S0、S10、S13、S16）。RTI 的 Hilbert／READ／成本細節、HiT 的軟權重／壓縮損失／群組引導各在對應段落以「在單篇教學中」連結指向。
- 系列筆記（`notes/`，皆已存在）：`tokc-lineage-hub`（S0、S16）、`tokc-hourglass-and-latent-interfaces`（S4、S9 的 HDiT、S10 的 ELIT）、`tokc-vit-pruning-and-merging`（S5）、`tokc-position-for-merged-tokens`（S5 的 $\log s$、S13 位置列）、`tokc-training-free-diffusion-merging`（S6）、`tokc-learned-routing-dynamic-dit`（S8）、`tokc-content-adaptive-tokenization`（S9 Quadformer、S10 Hilbert 序列化）、`tokc-dynamic-chunking-bytes-to-pixels`（S9 H-Net、S10 DC-DiT）。
- 單篇論文 hub：`rti-lineage-hub`、`hit-lineage-hub`（S10、S13）。既有筆記：`pixel-space-diffusion`、`x-prediction-manifold-view`、`generative-eval-metrics`（S1）、`aniso-pixel-patchified-setting`（S2）、`depth-redundancy-adjacent-methods`、`moe-conditional-computation-lineage`（S8）。連結格式 `<slug>.html`，連結文字用各筆記 frontmatter 的 `title`。

### 6.3 文案與排版

- 繁體中文、台灣用語、字面學術語氣；不用比喻、反問、宣傳形容詞；章節標題陳述句。專有名詞第一次出現附英文；S1 的先備詞彙之後，任何術語都在首次出現的段落定義（審查 7）。
- 三種陳述明確區分：論文結果（附 Sec./Eq./Tab.）、作者的解釋（「作者認為」「作者歸因於」）、本頁推導（「本頁推導」pill）。只讀摘要的數字一律用「僅摘要」pill（全頁只用這一種寫法）。
- 元件讀數不逐一加出處；每個圖版底部一行「讀數由本頁依 ○○ 即時計算」，C2、C3 的 spread、C4 另標「本頁推導／推定」；C4 標題下加「純玩具」說明（C4 規格）。
- 頁面上的論文數字只能來自本文件第 1 節與第 2 節參考答案；元件數值只能來自第 5 節。KaTeX 規則同參考頁（SVG 內用 Unicode；動態 DOM 呼叫 `renderMathInElement`；`.katex-display` 橫捲）。
- 用詞：「合併（merge）」「還原」「廣播（broadcast）」「殘差寫回」「彈性預算」「批次平均預算」「空間連續（連通）」「非局部」；不用「壓扁」「魔法」等。

### 6.4 時間線（S15）與延伸閱讀

- 來源 `notes/timeline.json`（23 筆，皆含 `year`、`title`、`authors_short`、`arxiv`、`role`、`source`），建置時原樣貼進 `<script type="application/json" id="timeline-data">`，不手改條目。EViT、AT-EDM、IBTM、FlexiDiT 不在檔案中，只出現在正文與對照表。
- 期別對應（常數 `TL_STAGE`，以 arXiv ID 為 key；不在表中歸「其他」）：
  - 第 0 期：2006.03236、2110.13711、2107.14795
  - 第 1 期：2106.02034、2210.09461
  - 第 2 期：2303.17604、2402.13573
  - 第 3 期：2404.02258、2410.03456、2412.16822、2601.01608、2212.11972、2209.12152
  - 第 4 期：2301.11093、2401.11605、2511.13720
  - 第 5 期：2304.00287、2412.09871、2507.07955
  - 第 6 期：2603.12245、2603.06351、2608.29281、2611.22962
- 卡片的「筆記」連結（常數 `TL_NOTE`）：依 hub 閱讀順序取第一篇 frontmatter `arxiv` 含此 ID 的 `tokc-*` 筆記；例外：2511.13720 → `pixel-space-diffusion`、2608.29281 → `rti-lineage-hub`、2611.22962 → `hit-lineage-hub`。結果：2006.03236、2107.14795、2110.13711、2212.11972、2209.12152、2301.11093、2401.11605、2603.12245 → `tokc-hourglass-and-latent-interfaces`；2106.02034、2210.09461 → `tokc-vit-pruning-and-merging`；2303.17604、2402.13573 → `tokc-training-free-diffusion-merging`；2404.02258、2410.03456、2412.16822、2601.01608 → `tokc-learned-routing-dynamic-dit`；2304.00287 → `tokc-content-adaptive-tokenization`；2412.09871、2507.07955、2603.06351 → `tokc-dynamic-chunking-bytes-to-pixels`。
- 「對照表」連結（`#ax-<key>`）：2110.13711 `hourglass`、2107.14795／2212.11972 `perceiver`、2106.02034 `dynamicvit`、2210.09461 `tome`、2303.17604 `tomesd`、2402.13573 `todo`、2404.02258／2412.16822 `mod`、2410.03456 `dydit`、2401.11605 `hdit`、2511.13720 `jit`、2507.07955 `hnet`、2603.12245 `elit`、2603.06351 `dcdit`、2608.29281 `rti`、2611.22962 `hit`；其餘不顯示。「關聯圖」連結只給 2608.29281（#paper-217）、2611.22962（#paper-585）、2511.13720（#paper-349）。
- 呈現：期別 chip（`aria-pressed`，可多選；**一個都沒選時顯示全部**；未選期別的卡片改用 `--ink-3` 文字與 `--surface-2` 底、不調 opacity，維持 ≥ 4.5:1）；卡片水平捲動、`scroll-snap`，依年份排序（同年保持檔案順序）；卡片內容：年份、期別標籤、標題、作者、`role`（三行截斷，全文放 `title`）、上述連結。RTI、HiT 卡片以 `--accent` 外框與「起點」標籤突出。左右捲動鈕；方向鍵只在卡片列取得焦點時作用；reduced-motion 時捲動 `auto`；JSON 解析失敗時顯示「時間線資料尚未提供」。
- S16 延伸閱讀依 hub 閱讀順序：`tokc-lineage-hub` → `tokc-hourglass-and-latent-interfaces` → `tokc-vit-pruning-and-merging` → `tokc-training-free-diffusion-merging` → `tokc-learned-routing-dynamic-dit` → `tokc-content-adaptive-tokenization` → `tokc-dynamic-chunking-bytes-to-pixels` → `tokc-position-for-merged-tokens` → `rti-lineage-hub`、`hit-lineage-hub` → 兩篇單篇教學 → 既有筆記 `pixel-space-diffusion`、`depth-redundancy-adjacent-methods`、`generative-eval-metrics`。每項附 frontmatter 的 `title`。

### 6.5 建置後自我檢查

沿用版面，截圖只看新元件（C1–C4、期別色帶、Hilbert 小圖、對照表、逐軸表、時間線）與 390px、深色模式；`?test` 全數通過；impeccable 0 項；執行 `extract_page_text.py` 確認 12 個 LO 的參考答案都在正文；確認 390px 下 C1 堆疊、C2 圖例列、S13／S14 改為卡片、C3 讀數表不撐破版面；以節流 CPU（DevTools 6×）確認 C3 計算中按鈕停用、主執行緒不卡住。

---

## 7. 風險與未決事項

| # | 項目 | 說明 | 處理 |
| --- | --- | --- | --- |
| R1 | 玩具特徵 | C3、C4 用原始像素代替模型特徵；EV、群組數只說明機制，不重現 RTI／HiT 的數字。平均連結用 $q=k$ 的餘弦，與 HiT 學得的不對稱投影不同。 | 每個圖版寫明「推定的簡化」；不與論文數字並列成同一張表。 |
| R2 | C3 預算不對等 | 方法 4、5 在單張圖上的 $K$ 與 $R$ 不同，EV 不能直接與精確 $R$ 的方法比；批次平均恰好等於 $R$ 是理想化（HiT 實際只以損失鼓勵）。方法 4 是本頁構造的對照，不是任何論文的方法。 | $K$ 欄緊鄰 EV；觀察 3 與「批次預算的理想化」段明寫；方法 4、5 放在「進階」。 |
| R3 | 時間步依賴 | 原始像素加噪聲（標準差 $1-t$）後所有分組都退化（設計時試算：以 $t=1$ 校準的固定門檻在 $t\le0.75$ 時 $K$ 幾乎全為 256），所以 C3 只用乾淨特徵、C4 才引入 $t$。HiT 只說群組數隨時間步改變，未報告方向。 | C3 不設 $t$；S11 正文以 RTI Fig. 2b（取樣進行時保留比例上升）與 DiffCR（雜訊大的時間步比例高）文字說明時間依賴。 |
| R4 | ToDo 成本假設 | ToDo Sec. 2.2 只寫「把 $D(\cdot)$ 套到 keys 與 values」，未說明投影在下採樣前後。 | C2 採保守假設（投影全長），圖版與 LO6 參考答案都寫出此假設。 |
| R5 | 成本模型 | $C(L)$ 未計 adaLN、雙流權重、文字流、合併／還原、Read/Write（RTI 稱後者 < 2%）與記憶體頻寬；ToMeSD 的實測（SD U-Net）不能與模型比較。$C(L)$ 乘加計 2，HiT／JiT 的 GFLOPs 乘加計 1（HiT Sec. 4.1），不可混用。 | C2 只疊 RTI B/16 的兩個牆鐘點，並註明差異來源（App. A）。 |
| R6 | 筆記與時間線 | 已產出（hub＋7 篇、23 筆時間線）；6.2、6.4 已依實際檔案填寫。備忘建議的 `tokc-rti-*`、`tokc-hit-*` 未建立，改連單篇 hub。 | 建置時若 `notes/` 再變動，重新比對 frontmatter 的 `arxiv` 更新 `TL_NOTE`。 |
| R7 | 只讀摘要的論文 | Funnel、Perceiver IO、DynamicViT、AT-EDM、IBTM、FlexiDiT、RIN、U-ViT、simple diffusion、BLT、JiT 只讀摘要。MoD、DyDiT 全文在 `research/papers/`，但本文件只引用其摘要層級的數字。FlexiDiT 機制未核對；ELIT 位置以「每群學得位置嵌入（未逐行核對）」表述。 | 頁面標「僅摘要」；對照表不列 FlexiDiT、Sparse Guidance。 |
| R8 | 與單篇教學重疊 | 217 的 C2（分割）、C4（成本）與 585 的 C1（平均連結）、C3（layout 成本）主題相近。 | 本頁元件都是「跨方法」：C2 比較四種位置策略、C3 在同一批次上並列多種規則、C4 比較還原方式；機制細節連到單篇。 |
| R9 | HDiT 未被引用 | RTI 與 HiT 都沒有引用 HDiT，是依主題補入。 | S9 一句說明「本頁依主題補入」。 |
| R10 | MiniT2I 無 arXiv | 兩篇引用時皆無編號；文字 token 數未知。 | C2 的 $n_{\text{txt}}$ 由使用者設定，預設 0；不寫 MiniT2I 的文字長度。 |
| R11 | prop 不必然更接近 dense | C1 在組內差異大時 prop 誤差可能大於 plain。 | 照實顯示；正文只主張「加 $\log s$ 等於複製 key」這個恆等式。 |
| R12 | 平均連結效能 | 低階手機上兩張圖的 dendrogram 首次約 0.1–0.3 秒以上。 | Worker（失敗時分段）、只算一次、截斷重放；計算中停用方法 4、5；方法 1–3 不受影響。 |
| R13 | 未完全採納的審查意見 | 審查 17 建議把 ToMe 的 $s$、$r$ 改名。$s$、$r$、$S_{ij}$、$R$ 都是原論文符號，改名會讓讀者無法對照原文。 | 保留原符號，在 3.0 符號表的「易混淆」欄逐一說明；其餘衝突（寬度 $d$ 對距離 $d_j$、常數 $c$ 對 $c(j)$、文字 $T$ 對時間 $t$）已改名為 $u_j$、$A_{hh}$ 等、$n_{\text{txt}}$。 |

---

## 8. 修訂紀錄（round 1，對應 `reviews/round1.md`）

| 審查 | 處理 |
| --- | --- |
| 1 | 1.5、1.6：39.0 改寫為「換成 Latent Array（同時失去分組、連續、位置）」。 |
| 2 | C4：β=1 改稱與 RTI Eq. 4「類比」，寫明 $g$ 作用在串接上、形式不同。 |
| 3 | 1.5、LO10、S12、C4 觀察 4：RTI 7.2 標為「取代整個 READ/WRITE」；只改還原的證據只用 HiT 7.97 對 6.61。 |
| 4 | 1.1、1.4 改為謹慎措辭，明寫 HiT 的對照組本身連續、未測試非局部；指向 1.7。 |
| 5 | C4：新增「純玩具」聲明；觀察 3 註明原始像素。 |
| 6 | C3：方法 4 改名「Hilbert 切割＋批次預算」並寫明非任何論文方法；新增「批次預算的理想化」段，寫出 HiT 的實際校準（1,024 個雜訊樣本、凍結）與損失只鼓勵平均。 |
| 7 | 新增 S1 先備（DiT、前向過程、x-prediction、CFG、FID、GenEval、牆鐘、LoRA、RoPE）；S3 只放軸；dst／src／unmerge 移到 S6、核心前特徵／殘差寫回移到 S12；Hilbert、平均連結、dendrogram、非局部在 S10 定義；Amdahl 在 S7 定義。 |
| 8 | 原 S9 拆為 S10（三種規則）與 S11（預算、EV、spread、C3）；C3 預設三種方法，另兩種在「進階」。 |
| 9 | 6.3 改為「論文數字只來自第 1 節與第 2 節，元件數值只來自第 5 節」。 |
| 10 | S7 正文先介紹 3＋11＋3 與 3/3/6/3/3 兩種配置；C2 每種策略與同 block 數的稠密模型比較。 |
| 11 | `common.py`：`thresh_batch`、`linkage_batch` 改為依排序恰好取配額（不用浮點門檻），平手確定；重跑後數值不變。 |
| 12 | dendrogram 只算一次（Worker／分段），依合併次數截斷；腳本斷言無反轉；計算中停用按鈕。 |
| 13 | C4 自行觀察可見性並以預設啟動、觸發背景計算。 |
| 14 | C3 canvas 鍵盤游標、Enter 選取、live region 宣告。 |
| 15 | 4.0 窄螢幕規則：C1 堆疊、C2 圖例列；S13 表格改卡片。 |
| 16 | 時間線：淡化改用 `--ink-3`／`--surface-2`（≥ 4.5:1）、未選即全部、方向鍵只在卡片列有焦點時作用。 |
| 17 | 3.0 符號表加「易混淆」欄；$d_j\to u_j$、$T\to n_{\text{txt}}$、C4 的 $a,b,c\to A_{yy},A_{yh},A_{hh}$；$s,r,R,S_{ij}$ 保留（R13）。 |
| 18 | Token Skip 加註預算不同與「低估差距」（1.4、LO7）。 |
| 19 | 「共同發現」改寫，加 GFLOPs 18.3 → 22.2 的混淆。 |
| 20 | 1.7 第 5 點：Fig. 7、MiniT2I-L/16、「低於約 3 秒／高於約 5 秒」。 |
| 21 | Perceiver IO 的「無位置」改為歸於 RTI 的 Latent Array 基線；對照表 G 欄同步。 |
| 22 | 全頁只用「僅摘要」；R7 修正 MoD、DyDiT 的敘述。 |
| 23 | LO6 參考答案寫出 K/V 投影全長的假設。 |
| 24 | 1.2 結論一標〔本頁推導〕並列出 ToMeSD Sec. 5 的作者解釋。 |
| 25 | 1.4 H-Net：downsampler 只保留邊界向量；dechunking 含 EMA 平滑與 straight-through 信心值。 |
| 26 | 1.5：名目 GFLOPs 139.4 對 201.6、乘加計 1；共同分組只在 T2I。 |
| 27 | C4 β 滑桿 [−1, 2]，β* 超出時夾值並顯示實際值。 |
| 28 | S13 標題「RTI 與 HiT 在八個設計軸上的選擇對照」；S15 標題「研究時間線」。 |
| 29 | S8、S9 各只教 1–2 個代表方法，其餘一行帶過並連對照表與筆記。 |
| 30 | 4.0：halo 用 `--bg`、canvas 依 DPR、斜線對比 ≥ 3:1、各動畫的 reduced-motion 處理。 |
| 31 | 第 2 節開頭說明無元件的 LO 與不設 LO 的章節是刻意的。 |
| 驗算 | 三支腳本加 `sys.path` 修正，可用 `python3 -I` 在任何目錄執行；C1、C4 輸出改為較多有效位數；第 5 節改為依數量級的容許誤差。 |

## 9. 修訂紀錄（readiness）

- §1.3 G 欄改為「ToMe 的對數大小項（S5 定義）」，S3 不出現 $s$。
- `notes/timeline.json` 2601.01608 的 `role` 改為「推論時利用 token 層級稀疏度作為引導訊號（依摘要）；HiT 的 group guidance 寫明受其啟發（HiT App. A.6）」——HiT 原文：「Inspired by Sparse Guidance [22], we use this variation in token capacity to construct a guidance signal」。
- §1.6 補 E（訓練）、H（擴散）兩列，S13 表現為「目的」加 A–H 八軸，共九列；原「文字」列併入 H。
- 非阻擋：RTI Eq. 4 改寫為 $h^{\text{in}}/h^{\text{out}}$；H-Net 目標比全寫 $N_{\text H}$；S10(a) 與 C3 方法 1 一致；C2 縱軸截斷規則；C2 ToMeSD 式假設寫在圖版。

## 10. 建置紀錄（build round）

- 建置來源在 scratchpad `tp/series_parts/`（body.html、app.js、extra.css），由 `tp/build_series.py` 組成 `build/index.html`（外殼 CSS 與 head 直接取自參考頁）。
- `?test`：63 passed, 0 failed（含 390px iframe 內 scrollWidth 檢查）；impeccable `--no-advisory` 0 項（只剩 numbered-section-labels advisory）。
- 與設計的差異：每個有 LO 的章節末加一段「本節重點」，讓 LO 參考答案在單一段落中完整出現；S03 符號表改用 `details.more`（`details.check` 只留給檢查題）；C3 讀數表用短方法名；C2 二次項比例改為細長條＋文字；C4 縱軸涵蓋 β∈[−1,2] 的完整拋物線（最多到 4 倍 max(E0,E1)）；C1 在窄螢幕每個面板最寬 320px。
- impeccable 修正：本節重點框與對照表卡片的左側色條改為全框＋底色；ToDo 圖例改用虛線（不用 repeating-gradient）。
