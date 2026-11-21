# 系列研究備忘：Transformer 的 token 縮減與解析度還原——從 ViT token merging 到 pixel-space DiT 的 RTI 與 HiT

> 研究主管備忘（供後續互動教學設計者使用）。起點論文：RTI（arXiv 2608.29281，Paper Master No.217）與 HiT（arXiv 2611.22962，No.585）。
> 撰寫日期：2026-11-21。所有 arXiv 編號、標題與年份均已以 arXiv API（export.arxiv.org/api/query?id_list=…）核對；
> 全文已下載者置於 `research/papers/<arXiv id>.pdf/.txt`，卡片中以「全文」標示；僅讀摘要者標示「僅摘要」。
> 數字一律取自原文；未能核對者列於第 8 節。

---

## 1. 核心問題

### 1.1 為什麼 token 數量決定成本

一個標準 Transformer block 在序列長度 L、寬度 d、MLP 擴張比 m 時，前向 FLOPs（乘加計為 2）約為：

- QKV 與輸出投影：8·L·d²
- MLP（兩層，隱藏寬 m·d）：4·m·L·d²
- 注意力分數 QKᵀ 與加權 AV：4·L²·d

合計 C(L) ≈ (8 + 4m)·L·d² + 4·L²·d。取 m = 4，二次項與線性項之比為 4L²d / 24Ld² = **L / (6d)**。
本研究以此公式試算（d = 768）：L = 256 時二次項僅占 5.3%，L = 1024 占 18.2%，L = 4096 占 47.1%。
因此有兩點值得在教學中明示：

1. 在 DiT 常見的長度（256–1024）下，成本大致與 token 數呈線性關係，**每減少一半 token，block 成本大致減半**；注意力的 O(N²) 只在高解析度（數千 token 以上）才成為主項。這正是 ToMeSD／ToDo 主打 2048² 大圖時加速倍率最高的原因。
2. 實測牆鐘時間可能比 FLOPs 下降得更快。RTI 附錄 A（Fig. 10）指出：在 RTX 4090、batch 4 下，量測時間在每個預算下都低於解析 FLOPs 曲線，因為這些長度下的注意力部分受記憶體頻寬限制。

### 1.2 為什麼 pixel-space DiT 的 token 特別多、也特別值得壓縮

- latent diffusion 先以 VAE 將影像壓縮（例如 8× 下採樣），再以 patch 2 切塊；pixel-space 模型必須直接面對像素格點。JiT（Li & He，arXiv 2511.13720，PM No.349）的作法是以大 patch（16 或 32）讓 token 數維持在可負擔範圍：256² 影像、patch 16 → 16×16 = 256 個 token；RTI 使用的 MiniT2I-B/16 在 512²、patch 16 下為 32×32 = **N = 1024** 個 token（RTI §3.1）。
- RTI 的論點（§1）：latent 模型在 VAE 中「一次付清」低階空間冗餘，pixel-space 模型則「在每一個 block 都付一次」。RTI 以凍結的 MiniT2I-B/16 探測（Fig. 2）：細節集中於少數 patch（細節最多的 15% patch 承載一半細節、前 50% 承載 88%）；把 1024 個 token 匯聚為 256 個區域後，中段 block 的表徵保留比例隨取樣進行而上升；以深度而言，保留比例在首個 block 為 61%，中段約 75%，往最後 block 再下降（§3.2）。
- HiT 的論點不同（§1）：pixel-space 生成器缺少 latent 模型在編碼器預訓練中獲得的「視覺階層」；以一 patch 一 token 貫穿整個骨幹，不會鼓勵模型把相關內容在更粗的層級上加以總結。因此 HiT 視 token 分組為**品質與訓練效率**的手段，而不只是推論加速。

### 1.3 每個方法都必須在以下設計軸上做選擇

| 設計軸 | 選項 |
|---|---|
| A. 縮減什麼（reduce） | 丟棄（drop/prune/skip）；合併（merge，平均或加權平均）；匯聚至學得 latent（pool to learned queries）；切塊（chunk，沿序列切出區段後以代表向量表示）；僅縮減 K/V（ToDo） |
| B. 分組如何決定（grouping） | 固定（步幅、網格、pixel-unshuffle）；啟發式相似度（ToMe 二部配對、餘弦相似度）；重要性分數（注意力、CFG 差異、router 分數）；學得（router、affinity、boundary predictor）；無參數但內容自適應（RTI 的 Hilbert 切割） |
| C. 在哪裡縮減（where） | 每個 block 前合併、block 後還原（ToMeSD）；逐層累進（ToMe、DynamicViT）；僅中段（hourglass 的 prelude/core/coda）；多層巢狀（HDiT、HiT） |
| D. 如何還原解析度（restore） | 不需還原（分類只用 CLS）；unmerge 複製（ToMeSD）；broadcast（區域平均更新廣播回每個 patch）；殘差寫回（RTI 的 Write、ELIT 的 Write cross-attention）；skip connection（U-ViT、HDiT 的 lerp、HiT 的投影 skip）；注意力上採樣（Hourglass） |
| E. 訓練方式 | training-free（ToMe 現成模型、ToMeSD、ToDo、AT-EDM）；在預訓練模型上微調/適配（DiffCR、DyDiT、FlexiDiT、RTI 的 LoRA）；從頭訓練（Hourglass、HDiT、RIN、H-Net、HiT；ELIT 與 DC-DiT 兩者皆可） |
| F. 預算（budget） | 固定比例/固定 token 數；每層/每時間步不同比例（DiffCR、AT-EDM 的 DSAP）；批次平均比例、單張可變（H-Net、HiT）；彈性（elastic）：一個 checkpoint 支援多種預算（ELIT、DC-DiT、FlexiDiT、RTI） |
| G. 位置處理（position） | 不處理；proportional attention（ToMe：logit 加 log s）；以保留 token 的原始 2D 座標（DC-DiT）；平均 RoPE 相量（RTI）；硬/軟質心 RoPE 或學得 RoPE（HiT 附錄 B.1）；latent 無位置（Perceiver/RIN） |
| H. 擴散特有問題 | 輸出是稠密的（每個 patch 都需要預測）；時間步依賴（雜訊大時更冗餘：DiffCR、DC-DiT、RTI Fig. 2b）；CFG 的條件/無條件分支需一致（ToMeSD 固定隨機性）或可刻意不對稱（Lite-CFG、group guidance、Sparse Guidance）；每一步都要重新分組（RTI 每步重切、HiT 每步重組） |

---

## 2. 時間軸與分期

> 以下每期列出「面臨的問題 → 提出的解法 → 前期做不到、本期補上的能力」。

### 第 0 期（2020–2021）：序列縮短的兩個原型
- **問題**：長序列使 Transformer 成本過高；但語言建模等任務需要逐 token 輸出，縮短後必須還原。
- **解法 1：固定步幅的縮短＋上採樣**。Funnel-Transformer（2020）逐步壓縮隱藏序列並以解碼器還原逐 token 表徵；Hourglass（Nawrot et al. 2021/2022）系統比較下採樣/上採樣方式，提出 shorten → 中段處理 → upsample＋殘差的沙漏結構。
- **解法 2：學得 latent 瓶頸**。Perceiver IO（2021）以固定數量的 latent 經 cross-attention 讀入任意長度輸入，再以輸出查詢讀出，使成本對輸入/輸出長度呈線性。
- **新增能力**：證明「中段在短序列上運算、兩端保持全長」可行。**限制**：分組完全固定（步幅）或完全不具空間位置（latent），與內容無關。

### 第 1 期（2021–2022）：ViT 分類中的 token 縮減
- **問題**：ViT 對每個 patch 一視同仁，背景 patch 浪費計算。
- **解法**：剪枝（DynamicViT 以預測模組學習重要性並以注意力遮罩可微地剪枝；EViT 依 CLS 注意力保留重要 token 並融合其餘）；學得 token（TokenLearner）；合併（ToMe，以二部軟配對逐層合併相似 token，不需訓練即可使用）。
- **新增能力**：內容相依的縮減；ToMe 首次做到「現成模型、免訓練、接近剪枝的速度但精度更高」。
- **限制**：分類只需要 CLS token，被丟棄/合併的 token **不需要還原**；直接搬到稠密預測（擴散）時沒有還原機制。

### 第 2 期（2023–2024）：免訓練的擴散 token 合併
- **問題**：擴散模型需要對每個 token 預測雜訊（稠密輸出）；且同一張圖在 CFG 下跑兩個分支、在多個時間步間重複運算。
- **解法**：ToMeSD（2023）定義 unmerge（合併後的 token 複製回所有來源位置），並以「每個 2×2 區域隨機挑一個 dst、且整個 batch 共用隨機性」修正 CFG 不一致；ToDo（2024）放棄相似度計算，僅對 K/V 做最近鄰下採樣、保留全部 query，因此不需要 unmerge；AT-EDM（2024）以注意力圖上的 G-WPR 排序剪枝，並依去噪步調整剪枝預算；Importance-Based Token Merging（2024）以 CFG 提供的重要性分數保護高資訊 token。
- **新增能力**：首次把 token 縮減用在稠密生成上而不需訓練；開始顧及時間步與 CFG。
- **限制**：合併群組由外觀相似度決定，可能把畫面兩端的 patch 併在一起（RTI 稱之為失去連續性與位置）；比例需手調；在大幅壓縮時品質崩壞。

### 第 3 期（2022–2025）：學得的路由與架構層級的瓶頸
- **問題**：免訓練方法的比例是手調、與層/時間步無關；架構本身仍對每個 token 等量計算。
- **解法 A（在既有 DiT 上微調路由）**：Mixture-of-Depths（2024，語言模型）以 top-k router 讓部分 token 跳過 block；DyDiT（2024）以時間步相依寬度＋空間動態 token 微調 DiT；DiffCR（2024）在每層加 router，並讓每層、每時間步的壓縮比例可微地學出；FlexiDiT（2025）把預訓練 DiT 轉為可用多種計算預算的模型；Sparse Guidance（2026）處理 token-sparse 訓練模型對 CFG 反應不佳的問題。
- **解法 B（架構層級的瓶頸）**：RIN（2022）以 latent 承擔大部分全域注意力並以 read/write 路由；U-ViT（2022）以長 skip 連接淺層與深層；HDiT（2024）把 Hourglass 結構帶到 pixel-space 擴散，以 pixel-unshuffle 合併 2×2 token、低解析度層用全域注意力、高解析度層用鄰域注意力，成本對像素數呈 O(n)。
- **新增能力**：比例可學、可隨層與時間步變化；架構層級的縮減讓 pixel-space 高解析度訓練可行。
- **限制**：Skip 類方法「刪除」而非「總結」（RTI 量測 Token Skip 只保留其分組三分之一的表徵）；HDiT 的合併仍為固定 2×2；latent 類方法缺乏空間位置。

### 第 4 期（2023–2025）：pixel-space 擴散的回歸
- **問題**：latent 擴散依賴另外訓練的 VAE；pixel-space 直接建模則 token 數過多。
- **解法**：simple diffusion（2023）調整雜訊排程並以下採樣避免高解析度特徵圖；HDiT（2024）以沙漏結構達成 O(n)；PixelFlow（2025）以串接流（cascade flow）控制成本；JiT（2025）以 x-prediction 搭配大 patch（16/32）讓單純的 Transformer 在像素上有效。
- **新增能力**：去除 tokenizer 的端對端生成。**遺留問題**：大 patch 是一種「固定、與內容無關」的壓縮；T2I 512² 仍有 1024 個 token，且所有 token 在每層、每步成本相同——這是 RTI 與 HiT 的共同起點。

### 第 5 期（2023–2025）：內容自適應的切分與動態切塊
- **問題**：固定網格對平坦區與細節區分配相同 token 數。
- **解法**：Quadformer／mixed-resolution tokenization（2023）以 quadtree＋顯著性評分器產生大小不一的 patch 馬賽克；BLT（2024）以下一個 byte 的熵切分 byte 為動態長度 patch；H-Net（2025）以相鄰向量的餘弦相似度預測邊界、搭配 EMA 平滑與 ratio loss，端對端學出切塊。另有 FlexTok、ElasticTok 等可變長度 tokenizer（附帶提及）。
- **新增能力**：token 數隨內容變化；H-Net 證明切塊可與主任務聯合訓練。**限制**：Quadformer 需從頭訓練且限於分類；H-Net 為一維因果序列；tokenizer 類方法改變的是輸入表示，無法用於已釋出的 checkpoint（RTI §2 的批評）。

### 第 6 期（2026）：擴散 Transformer 中的彈性與自適應 token——RTI 與 HiT 所在的位置
- **ELIT（2026-03）**：在 DiT 中插入可變長度的 latent 介面，以輕量 Read/Write cross-attention 在空間 token 與 latent 間搬移資訊；訓練時隨機丟棄尾端 latent，使 latent 依重要性排序，推論時可任意調整數量。
- **DC-DiT（2026-03）**：把 H-Net 的動態切塊帶到 2D latent DiT，router 以局部可預測性給出邊界機率；以 tail dropping 實現彈性推論，可由預訓練 DiT upcycle。
- **RTI（2026-08）**：在凍結的 pixel-space T2I 骨幹上，以 Hilbert 曲線順序在最大特徵落差處切割，得到連續、大小隨內容變化、帶位置的區域 token；Read/Write＋LoRA，預算隨機抽樣，一個 checkpoint 對應任意預算。
- **HiT（2026-11）**：從頭訓練的 pixel-space 階層 DiT，以學得 affinity＋average linkage 漸進分組（可空間不連續），批次平均壓縮預算，文字與影像共同分組。
- **本期的核心張力**：「分組必須空間連續且帶位置」（RTI）對「分組可非局部、由模型自行學出」（HiT）；「改造既有模型以加速」（RTI）對「從頭訓練以提升品質」（HiT）。詳見第 5 節。

---

## 3. 逐篇卡片

> 格式：標題｜年份｜arXiv｜PM id｜來源程度。設計軸代號同 1.3 節（A 縮減、B 分組、C 位置層、D 還原、E 訓練、F 預算、G 位置編碼、H 擴散議題）。

### 3.1 Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing
- 2020｜arXiv 2006.03236｜PM：無｜**僅摘要**
- 問題：多數任務只需單一向量表示，維持逐 token 全長表示是冗餘。
- 機制：逐步把隱藏序列壓縮為較短序列；省下的 FLOPs 再投入加深或加寬模型；需要逐 token 預測時以解碼器從縮短序列還原深層 token 表徵。
- 設計軸：A 池化縮短｜B 固定｜C 逐段累進｜D 解碼器還原｜E 從頭訓練｜F 固定。
- 數字：摘要僅稱在相當或更少 FLOPs 下優於標準 Transformer（無具體數字）。
- 限制：分組與內容無關。

### 3.2 Hierarchical Transformers Are More Efficient Language Models（Hourglass）
- 2021（ACL Findings NAACL 2022）｜arXiv 2110.13711｜PM：無｜全文
- 問題：長序列 Transformer 成本高；作者主張明確的階層結構是有效處理長序列的關鍵。
- 機制：shorten（縮短係數 k）→ 中段 Transformer → upsample＋殘差 → 全長 Transformer。為避免自回歸資訊洩漏，縮短前再右移 k−1 個 token（§2）。比較上採樣方式：重複（Funnel 作法，無法區分組內位置）、線性上採樣（每個組內位置各自一個 d×d 投影）、注意力上採樣 x = U(x, x′) + Attention(Q = U(x, x′), K = V = x′)，其中 U(x, x′) = x + LinearUpsampling(x′)（§2.3）。
- 設計軸：A 線性池化｜B 固定步幅｜C 中段｜D 注意力上採樣＋殘差｜E 從頭｜F 固定｜G 不處理。
- 數字：ImageNet32 生成 3.741 bits/dim，為自回歸 Transformer 新 SOTA（§3.4，Tab. 3）；enwik8 在相同計算下優於 Transformer-XL 基線。
- 限制：固定步幅；因果設定。
- 系譜意義：RTI 明言其沙漏形狀即來自此類階層 Transformer（RTI §2），HDiT 直接採用 Hourglass 結構。

### 3.3 Perceiver IO: A General Architecture for Structured Inputs & Outputs
- 2021｜arXiv 2107.14795｜PM：無｜**僅摘要**（機制輔以 RTI 附錄 A 對其基線實作的描述）
- 問題：架構難以擴展到大型輸入與輸出。
- 機制：以固定大小 latent 陣列經 cross-attention 讀入輸入，在 latent 上做自注意力，再以任意輸出查詢讀出；成本對輸入、輸出大小呈線性。
- 設計軸：A 匯聚至 latent｜B 學得（全域 cross-attention，無硬分組）｜C 全程｜D 輸出查詢讀出｜E 從頭｜F 固定｜G latent 無位置。
- 數字（RTI 重實作，Tab. 1）：在凍結 MiniT2I-B/16 上以 R = 256 latent 作為「Latent Array」基線，GenEval 僅 45.7（RTI 為 84.7）；RTI 量測其讀取注意力的空間散布為 11.35 格（整張圖為 12.24），表徵保留 −0.08（Tab. 4）。
- 限制：用於凍結骨幹時缺乏位置，RoPE 注意力讀不到位置；RTI 推估移除位置的代價達 39.0 GenEval 點（§4.2）。

### 3.4 DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification
- 2021｜arXiv 2106.02034｜PM：無｜**僅摘要**
- 問題：ViT 的最終預測只依賴少數資訊量大的 token。
- 機制：輕量預測模組依當前特徵估計 token 重要性，插入多層以階層式剪枝；以注意力遮罩阻斷被剪 token 與其他 token 的互動，使剪枝可端對端微分。
- 設計軸：A 丟棄｜B 學得重要性｜C 逐層累進｜D 不需還原（分類）｜E 訓練｜F 固定保留比例。
- 數字：階層式剪去 66% 輸入 token，FLOPs 降低 31%–37%，吞吐量提升 40% 以上，精度下降 0.5% 以內（摘要）。
- 限制：剪掉的 token 不再存在，無法用於稠密輸出。同期 EViT（arXiv 2202.07800）以 CLS 注意力保留重要 token、融合其餘 token（附帶提及）。

### 3.5 Token Merging: Your ViT But Faster（ToMe）
- 2022（ICLR 2023）｜arXiv 2210.09461｜PM：無｜全文
- 問題：剪枝會丟失資訊；聚類演算法太慢、需迭代。
- 機制（§3，二部軟配對 bipartite soft matching）：每個 block 內，(1) 把 token 交替分為 A、B 兩集合；(2) 每個 A token 連一條邊到 B 中最相似者（以 key 的餘弦相似度）；(3) 保留最相似的 r 條邊；(4) 將相連 token 合併（以 token 大小加權平均）；(5) 串回。每層減少 r 個 token。合併後以 **proportional attention**：A = softmax(QKᵀ/√d + log s)，s 為每個 token 代表的 patch 數，等同於把該 key 複製 s 份。
- 設計軸：A 合併｜B 啟發式相似度（key 餘弦）｜C 每層累進｜D 不需還原｜E 免訓練（亦可訓練中使用）｜F 每層固定 r｜G proportional attention。
- 數字：現成 ViT-L@512 與 ViT-H@518 吞吐量 2×、ViT-L 影片 2.2×，精度下降 0.2–0.3%（摘要）。
- 限制：分組依外觀，不顧空間位置；分類無需還原。
- 系譜意義：RTI 的「Feat Sim」基線與 HiT 的 log-size bias（附錄 B.1）皆來自此。

### 3.6 Token Merging for Fast Stable Diffusion（ToMeSD）
- 2023（CVPR Workshop）｜arXiv 2303.17604｜PM：無｜全文
- 問題：擴散需要每個 token 的輸出；ToMe 直接套用時品質大幅改變。
- 機制：在每個元件（self-attn、cross-attn、MLP）前合併、輸出後 **unmerge**——若 x*₁,₂ = (x₁ + x₂)/2，則 x′₁ = x′₂ = x*₁,₂（§3.1）；改以比例 r% 而非固定個數合併。src/dst 劃分：ToMe 的交替劃分會使 dst 形成規則欄位、等同沿列減半解析度；改為**每個 2×2 區域隨機挑一個 dst**，且必須**在整個 batch 固定隨機性**，使 CFG 的條件與無條件樣本以相同方式配置 dst（§4.1）。設計實驗：只對 self-attn 合併最划算；只套用在 token 最多的 block；早期步驟多合併、後期少合併有小幅改善（Tab. 3）。
- 設計軸：A 合併｜B 相似度＋隨機 2×2 dst｜C 每 block 前合併後還原｜D unmerge 複製｜E 免訓練｜F 固定比例（可隨步驟調整）｜H CFG 一致性、時間步。
- 數字：最多減少 60% token、最高 2× 速度、記憶體最多降低 5.6×（摘要）；SD v1.5、512²、2,000 張 ImageNet 類別圖：基線 FID 33.12（3.09 s/im），r = 50% 時 FID 33.02（Tab. 4）。剪枝（以 0 取代）則影像迅速崩壞（Fig. 4）。
- 限制：unmerge 為純複製，同組 token 完全相同；分組仍依相似度。

### 3.7 ToDo: Token Downsampling for Efficient Generation of High-Resolution Images
- 2024｜arXiv 2402.13573｜PM：無｜全文
- 問題：ToMe 的相似度矩陣為 O(n²)，merge–unmerge 循環在高合併率下損失細節。
- 機制：假設空間相鄰 token 相似，以最近鄰下採樣 D(·) 取代相似度配對（O(n)）；只對 key、value 下採樣，保留全部 query：Attention = softmax(Q·D(K)ᵀ/√d_k)·D(V)，因此輸出仍為全長，不需 unmerge（§2.2）。
- 設計軸：A 僅縮減 K/V｜B 固定網格｜C 注意力層｜D 不需還原（query 全長）｜E 免訓練｜F 固定比例（0.75、0.89）。
- 數字：一般尺寸加速至多 2×，2048×2048 等高解析度至多 4.5× 以上（摘要）。
- 限制：只省注意力（MLP 仍全長），在 N 不大時效益有限（見 1.1 節）。

### 3.8 Attention-Driven Training-Free Efficiency Enhancement of Diffusion Models（AT-EDM）
- 2024（CVPR）｜arXiv 2405.05252｜PM：無｜**僅摘要**
- 機制：以注意力圖執行時剪枝；單步內以 Generalized Weighted Page Rank（G-WPR）排序冗餘 token，並以相似度為基礎的恢復法為卷積層補回 token；Denoising-Steps-Aware Pruning（DSAP）依去噪步調整剪枝預算。
- 設計軸：A 剪枝＋相似度恢復｜B 注意力重要性｜D 相似度恢復｜E 免訓練｜F 隨時間步變化｜H 時間步依賴。
- 數字：未於摘要中讀到具體數字（DiffCR 以其為比較基線）。

### 3.9 Importance-Based Token Merging for Efficient Image and Video Generation
- 2024（ICCV 2025）｜arXiv 2411.16720｜PM：無｜**僅摘要**
- 機制：合併時優先保留高資訊 token；重要性分數取自現成訊號，例如擴散模型中的 classifier-free guidance。適用 Stable Diffusion、Zero123++、AnimateDiff 等。
- 設計軸：A 合併｜B 重要性（CFG）｜E 免訓練。
- 限制（RTI §2 的歸類）：仍依相似度或重要性分組，而非依 token 所在位置。

### 3.10 Mixture-of-Depths: Dynamically allocating compute in transformer-based language models
- 2024｜arXiv 2404.02258｜PM：無｜摘要＋RTI 對其基線之描述
- 機制：每層以 top-k router 決定哪 k 個 token 參與自注意力與 MLP，其餘沿殘差流跳過；k 事先固定，計算圖靜態、總計算可預測，但哪些 token 被處理是動態的。
- 設計軸：A 跳過（skip）｜B 學得 router｜C 交替層｜D 殘差流直通｜E 訓練｜F 固定容量。
- 數字：與基線在相同 FLOPs 與訓練牆鐘時間下表現相當，取樣時每步可快 50% 以上（摘要）。RTI 重實作的「Token Skip」（中段 block 稠密/稀疏交替，稀疏 block 只收前 12.5% 影像 token）：GenEval 82.3，但僅 1.19× 加速（Tab. 1）；只保留其分組三分之一的表徵（§4.2）；JiT 類別條件上 R = 64 時 RTI 以 62% token 計算得 16.69 FID，Token Skip 以 71% 得 19.11（附錄 B.2）。
- 既有筆記：`depth-redundancy-adjacent-methods` 已涵蓋 MoD。

### 3.11 Layer- and Timestep-Adaptive Differentiable Token Compression Ratios for Efficient Diffusion Transformers（DiffCR）
- 2024（CVPR 2025）｜arXiv 2412.16822｜PM：無｜全文（擷取）
- 機制：(1) 每個 DiT 層加輕量 router，與模型權重聯合微調以預測 token 重要性，取 top-k 處理，其餘 token 繞過整層（輸出乘以 router 分數，同 MoD）；(2) 層級可微壓縮比，自零初始化學出各層比例；(3) 時間步可微壓縮比（PixArt-Σ 20 步分為 4 區、每層 4 個可學比例）。學到的模式：雜訊越大的時間步比例越高。
- 設計軸：A 跳過｜B 學得 router｜C 每層｜D 殘差直通（快取）｜E 微調｜F 每層、每時間步學得｜H 時間步依賴。
- 數字：PixArt-Σ 上與未壓縮模型品質相當，同時節省 20.68% 延遲與 8.33% 記憶體（§4）。
- 限制：仍是「刪除」而非「總結」；需微調。同一作者群前作 DiffRate（arXiv 2305.17997）在 ViT 上讓壓縮比可微（附帶提及）。

### 3.12 Dynamic Diffusion Transformer（DyDiT）
- 2024｜arXiv 2410.03456｜PM：無｜**僅摘要**
- 機制：Timestep-wise Dynamic Width（依時間步調整模型寬度）＋Spatial-wise Dynamic Token（避免在不必要的空間位置計算）。
- 設計軸：A 跳過 token＋縮寬｜B 學得｜E 少量微調｜F 隨時間步。
- 數字：額外微調不到 3% 迭代，DiT-XL FLOPs 降低 51%、生成加速 1.73×，ImageNet FID 2.07（摘要）。DC-DiT 表示可與 DyDiT 疊加。

### 3.13 FlexiDiT: Your Diffusion Transformer Can Easily Generate High-Quality Samples with Less Compute
- 2025｜arXiv 2502.20126｜PM：無｜**僅摘要**
- 機制：將預訓練 DiT 轉為可在不同計算預算下處理輸入的「彈性」模型，打破每步固定計算的靜態範式（具體機制未讀全文，見第 8 節）。
- 數字：影像生成 FLOPs 降低 40% 以上而品質不降；影片至多省 75% 計算（摘要）。

### 3.14 Guiding Token-Sparse Diffusion Models（Sparse Guidance）
- 2026｜arXiv 2601.01608｜PM：無｜全文（摘要部分）
- 問題：以 token 稀疏訓練的擴散模型對 CFG 反應不佳。
- 機制：以 token 層級稀疏度（而非條件丟棄）作為引導訊號。
- 數字：ImageNet-256 FID 1.58，FLOPs 少 25%；在與基線同品質下至多省 58% FLOPs；並訓練 2.5B T2I 模型（摘要）。
- 系譜意義：HiT 的 group guidance 明言受其啟發（HiT 附錄 A.6）。

### 3.15 Scalable Adaptive Computation for Iterative Generation（RIN）
- 2022（ICML 2023）｜arXiv 2212.11972｜PM：無｜**僅摘要**
- 機制：Recurrent Interface Networks 把全域自注意力集中在一組 latent token，以 cross-attention 在 latent 與資料 token 之間讀寫（路由）；堆疊 RIN block 形成由下而上與由上而下的回饋；以前一步的 latent 進行 latent self-conditioning。
- 設計軸：A 匯聚至 latent｜B 學得（全域）｜D cross-attention 寫回｜E 從頭｜G latent 無位置。
- 系譜意義：ELIT 的 Read 層「依循 [RIN]」；RTI 的 Latent Array 基線即 Perceiver-IO／RIN 式。

### 3.16 All are Worth Words: A ViT Backbone for Diffusion Models（U-ViT）
- 2022（CVPR 2023）｜arXiv 2209.12152｜PM：無｜**僅摘要**
- 機制：把時間、條件、雜訊 patch 全部視為 token；淺層與深層之間以長 skip 連接。
- 數字：latent 擴散 ImageNet 256 FID 2.29；MS-COCO T2I FID 5.48（摘要）。結論之一：長 skip 至關重要，而下採樣/上採樣運算並非必要。
- 系譜意義：HiT 附錄 B.2 以 U-ViT 研究過的 skip 融合規則做消融；HDiT 改以可學 lerp。

### 3.17 Scalable High-Resolution Pixel-Space Image Synthesis with Hourglass Diffusion Transformers（HDiT）
- 2024（ICML 2024）｜arXiv 2401.11605｜PM：無｜全文
- 問題：DiT 對像素數 O(n²)，無法在 pixel space 直接訓練百萬像素影像。
- 機制：採用 Hourglass 結構；層數依主解析度決定，使最內層為 16×16 token；編碼端每層以 Pixel-UnShuffle 把 2×2 token 合併為一並投影，解碼端以 Pixel-Shuffle 分裂；skip 以可學係數線性插值合併：merged = f·x_skip + (1 − f)·x_upsampled；高解析度層用鄰域注意力、最內層用全域注意力，整體為 O(n)。
- 設計軸：A 合併（固定 2×2）｜B 固定｜C 多層巢狀｜D pixel-shuffle＋lerp skip｜E 從頭｜F 固定｜G 軸向 RoPE。
- 數字：FFHQ-1024² FID 5.23（85M 參數，50k 樣本；Tab. 3）；ImageNet-256 與現有模型競爭。
- 限制：分組與內容無關；每層 token 數固定。
- 系譜意義：pixel-space 沙漏的代表；ELIT 亦以 HDiT 為可插入的骨幹之一。注意：RTI 與 HiT **都未引用 HDiT**（第 8 節）。

### 3.18 One Model, Many Budgets: Elastic Latent Interfaces for Diffusion Transformers（ELIT）
- 2026｜arXiv 2603.12245｜PM：無｜全文（擷取）
- 問題：DiT 的每步 FLOPs 被解析度鎖死，且計算平均分配給所有空間 token。
- 機制：把 block 分為「短空間頭」→「latent 核心」→「空間尾」；Read：l_CA = l + CA(Q = l; K, V = s)，l_O = l_CA + MLP(l_CA)；Write 對稱。為降低成本採 **grouped cross-attention**：把空間 token 分為 G 個不重疊網格群組，latent 對應分群（每群 J = K/G 個）。訓練時每次迭代抽一個 J̃ 並丟棄各群尾端 latent，使 latent 依重要性排序；推論時任意選 J̃。主實驗用 16 個空間群組，512px 每群至多 64 個 latent，共 60 種推論預算；XL 模型 Read 在第 4 block、Write 在第 24 block。
- 設計軸：A 匯聚至 latent｜B 學得 cross-attention（群組為固定網格）｜C 中段｜D Write cross-attention｜E 從頭（亦插入 DiT、U-ViT、HDiT、MM-DiT/Qwen-Image）｜F 彈性（尾端丟棄）｜G 見第 8 節。
- 數字：ImageNet-1K 512px 上 FID 平均改善 35.3%、FDD 39.6%（摘要）。
- 系譜意義：RTI 稱其為「最接近」的工作，RTI 的 Read/Write 屬同一族，但把 latent 錨定在空間區域上。

### 3.19 Simple diffusion: End-to-end diffusion for high resolution images
- 2023（ICML 2023）｜arXiv 2301.11093｜PM：無｜**僅摘要**
- 四項發現：高解析度需調整雜訊排程；只需放大架構的特定部分；在特定位置加 dropout；**下採樣是避免高解析度特徵圖的有效策略**。
- 系譜意義：pixel-space 端對端擴散的前驅；HiT 將其列為 pixel-space 方法（[17]）。

### 3.20 Back to Basics: Let Denoising Generative Models Denoise（JiT）
- 2025｜arXiv 2511.13720｜**PM No.349**｜**僅摘要**＋兩篇起點論文的描述
- 機制：依流形假設主張直接預測乾淨資料（x-prediction）；以大 patch（16、32）的單純 Transformer 在像素上生成，無 tokenizer、無預訓練、無額外損失。
- 系譜意義：RTI 的 MiniT2I 為 JiT 家族 T2I 模型（RTI §3.1）；HiT 從 JiT 出發（RMSNorm、QK-norm、SwiGLU）並以 JiT 為主要基線。RTI 指出：因骨幹預測乾淨影像，其特徵追蹤影像結構，大特徵落差即標示內容邊界（§3.3）。
- 數字（HiT Tab. 1 轉引）：JiT-H/16 ImageNet-256 FID 1.86（600 epochs）、JiT-G/16 1.82。
- 既有筆記：`pixel-space-diffusion`、`x-prediction-manifold-view`。

### 3.21 Vision Transformers with Mixed-Resolution Tokenization（Quadformer）
- 2023（CVPR Workshop）｜arXiv 2304.00287｜PM：無｜全文（摘要）
- 機制：以 Quadtree 演算法加新的顯著性評分器，把影像切為大小不一的 patch 馬賽克，低顯著區以低解析度處理；架構同 vanilla ViT，加 2D 位置嵌入。
- 設計軸：A 輸入端合併｜B 啟發式顯著性＋quadtree｜C 輸入｜D 不需（分類）｜E 從頭｜G 2D 位置。
- 數字：控制計算預算下分類精度大幅提升（摘要無具體數字）。
- 系譜意義：RTI §3.3 指出 quadtree 把區域綁在二進位正方形且每步每張需建樹；Hilbert 曲線上**每個二進位（quadtree）格子恰為一段連續區間**，故切割 Hilbert 序列可表達任何 quadtree 切分以及 quadtree 達不到的切分。相關：QDM（arXiv 2503.12015）以 quadtree 做超解析擴散；NaViT（arXiv 2307.06304）原生解析度打包（附帶提及）。

### 3.22 Byte Latent Transformer: Patches Scale Better Than Tokens（BLT）
- 2024（ACL 2025）｜arXiv 2412.09871｜PM：無｜**僅摘要**
- 機制：依下一個 byte 的熵把 byte 切成動態長度 patch，資料可預測時用長 patch。
- 數字：首次在 FLOP 控制下擴展 byte 層級模型至 8B 參數、4T 訓練 byte（摘要）。
- 系譜意義：HiT §2 將 BLT 與 H-Net 並列為「沿序列形成自適應 chunk」的階層序列模型。

### 3.23 Dynamic Chunking for End-to-End Hierarchical Sequence Modeling（H-Net）
- 2025（ICLR 2026）｜arXiv 2507.07955｜PM：無｜全文（擷取）
- 問題：tokenization 是端對端語言模型的障礙；需要可學且與主任務聯合訓練的切塊。
- 機制：encoder → chunking → main network → dechunking → decoder，可遞迴多階段。Routing：q_t = W_q x̂_t、k_t = W_k x̂_t，p_t = ½(1 − cos(q_t, k_{t−1}))，b_t = 1{p_t ≥ 0.5}（p₁ = 1）；downsampler 直接選取邊界位置的向量。Smoothing：z̄_t = P_t ẑ_t + (1 − P_t) z̄_{t−1}（EMA，使低信心邊界被平滑）；upsampler 重複每個壓縮向量直到下一邊界，並以 STE 將信心值於前向取整。Ratio loss：L_ratio = N/(N−1)·((N−1)FG + (1−F)(1−G))，F 為實際選取比例（不可微）、G 為平均邊界機率（可微），F = G = 1/N 時達最小值 1。
- 設計軸：A 切塊（選代表）｜B 學得（相鄰相似度）｜C 多階段巢狀｜D 平滑＋重複上採樣＋殘差｜E 從頭｜F 平均比例（ratio loss）。
- 數字：一階段 byte 層級 H-Net 在計算與資料對齊下優於 BPE Transformer；兩階段可匹敵兩倍大的 token 模型；DNA 資料效率近 4×（摘要）。
- 系譜意義：HiT 明言受 H-Net 啟發（q/k 投影、ratio loss、巢狀 encoder/main/decoder）；DC-DiT 為其 2D 擴散版本；HiT 附錄 A.2 證明其 compression loss 與 H-Net ratio loss 代數等價（本研究已數值驗證）。

### 3.24 DC-DiT: Adaptive Compute and Elastic Inference for Visual Generation via Dynamic Chunking
- 2026｜arXiv 2603.06351｜PM：無｜全文（擷取）
- 機制：在 flattened latent 格點（相當於 patch 1）上，isotropic encoder 混合局部資訊 → router：投影至瓶頸 z，以 3×3 卷積預測器由鄰域估計 ẑ，殘差 r = z − ẑ 表示局部不可預測性，p_i = σ(f_score(r)_i) → 以 p_i > 0.5 硬遮罩選取邊界 token（STE）→ 加上依原始 2D 座標的正弦位置嵌入 → DiT 骨幹 → de-chunking 還原原格點 → 加上由邊界機率門控的 encoder 殘差 → decoder。彈性推論：對自然邊界集 B 依 p_i 排序，丟棄最低的 ρ|B| 個（tail dropping）；Lite-CFG 讓無條件分支丟更多。批次內以 sequence packing 處理不同長度。
- 設計軸：A 選取代表 token｜B 學得（局部可預測性）｜C 骨幹前｜D de-chunk＋門控殘差｜E 從頭或由 DiT upcycle｜F 彈性（ρ 為邊界集比例，非精確 token 數）｜G 原始 2D 座標正弦嵌入｜H 時間步自適應、Lite-CFG。
- 數字：ImageNet 類別條件上推論 FLOPs 至多降低 36.8%、FID 至多改善 37.8%（摘要）。
- 系譜意義：RTI 所稱最接近的工作之一；RTI 批評其 router 需學習、只能控制平均比例而非精確預算，且在已吸收冗餘的 VAE latent 上運作（RTI §2）。

### 3.25 Elastic Token Compression for Pixel-Space Diffusion Transformers（RTI）
- 2026｜arXiv 2608.29281｜**PM No.217**｜全文
- 問題：在不重新訓練的前提下，把已釋出的 pixel-space T2I 骨幹改造成可依任意預算加速的模型。
- 四項需求（§1，由 Fig. 2 探測導出）：分組必須**連續**、**大小隨內容**、**帶位置**、並**總結而非刪除**。
- 機制：
  - 只壓縮中段（§3.2）：MiniT2I-B/16 共 17 個雙流 block，核心為 [3, 13]（前後各 3 個全解析度 block）；L/16 為 23 個 block 中的 [3, 19]。文字流始終全長。
  - 區域劃分（§3.3）：π 為 Hilbert 曲線在 32×32 格點上誘導的順序；d_j = ‖h_{π(j+1)} − h_{π(j)}‖₂（j = 1…N−1），在 R−1 個最大 d_j 處切開，得到 R 段連續區間。無參數、O(N)，每個取樣步依當前 x_t 重切。
  - Read（Eq. 3）：α_j = softmax_{j∈c}(wᵀh_j)，z_c = Σ_{j∈c} α_j h_j + e_size[⌊log₂|c|⌋]，w 為單一可學向量；位置取區域內各 patch RoPE 相量的平均——相量平均在波長長於區域範圍時接近單位圓、短於區域範圍時趨近零，相當於依區域大小做低通。
  - Write（Eq. 4）：h_j^{(s₁+1)} = h_j^{(s₀)} + g([h_j^{(s₀)} ∥ z′_{c(j)} − z_{c(j)}])，g 為小型可學映射；同時構成繞過核心的 skip。
  - 訓練：骨幹凍結，只訓練 rank-32 LoRA（注意力與 MLP 投影）與 Read/Write，初始化為恆等；每步自 {64, 128, 256, 512} 抽 R；40K 步、約 120K 張對齊資料；B/16 可訓練參數 16.3M（5.9%）。
- 設計軸：A 加權池化｜B 無參數內容自適應（Hilbert 序上最大特徵落差）｜C 單層級中段核心｜D 殘差 Write｜E 凍結骨幹＋LoRA｜F 彈性、每張精確 R｜G 平均 RoPE 相量＋大小嵌入｜H 每步重切、文字流全長。
- 數字（Tab. 1，512²，GenEval，RTX 4090 牆鐘）：B/16 稠密 87.2；RTI R = 512 為 87.8（1.84×）、R = 256 為 84.7（2.15×），同預算 Feat Sim 為 84.7／80.5，Latent Array（R = 256）45.7，Token Skip 82.3（1.19×）。L/16 稠密 88.1；RTI R = 512 為 87.5（2.11×）、R = 256 為 86.3（2.59×）。R ≥ 384（2.00×）時與稠密差距落在雜訊內；加速在約 2.5× 飽和，R ≈ 192 以下省不到時間。未見過的預算 {96, 192, 384} 落在同一曲線（Fig. 8a）。消融：均勻切割代價 2.7（R = 128）；以 mean＋broadcast 取代 Read/Write 代價 7.2（§4.3）。分組散布：Feat Sim 5.16 格、RTI 0.49 格（Tab. 4）。MJHQ-30K FID：RTI 比 Feat Sim 好 8.6（R = 512）與 13.3（R = 256），但 R = 512 比稠密差 2.4（附錄 B.1）。JiT-L/16 類別條件 R = 193 FID 2.42（稠密 2.36）。Hilbert 對 raster 在 R = 64 PickScore −0.268（附錄 B.4）。
- 限制（§5）：加速有上限（prelude、coda、文字流不縮）；核心無法區分區域內 patch；並非每種分組都是 Hilbert 區間；預算由使用者而非內容決定。

### 3.26 Hierarchical Diffusion Transformer for End-to-End Image Generation（HiT）
- 2026｜arXiv 2611.22962｜**PM No.585**｜全文
- 問題：pixel-space 擴散在品質與訓練效率上落後兩階段 latent 模型；假設部分原因是缺乏可漸進總結內容的視覺階層。
- 機制：
  - 巢狀骨幹（§3.1）：N → K₁ → … → K_J → … → K₁ → N；預設兩個 2× 階段（ImageNet 256、p16：256 → 128 → 64），18 個 block 配置 3/3/6/3/3；各層級有 router 與 skip 投影 W_ℓ（初始化為恆等）。
  - 硬分組（§3.2）：q_i、k_i 為兩個學得投影並正規化，S_ij = ½(q_iᵀk_j + q_jᵀk_i)；自單一 token 起，反覆合併平均 affinity 最高的兩群（average linkage），保留 affinity ≥ τ 的合併，得 K 群；群內 token 可空間相距甚遠。
  - 軟權重（附錄 A.1）：e_ig = (1/|G_g|)Σ_{j∈G_g} S_ij，A_ig = softmax_g(e_ig/T_r)；P_ig = A_ig / Σ_j A_jg；Z = PᵀH，H̃ = A·M(Z) + H W_ℓᵀ。反向傳播時硬分割視為常數，梯度經軟權重回到 affinity 投影。
  - 壓縮損失（Eq. 4）：L_comp = 1 + (f_h − ρ)(f_s − ρ)/(ρ(1 − ρ))，ρ = 1/r；f_h 為整個 batch（跨裝置）總群數 / 總 token 數，f_s 以 sigmoid((s_{b,m} − τ)/T_b) 取代每次合併的門檻判定所得之可微估計；L = L_diff + λ L_comp，λ = 0.01、T_r = 0.05、T_b = 0.1；門檻 τ 在訓練前以 1,024 個雜訊樣本依序校準後凍結。
  - 條件：移除 adaLN，改以 8 個 class token＋1 個 time token 自第一個 block 輸入，不參與分組（附錄 A.4）。
  - T2I（附錄 A.5）：外層保留影像/文字各自的投影與 MLP；外層 encoder 之後，把影像與有效 caption 特徵串接後共同分組，一個群可同時含影像與文字；padding 與 time token 排除。
  - Group guidance（附錄 A.6）：v̂ = v_∅^{(r_u)} + s(v_c^{(r_c)} − v_∅^{(r_u)})，r_c ≥ 1 ≥ r_u 為群數倍率（透過調整門檻實現），不需額外網路評估；T2I 用 r_c = (1.2, 1.2)、r_u = (0.8, 1)、s = 6.5。
- 設計軸：A 軟加權池化｜B 學得 affinity＋階層聚類＋門檻｜C 多層巢狀（兩階段）｜D 軟混合讀回 A·M(Z)＋投影 skip｜E 從頭｜F 批次平均比例，單張/單步可變｜G 預設 compact 層不加 RoPE（全解析度保留 RoPE）｜H 每步重組、文字–影像共同分組、group guidance。
- 數字：ImageNet-256 FID 1.48（HiT-XL/16，600 epochs）；280 epochs 1.87，接近 JiT-H/16 600 epochs 的 1.86（Tab. 1）；FID ≈ 2.16 時訓練計算少 2.4×、更新步數少 2.3×（Fig. 1d）。T2I 512²：平均 65.93，MiniT2I-B/16 64.64，GFLOPs 139.4 對 201.6（減 31%），UniLiP-3B 65.76（Tab. 2）。消融（HiT-B/16，80 epochs）：只在硬群內池化 10.16–10.32、硬查表讀回 7.97、預設 6.61（Tab. 3）；2×2 為 6.61、4×2 為 9.54、單階段 4× 為 9.59（Tab. 4）；學得分組 6.61 對固定空間配對 7.11（GFLOPs 22.2 對 22.1，Tab. 7）；學得 RoPE＋per-group MLP 使主模型 6.61 → 6.25（Tab. 9）；HiT-B 22.2 GFLOPs/6.61 對 JiT-B 25.2/10.37（Tab. 8）。
- 限制（本研究整理）：GFLOPs 為以目標群數計的名目值，實際隨圖與步變化；average linkage 的計算成本未報告；以群數倍率提高推論群數不保證變好（附錄 A.6）。

---

## 4. 設計軸對照表

| 方法 | 年 | A 縮減 | B 分組 | C 位置層 | D 還原 | E 訓練 | F 預算 | G 位置編碼 | 空間連續 |
|---|---|---|---|---|---|---|---|---|---|
| Funnel | 2020 | 池化 | 固定 | 累進 | 解碼器 | 從頭 | 固定 | — | 是（相鄰） |
| Hourglass | 2021 | 線性池化 | 固定步幅 | 中段 | 注意力上採樣＋殘差 | 從頭 | 固定 | — | 是 |
| Perceiver IO | 2021 | 匯聚 latent | 學得全域 | 全程 | 輸出查詢 | 從頭 | 固定 | latent 無 | 否 |
| DynamicViT | 2021 | 剪枝 | 學得重要性 | 累進 | 不需 | 訓練 | 固定比例 | 原位置 | 否 |
| ToMe | 2022 | 合併 | key 相似度二部配對 | 每層累進 | 不需 | 免訓練 | 每層 r | log s | 否 |
| RIN | 2022 | 匯聚 latent | 學得全域 | 全程 | cross-attn 寫回 | 從頭 | 固定 | latent 無 | 否 |
| U-ViT | 2022 | 不縮減 | — | — | 長 skip | 從頭 | — | — | — |
| ToMeSD | 2023 | 合併 | 相似度＋隨機 2×2 dst | 每 block | unmerge 複製 | 免訓練 | 固定比例 | — | 否 |
| simple diffusion | 2023 | 下採樣 | 固定 | 架構 | U-Net 上採樣 | 從頭 | 固定 | — | 是 |
| Quadformer | 2023 | 輸入端大 patch | 顯著性＋quadtree | 輸入 | 不需 | 從頭 | 固定 | 2D 嵌入 | 是（正方形） |
| HDiT | 2024 | 合併 2×2 | 固定 | 多層巢狀 | pixel-shuffle＋lerp | 從頭 | 固定 | 軸向 RoPE | 是 |
| ToDo | 2024 | 僅 K/V | 固定網格 | 注意力 | 不需 | 免訓練 | 固定 | 原 Q | 是 |
| AT-EDM | 2024 | 剪枝 | 注意力 G-WPR | 每層 | 相似度恢復 | 免訓練 | 隨時間步 | — | 否 |
| IBTM | 2024 | 合併 | CFG 重要性 | — | unmerge | 免訓練 | 固定 | — | 否 |
| MoD | 2024 | 跳過 | 學得 router | 交替層 | 殘差直通 | 訓練 | 固定容量 | 原位置 | 否 |
| DyDiT | 2024 | 跳過＋縮寬 | 學得 | 每層 | 直通 | 微調 | 隨時間步 | 原位置 | 否 |
| DiffCR | 2024 | 跳過 | 學得 router | 每層 | 直通 | 微調 | 學得逐層/逐步 | 原位置 | 否 |
| BLT | 2024 | 切塊 | 熵 | 階層 | 解碼器 | 從頭 | 平均 | 序列 | 是（1D） |
| H-Net | 2025 | 切塊選代表 | 學得相鄰相似度 | 多階段 | EMA 平滑＋重複＋殘差 | 從頭 | ratio loss 平均 | 序列 | 是（1D） |
| JiT | 2025 | 大 patch | 固定 | 輸入 | 線性頭 | 從頭 | 固定 | RoPE | 是 |
| FlexiDiT | 2025 | （未核對） | — | — | — | 微調 | 彈性 | — | — |
| Sparse Guidance | 2026 | 稀疏 token | （訓練時稀疏） | — | — | 從頭 | 推論可調 | — | — |
| ELIT | 2026 | 匯聚 latent | 學得 cross-attn，固定網格群 | 中段 | Write cross-attn | 從頭/插入 | 彈性尾端丟棄 | （未核對） | 群內否、群間是 |
| DC-DiT | 2026 | 選代表 | 學得局部可預測性 | 骨幹前 | de-chunk＋門控殘差 | 從頭/upcycle | 彈性 ρ\|B\| | 2D 正弦 | 否（選點） |
| **RTI** | 2026 | 加權池化 | 無參數 Hilbert 最大落差 | 單層中段 | 殘差 Write g(h, Δz) | 凍結＋LoRA | 彈性、精確 R | 平均 RoPE 相量＋大小 | **是（保證）** |
| **HiT** | 2026 | 軟加權池化 | 學得 affinity＋average linkage | 兩層巢狀 | A·M(Z)＋投影 skip | 從頭 | 批次平均、單張可變 | 預設無（可選學得 RoPE） | **否（允許非局部）** |

---

## 5. RTI 與 HiT 的深入比較

### 5.1 在系譜中的位置
- 兩者共同繼承：沙漏/中段壓縮（Hourglass → HDiT；兩端全解析度、中段短序列）；pixel-space JiT 家族骨幹（x-prediction、patch 16）；MiniT2I 雙流 T2I 設定（RTI 直接改造 MiniT2I；HiT 以 MiniT2I 為 T2I 基線）；ToMe 的合併思想（RTI 作為 Feat Sim 基線，HiT 採 log-size bias 作為位置變體）。
- RTI 獨有的承襲：Perceiver-IO／RIN／ELIT 的 Read/Write 介面；遞迴深度模型的 prelude/core/coda 用語（Geiping et al.，arXiv 2502.05171）；點雲 Transformer 以 Hilbert 序列化（Point Transformer V3，arXiv 2312.10035）——RTI 強調 PTv3 用曲線排序注意力序列，RTI 則用它**定義分組本身**；Mixture-of-Depths 作為 Token Skip 基線。
- HiT 獨有的承襲：H-Net 的 q/k 路由投影、巢狀 encoder/main/decoder、ratio loss；GroupViT（arXiv 2202.11094）的階層語意分組；U-ViT 的 skip 融合；Sparse Guidance → group guidance。

### 5.2 逐軸差異
1. **目的**：RTI 為推論加速（凍結骨幹、2×–2.6× 牆鐘）；HiT 為品質與訓練效率（從頭訓練、FID 1.48、訓練計算約 40%）。
2. **分組**：RTI 無參數、每張精確 R 個、保證空間連續（Hilbert 區間），大小隨特徵落差；HiT 學得、群數由門檻決定而隨圖與時間步變化，允許空間不連續（例如兩根香蕉同群）。
3. **池化**：RTI 只在區域內做單查詢 softmax；HiT 每群對**所有** token 做軟加權（Tab. 3：限制在硬群內 FID 由 6.61 惡化至 10.16–10.32）。
4. **還原**：RTI 寫回的是區域 token 的**變化量** Δz = z′ − z，並以每個 patch 自身的前置特徵個別化（g）；HiT 以同一組軟權重把處理後的群特徵混回（A·M(Z)），再加投影 skip。兩者都保留核心前特徵，差別在於 RTI 的核心輸出被定義為「修正」，HiT 則為「重建＋skip」。RTI 消融中 mean＋broadcast 代價 7.2 GenEval；HiT 消融中硬查表讀回 FID 7.97 對軟混合 6.61——兩者都顯示「每個位置共用同一更新」的代價。
5. **位置**：RTI 證明位置不可或缺（Latent Array 缺位置 −39.0 GenEval），使用凍結 RoPE 能直接讀取的平均相量；HiT 預設在 compact 層**不加** RoPE，而學得 RoPE＋MLP 只把 FID 從 6.61 改善為 6.25。可能解釋（推論，未經實驗驗證）：凍結骨幹無法學習新的位置機制，必須提供它看得懂的位置；從頭訓練的模型可經由 skip 與全解析度層取得位置資訊。
6. **預算**：RTI 訓練時隨機抽 R，推論可設任意 R（含未見過者），是使用者可控的連續旋鈕；HiT 控制批次平均，單張內容決定群數，推論時以門檻倍率調整（group guidance 對條件/無條件分支不對稱）。
7. **層級**：RTI 一個層級（1024 → R → 1024）；HiT 兩個層級（256 → 128 → 64 → 128 → 256），三階段未見改善（附錄 B.4，6.66 對 6.61）。
8. **多模態**：RTI 文字流全長、不壓縮（因此成為加速上限的一部分）；HiT 文字與影像共同分組，湧現詞–區域關聯。
9. **共同發現**：兩端應保持全解析度。RTI Fig. 2c 顯示首末 block 對每個 patch 敏感；HiT Tab. 5 顯示把 block 從核心移到外層（2/2/10/2/2 → 3/3/6/3/3）使 FID 由 8.48 改善為 6.61。

### 5.3 表面矛盾：分組必須空間連續嗎？
- RTI：相似度合併使群組散布 5.16 格、在 R = 256 時 GenEval 比 RTI 低 4.2 點，結論是「失去連續性即失去位置」。
- HiT：學得（可非局部）分組優於固定空間配對（6.61 對 7.11）。
- 兩個比較的條件不同：RTI 的 Feat Sim 是**免學習**的相似度分組，在**凍結**、依賴 RoPE 的骨幹上；HiT 的對照組是**形狀固定**（水平再垂直配對）而非內容自適應的連續分組，且 HiT 的軟池化讓每個 token 都能貢獻給每個群。**尚無研究在同一設定下比較「內容自適應但連續」與「學得且非局部」兩種分組**——這是教學中可以明確點出的開放問題。

### 5.4 開放問題
1. 同一設定下比較 Hilbert 連續區域與學得非局部分組（從頭訓練與凍結改造兩種情境各一）。
2. 精確預算與內容自適應預算的結合：RTI 的 R 由使用者決定；HiT/H-Net/DC-DiT 的群數由內容決定但只控制平均。
3. 文字流壓縮：RTI 的加速上限部分來自文字流全長；HiT 的共同分組是否能移植到改造式方法。
4. 位置編碼：合併 token 的最佳位置表示（平均相量、質心、學得衰減 exp(−γω²v_g/2)）仍無定論，HiT 各變體差距小。
5. 與步數縮減的關係：RTI 附錄 B.3 顯示每張影像約 3 秒以下時 token 壓縮優於少步取樣，5 秒以上則相反；與蒸餾少步模型結合尚未測試。
6. 階層聚類的成本：HiT 的 average linkage 在 N 增大時的實際開銷未報告。
7. 2D 動態切塊的「區段」定義：H-Net 在 1D 以「到下一個邊界為止」定義 chunk，DC-DiT 在 2D 選取邊界 token；RTI 以 Hilbert 序把 2D 化為 1D，可視為把 H-Net 式切塊搬到 2D 的一種無參數解。

---

## 6. 可做成互動元件的機制（皆可於瀏覽器精確計算）

### W1. 成本模型：FLOPs 對 N，含稠密 prelude/coda（Amdahl）
- 公式：C_block(L) = (8 + 4m)·L·d² + 4·L²·d；二次/線性比 = L/(6d)（m = 4）。
- 中段壓縮加速：S(R) = [n_tot·C(N+T)] / [(n_pre + n_coda)·C(N+T) + n_core·C(R+T)]，T 為文字 token 數。
- 本研究試算（d = 768、n_pre = n_coda = 3、n_core = 11、N = 1024）：T = 0 時 S(512) = 1.55、S(256) = 2.03、S(64) = 2.59、上限 S(→0) = 2.83；若 T = 256（假設值），S(256) = 1.73、上限 2.17。RTI 實測 R = 256 為 2.15×、飽和約 2.5×，高於 FLOPs 模型，原因見 1.1 節第 2 點。元件可讓使用者調 d、N、T、核心長度，並疊上 RTI Tab. 1 的實測點，顯示 FLOPs 與牆鐘的差距。
- 試算程式：`research/cost.py`（成本模型與 ratio loss 等價驗證）、`research/hilb.py`（Hilbert 性質與 Dirichlet 核驗證）；設計者可直接轉為 JS。

### W2. ToMe 二部軟配對（玩具 token）
- 輸入：2D 平面上的 n 個 token（特徵 = 顏色或向量），交替分 A/B；每個 A 連到最相似的 B（餘弦），取前 r 條邊，依大小 s 加權平均合併，s 相加。
- 顯示 proportional attention：softmax(QKᵀ/√d + log s) 與「把 key 複製 s 份」的注意力完全一致（可在元件中並列驗證）。
- 延伸：切換 ToMeSD 的 dst 劃分（交替欄位 / 2×2 步幅 / 完全隨機 / 每 2×2 隨機一個），顯示交替劃分在 50% 合併時等同沿列減半解析度。

### W3. Hilbert 序區域劃分（RTI §3.3）
- Hilbert d → (x, y)（n 為 2 的冪）：
  ```
  x = y = 0; t = d
  for s = 1, 2, 4, …, n/2:
      rx = 1 & (t / 2); ry = 1 & (t XOR rx)
      if ry == 0: { if rx == 1: x = s-1-x; y = s-1-y }; swap(x, y)
      x += s·rx; y += s·ry; t /= 4
  ```
  本研究已驗證（32×32）：相鄰序號必為格點鄰居；每個對齊的 4×4 二進位格子恰為一段長 16 的連續區間。
- 劃分：d_j = ‖h_{π(j+1)} − h_{π(j)}‖，取 R−1 個最大 d_j 為切點（線性選擇即可）。對照：均勻切點（RTI 的 fixed boundaries）、raster 序、Morton 序（RTI 附錄 B.4：raster 在列尾斷開、Morton 在二進位區塊邊界跳躍）。
- 評量：EV = 1 − Σ_j‖h_j − h̄_{c(j)}‖² / Σ_j‖h_j − h̄‖²（RTI Eq. 5）；散布 spread_r = Σ_j w_rj‖p_j − p_r‖₂（Eq. 6）。可用一張真實或合成影像的 patch 顏色作為 h，讓使用者拖動 R 觀察平坦區合併、細節區細分。

### W4. 平均 RoPE 相量的低通效果
- 1D 長度 L 的連續區間、頻率 ω：|(1/L) Σ_{p=0}^{L−1} e^{iωp}| = |sin(Lω/2) / (L·sin(ω/2))|（Dirichlet 核；本研究已數值驗證）。
- 元件：橫軸頻率、縱軸平均相量長度，滑桿調區域長度，顯示「區域越大，高頻位置資訊越被抹除」，對應 RTI §3.4 的論述；可再對照 HiT 學得 RoPE 的衰減 exp(−γω²v_g/2)（附錄 B.1）。

### W5. 還原誤差：unmerge/broadcast 對殘差寫回
- 設理想的逐 patch 更新為 Δ_j。廣播（同區所有 patch 共用一個更新 u_c）的最小誤差在 u_c = 區內平均時取得，為 Σ_c Σ_{j∈c}‖Δ_j − Δ̄_c‖²（即理想更新的區內變異）。
- 殘差寫回 h_j + g([h_j ∥ Δz_c])：若 g 能利用 h_j 預測區內偏差，可低於上述下界。元件可用線性 g（g = Δz_c + B·(h_j − h̄_c)，B 以最小平方擬合）展示誤差下降；對照 RTI 消融（mean＋broadcast −7.2 GenEval）與 HiT 消融（硬查表 7.97 對軟混合 6.61）。

### W6. HiT 的 average linkage＋門檻＋軟權重
- 輸入：少量 token 的單位向量 q、k；S_ij = ½(q_iᵀk_j + q_jᵀk_i)；群間 affinity s(U, V) = (1/|U||V|) Σ_{i∈U} Σ_{j∈V} S_ij；反覆合併最大者直到最大值 < τ。
- 軟權重：e_ig 為 token 對群成員的平均 affinity，A_ig = softmax_g(e_ig/T_r)，P 為 A 欄正規化。滑桿調 τ、T_r，顯示群數 K 與非局部成群（可複製 HiT Fig. 3 的 6-token 範例結構，數值須自行設定，原圖數值為示意）。
- 壓縮損失曲面：L = 1 + (f_h − ρ)(f_s − ρ)/(ρ(1 − ρ))，與 H-Net 形式 r/(r−1)·[(r−1)f_h f_s + (1−f_h)(1−f_s)] 等價（本研究已數值驗證）。

### W7. H-Net 1D 動態切塊
- p_t = ½(1 − cos(q_t, k_{t−1}))、b_t = 1{p_t ≥ 0.5}；EMA 平滑 z̄_t = P_t ẑ_t + (1 − P_t) z̄_{t−1}；上採樣重複至下一邊界。
- 元件：一列可編輯向量（或字元），顯示邊界、壓縮比 F 與 G、ratio loss 值；並與 W3 並列，說明 RTI 等於「2D → Hilbert 1D → 依相鄰落差切塊」，但以精確 R 取代門檻 0.5。

### W8. 預算對品質/速度曲線（資料點直接取自論文）
- RTI Tab. 1：B/16（R, 加速, GenEval）＝（1024, 1.00, 87.2）、（512, 1.84, 87.8）、（256, 2.15, 84.7）；L/16＝（1024, 1.00, 88.1）、（512, 2.11, 87.5）、（256, 2.59, 86.3）；Feat Sim B/16（512, 1.84, 84.7）、（256, 2.15, 80.5）。
- RTI 附錄 B.3：同成本下少步取樣對 R = 256（1.8 s：GenEval 65.9 對 82.7；11.2 s：87.6 對 86.3）。
- HiT Tab. 4：（比例, GFLOPs, FID）＝（4×, 19.1, 9.59）、（2×4, 20.8, 6.89）、（4×2, 17.8, 9.54）、（4×4, 17.1, 10.78）、（2×2, 22.2, 6.61）。
- 元件：可切換「彈性 checkpoint 一條曲線」對「固定預算多個點」，顯示 RTI Fig. 8a 的未見預算也落在曲線上。

### W9. Group guidance／Lite-CFG 的不對稱預算
- v̂ = v_∅^{(r_u)} + s(v_c^{(r_c)} − v_∅^{(r_u)})；在 2D 玩具向量場上示意「細分支 − 粗分支」作為額外引導方向。屬概念示意，無法重現論文數字。

---

## 7. 既有資料庫論文與筆記

### 7.1 papers.jsonl 中的相關論文
| PM id | 論文 | 關係 |
|---|---|---|
| 217 | Elastic Token Compression for Pixel-Space Diffusion Transformers（RTI） | 起點（資料庫中無 arXiv 連結，url 為專案頁） |
| 585 | Hierarchical Diffusion Transformer for End-to-End Image Generation（HiT） | 起點 |
| 349 | Back to Basics: Let Denoising Generative Models Denoise（JiT） | 兩篇共同骨幹家族 |
| 232 | Rethinking the Design Space of Vanilla Pixel Diffusion Transformers | 相鄰（pixel DiT 設計空間；資料庫 related 列有 217、585、349）；未讀內容 |
| 50 | V-CoLA: Vision Token Compression with Linear Attention（arXiv 2610.11251） | 相鄰（VLM 視覺 token 壓縮）；未讀內容 |

其餘 24 篇（ToMe、ToMeSD、HDiT、H-Net、ELIT、DC-DiT 等）皆**不在**資料庫中（以 arXiv id 與標題關鍵字比對）。

### 7.2 既有研究筆記（slug）
- `pixel-space-diffusion`：pixel-space 陣營（JiT、PixelDiT、PixelGen、EPG）與 latent 必要性爭論——本系列第 4 期直接連結。
- `x-prediction-manifold-view`、`prediction-targets-overview`：JiT 的 x-prediction 論點。
- `aniso-pixel-patchified-setting`：patchified pixel transformer（D = 16×16×3）設定。
- `depth-redundancy-adjacent-methods`：Mixture-of-Depths 等深度/逐 token 計算分配。
- `moe-conditional-computation-lineage`：條件式計算脈絡（MoD 的上位概念）。
- `latent-diffusion-rae-overview`、`latent-diffusion-open-problems`、`representation-autoencoder-rae`：latent 端的對照。
- `generative-eval-metrics`：FID/GenEval 等評估。
- 未發現任何既有筆記涵蓋 token merging、hourglass/HDiT、H-Net 動態切塊、Hilbert 劃分或彈性預算。

### 7.3 建議新增的研究筆記（slug 前綴 `tokc-`）
1. `tokc-lineage-hub`：系列總覽與分期（本備忘第 1、2、4 節的精簡版）。
2. `tokc-vit-pruning-and-merging`：DynamicViT、EViT、TokenLearner、ToMe、DiffRate；二部軟配對與 proportional attention。
3. `tokc-training-free-diffusion-merging`：ToMeSD、ToDo、AT-EDM、Importance-Based Token Merging；unmerge、CFG 一致性、時間步排程。
4. `tokc-learned-routing-dynamic-dit`：DyDiT、DiffCR、FlexiDiT、Sparse Guidance（MoD 連結既有 `depth-redundancy-adjacent-methods`，不重寫）。
5. `tokc-hourglass-and-latent-interfaces`：Funnel、Hourglass、HDiT、Perceiver IO、RIN、ELIT；prelude/core/coda 與 Read/Write。
6. `tokc-dynamic-chunking-bytes-to-pixels`：BLT、H-Net、DC-DiT、HiT 的路由與 ratio loss 系譜。
7. `tokc-content-adaptive-tokenization`：Quadformer、NaViT、QDM、FlexTok、ElasticTok 與 Hilbert/Morton/raster 序列化。
8. `tokc-position-for-merged-tokens`：log-size bias、平均 RoPE 相量、硬/軟質心、學得 RoPE 衰減、2D 正弦座標。
9. `tokc-rti-region-token-interface`：RTI 機制、四項需求、基線對照與消融。
10. `tokc-hit-learned-hierarchy`：HiT 分組、壓縮損失、文字–影像共同分組、group guidance。

---

## 8. 未確認事項

1. **MiniT2I**（Wang, Zhao, Lu, Zhou, Ma, He 2026）在 arXiv 以標題搜尋無結果；兩篇起點論文引用時皆無 arXiv 編號。其架構細節（文字 token 數、MLP 形式）僅依 RTI/HiT 描述；W1 中 T = 256 為依 HiT「FLAN-T5-Large 序列長 256」所作的假設，實際有效文字 token 可能較少。
2. **FlexiDiT** 的具體機制（是否以改變 patch 大小達成彈性）未讀全文，對照表中標為未核對。
3. **ELIT** 的 latent 是否帶位置嵌入未核對；RTI 稱其 read 為全域查詢 O(NR)，但 ELIT 全文顯示採固定網格的 grouped cross-attention（16 個空間群組），兩者描述存在落差，教學時應以 ELIT 原文為準，並把 RTI 與 ELIT 的差異表述為「內容自適應且逐區域帶位置」對「固定網格群組」。
4. **DC-DiT** 的 de-chunking 在 2D 上如何把被丟棄位置補回（例如取最近代表或平滑）未讀到細節。
5. **AT-EDM、IBTM、DyDiT、RIN、Perceiver IO、Funnel、U-ViT、simple diffusion、BLT、JiT** 僅讀摘要（JiT 另有兩篇起點論文的轉述）；其數字以摘要為限。
6. **HDiT** 未被 RTI 或 HiT 引用，是本研究依主題補入；ELIT 把 HDiT 作為可插入的骨幹之一。
7. **HiT 的實際（非名目）FLOPs** 與 average linkage 開銷未報告；HiT Tab. 2 中 HiT-B/16 的 T2I 結果使用 D27（27 block）checkpoint 或 Tab. 2 所列參數 0.25B，文中 Tab. 2 與附錄 C/D 的 D27 描述是否為同一模型未完全釐清。
8. RTI 正文 Fig. 2c 敘述「首個 block 61%」而附錄 Fig. 12 敘述 JiT 家族保留比例 0.61–0.78，兩者量測對象不同（MiniT2I 對 JiT），引用時須標明模型。
9. W1 的 FLOPs 模型為簡化估算（未計 adaLN、雙流權重、文字流 MLP 分離、Read/Write 開銷；RTI 稱後者 < 2%），僅供教學示意，不可當作論文數字。
10. ToMeSD 的 s/im 數值在本研究擷取的表格中不完整（只讀到基線 3.09 s/im 與 FID 欄），引用加速倍率時以摘要「最高 2×」為準。
