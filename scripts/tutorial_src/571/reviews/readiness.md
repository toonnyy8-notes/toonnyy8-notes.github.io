NO-GO

## 理解複述（explain-back）

頁面的論證鏈：離線 RL 只能用固定資料（S1）；TD 的三個問題，C2 示範 α=1 需 H 次備份、極限 γ^(H−1)（S2）；以 returns-to-go R̂_t = Σ r 取代獎勵，序列為 (R̂, s, a) 交錯（S3，Eq. 2）；3K 個 token、每時間步一個嵌入、因果遮罩，s_t 位置預測 a_t（S4）；訓練為交叉熵／MSE，測試時 R̂ ← R̂ − r 並保留最後 K 步（S5）；圖上最短路徑與拼接（S6）；Atari 1% 與 D4RL 數字（S7、S8）；%BC、脈絡長度、信用分配（S9–S11）。

元件：C1 切換序列、模式、t、K、G，觀察 RTG 遞減、視窗與 token 數（訓練 3m、s_t 看到 3m−1）。C2 調 H、γ、α、n，觀察 V₁ 與已傳到狀態數，播放至 n†。C3 換 seed、資料量、生成方式、起點、L₀，觀察生成路徑、拼接與直方圖。C4a 輸入原始分數算正規化（預設 267.5）；C4b 切換表格並檢視重算差異。

## 阻擋項（blocking）

1. **符號先用後定義（G）。** 步驟 3（S3）寫「常數 $G-\hat R_1^{\text{train}}$」，但 §1.4 符號表將 $G$ 定義在 S5（「S5 | $G$ | 測試時給定的目標報酬」）。違反 §3 的「每個符號在首次使用的章節定義」。修法：S3 句子改寫為不用 G，或把定義移到 S3。

2. **符號與術語先用後定義（Hero）。** S0 的 SVG 顯示 $\hat R_1,s_1,a_1$，事實卡 3 寫「D4RL 平均（不含 Reacher）DT 74.7、CQL 63.9」。R̂ 於 S3、CQL 於 S2、D4RL 與 Reacher 於 S7 才定義。同樣違反 §3 規則。修法：hero 加「定義見 S3」等註記，或移除這些符號。

3. **C3 拼接判定自相矛盾。** 步驟 7 寫「不是前 $N_{\text{traj}}$ 條訓練軌跡中任何一條的連續子序列（全部軌跡都比對；…）」。「前 N_traj 條」與「全部軌跡都比對」兩種讀法在 N < 1000 時結果不同。verify/graph.py 的 `stitched` 只比對當次 N 條軌跡，graph_out.txt 中 NTRAJ 50/100/200/500 的拼接數（1/5/5/2）即依此得出，與 §5 一致。因此應刪去「全部軌跡都比對」，明文寫「前 N_traj 條」。

## 其他已核對項目（通過）

- 12 個學習目標皆有參考答案，且各自的教學章節（S1–S11）內文有對應內容。
- 9 個連結 slug 全部存在：571/notes 下 7 篇（dt-lineage-overview、dt-method-and-results、dt-offline-rl-baselines-datasets、dt-transformer-lm-background、dt-transformers-in-rl-and-trajectory-transformer、return-conditioned-sequence-modeling、dt-credit-assignment-lineage）；tut/568/notes 下 rl-mdp-q-learning-basics、offline-rl-distribution-shift。
- 上述 9 篇 frontmatter 皆含 slug、title、cluster、kind、tags、arxiv、sources、related。
- §5 數值抽查：C1 表（三組序列的 RTG、G、差值）與視窗數字手算一致；C2 的 0.747172、6.96e-10、0.367905、n†=59／199／399 與閉式解一致；C3 的 dist、直方圖（4,2,3,3,3,3,1 / 19）、拼接數、固定目標序列（起點 6：2,2,3,3,5,6,7,8,9,10）與 graph_out.txt 一致；mulberry32 的 JS 寫法與標準實作等價。
- timeline.json 的 12 個 arXiv ID 皆被 TL_GROUP 覆蓋。

未核對：C4 各表的原始數字未逐格與 paper.txt 比對（只驗算 Table 12→1、13→4 的正規化）。

## 非阻擋建議（up to 8）

1. timeline.json 位於 notes/ 內，但是資料檔而非 .md 筆記，缺 slug、cluster、kind、tags、sources、related。若檢查者把它算作 note，這一項會成為阻擋項。
2. C1 的 K 滑桿註記「論文 Atari 用 30、Gym 用 20」漏了 Pong 的 50（S4、步驟 10 有寫）。
3. S8 標題「多數任務分數最高」無法從設計文件驗證（只有平均值，沒有逐列數字）。
4. S11 標題「DT 幾乎不受影響」與 Table 7 Medium 的 −6.9 不完全相符。
5. C4b 的 Table 2、3、5、6、7 數字只給出出處行號，建置者需照 paper.txt 抄錄；§5 驗收未涵蓋這些格。
6. S4 的 6×6 遮罩矩陣未指明 t，應寫明 t=2（s_t = token 5）。
7. 符號重用：R 同時是獎勵函數、報酬 R_t、R̂；n 在 Eq. 1 是序列長度，在 C2 是備份次數；S2 閉式解中的 Bin、Pr 未定義。
8. LO9 參考答案「4 款中 3 款與 CQL 相當」與 Breakout 267.5 vs 211.1 的實際差距不符，建議改為「Breakout 較高，Pong、Seaquest 相當，Qbert 較低」。
