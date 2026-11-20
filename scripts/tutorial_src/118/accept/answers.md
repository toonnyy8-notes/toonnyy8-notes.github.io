## Goldfish 驗收作答（僅依 page.txt）

**LO1**：生成器對同一題抽 N 份解答，驗證器打分後選最高分一份。驗證器完全無用（分數與對錯無關）時，等於隨機挑一份，結果等於 Pass@1；驗證器完美時，只要 N 份中有一份正確就會被選到，結果等於 Pass@N。Pass@N = 1−(1−p)^N 只在各題 p 相同時適用（頁面 C1 假設）。
段落：「驗證器分數與正確性無關時，選最高分等於隨機挑一份」
信心：high

**LO2**：Best-of-N 解題率 = 挑選後測試集上答對的題目比率，評估「排序後能否選到正確解」；RM accuracy = 單份解答被分類為正確或錯誤的準確率，評估「逐點分類」。兩者相關但不同，不能直接比大小。頁面示範值（p=0.30、μ=1.5）：RM accuracy 77.3%，Best-of-16 為 84.7%，Best-of-1 為 30.0%（這些是頁面自訂玩具模型的數字）。
段落：「RM accuracy 只看單份解答是否被正確分類」
信心：high

**LO3**：判別式 RM 的分數 r_θ = sigmoid(z_cls)，即特殊 token cls 的 logit 過 sigmoid；損失為 Eq. 3 的 BCE：L = −E_correct log r(x,y⁺) − E_incorrect log(1−r(x,y⁻))，正確與錯誤資料各半。說它沒用到生成能力，是因為它只在 cls 的 logit 上輸出一個數字、不產生文字，所以不能先寫 CoT；Yang et al. 的輔助生成損失放在另一個頭上，訓練後丟棄。
段落：「它只在一個專用輸出（cls 的 logit）上給一個數字」
信心：high

**LO4**：分數 r_Direct(x,y) = p_θ(Yes | x, y, I)，I = 「Is the answer correct (Yes/No)?」，即下一個 token 是 Yes 的機率（Eq. 4）。訓練用 Eq. 2 的 SFT：正確解的目標是 Yes，錯誤解的目標是 No，即最大化 log p(Yes|·) 與 log p(No|·)。與判別式 RM 的關係（頁面「本頁推導」，非論文結論）：若其他 token 機率可忽略，p(Yes) ≈ sigmoid(z_Y − z_N)，正確解的損失 −log p(Yes) 形式就與 Eq. 3 的 BCE 相同；但論文沒有量測此近似，且兩者參數位置不同（cls logit vs. LM 頭的 Yes/No token）。
段落：「若其他 token 的機率可忽略，p(Yes)≈sigmoid」
信心：medium（分數與訓練為 high；數學關係為近似推導）

**LO5**：Eq. 5：L_GenRM(θ, D_verify) = L_SFT(θ, D_verify) + λ · L_SFT(θ, D_correct)。D_verify 是驗證資料（如 D_Direct、D_CoT），D_correct 是「題目 → 正確解」的生成資料，λ > 0 控制混合比例。預設 λ：演算法任務 1/3、GSM8K 1/4（App. D）。λ 過大的後果：解題資料過多會降低驗證表現（Fig. D.3，Gemma-7B GSM8K，λ 中間值最好）。
段落：「解題資料過多會降低驗證表現」
信心：high

**LO6**：GenRM-CoT 推論時依序做兩步：(1) 在題目與解答後接提示 I_CoT = 「Let's verify step by step.」，生成一段驗證理由 v_CoT（頁面例子用貪婪解碼）；(2) 再接上 I（「Is the answer correct (Yes/No)?」），以下一個 token 是 Yes 的機率為分數（Eq. 6）。
段落：「先在題目與解答後接「Let's verify step by step.」」
信心：high

**LO7**：Eq. 7 平均的是 K 條抽樣驗證理由各自的 Yes 機率（不是票數）：r_MajV@K = (1/K) Σ p(Yes | ..., v^(i)_CoT, I)。預設 K = 32（除非另註）。判別式 RM 不能這樣做，因為對同一輸入它只有一個確定的分數，沒有可以抽樣的推理路徑。
段落：「判別式 RM 對同一個輸入只有一個確定的分數」
信心：high

**LO8**：GSM8K 的訓練理由由 Gemini 1.0 Pro 產生：提示中附上一份得到正確答案的參考解答（Expected Answer），請它逐步批改待驗解答（Table A.2）。過濾方式：答案檢查器（字串比對或 SymPy）判定與理由最終 Yes/No 一致才保留；且只取「超過 50% 的驗證理由與答案檢查器一致」的解答的理由（App. C）。參考解答在推論時不用；它只用於產生訓練資料，微調與推論都不放入。
段落：「參考解答只用於產生訓練理由」
信心：high

**LO9**：三個 Best-of-N 主結果（頁面摘要）：演算法任務 5.0% → 45.3%；GSM8K 73.0% → 93.4%；MATH 28.0% → 44.6%；MMLU abstract algebra 37.9% → 53.5%。「MATH 多解 17%」的單位是百分點（絕對差），不是相對百分比：44.6 − 28.0 = 16.6 個百分點。
段落：「這個 17% 是百分點，不是相對百分比」
信心：medium（單位為 high；演算法任務起點 5.0% 與 Fig. 1 說明的 25.0% 不一致，見卡點 2）

**LO10**：模型變大時（Gemma 2B → 7B → 9B，MATH），GenRM、GenRM-CoT 與判別式 RM 都上升，生成式驗證器在每個大小都較好（Fig. 12）。推論運算增加（K 增加）時分數更高，微調過的 Gemma GenRM-CoT 勝過用同樣 CoT 與票數、但模型更強的 Gemini 1.0 Pro LLM-as-a-Judge（Fig. 11）。MMLU college math（Best-of-32）：判別式 RM 53.0%，GenRM-CoT 56.1%（Fig. 7）。
段落：「生成式驗證器在每個大小都較好」
信心：high

**LO11**：加權 SC：每份解答依驗證分數加權，對每個最終答案加總分數，選總和最大的答案。App. C 的修改：每個答案只加總分數最高的 K_top 份解答（GSM8K 用 K_top = 6，MATH 用 K_top = 4）。結果：GSM8K 幾乎沒有增益，16 份解答時只從 93.4% 到 93.5%（Fig. D.4，作者認為已飽和）；MATH 上 GenRM-CoT 加權的 SC 比判別式 RM 加權的 SC 少用 2.5 倍的解答達到相同表現（Fig. 8）。
段落：「只加總每個答案分數最高的 K_top 份解答」
信心：high

**LO12**：訓練 GenRM-CoT 需要三類資料：(1) 正確、錯誤各半的「題目–解答」配對與 Yes/No 標記（D_Direct）；(2) 每份解答的驗證理由（GSM8K 為參考解答引導合成再過濾，D_CoT）；(3) 「題目 → 正確解」的生成資料（Eq. 5）。與判別式 RM 的三項不同：(1) 分數來自語言模型頭的 Yes token，而非 cls 的 logit；(2) 損失是 next-token SFT，並與 λ 倍的解題資料混合；(3) 推論時可先生成理由，並平均 K 條理由。
段落：「分數來自語言模型頭的 Yes token」
信心：high

---

## 常見卡點（最多 5 項）

1. **三個 K 與 λ 容易混淆**：Eq. 7 的 K（理由數，預設 32）、weighted SC 的 K_top（GSM8K 6、MATH 4）、Eq. 5 的 λ（混合比例，演算法 1/3、GSM8K 1/4）是三個不同的量。
2. **Best-of-N 起點數字不一致**：演算法任務在摘要寫 5.0%，但 Fig. 1 說明寫 Best-of-32 為 25.0%；頁面也寫 Fig. 1 箭頭沒標起點，只能「推定」為 Pass@1。初學者無法判斷哪個是對的。
3. **頁面上的互動數字不是論文結果**：C1–C4 的 p、μ、q_e、h、K_top 與候選分數都是「示範值」或頁面自訂玩具模型，例如 Best-of-16 84.7% 與 RM accuracy 77.3% 不是論文數字，容易被誤讀成論文結果。
4. **Eq. 4 的機率是整個詞彙表的 softmax**：不是只在 Yes 與 No 之間正規化；頁面也說明評分時 γ = 1 是「推定」。初學者可能把 Yes 與 No 的兩個機率直接相除。
5. **「多數決」其實是機率平均，且判別式與 GenRM 的關係只是近似**：Eq. 7 平均的是 Yes 機率而不是票數；LO4 的 sigmoid 近似頁面也說「論文沒有量測」。讀者若沒注意到「本頁推導」標記，會把推導當成論文結論。
