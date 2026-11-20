1. P0 W3 / kappaSched：「rescale」「cap」只由 MDM 的 σ_max（式 (21)(27)(28)）定義，toy 卻提供「均勻」先驗搭配 rescale（5.3 表有數值）。USDM 下 σ、σ_max 無定義，開發者無法實作。修正：寫出均勻先驗的 kappaSched（直接把 κ=1−η/(1−α_s) 當定義並標「推定」），或停用均勻先驗的 rescale。
2. P0 S8 一節塞入 Duality、𝒯、式 (9)、τ、order statistics、Algo. 1、case 1/2、Floyd，只有 450–550 字，且 Gaussian latent、argmax、softmax 溫度、Φ⁻¹、截斷高斯都沒有先講。修正：拆成「Duality 與 softmax 稀疏性」「top-k 抽樣」兩節，或把 Floyd 與 𝒯 移到延伸；先寫為何 K 維向量不可行（記憶體）再給解法。
3. P1 S7 要在 500–600 字內放 2 表、FID、Table 9 限制、四點讀法、Takeaway 1–4；Gen. PPL、unigram entropy、nucleus p、log-linear/cosine schedule、class dropout 都未教。修正：S7 前加一段指標定義，或拆成語言／影像兩節；LO10 的 105.75 需有承載它的內文位置。
4. P1 Hero 事實卡與 κ 軸（0.95、0.99）在 S1 之前使用 ReMDM、[MASK] 先驗、FID、Gen. PPL、κ、corrector。修正：hero 卡片只用 S1 之前已定義的詞，κ 軸推薦值移到 S5/S7 才標。
5. P1 S2 在 400–500 字內放式 (2)、(3)（(3) 含 ⊙、⟨·,·⟩、Cat）、統一 Bayes 形式、x_θ、ancestral、NFE、自我修正，是全頁第一個難度陡升。q_s、q_{s|t}、α_{t|s} 在此首次出現，符號表在 S1 卻不含 q。修正：S1 補 Cat、⊙、⟨⟩、q_s 的定義；(3) 改放 details，內文只講統一形式。
6. P1 S5 同時放 ReMDM、FB/DFM、三種 schedule、κ<0、NELBO，沒有元件，且 LO8 的換算只在內文。NELBO 與取樣無關，會打斷主線。修正：NELBO 移到 S12；W2 的 σ 讀數擴成 κ↔σ 對照（含 κ<0 區段），讓 LO8 有互動支撐。
7. P1 W1 的「保留率」、W4a 的「重複 400 次」、W4b 的 τ 掃描未對應內文結論；LO5 第 (iv) 條件（κ 不依賴 z_t）無任何元件。修正：每個讀數在 S3/S8/S9 內文各用一句連到 LO，並刪掉沒對到 LO 的讀數。
8. P1 風格違規：「錯誤被鎖住」「補上未抽樣的項」「把 token 放回雜訊」「讓之後的步驟重新預測」「去噪器越準／上下文越正確」以及 S9 標題「補上」帶比喻。修正：改為「已解開的 token 不再更新」「加入未抽樣項的期望值」「以機率 (1−κ)(1−α_s) 重新抽樣 token」。「Floyd」無解釋。
9. P1 W4 的 K 滑桿範圍 10²–10⁵ 但 W4b 只允許 ≤50257、暴力法 ≤10⁴，面板會空白。log 滑桿的取值（整數？步進？）與 chip 50257 如何對應也沒寫。修正：滑桿上限 50257（另加 10⁵ 只供 W4a），寫明 K=round(10^v)，v 步進 0.05，chip 直接設 v=log10(50257)。
10. P1 KaTeX 與 SVG：熱圖、κ 軸、圖例、軸標（q_s、κ_t、ε、T）在 SVG 內的數學無規格。KaTeX auto-render 不處理 SVG 文字，動態重繪也不會重跑。修正：SVG 標籤用純 Unicode（κ_t、q_s）；需要 TeX 的放 HTML 疊層並在每次 render 後對該容器呼叫 renderMath。aria-label 勿含 TeX。
11. P1 暗色模式：`.tok-bad` 為「白底」硬編碼；熱圖深淺字以 p>0.55 判斷，暗色主題下表面色相反會判錯。修正：hatch 底用 var(--surface)；字色用 color-mix 結果的亮度或兩個 token 決定，並寫明暗色驗收。
12. P1 W3 軌跡圖 2048 個 SVG rect 加 hover；「←／→ 逐列」未指定焦點載體。修正：用單一 canvas 或每列一個 `<g>`，圖外容器 tabindex=0＋aria-activedescendant 或 roving tabindex；選列結果寫入固定 readout，不用 hover tooltip。
13. P2 W3 主執行緒：每個 T 段約 1e6–5e6 次運算且 psiTheta 每位置配置陣列，T=512 段可能凍結數百毫秒。修正：預配置 Float64Array 緩衝；T=256、512 再各拆成每 8 次 run 一個 setTimeout；加取消機制（按參數變動即中止）。
14. P2 W1 滑桿互動：α_t 推動 α_s 的規則只寫一向；α_s 下限隨 α_t 變動時 value 與 min 的更新順序未定。聯動滑桿（W3 的 t_off<t_on）同樣未寫被推動者與提示。修正：統一寫「改一者時，被限制者以 clamp 更新並 aria-live 一次」。
15. P2 W2 的預設 USDM `全錯` 先驗 x_θ[A]=0，P(A)=0.0352 小；「應觀察到」(1) 要求比較 κ=1 與 0.9 的 0.0352→0.0417，差異極小，初學者看不出。修正：預設 κ=0.5（0.0676），或讀數用並排「κ=1／目前 κ」兩欄；刻度另標下限 (1−κ)(1−α_s)/K。
