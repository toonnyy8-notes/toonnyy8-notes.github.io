# Round 1 review（fidelity F + pedagogy P 合併、去重）

驗算腳本 c1/c2/c3/c3_scan/paper_numbers 全部重跑，與 §5.1–5.3 一致；Eq. 8 括號（R7）已對照 PDF p.6，正確。Table 1–5、Fig. 5/7、10.98×、1.6×/3.1×、Eq. 4–8 轉錄均已核對無誤。

## P0（阻擋建構）
1. [P1] C3 η：三值滑桿語意不清 → 改成 3 選項 segmented control；重設時還原 `mulberry32(22)` 初始點；寫明「執行」到第 200 步是停止還是重來。
2. [P2] C3 reduced-motion 一次算完 200 步（約 2.6e7 次 kernel）會凍結主執行緒 → 兩種模式都用同一個 12ms 時間切片迴圈，reduced-motion 只是最後才畫。

## P1（錯誤或誤導）
3. [F1] Case 2（高斯）的 O(t²) 與 ‖∇a*‖<1 是論文 App. A 的**建模假設**，不是 Prop. 1 證出來的；1.4 step 4、LO5、C1 caption 標成假設，並註明 0.426 chip 不是「真實 h23→h7 非 Dirac」的證據。另，論文「Jacobian <1」你算出約 1.20，請在頁面如實呈現。
4. [F2] LO5「最大 Jacobian ∝ (1−t)/t² V(a|y)」與你自己的推導 (1−t)/t²·Δ²/4 不一致（K=4/8/16 時 t=0.05 的 J_max 為 72.8/17.3/4.28）。改寫為 J_max ∝ 模態間距²，論文的 Θ(V) 是模態分離時的漸近敘述。
5. [F3] Table 1 欄名「FID（無引導）」錯：論文是「without CFG」，RevFlow 列用了 REPA guidance，RAE 列沒有。欄名改「無 CFG」並加註；「1.96 對 2.16」說明是有 REPA 引導對無引導。
6. [F4] 重建比較有混淆：18.09 dB 是 DINOv2-B 最後一層的 RAE latent（App. A.1），RevFlow 是 DINOv3-L 第 7 層。step 1、hero、LO12 寫明 encoder 與層，並引用 Table 4 deep-only 19.87 dB（同 89M decoder）作為對照。
7. [F5 + P15] C2 時間嵌入：ω_j=(π/2)2^j 時 j=2、3 的 T_j(t+1)=T_j(t)，「兩段弧不重疊」只對前一兩組成立；論文沒有給 T 的形式。只畫第一對座標並標〔推定〕，或刪掉「不相交」觀察。C2 同時標 LO8、LO9，加一題 σ 的檢查題。
8. [F6] C2 σ：只顯示代價（corr 0.9→0.05），沒顯示好處（Table 4c：σ=0 FID 2.10 vs 1.96）；corr 來自隨機權重；論文只說擾動「訓練時」的 condition，未說取樣時是否加。LO9 與 caption 補上。
9. [P5] S4 太密且符號未教：score、Jacobian、Dirac、operator norm、條件平均／條件共變異數、score–noise identity。S4 開頭加先備段（字面定義），拆成 04a（Def. 1 + Prop. 1 敘述，補上 F11 的假設：E‖a‖²<∞、ε 與 (a,u) 獨立、t∈(0,1)、Jacobian 對 a_t 取）與 04b（兩種情形 + C1）；證明概要放 `<details>`（不要用 class="check"）。
10. [P6] S8：MMD 是什麼、IMQ kernel 公式、「單步輸出等於條件平均」都要寫在正文，C3 之前；不能只在 widget 裡。
11. [P7] 符號表放在 S2 最後，但 S1–S3 已用 z、c、f_θ、ℓ、h_ℓ、E_ℓ、c^ℓ、P_target → 移到 S2 開頭（或第一次使用處），S1 用到的先在 S1 定義。
12. [P3 + P4] C1 主執行緒：約 1.2e7 次 exp 同步計算 → 分段計算或降到 40 點／401 格；重算中右圖顯示舊曲線（變淡）。「0.445 倍」改寫為「t→0 時約 0.445 倍（t=0.10 為 0.45）」，並註明 0.426 是線性 R² 上界、0.958 是類別平均值、兩者在 1024 維 h7 上量而非 128 維 z^M（F10），所以是界的比值，不是實測比值。
13. [P8 + P9] C2：刪除未驗證的「|m|<10」宣稱，或在 selfTest 加全範圍斷言；mask 寫明 0-based r,c∈0..7、中心是否應為 3.5、token 數在 selfTest 斷言；E_s 索引與 Eq. 5 的 E_s(2) 統一說明。
14. [P10 + P11] 無障礙：C2 熱圖用 role="grid"（方向鍵、Home/End、Tab 離開）、色盲友善發散色階＋圖例、關閉數值時每格加 +/− 記號；每個 widget 只用一個 announce()，給定模板文字。C3 三群點要有不靠位置的標示。
15. [P12 + P13] KaTeX 不能進 SVG `<text>`：SVG 軸標用 Unicode 或 HTML overlay；動態 readout 改寫後呼叫 renderMath()。390px：寬公式（Eq. 5、8、10）包 `overflow-x:auto` + tabindex=0 容器；窄版 SVG 文字至少 12px。SVG hatch 與 --c-par/--c-perp 要有 dark mode 值。

## P2（不精確）
16. [P14] 文字風格：「剛性」先定義為「Jacobian 的最大值」後再用，或改寫成「最大 Jacobian 隨 t⁻² 增加」；「崩塌」→「輸出相同的條件平均」；「擠」→「集中在」；「階梯狀映射」改字面描述。
17. [F7] 塌縮推導：Eq. 9 在 t=1 是 0/0，改用 posterior-mean 形式；「對所有 ε 輸出同一個」只在 t=1 精確成立；與 FID≈280「相符」改為「一致，但論文未給原因」。
18. [F8] C3 玩具註記加一行：初始化 mean+0.05ε 提供了排斥梯度，真正塌縮的生成器輸出完全相同時 Σ∇k=0，必須學會把不同 ε 對到不同輸出；三群 40/40/40 是人為設定；論文還搭配 drifting 項與 FM 正則化。
19. [F9] VAE 結論恢復論文的保留語氣：「支持…；所測的 VAE 對照沒有受益」（一個 checkpoint、一種配方）。
20. [F12] 「Muon 與 AdamW，global batch 256」限定為 5-epoch 執行；80-epoch 設定論文未寫。
21. [F13] Eq. 10 期望跡 → 0（Dirac 目標）比論文更強，維持〔本頁推導〕標記，並說明「剛性」論證靠的是 sup-norm，不是 Eq. 10 第二行。
