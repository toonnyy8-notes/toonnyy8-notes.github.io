# No.405 審查第一輪（合併）

完整報告：reviews/fidelity.md（忠實度，15 項）、reviews/pedagogy.md（教學與實作，15 項）。以下為去重後的編號清單；P2 細項見兩份原報告，一併處理。

## P0
1. 研究筆記已完成：notes/ 內 hub `input-perturbation-lineage-overview` 與 7 篇 topic、notes/timeline.json。補上 S10、S11、Hero 的連結與時間軸，移除 6.6 的佔位。（pedagogy）
2. 6.1 的路徑改用 scripts/tutorial_src/405（不是 tut/405/）；外殼直接從 tutorials/prediction-targets-k-diff.html 複製，不依賴 238 的外部 CSS。（pedagogy）
3. ABAR 用截斷後 (1−β) 的累乘建，不用 cosine 閉式（閉式在 t=T 給 ᾱ=0）。（pedagogy）

## P1
4. LO4 與步驟 2：Tab. 1 的距離在完美網路下不是 0（論文自己提到 Gaussian 近似誤差；check.py 給 FD(δ=0)=4e-6），FID 還有有限樣本下限；100 步的 0.983 vs 0.972 不是效應。改寫參考答案與正文。（fidelity）
5. 等價資料不要叫「模糊」，是加性噪音。改步驟 6、LO10、C3 的用詞。（fidelity）
6. C3 觀察 1 誤導：在 x_t 空間擾動是固定的 (1+γ²) 噪音功率倍數，std 受 γ 約束；x0 空間的發散只來自除以很小的 √ᾱ。（fidelity）
7. 線性高斯斜率玩具分不出 IP 與 DDPM-y（斜率比約 1.004/0.997/0.995），不能支撐「平滑解釋 IP」；R4 宣稱 check.py 驗證過但其實沒有。刪掉或改成誠實的說法，若保留就在 check.py 補上計算。（fidelity）
8. x̂_0（由 Eq. 4 解出）與後驗平均形式在 S1 沒教，C1、S6 卻用到；S1、S6 太密，S6 拆開（Shapiro–Wilk、格點搜尋、加噪等式、玩具）。（pedagogy）
9. 定義先於使用：ν_t、e_t（S4 用、S6 定義）；Lipschitz 與 K（C3、LO10 用、S7 教）；Hero 卡片的 sFID、DDIM、η、FID、γ。（pedagogy）
10. S3 提前引用 Tab. 3；「最佳 T′ 為 100–300」與表格不符（最佳是 300，4/5 資料集；100 從未最佳）。C4 的「最佳 T′」欄要寫「格點內」，CelebA IP 的 1.27 在 T′=900。（pedagogy + fidelity）
11. KaTeX 不能放進 SVG `<text>`；動態插入的節點要呼叫 renderMath。（pedagogy）
12. C4 在 390px：對數軸上 80 與 100 重疊；Tab. 4 各資料集差 9 倍，需各自尺度；分頁的鍵盤操作要規定。（pedagogy）
13. Hero 把 CelebA FID 1.27 與 37.5% 訓練節省放在一起，兩者來自不同設定（Tab. 4：300K iter、200 步為 1.53）。「60–200 步達到」「全面下降」依據的差距只有 0.03–0.06，部分落在 R1 不一致之內，改寫得保守。（fidelity）

## P2（摘要）
14. C1 軸範圍、「下限」用詞相反、t₀=1 只剩一點；δ 一符號兩用；C3 的 0.2 參考線與 R3 矛盾。（pedagogy）
15. 「某個相容的 x0」不精確（鏈收斂到眾數或平均，t=1000 時 L1=E|x0|）；σ_t 標準差讀法與 respacing 重算 β 的說法要標 〔本頁推導〕；Tab. 9 的 540K 與 Tab. 4 的 480K/500K 列入 R1。（fidelity）
16. 擬人化用語、reduced-motion、深色模式顏色與斜線紋理、C2 的 hover/focus。（pedagogy）
