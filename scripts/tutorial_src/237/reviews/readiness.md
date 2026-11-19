NO-GO

## Explain-back

頁面從 Li et al. 的 Thm 3.1 出發：小 σ 時 log p_σ 的 σ⁻² 項只含到流形距離 d_M，資料密度 p_data 只在 Θ(1) 項出現（Eq. 6、7，S3）。由此得到 Thm 4.1 的兩個門檻：生成分數誤差 o(σ⁻²) 足以集中到流形，o(1) 才能還原 p_data（S4）。通用工具是 Gibbs 極限 exp(−f_θ/θ)（Thm B.1，S5）。TS Langevin 把分數乘 σ^α、θ = σ^(2−α)，在 max{−β,0} < α < 2 時極限為流形上對黎曼體積的均勻分佈（Thm 5.1，S6）；非梯度情形以 WKB 三階比對得到同一結論（Thm 5.2，S7）。逆問題只縮放先驗分數，得到均勻先驗的後驗（Thm 6.1，S8）。
元件：C1 以 ρ、φ、σ 顯示 Eq. 6 四項與分數的法向／切向分量（單位圓封閉式）。C2 選形狀、σ、α、誤差、β、λ，數值計算平穩分佈並給出 TV 與最大／最小比。C3 以 (β, α) 平面標出定理保證區。C4 以粒子模擬比較標準 Langevin 與 TS 的角度直方圖，誤差可選。

## Blocking

1. 符號先用後定義。S1 列「Eq. 1、3（VE／VP）」，Eq. 3 含 M、Φ(u)、u、p_data，但 Φ、M、p_data 的定義在 S2（符號表在 S2 末尾）。這違反頁面自訂的規則「每節所用符號都在該節或更早定義」（design.md 第 232 行）。
2. 驗收表缺容許誤差。5.3（C3）表格 `| (β, α) | 集中 | 還原 p_data（α = 0） | TS 均勻 | θ | w | σ^{α+β} | e^{2κσ^α}（漸近式） |` 上方只有「論文只證明充分條件；定理外一律標『不保證』」，沒有任何容許值。5.1、5.2、5.4 都有容許值，5.3 沒有。

## Checked and passing

- 12 個學習目標（LO1–LO12）都有參考答案，且對應的章節（S1–S9）都有內文。LO5 的數值（−1000.455、3.636、比值 275）與 C1 表一致。
- 設計中連結的 11 個 slug 都存在：9 個在 `tutorial_src/237/notes/`，`eps-prediction-ddpm-score` 與 `x-prediction-manifold-view` 在 `scripts/notes_split/`。
- 9 篇 `notes/*.md` 與上述兩篇的 frontmatter 都有 slug、title、cluster、kind、tags、arxiv、sources、related。
- `timeline.json` 目視檢查為 15 筆，欄位齊全；6.6 的 TL_GROUP 涵蓋全部 15 個 arXiv ID。
- Table 2 的「30 組比較只有一處 TS 不如 PC」已逐格核對，結論成立。C1 與 C3 的幾個數值抽查也一致。

## Non-blocking

1. S3 正文的 Eq. 6 含 H(x)，H 只在收合區塊的 Eq. 12 以 Ĥ 定義。LO4 的「H(x)（曲率）」在內文沒有定義。
2. S3 的單位圓封閉式用到 e_0，只說明了 R 在此定義。
3. S10 的虛擬碼用到 δ、ξ，正文沒有定義。
4. 5.2 的「≈ 10^83」列無法用「±0.5%」的最大／最小比容許值比較，建議註明此列只驗量級。
5. S6 的判定行「無誤差與 Θ(1) 誤差：α > 0 即在範圍內」對 Θ(1) 誤差需要選 β < 0 且接近 0。正文應寫明這個選取，否則讀者會誤以為 Θ(1) 對應 β = 0。
6. 兩篇 `notes_split` 筆記（`eps-prediction-ddpm-score`、`x-prediction-manifold-view`）不在設計指定的 `notes/` 目錄。建置時需確認來源路徑，避免漏掉。
7. Hero 副標使用「Langevin」與「α > −β」，但卡片已避開這些術語。副標可再簡化。
8. R13 要求建置前重讀 `notes/`。目前 `notes/` 有 9 篇，建置時須再核對 TL_GROUP 與 S12 延伸閱讀清單。
