# 255 設計文件審查（教學法＋建置）

1. **P0｜S5＋S6 的 Z_t 與 C3 錯位**。「S5 … Z_t 化簡〔本頁推導〕與兩個推論。LO6、LO7」、「單步面板」放在 S6。LO7 在 S5 以純公式教，驗證元件卻在下一章。問題：控制項先於概念，S5 又塞 Markov chain、平穩分佈、detailed balance、Def. 1、提議、Z_t，550–650 字放不下。修正：S5 拆成 5a（MH 與 Eq. 9）、5b（隨機重取與 Z_t）；單步面板移到 5b 後，成為獨立小元件 C3a；C3 只留 Alg. 1 整體分佈。
2. **P0｜Hero 與 S1 使用未教的詞**。事實卡 3 出現「Metropolis-Hastings」「GRPO」「token 成本 8.84 倍」；卡 1 已用 p(x)^α。問題：讀者尚未學到 MH、GRPO。修正：Hero 卡只用已定義的詞，MH、GRPO、8.84 改為「後文說明」。
3. **P1｜S1 過密**。「450–550 字」要涵蓋 token、next-token、Eq. 1、log-likelihood、溫度、RL 後訓練、GRPO、pass@k、銳化假說、LO1 答案段與符號表。修正：放寬到 700–800 字並拆 1a（序列機率與溫度）、1b（RL 背景與 pass@k）；pass@k 移到 S8 首次用到處再定義。
4. **P1｜S3 過密且有難度跳躍**。「600–700 字」含 Eq. 4–8、F 分解、相等條件、E3、Example 1、E2、C1、LO3、LO4。Eq. 7/8 的 $\sum_{x_{>t}}$ 與 F 先於數值例。修正：先 Example 1 手算（C1），再引入 Eq. 7/8 與 F；E3 移到 F 之後；拆成 3a、3b。
5. **P1｜S4 難度再跳一級**。Prop. 3 的 ε、ε′、N 與 Def. 2–4 在 400–500 字內，C2 有四個互相約束的滑桿。修正：C2 預設只露 ε、N，ε′ 與 α 收進「進階」；Eq. 14 先以 Example 1 代入再一般化。
6. **P1｜TV 未在內文教過**。符號表寫「只在 C3 使用」，內文沒定義。修正：S6 內文以一句話定義 TV 與其範圍 [0,1]，說明 C3 是精確分佈傳遞，不是抽樣。
7. **P1｜文字風格違反字面規則**。「偏好」（冪分佈偏好 token）、「困在低 likelihood 的未來」、「為未來高 likelihood 規劃」、「銳化」「尖」「勝出」「輸給」、「◀ 偏好」。這些是擬人或比喻。修正：首次出現時定義「偏好＝該 token 得到較大的相對權重」並全頁統一；「困在…」「規劃」只出現在「作者認為」且改寫成「後續序列的 likelihood 較低」；「銳化」附定義後使用，「輸給／勝出」改「較低／較高」。
8. **P1｜x_0 與 T 的符號衝突**。Example 1 的 x_0 是第一個生成 token；C3 的 x_0 是固定提示；T 在 E3 是末位置，在成本是平均長度 679。修正：Example 1 與 C1 改寫成 token 1、token 2，或在 S3、S6 各加一行說明；符號表加註。
9. **P1｜C1 α* 與條件機率寫死**。α* = ln2/ln(p_ab/p_ba) 只在 p_ba=p_bb 成立；F 以列和正規化，列和為 0 會除以零。目前只有兩組固定表，所以不會出事。修正：註明兩組表都滿足 p_ba=p_bb，`ex1` 對列和 0 回傳 F=1，並以 `Math.pow(0,a)` 檢查（a>0 為 0）。
10. **P1｜C3 缺快取與請求規格**。「改變 N_MCMC 只需重用已建好的矩陣」，但矩陣在 worker 內，訊息只有 `{reqId,V,alpha,tau,B,seed,nList}`。滑桿拖動會每次重建。修正：worker 以 (V,α,τ,B,seed) 為鍵快取；N 滑桿只送 `{key,nList}`；拖動以 150ms debounce 發送；另寫 V=3、B=6 的耗時預算與退路進度文字。
11. **P1｜toyCore 序列化風險**。`toyCore.toString()` 注入 Blob，若建置流程會壓縮或轉譯，內部引用會斷。修正：在 selfTest 加「worker 結果與主執行緒 TOY 的 TV 相同」；建置不得壓縮該函式。
12. **P1｜390px 與動態 KaTeX**。「讀數由本頁依 Eq. ○ 計算」、Z_t 乘積與 Eq. 12 在 390px 只靠 `overflow-x:auto`，無橫捲提示。Eq. 7/8 並列需拆成兩個 display。符號表在 `<details>` 內含 KaTeX，且兩欄表在窄螢幕。修正：display 加右側漸層陰影提示；明定哪些動態文字用 Unicode，哪些用 KaTeX 並在更新後呼叫 `renderMath()`；符號表改為 `dl` 堆疊。
13. **P2｜無障礙細節缺口**。分段按鈕未寫 `role="radiogroup"` 與方向鍵；「◀ 偏好」靠符號可行，但 C3 四種標記的最小尺寸與對比未指定；單步「下一步」按鈕沒有宣告結果的 live 文字；`aria-label` 在滑桿拖動時同步更新可能洗版。修正：寫明 radiogroup、方向鍵；標記最小 10px；按鈕按下時 `announce` 一次；滑桿 `change` 才更新 SVG `aria-label`。
14. **P2｜reduced-motion 與深色模式**。「計算中…保留舊圖（opacity .5）」若有 transition 需在 reduced-motion 取消；斜線填滿、虛線與參考線的深色 token 未列。修正：列 `--c-pow`、`--c-temp` 在深色下的值與斜線 pattern 的 stroke token；opacity 變化不加 transition。
15. **P2｜學習目標與元件對應**。LO7 的「AlpacaEval τ=0.5 不適用」、LO8 的「第 10 行不凍結前綴」沒有元件對應，C3 的 B 設定未連到任何 LO。修正：C3 加「B 與 N 對 TV 的影響」一行觀察，並在 LO8 答案段加 B 的影響；或刪除 B 控制項。
