# 驗收評分 (138 GPPO)

Score: 11/12 correct, 1 partial (LO10). 無 wrong。

## Per-LO
- LO1 correct: 支撐條件、E_μ[r]=1、E_μ[rA]=E_π[A] 全中。
- LO2 correct: 收益項與 TV² 懲罰項正確。
- LO3 correct: ½E|r-1|、2D_TV 單樣本估計、剪裁等價皆對。
- LO4 correct: A>0 在 r>1+ε，A<0 在 r<1-ε。
- LO5 correct: 一次項期望消失、二次近似、單樣本仍含一次項。
- LO6 correct: 定義與 1/4/2/1 皆對。
- LO7 correct: C_f>0、一次方、Eq.11 無保證。
- LO8 correct: δ=f(0.8)、四區間、TV/χ² 相同導致 11 種。
- LO9 correct: μ²f(r) 期望、非無偏、App. B 只有二階動差。
- LO10 partial: DAPO、RIPO 全對；DPPO 先寫對 1±δ/μ，卻又自創「1±δ/(2μ)」與「µ=0.1 時 [0*, 2.0]」，與頁面表格 (TV–Binary = 1±δ/μ) 矛盾。
- LO11 correct: r≤1/μ、0.0969 / 0.0955、假設性疊加。
- LO12 correct: 數字、兩個反例、無 seed/變異皆對。

## Stuck point 5 (4x3 表格錯位): 非資料或版面 bug
build/index.html 第 766-772 行：thead 為 散度/MC/Atomic/Binary，χ² 列的第三格（Atomic 欄）為 `1±√(δ/μ) <small>RIPO</small>`，第四格（Binary）為「本頁推導」。RIPO = (χ², Atomic) 正確，欄位對齊。reader 看的是攤平的 page.txt，`<small>` 標籤接在公式後、下一格之前，攤平後看起來像屬於下一欄。這是純文字攤平的假象；真實渲染不錯位。唯一資料疑點：TV 列 MC 欄標「以 TV2 計」寫在 PPO/GRPO，Atomic 欄也寫「以 TV2 計」，易混（見修正 2）。

## Fix list
1. (LO10) 在「TV–Atomic(δ) 與 TV–Binary(2δ) 相同」段後加一句：「表中 TV–Binary 欄的 1±δ/μ 就是 DPPO 的區間，不再除以 2；½ 只影響與 TV–Atomic 的換算。」並在 §3.4 段末明寫「DPPO 的區間：1±δ/μ」。
2. 表格 TV 列：「以 TV2 計」改成「以 2D_TV 計（半徑 δ）」，與 DPPO「以 TV 計」並列時說明兩者差兩倍，避免讀者誤推 2μ。
3. (stuck 5，可選) 表格每個方法標籤改放在公式前或獨立成第二行（`<br><small>`），並以粗體標示，避免攤平或窄螢幕時被誤認為相鄰欄；在表上方加一句「RIPO 在 χ² 列、Atomic 欄」。
4. (stuck 4) C_f 的 inf 只給結果：在 S5 加一句「四個值是格點數值計算，非解析推導，讀者只需要知道使用條件 C_f>0」。
5. (stuck 2) TV 的 ½ 造成 [0.6,1.4] 與 [0.8,1.2] 兩種讀法：在 S8 校準段首次出現 ½ 處，直接放這個 δ=0.2 的對照小例，並標明本頁採 |r-1|。
6. (stuck 1) μ 同時作符號與機率值：S1 符號表加一行「μ(a|s) 是舊策略給動作 a 的機率，同時也是 Atomic/Binary 公式中的 μ」。
7. (stuck 3) 不需要改，LO4 答對；可在 S3 補一個正負優勢各一行的小圖示但非必要。
