NO-GO

## Explain-back

本頁教：離線 RL 的外插誤差（只最大化擬合的 Q 會選到資料外、被高估的動作）→ TD3 actor 目標 E[Q(s,π(s))]（Eq. 1）→ 加入 BC 項得 max E[λQ − (π−a)²]（Eq. 3）→ λ = α / mean|Q|，α = 2.5，mini-batch 上估計且不微分（Eq. 5）→ 狀態以資料集 μ、σ 正規化（Eq. 4）。結果：D4RL 15 個任務總分 979.3，Fisher-BRC 974.6；訓練 39 分鐘，Fisher-BRC（PyTorch）2 小時 8 分。

互動元件：
- C1：α 滑桿 0–10（步長 0.1，預設 2.5）與 seed 重抽。顯示真實值 q、擬合 Q̂、J 平移曲線與三個標記。讀者應見 α = 2.5 時 π* = 0.381，只最大化 Q̂ 時選 a = 1（真實值 4.03），α ≥ 3.4 時停在上界 1。
- C2：log10 c 與 b 滑桿，比較正規化 λ 與固定 λ。讀者應見正規化時 π* 恆為 0.381，固定 λ 時 π* 隨 c 增加而右移。
- C4：分頁切換 Table 2、Table 3、Table 9 的長條與數字，可篩選資料集與演算法。

Explain-back 完整可讀，C1、C2、C4 的輸入、預設與預期觀察都能對上 design.md 第 4 節。

## Blocking issues

1. **α 在定義前使用。** design.md 第 1 節符號表（L44）寫明 α「首次定義」於 S6，但 α 在更早或同為 S5 的正文與元件中已出現：
   - 1.1 總結（L17）：「$\alpha=2.5$」。
   - 1.3 貢獻 2（L27）：「唯一新增的超參數是 $\alpha$（Sec. 5、Sec. 7）」，這是 S5 內容。
   - LO6 參考答案（L176）：「TD3+BC（α = 2.5）」，LO6 所在章節為 S5。
   - C1 `#c1b`（L241）：滑桿 α，章節為 S5。
   修正方式：把 α 的首次定義改為 S5，或把 S5 內的 α 改成先在 S5 定義。
2. **兩個連結 slug 不在指定目錄。** `rl-mdp-q-learning-basics` 與 `offline-rl-distribution-shift` 只存在於 `tutorial_src/568/notes/`，不在 `570/notes/` 或 `notes_split/`（L195、L383）。design.md R8（L409）已承認這個風險。在 568 上線前這兩個連結都會失效。修正方式：確認 568 已上線，或先移除這兩項。
3. **C2 的「Q 平均絕對值接近 0」警示沒有判定式。** L256 寫「接近 0」並提到「b 接近 −4.4 時」，但沒有數值門檻，前端無法實作精確判定。另外，以資料點計算時 mean|Q̂ + b| 不會真的為 0；由 C1 擬合值（資料點 Q̂ 約在 3.95 到 5.16 之間）估計，最小值約 0.2 量級，因此 `1e-12` 的截斷分支實際上不會觸發。這是估計，未實際執行 `c2.py` 核對。修正方式：指定門檻（例如 mean|Q̂+b| < 某數）並以 `c2.py` 核對後寫入第 5 節。

## Non-blocking notes

1. 步驟 4 標題引用「Eq. 2–3」，但正文只展示 Eq. 3，Eq. 2 未出現。
2. C1 觀察 (3) 寫「α ≈ 3.3 之後停在動作上界 1」，而第 5 節表中 α = 3.3 時 π* = 0.977；建議寫 α ≈ 3.32（π_int = 1 於 α = 3.3212）。
3. Hero 事實卡與 S10 用 Fisher-BRC 的 PyTorch 時間 2h 8m，而 Table 3 中 TensorFlow 版為 2h 12m；建議在 Hero 寫明「PyTorch 重寫版」。
4. 6.1 依賴 `tut/406/build/index.html` 存在；建置前需先確認。
5. 第 5 節數字（c1_out、c2_out、c4_out）未在本次審查中重新執行，只核對了 design.md 內的算術。
6. 已核對：timeline.json 共 14 筆，且 6.4 的 TL_GROUP 涵蓋全部 14 個 arXiv ID；Table 2 的 BC、TD3+BC、Fisher-BRC 欄加總與 ± 之和與文內數字一致（595.3、979.3、975.2、108.3，R3 已註記）；Table 9 與 Table 3 的加總一致。
7. 已核對：`tutorial_src/570/notes/` 下 7 篇筆記的 frontmatter 皆含 slug、title、cluster、kind、tags、arxiv、sources、related，符合驗收條件。
8. 已核對：design.md 中其餘連結的 570 筆記（`ddpg-td3-actor-critic`、`batch-constrained-policy-constraint-methods`、`advantage-weighted-bc-methods`、`rl-plus-behavior-cloning-prior-work`、`rl-implementation-details-reproducibility`、`td3bc-algorithm-and-d4rl`、`td3bc-lineage-overview`）皆存在。
