# 438 設計文件審查：教學法與建置就緒度（Goldfish）

1. **P0 · C4 計算** 「係數 $c_j=\mathrm{KL}_2/\Delta v^2$」：$\Delta v$ 滑桿範圍 [-1,1]、步長 0.01，會停在 0，除以零得 NaN。修正：直接用閉式 $c_j=\frac1{2T}(\frac{\sigma(1-t)}{2t}+\frac1\sigma)^2$ 計算，不除 $\Delta v^2$。

2. **P0 · 6.4 佔位必須補**（Hero、S1、S9、S10 都依賴）。依現有筆記：(a) hub slug = `flowgrpo-lineage-hub`（cluster「RL, Reasoning & Self-Distillation」）；(b) 對照表：hub 步驟 1 `flowgrpo-flow-matching-backbone`→S2；2 `flowgrpo-diffusion-sde-background`→S4；3 `flowgrpo-denoising-mdp-and-grpo`→S2（MDP）、S3；4 `rl-finetuning-text-to-image-diffusion`→S1、S3、S7；5 `flow-ode-to-sde-for-rl`→S4、S5；6 `flowgrpo-design-choices-and-results`→S6、S7；既有筆記 `flow-matching-velocity-target`→S2、`rlvr-grpo-credit-assignment`→S3、`rl-mdp-q-learning-basics`→S2；(c) `timeline.json` 只有 13 筆，且缺 DDPO 以外的 RL 微調（DPOK、ReFL、Diffusion-DPO、PPO 1707.06347、R1 2501.12948 等筆記有引用但時間線沒有），設計預期的「擴散 RL／偏好對齊」類別只有 2305.13301、2305.18290 兩筆；DDPM/DDIM、SD3、AWR、Adjoint Matching 不在 S9 敘事鏈中，需決定歸類。時間線無 slug 欄位，無法直接對應筆記；`role`、`source`（含 `paper.txt §`）是英文／內部資訊，不應原樣顯示。hub 內文提到「見 `timeline.json`」，上線頁不可保留此句。notes 的 `slug.html` 相對路徑需對 `tutorials/` 位置確認。另更新 R10。

3. **P1 · C3 先於概念（S4 內）** C3 模擬 Eq. 9、$\sigma_t=a\sqrt{t/(1-t)}$、$\Delta t$、$t'$，但這些在 S5 才教；$W_1$ 也未定義。修正：C3 移到 S5 的 Eq. 9 之後，或把 Eq. 9 與 $\sigma_t$ 前移到 S4；用一句字面話定義 $W_1$（兩個累積分布曲線之間的面積）。

4. **P1 · S3 的 $\rho$ 與裁剪先於策略** $\rho$ 是兩個策略機率之比，但策略分布（高斯）在 S5 才出現；S2 又把 $\pi$ 寫成機率分布，卻說轉移確定性（ODE 下沒有分布）。大綱沒有教：為何有 $\theta_\text{old}$、訓練開始時 $\rho=1$、裁剪的目的、$\hat A\cdot\rho$ 為何提高機率。修正：S2 加一句「ODE 下策略退化為確定性，S4 起改為隨機」；S3 加一段 $\rho$ 與 $\theta_\text{old}$ 的字面說明（把 $\rho$ 暫當一個數）。

5. **P1 · 字數預算不可行** S3 600–700 字要塞 Eq. 3、4、5、LO4–6、兩個元件；S4 500–600 字要塞四個術語、LO7、LO8、玩具說明。LO 答案必須在內文，不能靠 `<details>`。修正：S3 拆成「優勢」「裁剪」兩節，S4 拆成「為何要隨機」「如何構造 SDE」，或把預算提高到約 1.5 倍並再審。

6. **P1 · 術語先用後定義** Hero 第三張卡的 GenEval 到 S7 才定義；S1 的三個貢獻已用 ODE/SDE、KL、reward hacking、DDPO、DPO；S1 末尾符號表含 $\rho$、$\sigma_t$、score 等後面才教的符號，與「使用前定義」矛盾。修正：Hero 加 GenEval 半句定義；S1 貢獻用字面描述；符號表標「後續章節才使用」或移到 S8。

7. **P1 · LO 與內文不對齊** LO7 答案含「估計散度」，全文未教 divergence；LO8 含 Fokker–Planck，只有「各一句」。S8（7 個常見錯誤）、S9、S10、C4「$a$ 小則 $\rho$ 極端」、C3「$a$ 大且 $T$ 小失控」沒有對應 LO。修正：答案改字面敘述或補一句定義；新增 LO13（S8 至少辨識三個錯誤）並把 C3、C4 觀察項掛到 LO8、LO10。

8. **P1 · 文風** 「不受裁剪保護」「往上推／往下推」「噪聲項失控」「整組…不提供梯度」含擬人或比喻。改為「裁剪不限制此方向的更新」「機率增大／減小」「數值發散（超過 10⁶）」。

9. **P1 · 元件資料範圍未定** 前端無法只靠規格畫出：C1 RMSE 軸範圍／刻度與右軸、C3 直方圖 y 軸（count/(M·binwidth) 還是自動）、C4 密度圖 y 軸。C1 二元模式曲線依賴 $p$，但拖動 $p$ 時只更新單組，曲線會是舊的而無標示。修正：補軸範圍；曲線過期時顯示「p 已變更，曲線更新中」。C3 的 $W_1$ 「放開後才更新」同理，拖動中標「過期」。

10. **P1 · C2 邊界** 判斷 $\rho\hat A\le\mathrm{clip}(\rho)\hat A$ 用浮點相等，滑桿以 0.01 累加會有 1.2000000000000002，狀態文字在邊界閃爍；曲線 201 點取樣也畫不準 $1\pm\varepsilon$ 的折點。修正：滑桿以整數百分位儲存；加入 $1\pm\varepsilon$ 兩個精確點；比較加 1e-9 容差。

11. **P1 · C4 與 C2 不一致** 「是否在 [0.8,1.2]」寫死 $\varepsilon=0.2$，C2 的 $\varepsilon$ 可調；$j=0$（$t=1$）時 $\sigma$ 用 $t'$、$(1-t)/(2t)$ 用真實 $t$，未明說。修正：共用 $\varepsilon$ 狀態或標明固定值；寫明哪些項用 $t'$。

12. **P1 · 動態數學與 SVG** KaTeX 不能放進 SVG；規格只禁組合字元，沒列 SVG 與讀數可用的符號（$\rho,\varepsilon,\sigma,\Delta t,\pi_\theta,p_t$）。KaTeX 的 `<label>` 會同時有 MathML 與 HTML，讀屏重複朗讀；CDN 失敗時會顯示原始 `$…$`。修正：列出 Unicode 對照表（如「πθ」「p_t」）；KaTeX 輸出加 `aria-hidden` 並另給 `aria-label`；加入失敗後備。

13. **P2 · live region** C3 讀數約 12 個值，500ms debounce 後整段朗讀仍太長。修正：live region 只放一句摘要（例：「j=10，SDE W₁=0.043」），完整讀數不放 live；「計算中」不設 live。

14. **P2 · 390px** Table 2 有 9 欄，橫捲容器需 `tabindex=0`、`role="region"`、`aria-label` 與捲動提示；Eq. 5、Eq. 9、KL 式「< 600px 拆兩行」要說明是兩份 DOM 以 CSS 切換還是 `aligned` 換行，避免重複被朗讀；C1 $G=48$ 長條在約 340px 寬下每根約 7px，需最小寬度。

15. **P2 · 建置依賴與深色模式** 整個外殼「沿用 447」，本文件不含 tokens 的實際值（`--c-ref`、`<pattern id="hatch">` 的深色筆色、直方圖淡色填色）；「不足時回寫 6.2」但 6.2 沒有色表，selfTest 也無法檢測深色。修正：列出 light／dark 色值與對比數字；hatch 用 `currentColor`；人工檢查深色截圖並寫入 6.6。
