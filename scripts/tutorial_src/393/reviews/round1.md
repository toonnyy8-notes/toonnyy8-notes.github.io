# Round 1 — merged review (No.393)

Order of work: P0 first (pedagogy #1 W2 control matrix), then all P1 (fidelity #1–2, pedagogy #2–13), then P2. Overlaps: fidelity #7 and pedagogy #15 both touch W4 — handle together; fidelity #3 and pedagogy #1/#2 both touch W2/S8 — handle together. Also: verify scripts need PYTHONPATH=. (python -I breaks import dmath) — make them runnable standalone (sys.path insert) and seed check_sign.py.

## A. Fidelity (F1–F13)
F1. P1. Loc: LO6 answer, S5, hero card 2 ("逐點取最大座標，得到的不是離散擴散的路徑"), W3. Paper: A.3 (39)-(40) conditions on the full w_s, not on z_s, and concludes only "does not necessarily follow" / §3.1 "might not". Conditioning on w_s says z_t is not a function of z_s alone; it does not disprove a z-chain. Page states it as proven. My MC (K=3, alpha~ = 0.9/0.6/0.3, Gaussian chain (37), 2M runs, x fixed): P(z_t=z_s | z_s=x)=0.591, | z_s≠x = 0.454, USDM (38) gives 0.611 for both; Markov test P(z_u=z_t | z_t≠x) = 0.511 if z_s=z_t, 0.480 otherwise. So the conclusion holds, but by a different argument. Fix: tag the z_s-conditioned MC as 〔本頁推導〕, say A.3's argument is conditional on w_s, soften to the paper's "不一定"; W3's TV 0.1018 is a w_s-conditional gap, label it so.
F2. P1. Loc: hero card 3, contribution 2 "低變異的課程學習使訓練速度加倍", 1.5-4. Paper §5.1: Duo at 510K (35.2) beats UDLM at 1M (36.7); Table 3: w/o CL 35.0, and (17) is "about 1.7 points and the remainder from the curriculum". Duo@510K already includes (17), and 35.2 is worse than w/o-CL@1M (35.0), so the CL-only speedup is not shown. Fix: "Duo (CL + (17)) at 510K steps beats UDLM at 1M; the 2x is not isolated to the curriculum".
F3. P2. Loc: Step 8 / LO12 "與 α̃²=1−t 相符〔本頁推導〕". Paper C.7 only says [β,γ]=[0.03,0.15] and §5 α̃_t=1-t. The match is a numerical coincidence and fits K=30522 only (0.0469/0.9337); K=50257 gives 0.0368/0.919. Label 推定, not 推導; say "paper says similar for both tokenizers".
F4. P2. Loc: S6 / Step 6 Thm 3.1. Paper A.4: base case needs prior p̄_1=N(0,I_K), (15) divides by p_t(z_t), and (44) is solved for x_θ with no proof that it lies in Δ^K. The "任一高斯反向過程" statement omits these. Fix: add one line of assumptions and note existence of a simplex-valued (44) is not shown.
F5. P2. Loc: Step 6 (c). Paper A.5 applies D_KL(q_data‖p̄) with q_data a pmf and p̄ a density, where the KL is +∞ (not absolutely continuous). Say "inequality is vacuous as written", not just "logs not comparable".
F6. P2. Loc: Step 7 "論文稱之為 Rao-Blackwellized；從推導看是逐點等價改寫". Correct, but Table 3's last row is retrained UDLM ("Reduces to UDLM", ‡), so 36.7 vs 35.0 may include other implementation differences. Add that caveat to 1.5-3.
F7. P2. Loc: W4 "教的概念…USDM 可反覆修改" with oracle x_θ=x. Paper Fig. 5c / §5.2: self-correction means fixing a denoiser's earlier mistakes; with an oracle there are none, and change counts grow with T_steps (3,8,8 at T=64). Say the widget shows revisability, not self-correction.
F8. P2. Loc: Step 9 "Alg. 2 改用 EMA 為教師". Paper B.2.2: "EMA weights … obtained during pre-training"; Alg. 2 pseudo-code uses the running θ_ema. State both readings.
F9. P2. Loc: Step 8 "排程". Paper §4.1 prose: "temperature is gradually decreased … initially fully relaxed"; §5.1 is a two-stage tau=0.001 then 0. Note that the prose and the recipe differ (design W5 text half-covers this).
F10. P2. Loc: Step 7 sign note, "最大值 −7×10⁻⁸". check_sign.py is unseeded; my run gave −1.05e-5. Cite "≤0 in all 20000", not the number.
F11. P2. Loc: Step 10 "取樣 … 64×／128×". Paper compares Greedy-Tail 8-step (69.58, H 5.30) with base ancestral 77.69 (H 5.55); the like-for-like base Greedy-Tail 1024 is 71.72, H 5.22. Add so the entropy comparison is not mixed across samplers.
F12. P2. Loc: Step 10 Table 1 "與 MDLM 差距 2.9／1.9／2.0". MDLM LM1B 31.8 is † from Arriola et al., not retrained, unlike UDLM ‡; note source mismatch.
F13. P2. Loc: symbol table K=30522 / 50257. Tokenizer vocab sizes are not in the paper (flagged) but T depends strongly on K (T(0.97): 0.452 vs 0.409); say the paper does not give K and that implementations may pad.

## B. Pedagogy & build (B1–B15)
B1. **P0** W2 與 S8 的訓練窗 — 「帶 A／帶 B」「橫軸 α̃/α̃²」「放大 0.8–1.0」；「α̃²」軸下兩帶各畫在哪個座標未定義（帶 A 是否改為 [0.7225,0.9409]？放大範圍是 α̃ 還是 α̃²？）。問題：無法照規格實作，且「若讀作 α̃²=1−t」標籤位置矛盾。修正：列表寫出 {軸, 帶 A, 帶 B, zoom} 四種組合的端點與標籤。
B2. **P1** W2（S4）預設「顯示論文訓練窗」開 — 讀者要到 S8 才學到訓練窗與 [β,γ]。修正：S4 預設關閉、不顯示帶；帶與 α̃² 軸只在 S8 以第二個實例（或啟用旗標）出現。
B3. **P1** S3/W1 與 LO3 — 讀數與答案用 $T$、$\mathrm{Cat}(Tx+(1-T)\mathbf 1/K)$，但 T 在 S4 才定義。修正：S3 先以「保留機率 $\rho$」命名，S4 再說 $T$ 是 $\rho$ 的積分式；或把 (10)(11) 的定義提前到 S3 末。
B4. **P1** S2/S5 — 「邊際」「轉移核」「Markov 鏈」「速率矩陣」「ODE」「NELBO」「推前」均未先教；R12 說 (2)(12) 只要求懂線性 ODE，但 S2 仍列速率矩陣，S6–S8 反覆用 NELBO／KL／data-processing inequality。修正：S2 加一段「邊際 vs 聯合路徑」定義（S5 的前提），NELBO 以一句「損失＝期望 KL」定義並給 LO；速率矩陣移入 `<details>`。
B5. **P1** S4、S9 過密 — S4 同時塞 (11)、K 大壓縮、(33)→(36)→(12)、(13)、W2（目標 450–550 字）；S9 塞 consistency distillation、PF-ODE、DDIM (57)、(22)、(23)、Alg.1/2、W4（550–650 字），而 PF-ODE／DDIM／一致性蒸餾從未被教。修正：S4 拆成「T」與「ODE 與對偶」兩章；S9 拆成「DDT 軌跡」與「DCD 蒸餾」，先用 3–4 句教 PF-ODE 與 consistency distillation。
B6. **P1** S1 符號表 — 一次列 κ、τ、δ、N、M、x̄ 等後文才用的符號。修正：表只放 S1–S3 用到的，其餘各章首次使用處補一行定義。
B7. **P1** S10 標題「蒸餾後在少步取樣勝過 MDLM」 — 缺 1.5-5 的限定（僅 T≤32；Greedy-Tail 熵較低）；S6 標題「概似不低於高斯」缺「構造出的取樣器對」。修正：標題加限定語。
B8. **P1** S10／LO11 — Gen PPL、熵、Greedy-Tail、「64×／128×」未在 S10 大綱定義（定義只在 1.4 步驟 10）。修正：S10 首段定義兩個指標與評估模型（GPT-2 Large）。
B9. **P1** 文案違反字面規則 — 「神諭」（W4）、「帶」「課程」「teacher/student」無首次定義、「carry-over」。修正：「神諭」改「已知真值的 denoiser $x_\theta=x$」；「課程學習」首見處寫「依難度由易到難調整輸入」；英文術語首見附中文。
B10. **P1** W5 — 「K 為數萬時 τ=0.001 更接近 one-hot」缺依據（K 大時最大雜訊座標更大，邊際反而變小）；固定一組 ε 且 K=8，無法回答課程的核心問題。修正：刪除該句，改顯示「邊際 $(w_x-\max_{j\ne x}w_j)/\tau$」並放可選 K（8／30522 以最大雜訊座標期望值 σ√(2lnK) 估計）；5.5 「所有情形 arg max 皆正確」註明僅限該 seed。
B11. **P1** SVG `<pattern>` — 「放在每張圖的 `<defs>`」：多張 inline SVG 的 id 相同會互相覆蓋；深色模式下 pattern 內 fill 不吃外層 CSS。修正：id 加圖版前綴（`w1-hatch`），pattern 內用 `class` 取 `var(--c-sig)`。
B12. **P1** 動態 DOM 的 KaTeX — 讀數框、aria-label 含 α̃、σ̃；每次 input 重跑 `renderMathInElement` 會閃爍且慢；Unicode 組合字元 α̃ 在 JetBrains Mono 易錯位。修正：標籤靜態預渲染，只更新數字 span；SVG 與 aria 用 `α~` 或「alpha-tilde」純文字。
B13. **P1** 無障礙 — W2 讀數框 7 項 `aria-live` 每 500ms 播報、W1 aria-label 不會被朗讀；α̃ 滑桿步長 0.001（1000 檔）鍵盤不可用；W4 播放無暫停鍵。修正：live region 只播 $T$ 與 $P(z=x)$ 且 `change` 時；加 ±0.01 的步進鈕或 PageUp/Down 設大步；W4 加「暫停」；W1 加 live region。
B14. **P2** 390px — 只有 (4)(17)(21)(39) 規定 `aligned`；(11)(13)(28)(49)(12) 同樣過寬；W4 兩面板各 5 欄橫捲（約 400px）未說明是否同步捲動；Table 2/7 與 S1 四欄表需橫捲提示。修正：補齊式清單；兩面板共用一個捲動容器；捲動容器加右側漸層。
B15. **P2** 規格缺口 — W4 `hist` 形狀、「改變 k 次」定義、`n` 預設 =T_steps（一進頁面就是完成狀態，看不到雜訊起點）；seed 或 T_steps 變更時 n 是否重置；reduced-motion 只規範 W4，W1–W3、W5 的長條 transition 未規範；JS 失敗時各圖版無 noscript 後備；`selfTest` 的 `scrollWidth` 未指定視窗寬；5.1 對 T≈1e-4 的 ±2e-4 容許等於沒驗；跨引擎 `Math.log/cos` 末位差使「逐位相同」不成立（改用容許值）。修正：逐項補上。
