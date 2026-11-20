# Acceptance grade

**Score: 12/12**

## Verdicts
- LO1 correct: hypothesis (human patterns limit exploration, humans skip reflection) and final-answer-plus-format reward both match.
- LO2 correct: group mean/std replaces the value model; both reasons (size and cost, hard partial-output prediction) given.
- LO3 correct: 0.25 / 0.4330, +1.732, -0.577, all-correct gives std 0 and advantage 0 (labelled as the page's treatment).
- LO4 correct: both zero-gradient conditions, eps = 10, and the author's rationale.
- LO5 correct: x - log x - 1 >= 0, unbiased for the KL value, loss vs reward placement, 400 steps, beta = 0.001.
- LO6 correct: accuracy + format with equal weight (Eq. 4); reward hacking, retraining cost and pipeline complexity.
- LO7 correct: 32 x 16, 32,768 to 65,536 after 8.2k, 15.6 to 77.9, cons@16 86.7.
- LO8 correct: length grows, reflection words 5-7x as a count, "wait" surges after step 8000.
- LO9 correct: four stages with start model and reward; 1,700 steps, T 0.7, last 400 steps preference. The reference's eps = 10 for stage 2 is not required by the question.
- LO10 correct: IF-Eval 46.6 to 71.7, AIME 77.9 to 59.0, cold-start data volume; AlpacaEval +25.5, ArenaHard +16.7.
- LO11 correct: formula, T 0.6 / top-p 0.95, k = 64 for AIME, 79.8 vs 79.2.
- LO12 correct: 72.6 vs 47.0, distillation conclusion, early 7B/16B failures and capacity dependence.

No partial or wrong answers, so no LO-level text fixes.

## Stuck points
1. Eq. 1 rho at sequence vs token level: real but minor. The page says the implementation detail is unstated. Optional one-line note.
2. All-correct/all-wrong advantage = 0 is the page's own choice. Already labelled as such, so no fix.
3. KaTeX message: NOT a reader-visible problem. build/index.html line 458 has `<p class="katex-warn">`, with CSS `display:none` (line 387). It is shown only when the `nokatex` class is set, which happens in the CDN script `onerror` handler (lines 12-13). The text appears in page.txt only because the extraction ignored CSS and the CDN did not load. Normal readers with network access never see it. Offline readers get the intended fallback. No fix.
4. C4 "1746 -> 2029": NOT a sort or data bug. `stageDelta` (line 1110) deliberately excludes Codeforces Rating, so the lists contain only |delta| >= 1.0 score deltas. The `Codeforces Rating: cf[base] -> cf[n]` line (line 1119) is a separate div printed right after the 下降前 3 list. The reader read it as a list item. The data is correct (Table 3 R1 column = 2029, Dev3 = 1746). It is a layout and labelling problem, so fix it (fix 1).
5. 86.7% appears three times with different metrics: real confusion risk.

## Fixes
1. C4 card (build/index.html line 1119): give the Codeforces line its own label and visual separation, for example `<div class="x">另列（Elo 分數，不列入上方排名）Codeforces Rating：1746 → 2029（上升）</div>`, with a margin or border above it. Also drop the `<li>無</li>` ambiguity by writing "下降：僅 C-SimpleQA -3.2".
2. Pin the metric on each 86.7% occurrence: "86.7%（R1-Zero，cons@16 自洽解碼）", "86.7%（R1，64 樣本多數決）", "86.7%（Distill-Llama-70B，cons@64）". Consider one small table in S8 listing pass@1, pass@64 and majority vote for R1.
3. S4a, after the clip text: "論文未說明 rho 與裁剪是在整段輸出還是逐 token 計算；此處的『截斷大量 token 的梯度』是作者原話。"
4. Optional: at the first use of "rollout" and "off-policy", add a short parenthetical, for example "rollout（用目前策略實際生成一批輸出）" and "off-policy（用舊策略產生的資料更新新策略）".
