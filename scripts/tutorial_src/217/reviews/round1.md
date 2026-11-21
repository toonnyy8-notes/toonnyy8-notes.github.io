# Round 1 reviews (fidelity + pedagogy, merged)

Fidelity: all five verify scripts rerun identical to out_*.txt (note: run with PYTHONPATH=. — they fail under `python -I` on `import common`). Tab. 1, 2, 4, Eq. 1–6, App. C.2, B.2, B.3 spot-checked correct. No P0.

## P1 — paper fidelity

1. **JiT speedups labelled measured.** C4 title 「論文量測（Tab. 1、Fig. 16/18、App. B.2）」 and points JiT 64→1.61, 130→1.33, 193→1.14. Fig. 11 caption: "the speedup is the analytic one implied by token compute, as these backbones were not timed." Label 「解析 token 計算量（未計時）」; the 1.600/1.326/1.140 match is a consistency check, not validation.
2. **Speed ceiling over-generalised.** 步驟 12 #7 「R<192 不再加速」, LO10. Sec. 4.2 states this for B/16 only; L/16 rises below 192 (2.59×, 2.86×, 3.01× at R=256/128/64). Scope to B/16; L/16 ceiling higher.
3. **Iso-cost window caveat.** LO10, 步驟 9 「同成本 R≈461–572」, 5.4 「B/16 [1,15] 8.500」. B.5: window counts image tokens only (full text stream pushes true iso-throughput below R=461), and [1,15] was not timed; its ceiling "lies above the 2.5× saturation". Add caveat; do not display 8.5× as a ceiling (at most "above 2.5×"). Also C4 axis: auto-scale to the ceiling rather than fixed 1×–4×.
4. **Position-removal cost overstated (both reviewers).** 步驟 12 #5 and LO11 attribute 45.7 / 39.0 to removing position. Sec. 4.3: the latent-array swap "removes grouping together with position, so its cost is an upper bound on the contribution of anchoring alone." Say 「移除分組與位置（上界）；位置單獨的貢獻論文未分離」.
5. **C2 t slider.** Toy features = raw pixels, so t=0.5 means noisy input pixels, whereas the paper's features track clean-estimate structure (Sec. 3.3, Fig. 2b). Say on the plate the slider mixes noise into input pixels, not model feature noise (or use a blurred clean-estimate proxy).
6. **Toy EV contradicts the 26% lesson.** Toy recovers 99.5% at R=256 and ranks adaptive above Feat Sim; paper C.1: ~26% not recovered at R=256, Tab. 4 Feat Sim retained 0.74 ≥ RTI 0.73. Caption next to the readout: the toy is far more compressible than real features; no ranking conclusions from EV.

## P1 — pedagogy / implementation

7. **S2 and Tab. 4 use untaught terms.** S2 uses blocks/core (S4) and Hilbert cuts (S5/S6); S1 doesn't define 區塊. Tab. 4 / C2 use retained, size–detail, Feat Sim, Latent Array, Token Skip before their definitions (only in step 8 / S8/S12). Fix: S1 adds 「DiT 由 17 個區塊串接」 and 「一個取樣步＝一次前向計算」; S2 teaches only EV with equal 2×2 mean pooling; S3 gets one-sentence definitions of the three baselines plus retained and size–detail; C2's Feat Sim/Skip in `<details>` or a second tab.
8. **size–detail "恆為 0" for equal cuts** fails when R doesn't divide N (R=96/192/384 give sizes 10 and 11). Say 「R 整除 N 時為 0」; add an R=96 selfTest row.
9. **S7 too dense; phasors unprepared.** S7 teaches region softmax, size embedding, RoPE phasors and low-pass, Dirichlet formula, WRITE, skip, ablations, complexity; e^{iωp} and 低通 unprepared; WRITE has no widget. Split S7a READ / S7b position (RoPE, start with a two-position rotation-angle numeric example; C3 only here) / S7c WRITE.
10. **S1 overload; symbol table.** Split the symbol table by section, placed where used.
11. **Hero cards use undefined terms** (區塊, Hilbert 曲線, R−1, 區域 token): rewrite in plain words, e.g. 「把影像 patch 排成一條線，在內容變化最大處切開」.
12. **Metaphor / overclaim (both reviewers).** 「付相同代價」→「每個 patch 的計算量相同」; 「VAE 一次付清冗餘」, 「連續旋鈕」→「可調參數」, 「抗壓縮」; 「任何預算」/「適用所有預算」→「訓練範圍（64–512）內的預算」 (Sec. 4.3 "within this range the budget is a continuous knob").
13. **C2 keyboard / canvas.** Canvas isn't focusable: add tabindex=0 overlay with a drawn focus box; boundary lines handle devicePixelRatio (two-layer: translucent white outer + ink inner); the 1023-bar chart at 390px → canvas with per-pixel max binning; drop per-pixel "亮底用 ink".
14. **Theme redraw.** All canvas/SVG charts redraw via one redrawAll() on theme change (pm-theme / prefers-color-scheme), reading tokens with getComputedStyle.
15. **C4 controls.** Log slider: integer index mapping with defined rounding; switching backbone resets R to its default (JiT 130) and shows only that backbone's points; the step-count block fixed at L/16 (288/112 ms) must be labelled 「固定為 L/16」 or moved out.
16. **C3 readouts / symbol clashes.** 參與比 and 「保留頻率數（|m|≥0.9）」 untaught — tag 推定 and define in S7b, or use known quantities (max in-region weight). Rename clashing symbols: n (grid side vs token count), c (region vs cost — use ρ), r (index vs n/N).
17. **KaTeX at 390px.** Wrap every display equation in overflow-x:auto (tabindex=0); break Eq. 3 on small screens; render math in dynamically inserted `<details>`.

## P2

18. C4 band claim 「量測點位於模型帶上方或附近」: at τ=0, L/16 R=64 measured 3.01× is below both a=0 (3.23) and a=1 (3.79) lines; say inside the band only for τ>0 or R≥128.
19. Feat Sim mean-centering and anchor ⌊rN/R⌋ are not in App. A ("R evenly spaced anchor tokens… nearest anchor by cosine similarity"): tag 推定, note centering changes the ranking.
20. Tab. 4 context: retained is "without any learned WRITE… a lower bound", averaged over core [3,13]; step 3 row 連通 should also list Latent Array (Sec. 4.2: loses "contiguity and position at once"); caption 「凍結特徵」 needs this caveat.
21. 步驟 12 #1, #3, #6 are design choices (ends justified by Fig. 2c reasoning, per-step recut is App. A description, single-budget training untested) — reword as 「論文的設計選擇」, not experimentally shown errors.
22. Dirichlet low-pass 〔本頁推導〕: add 「（旁瓣與混疊除外，僅限直線段）」; C3's 2-D Hilbert runs are not covered by the straight-run formula.
23. C1: don't cite positions 8..15 as the Fig. 14a run (paper's example is an unaligned 8-run at the mean region size for R=128 showing Morton jump and raster wrap); 8..15 is aligned and connected in all orders.
24. C3 chips 「16 patch」「64 patch」: keep current p (clamp); draw two range lines (x and y extents) distinguished by line style and label.
25. 6.3 rule 「第 1 節以外不新增論文數字」 conflicts with S8–S12; restate as 「不新增設計文件之外的數字」. No animation anywhere: optionally make the hero Hilbert curve draw-on (respecting reduced-motion).
26. verify scripts: make them runnable with `python3 -I` (add sys.path insert of the script dir) or document PYTHONPATH=.
