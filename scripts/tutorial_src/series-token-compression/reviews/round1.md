# Round 1 reviews (fidelity + pedagogy, merged)

Fidelity: c1_tome.py, c2_cost.py, c3_group.py run (from verify/, not under `python -I` — they import sibling `common`; make them -I safe) and match out_*.txt byte-for-byte. Hand-checked C(L), quadratic shares, core-only 2.030× (N=1024, d=768, κ=0.25). RTI Tab.1/Sec 4.2/App B.3, HiT Tab.2/3/5/7/9, Fig.1d, ToMeSD Tab.4, ToDo, Hourglass 3.741, HDiT 5.23, DyDiT, DiffCR, Sparse Guidance, ELIT, DC-DiT, H-Net Eq.4/10 match. No P0.

## P1 — fidelity

1. **「移除位置代價 39.0」 (1.5, 1.6 row G).** RTI Sec 4.2: 39.0 is the Latent Array row (84.7 vs 45.7), which "forgoes grouping altogether, and with it contiguity and position at once". Write 「換成 latent array（同時失去分組、連續、位置）−39.0」.
2. **C4 「RTI Eq. 4 中 g 取恆等的特例」.** Eq. 4 is h_j + g([h_j‖z'_c − z_c]) with learned g on a concatenation; ŷ = ȳ_c + β(h − h̄_c) is not of that form. Say 「類比」.
3. **「RTI 平均＋廣播代價 7.2」 as restoration-only evidence (LO10, S10, C4 obs. 4).** Sec 4.3 replaces READ and WRITE together; confounded with pooling. Label 「取代整個 READ/WRITE」; only HiT 7.97 vs 6.61 isolates restoration.
4. **「給出相反的實驗證據」 (1.1) / 「核心張力」 (1.4).** HiT Tab.7/Sec 4.5 compare against a fixed horizontal-then-vertical pairing that is contiguous and keeps learned q/k and soft pooling, dropping only the tree and L_comp. HiT never tests non-contiguity. Reword to the cautious 1.7 #1 wording.
5. **C4 lesson "large noise ⇒ small β*, residual 帶入噪聲".** Nothing like this in the papers; RTI's pre-pool h_j are deep features, g is learned. State on the page it is a pure toy, not RTI behaviour.
6. **C3 methods 4–5 「δ/τ 對批次校準，群組數總和 = 2R」.** HiT calibrates τ once on 1,024 fixed noisy training examples and freezes it (Sec 4.1, App A.3); the batch mean is only encouraged by L_comp (Eq. 4). H-Net/DC-DiT use p_t ≥ 0.5 on a learned cosine, not a Hilbert-gap δ. Drop the 「H-Net／DC-DiT 式」 label for method 4; note the exact batch mean is an idealisation.

## P1 — pedagogy / implementation

7. **Prerequisites and early terms.** DiT, CFG, FID, GenEval, LoRA, RoPE, x-prediction, x_t = t x + (1−t)ε, Hilbert curve, Amdahl, wall-clock, 連續區域, 非局部 used before definition; C4 uses t, ε, 核心前特徵, 理想核心 unintroduced. Add a prerequisites section before S1 (forward process, x-prediction, CFG, metrics); keep only axes A–H in S2; move dst/src/unmerge to S4–S5 and 殘差寫回 to S10; define each term at first use.
8. **S9 + C3 too dense.** ELIT, DC-DiT, RTI Eq.1–2, HiT Eq.2/8, Hilbert, connectivity, EV, spread, batch budget, five rules, δ/τ calibration; Hilbert, d_j, average linkage, dendrogram, EV/spread untaught before C3. Split S9a (grouping rules: Hilbert, d_j, average linkage, one paragraph each) / S9b (budget and metrics). C3 shows three methods by default, the rest behind 「進階」.
9. **6.3 「第 1 節以外不新增論文數字」** contradicts LO5/7/9/10 and S5/7/8/9. Change to "numbers only from design §1 and §5".
10. **C2 layouts before they are taught.** 「3＋11＋3」 and HiT 3/3/6/3/3 appear in S6, before S9–S11; HiT has 18 blocks vs 17. Explain both layouts in S6 prose and compare each strategy against a dense model with the same block count (or move those curves after S11).
11. **C3 ties / float boundaries.** Checker image has many tied d_j, so `≥ δ` cuts more than 2R−2 times; τ equal to an actual merge score is float-fragile. Sort by (−d_j, j) and take exactly the first 2R−2 cuts; cut the dendrogram by merge count (or compare with 1e-12 tolerance).
12. **C3 performance.** Rerunning on every R/method change (5.6M comparisons × 2 images) on the main thread is 0.1–0.3 s+ on low-end phones. Average linkage has no inversions, so truncation equals rerunning: compute the full dendrogram once (Worker or chunked; ?test runs it synchronously), truncate per R, show 「計算中」 and disable buttons while running.
13. **C4 depends on C3 state.** If the reader jumps to S10 via the TOC, C3 hasn't run. C4 observes its own visibility and triggers C3's computation with defaults.
14. **C3 keyboard.** Canvas tabindex=0, 16×16 cursor with arrow keys, Enter selects, live region announces 「patch (r,c) 屬於第 k 組，共 n 個」.
15. **390px.** C1's three panels stack at 100% below 768px; C2 direct labels → legend row on narrow screens; S11 table → cards like S12.
16. **S13 timeline.** Faded opacity .35 fails AA → opacity ≥ .6 or a muted token with ≥4.5:1; "none selected" = show all; arrow keys act only while the card row has focus.
17. **Symbol collisions.** d (width) vs d_j; c (constant) vs c(j); T (text tokens) vs t; s (size) vs S_ij; r (merges) vs R. Rename (e.g. g_j, σ, n_txt, u) and add a 「易混淆」 column.

## P2

18. Token Skip 「同預算 RTI 低 2.4」 (1.4, 1.5, LO7): Tab.1 footnote "∗ Not a budget…" (576∗ tokens); Sec 4.2: 2.4 "understates its gap, since Token Skip runs at 1.19× to our 2.15×". Add caveat.
19. 「兩者共同發現：兩端保持全解析度（…HiT Tab. 5）」: Tab.5 moves 2/2/10/2/2 → 3/3/6/3/3 (FID 8.48 → 6.61) but GFLOPs also rise 18.3 → 22.2; HiT is N→…→N by construction. Add the compute confound, soften.
20. 1.7 #5 App. B.3: Fig.7 uses MiniT2I-L/16, RTI R=256; regimes "below about three seconds" / "above about five seconds". Name the backbone, use those wordings.
21. Perceiver IO 「latent 沒有空間位置」 (1.4, S3): not in its abstract; ELIT gives latents learned positional embeddings per group; RTI Sec 4.2 says this only for its own latent-array baseline. Remove or attribute to RTI's baseline.
22. Tags 「摘要」 vs 「僅摘要」: use one tag; MoD/DyDiT full texts are in research/papers (R7 calls them abstract-only) — fix R7.
23. LO6 ToDo 「只省注意力二次項…1.16×」 depends on R4's assumption (K/V projections full length; ToDo Sec 2.2 silent). Put the assumption in the reference answer.
24. 1.2 「這解釋 ToMeSD 與 ToDo 在 2048² 加速最多」: ToMeSD attributes small-image gains to the model not being the bottleneck. Mark 〔本頁推導〕.
25. H-Net upsampling: Sec 2.2.2 also has a smoothing module and a straight-through confidence gate; the downsampler keeps only boundary vectors. Add one clause.
26. HiT T2I 139.4 GFLOPs is nominal (one multiply-add = one op) vs MiniT2I 201.6 in the same table; joint text–image grouping is T2I only. Add "nominal", both figures, mark T2I only.
27. C4 β slider [−0.5, 1.5]: β* can fall outside. Widen to [−1, 2], clamp, show the actual β*.
28. Wording: S11 「選了相反的一邊」 → 「RTI 與 HiT 在八個設計軸上的選擇對照」; S13 title 「研究時間線」 (no implementation leak).
29. S7/S8 read as literature lists (6–7 methods each): teach one or two representative methods per section; rest to the table and further reading.
30. Dark mode / motion: C3 boundary halo uses --bg not white; DPR-scaled canvases; hatching/swatch contrast specified; list each animation's reduced-motion fallback.
31. Note: no widget is tied to LO2, LO3, LO7, LO8, LO11, and S12/S13 map to no LO — acceptable if intended; say so.

Notes now exist: scripts/tutorial_src/series-token-compression/notes/ (tokc-lineage-hub + 7 topics, timeline.json with 23 verified entries) — fill the note links and timeline. Paper hubs: rti-lineage-hub (217/notes), hit-lineage-hub (585/notes).
