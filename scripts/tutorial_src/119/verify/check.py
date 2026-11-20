"""Verification for tutorial 119 (LOPD). Same algorithms as the page (see design.md §4, §6).
Run: python3 check.py > check.out"""
import math, random

# ---------- shared math (page: softmax, logSoftmax, kl) ----------
def softmax(u):
    mx = max(u); e = [math.exp(x - mx) for x in u]; s = sum(e); return [x / s for x in e]
def kl(a, b):  # sum a log(a/b), terms with a<1e-15 count as 0
    return sum(x * math.log(x / y) for x, y in zip(a, b) if x >= 1e-15)

# ---------- C1: top-M + tail reverse KL ----------
U1 = [2.0, 1.2, 0.6, 0.1, -0.3, -0.8, -1.2, -1.5]
Z1 = [0.2, 1.5, -0.5, 1.0, 0.4, -1.0, 0.8, -0.6]
def c1(s, M, perm=False):
    pT = softmax(U1)
    pS = softmax([u + s * (z - u) for u, z in zip(U1, Z1)])
    order = sorted(range(8), key=lambda i: (-U1[i], i))
    sup = sorted(order[:M]); tail = [i for i in range(8) if i not in sup]
    if perm and len(tail) >= 2:
        vals = [pS[i] for i in tail][::-1]
        pS = pS[:]
        for i, v in zip(tail, vals): pS[i] = v
    tS = sum(pS[i] for i in tail); tT = sum(pT[i] for i in tail)   # direct sums, no 1-sum
    rev_tr = sum(pS[i] * math.log(pS[i] / pT[i]) for i in sup)
    if tS >= 1e-15: rev_tr += tS * math.log(tS / tT)
    return dict(sup=[i + 1 for i in sup], tailS=tS, tailT=tT, rev=kl(pS, pT), fwd=kl(pT, pS), revTr=rev_tr,
                pS=pS, pT=pT)

print("== C1 distributions")
r = c1(1, 3)
print("pT", [f"{x:.4f}" for x in r['pT']]); print("pS(s=1)", [f"{x:.4f}" for x in r['pS']])
print("pS(s=0.5)", [f"{x:.4f}" for x in c1(0.5, 3)['pS']])
print("== C1 table: s, M, revFull, fwdFull, revTrunc, tailS, tailT")
for s in (1.0, 0.5, 0.0):
    for M in (1, 2, 3, 5, 7, 8):
        r = c1(s, M)
        print(f"s={s} M={M} rev={r['rev']:.4f} fwd={r['fwd']:.4f} revTr={r['revTr']:.4f} tailS={r['tailS']:.4f} tailT={r['tailT']:.4f}")
print("== C1 perm (s=1)")
for M in (1, 3, 5):
    a, b = c1(1, M), c1(1, M, True)
    print(f"M={M} revTr {a['revTr']:.4f} -> {b['revTr']:.4f} ; rev {a['rev']:.4f} -> {b['rev']:.4f}")

# ---------- C2: delta ledger ----------
P = [0.40, 0.30, 0.20, 0.10]; Q0 = [0.55, 0.15, 0.25, 0.05]; RB = [0.8, 0.3, 0.6, 0.1]
def c2(s, rb):
    q = softmax([(1 - s) * math.log(p) + s * math.log(t) for p, t in zip(P, Q0)])
    Qa = [2 * r - 1 for r in rb]
    d = [math.log(qi / pi) for pi, qi in zip(P, q)]
    Ed = sum(pi * di for pi, di in zip(P, d)); K = kl(P, q)
    Qb = sum(pi * x for pi, x in zip(P, Qa))
    c = sum(pi * x * di for pi, x, di in zip(P, Qa, d))
    cov = sum(pi * (x - Qb) * di for pi, x, di in zip(P, Qa, d))
    return dict(q=q, d=d, Ed=Ed, K=K, Qb=Qb, c=c, cov=cov, first=-Qb * K)
print("== C2")
for s, rb in ((1, RB), (0.5, RB), (0, RB), (1, [0.5] * 4), (1, [0.9] * 4), (1, [0.1] * 4)):
    r = c2(s, rb)
    print(f"s={s} rb={rb} q={[round(x,4) for x in r['q']]} d={[round(x,4) for x in r['d']]} "
          f"Ed={r['Ed']:.4f} KL={r['K']:.4f} Qbar={r['Qb']:.4f} c={r['c']:.4f} -Qbar*KL={r['first']:.4f} cov={r['cov']:.4f}")

# ---------- C3: teacher-logit dynamics, student fixed ----------
PRESETS = {'A': [0.8, 0.3, 0.6, 0.1], 'B': [0.5, 0.5, 0.5, 0.5], 'C': [0.7, 0.7, 0.7, 0.0]}
ETA, ETA_B, T3, BMAX = 1.0, 0.5, 600, 5.0
def c3(preset, mode, beta=0.0, m=0.05):
    Qa = [2 * r - 1 for r in PRESETS[preset]]; Qb = sum(p * x for p, x in zip(P, Qa))
    u = [math.log(x) for x in Q0]; b = beta if mode == 'fixed' else 0.0; hist = []
    for i in range(T3 + 1):
        q = softmax(u); K = kl(P, q); D = sum(p * x * math.log(qi / p) for p, x, qi in zip(P, Qa, q))
        hist.append((q, K, D, b))
        if i == T3: break
        g = [(qi - p) - b * (p * x - qi * Qb) for p, x, qi in zip(P, Qa, q)]
        u = [a - ETA * gg for a, gg in zip(u, g)]
        mx = max(u); u = [max(-30.0, a - mx) for a in u]
        if mode == 'dual': b = min(BMAX, max(0.0, b + ETA_B * (m - D)))   # page guard: beta capped at 5
    w = [p * (1 + (beta if mode == 'fixed' else hist[-1][3]) * x) for p, x in zip(P, Qa)]
    w = [max(0.0, x) for x in w]; sw = sum(w); qs = [x / sw for x in w]
    return hist, qs
print("== C3 (steps 0,25,100,600): q, KL, Delta, beta ; q* uses final beta")
cases = [('A', 'fixed', 0.0, 0), ('A', 'fixed', 0.5, 0), ('A', 'fixed', 1.0, 0), ('B', 'fixed', 1.0, 0), ('C', 'fixed', 1.0, 0),
         ('A', 'dual', 0, 0.0), ('A', 'dual', 0, 0.05), ('A', 'dual', 0, 0.2), ('B', 'dual', 0, 0.05), ('C', 'dual', 0, 0.05), ('C', 'dual', 0, 0.2)]
for pr, mode, be, m in cases:
    h, qs = c3(pr, mode, be, m)
    print(f"{pr} {mode} beta={be} m={m}  q*={[round(x,4) for x in qs]}")
    for i in (0, 25, 100, 600):
        q, K, D, b = h[i]
        print(f"   step {i}: q={[round(x,4) for x in q]} KL={K:.4f} Delta={D:.4f} beta={b:.4f}")

print("== C3 stability sweep: all presets x (fixed beta 0..1 step .05, dual m 0..0.2 step .01); max |q600-q599|, max beta")
worst = 0; wcase = None; bmax = 0
for pr in PRESETS:
    for k in range(21):
        h, _ = c3(pr, 'fixed', k * 0.05)
        dq = max(abs(a - b) for a, b in zip(h[600][0], h[599][0]))
        if dq > worst and not (pr == 'C' and k * 0.05 > 0.95): worst, wcase = dq, (pr, 'fixed', k * 0.05)
    for k in range(21):
        h, _ = c3(pr, 'dual', 0, k * 0.01)
        dq = max(abs(a - b) for a, b in zip(h[600][0], h[599][0])); bmax = max(bmax, max(x[3] for x in h))
        if dq > worst: worst, wcase = dq, (pr, 'dual', k * 0.01)
print("max |q600-q599| = %.2e at %s (C fixed beta=1 excluded: q4 -> 0 slowly, monotone) ; max beta over dual sweep = %.4f" % (worst, wcase, bmax))
h, _ = c3('B', 'dual', 0, 0.2); print("B dual m=0.2: beta@25=%.4f beta@50=%.4f beta@600=%.4f (cap)" % (h[25][3], h[50][3], h[600][3]))

# ---------- C4: score -> problem counts ----------
ROWS = [  # id, backbone, bench, N, LOPD pct, baseline name, baseline pct
 ('q4-bfcl', 'Qwen3-4B', 'BFCL-v3 Avg', 800, 27.38, 'GRPO', 25.25),
 ('q8-bfcl', 'Qwen3-8B', 'BFCL-v3 Avg', 800, 29.88, 'GRPO', 29.00),
 ('q4-lcb5', 'Qwen3-4B', 'LiveCodeBench v5', 279, 50.54, 'GRPO', 50.18),
 ('q4-lcb6', 'Qwen3-4B', 'LiveCodeBench v6', 131, 45.04, 'GRPO', 44.27),
 ('q4-lcb', 'Qwen3-4B', 'LiveCodeBench Avg', 410, 48.78, 'GRPO', 48.29),
 ('q4-he', 'Qwen3-4B', 'HumanEval+', 164, 87.80, 'SDFT', 87.20),
 ('q4-mbpp', 'Qwen3-4B', 'MBPP+', 378, 78.57, 'Skill-SD／SDFT', 76.98),
 ('q4-ep', 'Qwen3-4B', 'EvalPlus Avg', 542, 81.36, 'SDFT', 80.07),
 ('o7-lcb5', 'Olmo3-7B', 'LiveCodeBench v5', 279, 53.41, 'SDPO', 49.82),
 ('o7-lcb6', 'Olmo3-7B', 'LiveCodeBench v6', 131, 45.80, 'GRPO', 45.80),
 ('o7-lcb', 'Olmo3-7B', 'LiveCodeBench Avg', 410, 50.98, 'GRPO', 48.29),
 ('o7-he', 'Olmo3-7B', 'HumanEval+', 164, 90.24, 'GRPO／SDFT', 89.02),
 ('o7-mbpp', 'Olmo3-7B', 'MBPP+', 378, 73.28, 'SDFT', 73.02),
 ('o7-ep', 'Olmo3-7B', 'EvalPlus Avg', 542, 78.41, 'SDFT', 77.86),
]
def c4(row):
    _, _, _, N, a, _, b = row
    na, nb = round(a * N / 100), round(b * N / 100)
    ph = a / 100; se = math.sqrt(ph * (1 - ph) / N) * 100
    return dict(na=na, nb=nb, ra=a * N / 100, rb=b * N / 100, dn=na - nb, dpp=a - b, se=se, z=(a - b) / se)
print("== C4: count = round(pct*N/100); raw shown to check near-integer; SE in pp")
for row in ROWS:
    r = c4(row)
    print(f"{row[0]:8s} N={row[3]:4d} LOPD {row[4]:.2f}->{r['ra']:.3f}~{r['na']}  {row[5]} {row[6]:.2f}->{r['rb']:.3f}~{r['nb']}  "
          f"diff={r['dn']} ({r['dpp']:.2f}pp) SE={r['se']:.2f}pp diff/SE={r['z']:.2f}")

# ---------- paper arithmetic ----------
print("== aggregates")
T1 = {  # backbone/method: Env, Base, MissF, MissP, Long, BFCLavg, MStep, MTurn, ACEavg
 'q4 Vanilla': (48.6, 28.50, 20.50, 20.50, 22.00, 22.88, 48.3, 52.2, 50.6),
 'q4 GRPO': (61.8, 33.00, 20.50, 24.50, 23.00, 25.25, 58.3, 54.4, 56.0),
 'q4 Skill-SD': (59.1, 33.00, 19.00, 24.00, 22.50, 24.63, 56.6, 55.6, 56.0),
 'q4 OPSD': (51.2, 31.50, 19.50, 20.50, 29.00, 25.13, 41.6, 53.3, 48.6),
 'q4 SDPO': (50.6, 21.50, 14.00, 13.50, 14.00, 15.75, 45.0, 33.3, 38.0),
 'q4 SDFT': (50.3, 31.50, 17.00, 17.00, 21.50, 21.75, 53.3, 47.8, 50.0),
 'q4 LOPD': (63.7, 32.50, 26.50, 24.50, 26.00, 27.38, 56.6, 63.3, 60.6),
 'q8 Vanilla': (49.2, 34.00, 28.50, 25.50, 25.50, 28.38, 61.6, 50.0, 54.6),
 'q8 GRPO': (57.3, 36.00, 28.00, 27.00, 25.00, 29.00, 65.0, 53.3, 58.0),
 'q8 Skill-SD': (60.2, 33.00, 27.00, 25.00, 24.50, 27.38, 63.3, 51.1, 56.0),
 'q8 OPSD': (52.0, 30.50, 25.00, 23.50, 24.00, 25.75, 60.0, 47.8, 52.7),
 'q8 SDPO': (55.3, 29.00, 24.50, 23.00, 23.50, 25.00, 56.7, 48.9, 52.0),
 'q8 SDFT': (56.2, 31.50, 26.00, 24.50, 25.50, 26.88, 65.0, 47.8, 54.7),
 'q8 LOPD': (66.4, 37.00, 29.00, 27.50, 26.00, 29.88, 73.3, 55.6, 62.7),
}
bad = 0
for k, v in T1.items():
    bf = sum(v[1:5]) / 4; ace = (20 * v[6] + 30 * v[7]) / 50
    if abs(bf - v[5]) > 0.006 or abs(ace - v[8]) > 0.06: bad += 1; print("  mismatch", k, round(bf, 3), round(ace, 3))
print("Table1 BFCL simple mean / ACE 20:30 weighted mismatches:", bad)
T2 = {
 'q4 Vanilla': (47.67, 41.22, 45.61, 85.37, 75.93, 78.79), 'q4 GRPO': (50.18, 44.27, 48.29, 86.59, 76.19, 79.34),
 'q4 Skill-SD': (49.82, 41.98, 47.32, 85.98, 76.98, 79.70), 'q4 OPSD': (42.65, 35.11, 40.24, 85.36, 76.72, 79.33),
 'q4 SDPO': (40.86, 34.35, 38.78, 84.76, 74.87, 77.86), 'q4 SDFT': (48.75, 43.51, 47.07, 87.20, 76.98, 80.07),
 'q4 LOPD': (50.54, 45.04, 48.78, 87.80, 78.57, 81.36),
 'o7 Vanilla': (47.31, 44.27, 46.34, 86.59, 70.37, 75.28), 'o7 GRPO': (49.46, 45.80, 48.29, 89.02, 72.75, 77.67),
 'o7 Skill-SD': (49.10, 45.03, 47.80, 87.20, 72.22, 76.75), 'o7 OPSD': (44.09, 45.04, 44.39, 87.80, 71.96, 76.75),
 'o7 SDPO': (49.82, 43.51, 47.80, 88.41, 72.75, 77.49), 'o7 SDFT': (47.67, 44.27, 46.58, 89.02, 73.02, 77.86),
 'o7 LOPD': (53.41, 45.80, 50.98, 90.24, 73.28, 78.41),
}
bad = 0
for k, v in T2.items():
    l = (279 * v[0] + 131 * v[1]) / 410; e = (164 * v[3] + 378 * v[4]) / 542
    if abs(l - v[2]) > 0.006 or abs(e - v[5]) > 0.006: bad += 1; print("  mismatch", k, round(l, 3), round(e, 3))
print("Table2 LCB 279:131 / EvalPlus 164:378 weighted mismatches:", bad)
print("== deltas")
print("4B best-baseline margins: Env %.1f BFCL %.2f ACE %.1f" % (63.7 - 61.8, 27.38 - 25.25, 60.6 - 56.0))
print("8B best-baseline margins: Env %.1f BFCL %.2f ACE %.1f" % (66.4 - 60.2, 29.88 - 29.00, 62.7 - 58.0))
print("8B vs vanilla: BFCL %.2f Env %.1f" % (29.88 - 28.38, 66.4 - 49.2))
print("8B Env over OPSD/SDFT/Skill-SD %.1f/%.1f/%.1f ; ACE %.1f/%.1f/%.1f" % (66.4 - 52.0, 66.4 - 56.2, 66.4 - 60.2, 62.7 - 52.7, 62.7 - 54.7, 62.7 - 56.0))
print("Code margins: q4 LCB %.2f EP %.2f ; o7 LCB %.2f EP %.2f" % (48.78 - 48.29, 81.36 - 80.07, 50.98 - 48.29, 78.41 - 77.86))
print("== Table 3 / Fig 4 / dual bound")
print("first-step shorter: %.4f" % (1 - 6210 / 9937))
print("naive reward/(steps*calls): vanilla %.4f  base+comp %.4f  LOPD %.4f" % (0.486 / (11.12 * 3.50), 0.631 / (16.31 * 1.21), 0.637 / (17.04 * 1.11)))
print("Fig4 fractions: 320/1600=%.2f 576/1600=%.2f 480/1600=%.2f" % (320 / 1600, 576 / 1600, 480 / 1600))
print("A=2r-1: r=0.486 -> %.3f ; r=0.637 -> %.3f" % (2 * 0.486 - 1, 2 * 0.637 - 1))
print("beta growth per step at Delta=0: %.3f ; steps=1600/32=%d ; beta max %.3f" % (0.5 * 0.05, 1600 // 32, 0.5 * 0.05 * (1600 // 32)))
print("Fig5a K: 8 56.5 16 56.3 32 63.7 64 60.6 128 62.9 ; dips: 63.7-60.6=%.1f 62.9-60.6=%.1f" % (63.7 - 60.6, 62.9 - 60.6))

# ---------- derivation checks (page-derived formulas) ----------
print("== derivation checks (finite differences, random cases, seed 1)")
rnd = random.Random(1); worst = [0] * 5
for _ in range(200):
    V = rnd.randint(3, 8); z = [rnd.gauss(0, 1.5) for _ in range(V)]; u = [rnd.gauss(0, 1.5) for _ in range(V)]
    Qa = [rnd.uniform(-1, 1) for _ in range(V)]; beta = rnd.uniform(0, 0.99); h = 1e-6
    p, q = softmax(z), softmax(u); K = kl(p, q); Qb = sum(a * b for a, b in zip(p, Qa))
    Dl = lambda qq: sum(pi * x * math.log(qi / pi) for pi, x, qi in zip(p, Qa, qq))
    for k in range(V):
        up = u[:]; up[k] += h; um = u[:]; um[k] -= h
        fd = (kl(p, softmax(up)) - kl(p, softmax(um))) / (2 * h); worst[0] = max(worst[0], abs(fd - (q[k] - p[k])))
        fd = (Dl(softmax(up)) - Dl(softmax(um))) / (2 * h); worst[1] = max(worst[1], abs(fd - (p[k] * Qa[k] - q[k] * Qb)))
        zp = z[:]; zp[k] += h; zm = z[:]; zm[k] -= h
        fd = (kl(softmax(zp), q) - kl(softmax(zm), q)) / (2 * h)
        worst[2] = max(worst[2], abs(fd - p[k] * ((math.log(p[k]) - math.log(q[k])) - K)))
    qs = [pi * (1 + beta * x) for pi, x in zip(p, Qa)]; s = sum(qs); qs = [x / s for x in qs]
    g = [(qi - pi) - beta * (pi * x - qi * Qb) for pi, qi, x in zip(p, qs, Qa)]; worst[3] = max(worst[3], max(map(abs, g)))
    # log-sum: truncated KL <= full KL for random top-M of teacher
    M = rnd.randint(1, V - 1); order = sorted(range(V), key=lambda i: -q[i]); sup = order[:M]; tail = order[M:]
    tS = sum(p[i] for i in tail); tT = sum(q[i] for i in tail)
    tr = sum(p[i] * math.log(p[i] / q[i]) for i in sup) + tS * math.log(tS / tT)
    worst[4] = max(worst[4], tr - K)
print("max |dKL/du - (q-p)| = %.2e" % worst[0]); print("max |dDelta/du - (pQ - qQbar)| = %.2e" % worst[1])
print("max |dKL/dz - p[(log p - log q) - KL]| = %.2e" % worst[2]); print("max |grad at q*| = %.2e" % worst[3])
print("max (truncKL - fullKL) = %.2e (must be <= 0)" % worst[4])
