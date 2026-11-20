"""Acceptance numbers for tutorial 138 (GPPO). Pure Python, same algorithms the page uses.
Run: python3 check.py   (prints every table in design.md section 5; asserts the key identities)
"""
import math

LN = math.log
# ---------- generators (r>0). KL forms: 'abslog' = |log r| (paper 3.4 usage, not a valid convex f),
# 'nonneg' = r-1-log r (valid f, same divergence as -log r), 'neglog' = -log r (Eq. 16 literal)
F = {
    'TV':  lambda r: 0.5 * abs(r - 1),      # Eq.14: TV = 1/2|r-1|
    'TV2': lambda r: abs(r - 1),            # TV2 = |r-1| (= 2*TV); with delta = TV2(0.8) = 0.2 gives the same sets as TV with 0.1
    'chi2': lambda r: (r - 1) ** 2,
    'H':   lambda r: (math.sqrt(r) - 1) ** 2,
    'KLabs': lambda r: abs(LN(r)) if r > 0 else math.inf,
    'KLnn': lambda r: (r - 1 - LN(r)) if r > 0 else math.inf,
    'KLneg': lambda r: -LN(r) if r > 0 else math.inf,
}

def bisect(g, lo, hi, it=100):
    """g(lo)<=0<g(hi) (or reverse); returns root."""
    glo = g(lo)
    for _ in range(it):
        m = 0.5 * (lo + hi)
        gm = g(m)
        if (gm <= 0) == (glo <= 0):
            lo, glo = m, gm
        else:
            hi = m
    return 0.5 * (lo + hi)

def region(Dhat, delta, rmax_feas):
    """Sublevel set {r in (0, rmax_feas]: Dhat(r) <= delta} for Dhat convex with min 0 at r=1."""
    g = lambda r: Dhat(r) - delta
    # upper
    if g(rmax_feas) <= 0:
        hi, hi_cap = rmax_feas, True
    else:
        hi, hi_cap = bisect(g, 1.0, rmax_feas), False
    eps = 1e-12
    if g(eps) <= 0:
        lo, lo_cap = 0.0, True
    else:
        lo, lo_cap = bisect(g, eps, 1.0), False
    return lo, hi, lo_cap, hi_cap

def dhat(fname, est, mu):
    f = F[fname]
    if est == 'MC':
        return f
    if est == 'Atom':
        return lambda r: mu * f(r)
    if est == 'Bin':
        def b(r):
            q = (1 - mu * r) / (1 - mu)
            if q <= 0:
                return mu * f(r) + (1 - mu) * f(1e-300) if q == 0 else math.inf
            return mu * f(r) + (1 - mu) * f(q)
        return b

def cell(fname, est, mu, rmin=0.8):
    delta = F[fname](rmin)
    feas = 1.0 / mu if est != 'MC' else 1e6  # MC: no mu; page plots r in (0, 1/mu] anyway
    if est == 'MC':
        feas = 1.0 / mu
    return region(dhat(fname, est, mu), delta, feas), delta

# ---------- 1. PPO clip (C1)
def ppo(r, A, eps):
    c = min(max(r, 1 - eps), 1 + eps)
    return min(r * A, c * A)

def ppo_grad(r, A, eps):  # d/dr, zero where clipped branch active
    c = min(max(r, 1 - eps), 1 + eps)
    return 0.0 if c * A < r * A else A

print('== C1 PPO-Clip (eps=0.2)')
for A in (1, -1):
    row = [(r, round(ppo(r, A, .2), 3), ppo_grad(r, A, .2)) for r in (0.5, 0.8, 1.0, 1.2, 1.5)]
    print(' A=%+d' % A, row)
assert ppo_grad(1.5, 1, .2) == 0 and ppo_grad(0.5, 1, .2) == 1
assert ppo_grad(0.5, -1, .2) == 0 and ppo_grad(1.5, -1, .2) == -1

# ---------- 2. MC regions (C2)
print('\n== C2 MC regions: delta=f(r_min), [r_min, r_max]')
for rmin in (0.8, 0.7, 0.9):
    out = []
    for fn in ('TV2', 'TV', 'chi2', 'H', 'KLabs', 'KLnn'):
        (lo, hi, _, _), d = cell(fn, 'MC', 1e-6, rmin)
        out.append('%s d=%.6f [%.4f, %.4f]' % (fn, d, lo, hi))
    print(' r_min=%.2f: ' % rmin + ' | '.join(out))
(lo, hi, _, _), d = cell('KLneg', 'MC', 1e-6)
print(' KL literal -log r: d=%.6f  region [%.4f, %s]' % (d, lo, 'inf(no upper bound)' if hi > 1e5 else hi))
print(' TV Eq.14 literal (1/2|r-1|, delta=0.2):', [round(x, 4) for x in region(F['TV'], 0.2, 1e6)[:2]])
print(' DAPO text: delta=0.23 -> [%.4f, %.4f]' % (math.exp(-.23), math.exp(.23)))

# ---------- 3. Vajda (C3)
def g_beta(f, V, b):
    return b * f(1 + V / b) + (1 - b) * f(1 - V / (1 - b))

def L_f(f, V, J=1000):
    best = math.inf
    for j in range(J):
        b = (1 - V) * (j + 0.5) / J
        v = g_beta(f, V, b)
        if v < best:
            best = v
    return best

FV = {'TV': F['TV'], 'chi2': F['chi2'], 'KL': F['KLneg'], 'H': F['H']}
print('\n== C3 Vajda: L_f(V) on beta-midpoint grid J=1000; C_f = min_{V=i/500, i=1..499} L_f/V^2')
CF = {}
for name, f in FV.items():
    best, argV = math.inf, None
    for i in range(1, 500):
        V = i / 500
        q = L_f(f, V) / V ** 2
        if q < best:
            best, argV = q, V
    CF[name] = best
    Ls = ' '.join('L(%.2f)=%.5f' % (V, L_f(f, V)) for V in (0.1, 0.25, 0.5, 0.75))
    print(' %-4s C_f=%.4f at V=%.3f | %s' % (name, best, argV, Ls))
assert abs(CF['chi2'] - 4) < 1e-3 and abs(CF['KL'] - 2) < 1e-2 and abs(CF['H'] - 1) < 1e-2 and abs(CF['TV'] - 1) < 3e-3

# mulberry32 + Dirichlet(1) pairs, same as page
def mulberry32(seed):
    a = seed & 0xFFFFFFFF
    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt

def dirichlet(rnd, K):
    g = [-LN(1 - rnd()) for _ in range(K)]
    s = sum(g)
    return [x / s for x in g]

def pairs(seed, K, N=400):
    rnd = mulberry32(seed)
    return [(dirichlet(rnd, K), dirichlet(rnd, K)) for _ in range(N)]

def Df(f, P, Q):
    return sum(q * f(p / q) for p, q in zip(P, Q))

print(' random pairs (seed=1, N=400): violations of D_f >= L_f(V)*(1-1e-3), min D_f/L_f')
for K in (2, 3, 6):
    ps = pairs(1, K)
    for name, f in FV.items():
        viol, mr = 0, math.inf
        for P, Q in ps:
            V = 0.5 * sum(abs(p - q) for p, q in zip(P, Q))
            if V < 1e-9:
                continue
            L = L_f(f, V, 400)
            d = Df(f, P, Q)
            mr = min(mr, d / L)
            if d < L * (1 - 1e-3):
                viol += 1
        print('  K=%d %-4s viol=%d  min ratio=%.4f' % (K, name, viol, mr))
_r = mulberry32(1); print(' mulberry32(1) first 3:', [round(_r(), 9) for _ in range(3)])
P0, Q0 = pairs(1, 3)[0]
print(' first pair seed=1,K=3: P=%s Q=%s' % ([round(x, 4) for x in P0], [round(x, 4) for x in Q0]))

# log-y axis range for C3 (review pedagogy 4): [1e-4, 1e4] must cover L_f on V in [0.02, 0.98] and the scatter
for name, f in FV.items():
    mx = max(Df(f, P, Q) for K in (2, 3, 6) for P, Q in pairs(1, K))
    lo, hi = L_f(f, 0.02), L_f(f, 0.98)
    print('  axis %-4s L(0.02)=%.2e L(0.98)=%.3g scatter max=%.3g' % (name, lo, hi, mx))
    assert 1e-4 <= lo and hi <= 1e4 and mx <= 1e4

# binary pair attains bound
V, b = 0.3, 0.5
for name, f in FV.items():
    P, Q = [b + V, 1 - b - V], [b, 1 - b]
    assert abs(Df(f, P, Q) - g_beta(f, V, b)) < 1e-12
print(' binary pair V=0.3, beta=0.5: g_beta =', {n: round(g_beta(f, .3, .5), 5) for n, f in FV.items()},
      ' L_f(0.3) =', {n: round(L_f(f, .3), 5) for n, f in FV.items()})

# ---------- 4. regions vs mu (C4)
# Two calibrations. 'page' (default): delta = f(0.8); TV as |r-1| (delta 0.2, same sets as 1/2|r-1| with 0.1);
#   H delta = 0.011146; KL: MC/Atom use |log r| (page reading of paper 3.4, NOT an f-divergence),
#   Binary uses Eq.20 with f = -log r (Bernoulli KL), delta = -log 0.8.
# 'literal' (paper 4.1 values taken at face value): TV f = 1/2|r-1|, delta 0.2; chi2 0.04; H 0.01;
#   KL f = -log r, delta 0.22 (MC/Atom then have no upper bound; capped at 1/mu).
CAL = {
 'page':    {'TV': ('TV2', 'TV2', 'TV2', 0.2), 'chi2': ('chi2',)*3 + (0.04,), 'H': ('H',)*3 + (F['H'](0.8),),
             'KL': ('KLabs', 'KLabs', 'KLneg', -LN(0.8))},
 'literal': {'TV': ('TV',)*3 + (0.2,), 'chi2': ('chi2',)*3 + (0.04,), 'H': ('H',)*3 + (0.01,),
             'KL': ('KLneg',)*3 + (0.22,)},
}
def cell2(cal, div, est, mu):
    fm, fa, fb, delta = CAL[cal][div]
    fn = {'MC': fm, 'Atom': fa, 'Bin': fb}[est]
    D = dhat(fn, est, mu)
    # monotonicity on each side of r=1 (bisection has a unique root per side); checked on a grid
    xs = [1 + (1 / mu - 1) * k / 200 for k in range(201)]
    ys = [D(x) for x in xs]
    assert all(y2 >= y1 - 1e-12 for y1, y2 in zip(ys, ys[1:]) if math.isfinite(y2)) or fn == 'KLneg', (cal, div, est, mu)
    xl = [1e-6 + (1 - 1e-6) * k / 200 for k in range(201)]
    yl = [D(x) for x in xl]
    assert all(y2 <= y1 + 1e-12 for y1, y2 in zip(yl, yl[1:]) if math.isfinite(y1)), (cal, div, est, mu)
    return region(D, delta, 1.0 / mu)

def fmt(c):
    lo, hi, lc, hc = c
    return '[%s, %s]' % ('0*' if lc else '%.4f' % lo, ('1/mu*' if hc else ('%.4f' % hi if hi < 1e4 else '%.3g' % hi)))

for cal in ('page', 'literal'):
    print('\n== C4 regions, calibration=%s' % cal)
    for mu in (0.5, 0.1, 0.01, 0.001):
        print(' mu=%g  (feasible r <= 1/mu = %g)' % (mu, 1 / mu))
        for div in ('TV', 'chi2', 'H', 'KL'):
            print('   %-5s ' % div + ' | '.join('%s %s' % (e, fmt(cell2(cal, div, e, mu))) for e in ('MC', 'Atom', 'Bin')))
# closed-form cross-checks (page calibration)
for mu in (0.5, 0.1, 0.01):
    lo, hi, _, hc = cell2('page', 'TV', 'Bin', mu); assert hc or abs(hi - (1 + 0.1 / mu)) < 1e-9
    lo, hi, _, hc = cell2('literal', 'TV', 'Bin', mu); assert hc or abs(hi - (1 + 0.2 / mu)) < 1e-9   # DPPO 1 +- delta/mu
    lo, hi, _, hc = cell2('literal', 'TV', 'Atom', mu); assert hc or abs(hi - (1 + 0.4 / mu)) < 1e-9
    lo, hi, _, hc = cell2('page', 'chi2', 'Bin', mu); assert hc or abs(hi - (1 + math.sqrt(0.04 * (1 - mu) / mu))) < 1e-9
    lo, hi, _, hc = cell2('page', 'chi2', 'Atom', mu); assert hc or abs(hi - (1 + math.sqrt(0.04 / mu))) < 1e-9
for d in (-LN(0.8), 0.22):
    print(' delta=%.4f: |log r|-Atom upper > 10 when mu < %.4f ; lower < 0.5 when mu < %.4f' % (d, d / LN(10), d / LN(2)))
print(' chi2-Atom upper > 10 when mu < %.6f ; TV-Atom(page) upper > 10 when mu < %.4f' % (0.04 / 81, 0.2 / 9))
# Binary is the same FUNCTION under f -> f + c(r-1) ...
for mu in (0.3, 0.05):
    for r in (0.5, 1.7):
        if mu * r < 1:
            assert abs(dhat('KLnn', 'Bin', mu)(r) - dhat('KLneg', 'Bin', mu)(r)) < 1e-12
# ... but delta = f(0.8) differs by representation, so the REGION differs (review P0)
for mu in (0.5, 0.1):
    a = region(dhat('KLneg', 'Bin', mu), -LN(0.8), 1 / mu)
    bb = region(dhat('KLnn', 'Bin', mu), F['KLnn'](0.8), 1 / mu)
    print(' mu=%g Bernoulli-KL Binary: delta=-log0.8 %s vs delta=f_nn(0.8)=%.6f %s' % (mu, fmt(a), F['KLnn'](0.8), fmt(bb)))
    assert abs(a[1] - bb[1]) > 0.1
print(' Binary: same function under f+c(r-1), different region under delta=f(0.8) : ok')
for mu in (0.3, 0.05):
    a = region(dhat('TV2', 'Atom', mu), 0.2, 1 / mu)[:2]
    bb = region(dhat('TV2', 'Bin', mu), 0.4, 1 / mu)[:2]
    assert all(abs(x - y) < 1e-9 for x, y in zip(a, bb))
print(' TV-Atom(delta) == TV-Bin(2 delta) : ok')

# ---------- 5. estimator expectations (C5)
MU = [0.5, 0.25, 0.12, 0.08, 0.04, 0.01]
F5 = {'TV': F['TV'], 'chi2': F['chi2'], 'KL': F['KLnn'], 'H': F['H']}

def c5(lam, fname):
    f = F5[fname]
    w = [m ** (1 - lam) for m in MU]
    Z = sum(w)
    PI = [x / Z for x in w]
    rows = []
    for m, p in zip(MU, PI):
        r = p / m
        h = f(r)
        b = m * f(r) + (1 - m) * f((1 - p) / (1 - m))
        rows.append((m, p, r, h, m * h, b, m * h * h, m ** 3 * h * h))
    D = sum(m * h for m, p, r, h, *_ in rows)
    EA = sum(m * a for m, p, r, h, a, *_ in rows)
    EB = sum(row[0] * row[5] for row in rows)
    S2MC = sum(row[6] for row in rows)
    S2A = sum(row[7] for row in rows)
    return PI, rows, D, EA, EB, S2MC, S2A

print('\n== C5 mu=%s, pi ∝ mu^(1-lambda)' % MU)
for lam in (0.0, 0.3, 0.6):
    for fn in F5:
        PI, rows, D, EA, EB, S2MC, S2A = c5(lam, fn)
        print(' lam=%.1f %-4s D_f=E[MC]=%.5f  E[Atom]=%.5f  E[Bin]=%.5f  E[MC^2]=%.5f  E[Atom^2]=%.6f  Var_MC=%.5f Var_Atom=%.6f'
              % (lam, fn, D, EA, EB, S2MC, S2A, S2MC - D * D, S2A - EA * EA))
        assert EA <= EB + 1e-12 and EB <= D + 1e-12  # mu f <= Bin <= D_f for f>=0
PI, rows, *_ = c5(0.3, 'chi2')
print(' lam=0.3 pi =', [round(x, 4) for x in PI], ' r =', [round(r[2], 4) for r in rows])
print(' per-action chi2 (lam=0.3): v_MC =', [round(r[6], 5) for r in rows], ' v_Atom =', [round(r[7], 7) for r in rows])

# ---------- 6. Table 1
print('\n== Table 1 recompute')
T = {
 'math': {'TV-MC': [34.6, 20.8, 33.3, 21.9], 'TV-Bin': [46.7, 32.5, 42.5, 34.2], 'TV-Atom': [45.0, 28.3, 40.8, 30.9],
          'H-MC': [37.9, 30.8, 35.4, 23.1], 'H-Bin': [42.1, 32.9, 37.5, 29.0], 'H-Atom': [42.9, 32.5, 43.3, 29.0],
          'chi2-MC': [34.6, 20.8, 33.3, 21.9], 'chi2-Bin': [45.0, 29.2, 36.7, 28.1], 'chi2-Atom': [47.9, 32.5, 47.5, 34.2],
          'KL-MC': [45.0, 27.5, 37.5, 28.5], 'KL-Bin': [47.1, 29.6, 42.9, 29.0], 'KL-Atom': [48.9, 33.9, 48.7, 34.4],
          'GSPO': [42.9, 28.7, 36.3, 32.5], 'GMPO': [41.7, 30.0, 35.8, 33.0], 'SAPO': [37.3, 28.7, 41.7, 32.3],
          'CISPO': [31.7, 23.8, 38.3, 28.0], 'Clip-Cov': [36.3, 22.9, 36.6, 30.2], 'Base': [3.8, 3.3, 15.0, 10.8]},
 'code': {'TV-MC': [61.9, 57.4, 36.2, 54.2], 'TV-Bin': [63.5, 58.6, 37.6, 55.6], 'TV-Atom': [64.2, 60.5, 38.1, 56.7],
          'H-MC': [66.4, 58.2, 36.3, 56.8], 'H-Bin': [66.3, 60.2, 39.2, 57.7], 'H-Atom': [68.2, 63.7, 40.3, 60.3],
          'chi2-MC': [61.9, 57.4, 36.2, 54.2], 'chi2-Bin': [64.8, 59.3, 38.6, 56.3], 'chi2-Atom': [68.2, 62.3, 39.3, 58.9],
          'KL-MC': [64.0, 59.1, 37.4, 56.8], 'KL-Bin': [66.4, 61.4, 38.3, 57.3], 'KL-Atom': [70.7, 65.1, 42.3, 60.8],
          'GSPO': [63.8, 57.9, 36.6, 56.3], 'GMPO': [64.0, 59.2, 37.4, 56.8], 'SAPO': [63.7, 57.0, 37.5, 55.9],
          'CISPO': [61.6, 54.6, 35.1, 53.9], 'Clip-Cov': [62.9, 57.4, 36.0, 55.5], 'Base': [32.3, 32.1, 16.5, 32.1]},
}
PRINTED = {
 'math': {'TV-MC': 27.7, 'TV-Bin': 39.0, 'TV-Atom': 36.3, 'H-MC': 31.8, 'H-Bin': 35.4, 'H-Atom': 36.9, 'chi2-MC': 27.7,
          'chi2-Bin': 34.8, 'chi2-Atom': 40.5, 'KL-MC': 34.6, 'KL-Bin': 37.2, 'KL-Atom': 41.5, 'GSPO': 35.1, 'GMPO': 35.1,
          'SAPO': 35.0, 'CISPO': 30.5, 'Clip-Cov': 31.5, 'Base': 8.2},
 'code': {'TV-MC': 52.4, 'TV-Bin': 53.8, 'TV-Atom': 54.9, 'H-MC': 54.4, 'H-Bin': 55.9, 'H-Atom': 58.1, 'chi2-MC': 52.4,
          'chi2-Bin': 54.8, 'chi2-Atom': 57.2, 'KL-MC': 54.3, 'KL-Bin': 55.9, 'KL-Atom': 59.7, 'GSPO': 53.7, 'GMPO': 54.4,
          'SAPO': 53.5, 'CISPO': 51.3, 'Clip-Cov': 53.0, 'Base': 28.3}}
for dom, tab in T.items():
    avg = PRINTED[dom]
    for k, v in tab.items():  # printed Avg agrees with column mean to rounding
        assert abs(sum(v) / 4 - avg[k]) <= 0.051, (dom, k)
    base = avg['TV-MC']
    rel = {k: (avg[k] / base - 1) * 100 for k in avg}
    print(' [%s] avg:' % dom, avg)
    print('   rel%%: KL-Atom %.2f  chi2-Atom %.2f  GMPO %.2f' % (rel['KL-Atom'], rel['chi2-Atom'], rel['GMPO']))
    for est in ('Bin', 'Atom'):
        rels = []
        for dv in ('TV', 'H', 'chi2', 'KL'):
            m, e = avg[dv + '-MC'], avg[dv + '-' + est]
            rels.append((e / m - 1) * 100)
        mc_avg = sum(avg[dv + '-MC'] for dv in ('TV', 'H', 'chi2', 'KL')) / 4
        e_avg = sum(avg[dv + '-' + est] for dv in ('TV', 'H', 'chi2', 'KL')) / 4
        print('   %s vs MC: per-divergence %s mean=%.1f%% ; ratio of means=%.1f%%'
              % (est, [round(x, 1) for x in rels], sum(rels) / 4, (e_avg / mc_avg - 1) * 100))
    gm = avg['GMPO']
    below = [k for k in tab if ('Bin' in k or 'Atom' in k) and avg[k] < gm]
    print('   rel%% all:', {k: round(v, 2) for k, v in rel.items()})
    print('   Atomic/Binary cells below GMPO avg %.1f: %s' % (gm, below))
    print('   KL-Atom - chi2-Atom(RIPO) = %.1f points; KL-Atom - TV-MC = %.1f points'
          % (avg['KL-Atom'] - avg['chi2-Atom'], avg['KL-Atom'] - avg['TV-MC']))
print('\nall asserts passed')
