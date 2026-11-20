# Verification for tutorial 129 (score centering). Same algorithms as the page (design.md §4).
# All toy models use logit coordinates: theta = logits z of one prefix, s_v = e_v - p.
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def softmax(z):
    z = np.asarray(z, float); e = np.exp(z - z.max()); return e / e.sum()

# ---------------- C1: drift / signal decomposition ----------------
C1_Z = np.array([1.2, 0.6, 0.0, -0.8])
C1_XI = np.array([-0.5, 0.8, 0.2, -0.5])
C1_PRESET = {'const': [1, 1, 1, 1], 'b01': [0.9, 0.2, 0.6, 0.1], 'pm1': [0.8, -0.6, 0.2, -0.8]}

def c1(eps, Q):
    Q = np.asarray(Q, float); p = softmax(C1_Z); q = softmax(C1_Z + eps * C1_XI)
    Rq, Rp = q @ Q, p @ Q
    sbar = q - p
    pg = q * Q - Rq * p
    drift = Rq * sbar
    signal = q * (Q - Rq)
    onp = p * (Q - Rp)
    r = p / q
    tis = q * np.minimum(r, 2) * Q - (q * np.minimum(r, 2) * Q).sum() * p
    assert np.allclose(pg, drift + signal, atol=1e-12)
    assert abs(p.sum() - 1) < 1e-12 and abs((p * 0 + (p @ np.eye(4) - p)).sum()) < 1e-12
    return dict(p=p, q=q, Rq=Rq, Rp=Rp, sbar=sbar, pg=pg, drift=drift, signal=signal, onp=onp, tis=tis)

print('=== C1 ===')
for name, Q in C1_PRESET.items():
    for eps in [0.0, 0.5, 1.0]:
        o = c1(eps, Q)
        print(f'{name} eps={eps}: p={o["p"]} q={o["q"]} Eq[R]={o["Rq"]:.4f}')
        print(f'   sbar={o["sbar"]} |sbar|={np.linalg.norm(o["sbar"]):.4f}')
        print(f'   PG={o["pg"]} |PG|={np.linalg.norm(o["pg"]):.4f}')
        print(f'   drift={o["drift"]} |drift|={np.linalg.norm(o["drift"]):.4f}')
        print(f'   signal(=SC)={o["signal"]} |signal|={np.linalg.norm(o["signal"]):.4f}')
        print(f'   onpolicy={o["onp"]} |onp|={np.linalg.norm(o["onp"]):.4f} |SC-onp|={np.linalg.norm(o["signal"]-o["onp"]):.4f}')
# shift invariance check: pm1 = 2*b01 - 1 ; SC(pm1) = 2*SC(b01)
a, b = c1(0.5, C1_PRESET['b01']), c1(0.5, C1_PRESET['pm1'])
assert np.allclose(b['signal'], 2 * a['signal'])
print('shift check ok: SC(pm1)=2*SC(b01); PG(pm1)-2PG(b01) = -sbar*1 :', b['pg'] - 2 * a['pg'], -a['sbar'])

# ---------------- C2: group centering, two-step prefix ----------------
def c2(a, b, piA, G):
    R01 = piA * a + (1 - piA) * b        # E_q[R] under +1/0
    out = {}
    out['b01'] = (a, b)
    out['pm1'] = (2 * a - 1, 2 * b - 1)
    g = 1 - 1 / G
    out['gc'] = (g * 2 * (a - R01), g * 2 * (b - R01))   # group-centered from +-1 rewards
    out['sc'] = (0.0, 0.0)
    return R01, out

def c2_mc(a, b, piA, G, n=400000, seed=0):
    rng = np.random.default_rng(seed)
    y1 = rng.random((n, G)) < piA
    pr = np.where(y1, a, b)
    R = np.where(rng.random((n, G)) < pr, 1.0, -1.0)
    A = R - R.mean(1, keepdims=True)
    return A[y1].mean(), A[~y1].mean()

print('=== C2 ===')
for (a, b, piA, G) in [(0.8, 0.2, 0.5, 8), (0.8, 0.2, 0.5, 2), (0.8, 0.2, 0.8, 8), (0.9, 0.6, 0.5, 8), (0.5, 0.5, 0.5, 8)]:
    R01, o = c2(a, b, piA, G)
    wavg = {k: piA * v[0] + (1 - piA) * v[1] for k, v in o.items()}
    rms = {k: np.sqrt(piA * v[0] ** 2 + (1 - piA) * v[1] ** 2) for k, v in o.items()}
    mc = c2_mc(a, b, piA, G)
    print(f'a={a} b={b} piA={piA} G={G} R01={R01:.3f}')
    for k in o: print(f'   {k}: cA={o[k][0]:+.4f} cB={o[k][1]:+.4f} wavg={wavg[k]:+.4f} rms={rms[k]:.4f}')
    print(f'   MC check gc: {mc[0]:+.4f} {mc[1]:+.4f}')
    assert abs(mc[0] - o['gc'][0]) < 0.01 and abs(mc[1] - o['gc'][1]) < 0.01

# ---------------- C3: feedback-loop logit bandit (ILLUSTRATION) ----------------
C3_XI = np.array([0.6, -0.2, 0.1, -0.3, -0.2])
C3_Q = {'const': np.ones(5), 'b01': np.array([1, 1, 0, 0, 0.0])}

def H(p): return float(-(p * np.log(np.maximum(p, 1e-300))).sum())

def c3(eps, mode, eta=1.0, T=3000):
    Q = C3_Q[mode]; z0 = np.zeros(5); d = eps * C3_XI
    qfix = softmax(z0 + d)
    z = {m: z0.copy() for m in ['pg', 'fix', 'sc', 'onp']}
    hist = {m: [] for m in z}
    for t in range(T + 1):
        for m in z:
            p = softmax(z[m])
            hist[m].append((H(p), float(p @ Q), float(np.linalg.norm(z[m] - z0))))
            if t == T: continue
            if m == 'pg':  q = softmax(z[m] + d); g = q * Q - (q @ Q) * p
            elif m == 'fix': q = qfix; g = q * Q - (q @ Q) * p
            elif m == 'sc':  q = softmax(z[m] + d); g = q * (Q - q @ Q)
            else:            g = p * (Q - p @ Q)
            z[m] = z[m] + eta * g
    return hist

def thalf(h):
    H0 = h[0][0]
    for t, x in enumerate(h):
        if x[0] <= H0 / 2: return t
    return None

print('=== C3 (illustration) ===')
print('H0 =', np.log(5))
for mode in ['const', 'b01']:
    for eps in [0.05, 0.1, 0.2, 0.5, 1.0]:
        h = c3(eps, mode)
        row = ' '.join(f'{m}:H={h[m][-1][0]:.3f},acc={h[m][-1][1]:.3f},|dz|={h[m][-1][2]:.2f}' for m in h)
        hs = {m: [round(h[m][t][0], 3) for t in (100, 1000, 3000)] for m in h}
        print(f'{mode} eps={eps}: {row}  t_half(pg)={thalf(h["pg"])}')
        print(f'      H@100/1000/3000 {hs}')
# fixed teacher equilibrium: const -> p = qfix; b01 -> p = q*Q/(q.Q)
for eps in [0.1, 1.0]:
    qf = softmax(eps * C3_XI)
    print('fixed teacher target const H=', round(H(qf), 4), ' b01 H=', round(H(qf * C3_Q['b01'] / (qf @ C3_Q['b01'])), 4))

# ---------------- C4: top-k reconstruction + IS composition ----------------
C4_V = 20
C4_Z = -0.4 * np.arange(C4_V)
C4_XI = np.array([0.31, -1.12, 0.54, 0.87, -0.43, 1.25, -0.76, 0.12, -1.38, 0.66,
                  0.95, -0.21, -0.88, 1.47, 0.03, -0.59, 1.06, -1.21, 0.38, -0.27])
EPS = 1e-6
F = {'sc': lambda r: np.ones_like(np.asarray(r, float)),
     'tis': lambda r: np.minimum(r, 2.0),
     'mis': lambda r: np.where((np.asarray(r) >= 0.5) & (np.asarray(r) <= 5.0), r, 0.0),
     'is': lambda r: np.asarray(r, float)}

def c4(sigma, k, fname):
    f = F[fname]
    p = softmax(C4_Z); q = softmax(C4_Z + sigma * C4_XI)
    order = np.argsort(-q, kind='stable'); H_ = np.sort(order[:k]); T_ = np.sort(order[k:])
    PT = max(p[T_].sum(), EPS); QT = max(q[T_].sum(), EPS)
    rho = QT / PT
    alpha = rho * float(f(np.array(1 / rho)))
    w = f(p / q)
    E = np.eye(C4_V); S = E - p           # row v = s_v
    exact = (q * w) @ S                    # E_q[w s]
    coef = q[H_] * w[H_] - alpha * p[H_]
    hat = coef @ S[H_]                     # head-only formula (Eq. 8 / 10)
    qhat = q.copy(); qhat[T_] = rho * p[T_]
    what = f(p / qhat)
    full_hat = (qhat * what) @ S           # E_qhat[w s], full-vocab sum
    sbar = q - p
    ns = np.linalg.norm(sbar)
    return dict(rho=rho, alpha=alpha, QT=q[T_].sum(), PT=p[T_].sum(),
                exact=np.linalg.norm(exact), hat=np.linalg.norm(hat),
                resid=np.linalg.norm(exact - hat), rel=(np.linalg.norm(exact - hat) / ns if ns > 0 else 0.0),
                ident=np.linalg.norm(hat - full_hat), sbar=ns)

print('=== C4 ===')
p = softmax(C4_Z)
for sigma in [0.0, 0.5, 1.0]:
    q = softmax(C4_Z + sigma * C4_XI)
    print(f'sigma={sigma}: p[:5]={p[:5]} q[:5]={q[:5]} max r={np.max(p/q):.3f} min r={np.min(p/q):.3f} |sbar|={np.linalg.norm(q-p):.4f}')
    for fname in ['sc', 'tis', 'mis', 'is']:
        for k in [1, 4, 8, 16, 20]:
            o = c4(sigma, k, fname)
            print(f'  {fname:3s} k={k:2d}: rho={o["rho"]:.4f} alpha={o["alpha"]:.4f} QT={o["QT"]:.4f} PT={o["PT"]:.4f} '
                  f'|exact|={o["exact"]:.4f} |hat|={o["hat"]:.4f} resid={o["resid"]:.5f} rel={o["rel"]:.4f} ident={o["ident"]:.1e}')
            assert o['ident'] < 1e-12
print('all asserts passed')

# ---- C1b extra: TIS (f=min(r,2)) under constant reward, where clip binds ----
print('=== C1b TIS ===')
for eps in [0.5, 1.0, 1.5, 2.0]:
    p = softmax(C1_Z); q = softmax(C1_Z + eps * C1_XI); r = p / q
    for name in ['const', 'b01']:
        o = c1(eps, C1_PRESET[name])
        print(f'eps={eps} {name}: r={r} |TIS|={np.linalg.norm(o["tis"]):.4f} |TIS-onp|={np.linalg.norm(o["tis"]-o["onp"]):.4f}')
lo, hi = 0.0, 2.0
for _ in range(60):
    m = (lo + hi) / 2
    p = softmax(C1_Z); q = softmax(C1_Z + m * C1_XI)
    (lo, hi) = (lo, m) if (p / q).max() > 2 else (m, hi)
print(f'TIS clip starts binding at eps ~ {hi:.3f}')
print('C3 sc b01 t to H<=0.35:', [(e, next(t for t, x in enumerate(c3(e, "b01")["sc"]) if x[0] <= 0.35)) for e in [0.1, 0.5]])
print('done')

# ---- round 1 additions ----
print('=== round1 ===')
# C2: G -> infinity limit vs +-1 values (review F1)
for piA in [0.5, 0.8]:
    _, o = c2(0.8, 0.2, piA, 10**6)
    print(f'C2 piA={piA} G=1e6: gc cA={o["gc"][0]:+.4f} cB={o["gc"][1]:+.4f}  pm1 cA={o["pm1"][0]:+.4f} cB={o["pm1"][1]:+.4f}')
_, o = c2(0.8, 0.2, 0.8, 10**6); assert abs(o['gc'][0] - 0.24) < 1e-5
_, o = c2(0.8, 0.2, 0.5, 10**6); assert abs(o['gc'][0] - 0.6) < 1e-5
# C3: t_half for every curve ('未達到' = None), log-x readouts unchanged
for mode, lam in [('const', 0.1), ('const', 1.0), ('b01', 0.1), ('b01', 0.5)]:
    h = c3(lam, mode)
    print(f'C3 {mode} lam={lam}: t_half', {m: thalf(h[m]) for m in h}, ' |z_N-z0| pg', round(h['pg'][-1][2], 2))
# C3 t_half ~ 1/lambda check (toy property, not a paper claim)
ts = {lam: thalf(c3(lam, 'const')['pg']) for lam in [0.02, 0.05, 0.1, 0.2]}
print('C3 const t_half*lam:', {k: round(k * v, 2) for k, v in ts.items()})
# C4: sigma=0 relative residual must be guarded ('—'), not NaN
o = c4(0.0, 8, 'sc'); print('C4 sigma=0 sbar norm', o['sbar'], '-> page shows "—"'); assert o['sbar'] < 1e-12
print('round1 asserts passed')
