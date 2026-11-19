"""Acceptance numbers for tutorial 260 (SDPO). Same algorithms the page will use. Run: python3 -I check.py"""
import math

# ---------- C1: GRPO group advantage ----------
def grpo_adv(r):
    m = sum(r) / len(r)
    return [ri - m for ri in r]
def collapse_prob(p, G):
    return (1 - p) ** G + p ** G
print("C1 adv r=[1,0,0,1,0,0,0,0]:", grpo_adv([1,0,0,1,0,0,0,0]))
for G in (4, 8, 16):
    print(f"C1 collapse G={G}:", {p: round(collapse_prob(p, G), 4) for p in (0.01, 0.05, 0.1, 0.3, 0.5, 0.9)})

# ---------- C2: toy token table (illustrative, NOT from paper) ----------
# each row: token, [student top-3 probs], [teacher top-3 probs]; candidate 0 = sampled token
C2 = [
 ("def",               [.97,.01,.01],   [.97,.01,.01]),
 (" numbers_up_to_n",  [.90,.05,.03],   [.90,.05,.03]),
 ("(n):",              [.98,.01,.005],  [.97,.02,.005]),
 (" return",           [.80,.10,.05],   [.82,.09,.05]),
 (" list",             [.70,.25,.03],   [.68,.27,.03]),
 ("(range",            [.95,.02,.01],   [.95,.02,.01]),
 ("(1,",               [.92,.03,.03],   [.90,.02,.06]),
 (" n",                [.96,.02,.01],   [.93,.05,.005]),
 (" +",                [.85,.14,.005],  [.06,.92,.01]),
 (" 1",                [.97,.02,.005],  [.40,.55,.01]),
 ("))",                [.97,.02,.005],  [.95,.03,.01]),
]
def tail(ps): return max(0.0, 1 - sum(ps))
seqA = []
print("\nC2 per-position: A(y_t)=log q/pi ; topK(3)+tail KL ; top1+tail KL")
for tok, pi, q in C2:
    a = math.log(q[0] / pi[0]); seqA.append(a)
    def kl(pp, qq):
        s = sum(x * math.log(x / y) for x, y in zip(pp, qq))
        tp, tq = tail(pp), tail(qq)
        if tp > 0: s += tp * math.log(tp / tq)
        return s
    print(f"  {tok!r:20} A={a:+.3f}  KL3={kl(pi,q):.4f}  KL1={kl(pi[:1],q[:1]):.4f}  tails pi={tail(pi):.3f} q={tail(q):.3f}")
print("C2 sequence-level mean A:", round(sum(seqA) / len(seqA), 4), " sum:", round(sum(seqA), 4))
print("C2 per-logit A at ' +':", [round(math.log(y / x), 3) for x, y in zip(C2[8][1] + [tail(C2[8][1])], C2[8][2] + [tail(C2[8][2])])])
for c in (0, 3, 8):
    print(f"C2 GRPO adv of failed rollout, c={c}/8 successes:", -c / 8)

# ---------- C3: single-position logit distillation ----------
TOK = [" +", "))", " -", ")", " *", ","]
PI0 = [.85, .14, .005, .003, .001, .001]
Q   = [.06, .92, .01, .008, .0015, .0005]
assert abs(sum(PI0) - 1) < 1e-12 and abs(sum(Q) - 1) < 1e-12
def softmax(z):
    m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e); return [v / s for v in e]
def rkl(p, q): return sum(a * math.log(a / b) for a, b in zip(p, q) if a > 0)
def loss_grad(z, q, div):
    p = softmax(z)
    if div == "rkl":
        g = [math.log(a / b) for a, b in zip(p, q)]; L = sum(a * gi for a, gi in zip(p, g))
    elif div == "fkl":
        L = rkl(q, p); return L, [a - b for a, b in zip(p, q)]
    else:  # js, beta = 1/2
        m = [(a + b) / 2 for a, b in zip(p, q)]
        L = 0.5 * rkl(p, m) + 0.5 * rkl(q, m)
        g = [0.5 * math.log(a / c) for a, c in zip(p, m)]
    gbar = sum(a * gi for a, gi in zip(p, g))
    return L, [a * (gi - gbar) for a, gi in zip(p, g)]
# finite-difference check of analytic gradients (script only)
z0 = [math.log(v) for v in PI0]
for div in ("rkl", "fkl", "js"):
    L, g = loss_grad(z0, Q, div); h = 1e-6
    for k in range(6):
        zp = z0[:]; zp[k] += h; zm = z0[:]; zm[k] -= h
        fd = (loss_grad(zp, Q, div)[0] - loss_grad(zm, Q, div)[0]) / (2 * h)
        assert abs(fd - g[k]) < 1e-6, (div, k, fd, g[k])
print("\nC3 analytic gradients match finite differences")
# advantage-with-baseline identity for reverse KL: -dL/dz_k = pi_k (A_k - E_pi[A])
L, g = loss_grad(z0, Q, "rkl"); A = [math.log(b / a) for a, b in zip(PI0, Q)]
EA = sum(a * x for a, x in zip(PI0, A))
assert all(abs(-gk - p * (Ak - EA)) < 1e-12 for gk, p, Ak in zip(g, PI0, A)); assert abs(EA + L) < 1e-12
print("C3 step0: A=", [round(x, 3) for x in A], " E_pi[A]=", round(EA, 4), " -grad=", [round(-x, 4) for x in g])
for div in ("rkl", "fkl", "js"):
    for eta in (0.5, 1.0, 2.0):
        z = z0[:]; out = {}
        for s in range(201):
            if s in (0, 1, 10, 50, 200):
                p = softmax(z); out[s] = (round(loss_grad(z, Q, div)[0], 4), round(p[1], 4), round(rkl(p, Q), 4))
            _, g = loss_grad(z, Q, div); z = [a - eta * b for a, b in zip(z, g)]
        print(f"C3 {div} eta={eta}: step:(loss, pi['))'], revKL)", out)
# top-K + tail approximation (Eq. 11), topK w.r.t. student (index order on ties)
order = sorted(range(6), key=lambda i: (-PI0[i], i))
exact = rkl(PI0, Q)
for K in range(1, 7):
    S = order[:K]; R = order[K:]
    s = sum(PI0[i] * math.log(PI0[i] / Q[i]) for i in S)
    tp = sum(PI0[i] for i in R); tq = sum(Q[i] for i in R)
    if tp > 0: s += tp * math.log(tp / tq)
    print(f"C3 topK K={K}: approx={s:.5f} exact={exact:.5f} gap={exact - s:.5f}")
    assert s <= exact + 1e-12

# ---------- C4: trust-region teacher (Eq. 10/16) + EMA weight ----------
QREF = Q
QTH = [.70, .25, .02, .02, .005, .005]
assert abs(sum(QTH) - 1) < 1e-12
def tr(alpha):
    lz = [(1 - alpha) * math.log(a) + alpha * math.log(b) for a, b in zip(QREF, QTH)]
    m = max(lz); e = [math.exp(v - m) for v in lz]; s = sum(e); return [v / s for v in e]
print()
prev = -1
for al in (0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1):
    qs = tr(al); k1 = rkl(qs, QREF); k2 = rkl(qs, QTH)
    assert k1 >= prev - 1e-12; prev = k1
    print(f"C4 alpha={al}: q*[' +']={qs[0]:.4f} q*['))']={qs[1]:.4f} KL(q*||qref)={k1:.5f} KL(q*||qth)={k2:.5f}")
for al in (0.01, 0.05):
    print(f"C4 EMA weight of theta_0 (1-a)^n, a={al}:", {n: round((1 - al) ** n, 4) for n in (10, 20, 50, 80, 100, 200)})

# ---------- C5: discovery@k ----------
def disc(p, k): return -math.expm1(k * math.log1p(-p))
def k_for(p, target): return math.ceil(math.log1p(-target) / math.log1p(-p))
def tmean(p, B=2750): return -math.expm1(B * math.log1p(-p)) / p
def passk_est(n, c, k):
    if n - c < k: return 1.0
    prod = 1.0
    for i in range(k): prod *= (n - c - i) / (n - i)
    return 1 - prod
pv = 1 - 0.97 ** (1 / 64); ph = 1 - 0.5 ** (1 / 64)
print(f"\nC5 p threshold very hard (pass@64<0.03): {pv:.6e} -> expected successes in 2944: {2944*pv:.3f}")
print(f"C5 p threshold hard (pass@64<0.5): {ph:.6e} -> expected successes in 2944: {2944*ph:.3f}")
for p in (1e-5, 1e-4, pv, 1e-3, ph, 0.05):
    k22 = k_for(p, 0.22)
    print(f"C5 p={p:.3e}: pass@64={disc(p,64):.4f} disc@2750={disc(p,2750):.4f} k(22%)={k22 if k22<=2750 else '>2750'} k(50%)={k_for(p,.5)} tmean={tmean(p):.1f}")
for c in (0, 1, 2, 5, 32):
    print(f"C5 estimator n=2944 c={c}: pass@64={passk_est(2944,c,64):.4f} pass@2750={passk_est(2944,c,2750):.4f}")

# ---------- paper aggregate check (Sec 1: 70.2 vs 66.6) ----------
T3 = {"sdpo": [73.2,80.9,66.6,75.6,50.6,56.8,72.1,78.4,68.0,68.5, 68.0,80.0,59.9,66.1,48.0,52.8,73.7,79.1,60.8,62.1],
      "grpo": [65.9,74.5,63.8,72.7,35.1,59.9,74.3,77.1,64.9,67.7, 39.7,56.7,55.3,63.3,35.6,55.8,70.9,75.0,56.4,65.0]}
T7 = {"sdpo": [73.2,80.9,70.6,80.6,50.6,56.8,72.1,78.3,56.4,68.5, 68.0,80.0,60.3,71.4,48.0,52.8,75.3,79.2,57.3,62.5],
      "grpo": [65.9,74.5,62.9,74.5,35.1,59.9,74.3,77.1,61.7,68.1, 53.1,67.7,55.3,63.3,35.6,55.8,73.8,78.1,56.4,65.0]}
for name, T in (("T3", T3), ("T7", T7)):
    for m, v in T.items():
        print(f"agg {name} {m}: 5h mean={sum(v[1::2])/10:.2f} 1h mean={sum(v[0::2])/10:.2f}")
print("Table 8 ratios:", round(820.8/255.8, 2), round(1095.4/343.9, 2))
print("Table 10 Q3: 20 steps x 16 =", 20*16)
