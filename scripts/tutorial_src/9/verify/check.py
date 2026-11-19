"""Acceptance numbers for the Lafa tutorial widgets (design.md section 5).

Same algorithms, constants, PRNG and draw order as the page. Stdlib only.
Run: python3 -I check.py
"""
import math

# ---------- PRNG (identical to the page: mulberry32 + Box-Muller cos branch) ----------
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

def gauss(r):
    u1, u2 = r(), r()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)

# ---------- toy shared by C1 and C4 ----------
VOCAB = ["hot", "ice", "sour", "dog", "cream"]
EMB = [(math.cos(math.pi / 2 - 2 * math.pi * v / 5), math.sin(math.pi / 2 - 2 * math.pi * v / 5)) for v in range(5)]
PAIRS = [((0, 3), 0.4), ((1, 4), 0.3), ((2, 4), 0.3)]  # hot dog, ice cream, sour cream

def tau(i, t, lam):
    """tokenwise time; i = 0 (position 1) or 1 (position 2)."""
    if i == 0:
        return 1 - (1 - t) ** (1 + 2 * lam)
    return t

def denoiser(x, taus, sigma):
    """exact D_t(x)^i over VOCAB for the 2-token toy. x = [(x1,y1),(x2,y2)]."""
    logw = []
    for (c, p) in PAIRS:
        s = math.log(p)
        for i in range(2):
            var = max((1 - taus[i]) ** 2 * sigma ** 2, 1e-24)
            e = EMB[c[i]]
            dx = x[i][0] - taus[i] * e[0]
            dy = x[i][1] - taus[i] * e[1]
            s -= (dx * dx + dy * dy) / (2 * var)
        logw.append(s)
    m = max(logw)
    w = [math.exp(l - m) for l in logw]
    z = sum(w)
    w = [v / z for v in w]
    D = [[0.0] * 5, [0.0] * 5]
    for (c, _), wc in zip(PAIRS, w):
        D[0][c[0]] += wc
        D[1][c[1]] += wc
    return D

def meanemb(p):
    return (sum(p[v] * EMB[v][0] for v in range(5)), sum(p[v] * EMB[v][1] for v in range(5)))

def argmax(p):
    best = 0
    for v in range(1, len(p)):
        if p[v] > p[best]:
            best = v
    return best

def particles(seed, n):
    r = mulberry32(seed)
    return [[(gauss(r), gauss(r)), (gauss(r), gauss(r))] for _ in range(n)]

def flow_decode(g, sigma, lam, K):
    """Eq. 8 with exact denoiser; returns (y1, y2, trajectory)."""
    x = [(sigma * g[0][0], sigma * g[0][1]), (sigma * g[1][0], sigma * g[1][1])]
    traj = [x]
    p = None
    for k in range(K):
        tk, tk1 = k / K, (k + 1) / K
        ta = [tau(i, tk, lam) for i in range(2)]
        tb = [tau(i, tk1, lam) for i in range(2)]
        p = denoiser(x, ta, sigma)
        nx = []
        for i in range(2):
            a = (tb[i] - ta[i]) / (1 - ta[i])
            m = meanemb(p[i])
            nx.append((x[i][0] + a * (m[0] - x[i][0]), x[i][1] + a * (m[1] - x[i][1])))
        x = nx
        traj.append(x)
    return argmax(p[0]), argmax(p[1]), traj

def c1(sigma, lam, K, seed=7, n=500):
    ps = particles(seed, n)
    counts = {}
    for g in ps:
        y1, y2, _ = flow_decode(g, sigma, lam, K)
        counts[(y1, y2)] = counts.get((y1, y2), 0) + 1
    valid = sum(counts.get(c, 0) for c, _ in PAIRS) / n
    tv = 0.5 * sum(abs(counts.get(c, 0) / n - dict(PAIRS).get(c, 0.0))
                   for c in set(list(counts) + [c for c, _ in PAIRS]))
    freq = {c: counts.get(c, 0) / n for c, _ in PAIRS}
    return valid, tv, freq, counts

# ---------- C2 attention masks ----------
def mask(mode, N, B, R=2):
    """returns list of (query_label, key_label) allowed; labels index rows/cols."""
    if mode == "infer":  # Fig. 3: prefix N clean, block B noisy
        rows = [("c", i) for i in range(N)] + [("x", i) for i in range(B)]
        def ok(q, k):
            if q[0] == "c":
                return k[0] == "c" and k[1] <= q[1]
            return True
    elif mode == "train":  # Fig. 4b: clean R*B then noisy R*B
        rows = [("c", i) for i in range(R * B)] + [("x", i) for i in range(R * B)]
        def ok(q, k):
            if q[0] == "c":
                return k[0] == "c" and k[1] <= q[1]
            r = q[1] // B
            if k[0] == "c":
                return k[1] < r * B
            return k[1] // B == r
    else:  # Fig. 9b fused: prefix c (N), draft y (B), blocks z(1..B) each B
        rows = [("c", i) for i in range(N)] + [("y", i) for i in range(B)] + [("z", j * B + i) for j in range(B) for i in range(B)]
        def ok(q, k):
            pos = lambda u: u[1] if u[0] == "c" else N + u[1]
            if q[0] in "cy":
                return k[0] in "cy" and pos(k) <= pos(q)
            j = q[1] // B  # block index 0..B-1 -> candidate z^(j+1)
            if k[0] == "c":
                return True
            if k[0] == "y":
                return k[1] <= j  # c + y_{1:j+1}
            return k[1] // B == j
    n = sum(1 for q in rows for k in rows if ok(q, k))
    return len(rows), n

# ---------- C3 tokenwise time ----------
EPS = 0.05
M = 200
def ell(q, t):
    return EPS + (1 - t) ** q

def tau_binned(q, t):
    L = [ell(q, (m + 0.5) / M) for m in range(M)]
    S = sum(L)
    F = [0.0]
    for v in L:
        F.append(F[-1] + v / S)
    if t >= 1:
        return 1.0
    # last m with F[m] <= t
    lo, hi = 0, M
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if F[mid] <= t:
            lo = mid
        else:
            hi = mid
    m = lo
    return (m + (t - F[m]) / (F[m + 1] - F[m])) / M

def F_exact(q, t):
    tot = EPS + 1 / (q + 1)
    return (EPS * t + (1 - (1 - t) ** (q + 1)) / (q + 1)) / tot

def tau_exact(q, t):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if F_exact(q, mid) < t:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def taudot_exact(q, t):
    """d tau / dt = (int_0^1 ell) / ell(tau(t))  [本頁推導]"""
    return (EPS + 1 / (q + 1)) / ell(q, tau_exact(q, t))

# ---------- C4 two-time denoiser ----------
NF = 2000
def c4(g, sigma, lam, s, t, u=None):
    js, jt = round(s * NF), round(t * NF)
    ju = (js + jt) // 2 if u is None else round(u * NF)
    x0 = [(sigma * g[0][0], sigma * g[0][1]), (sigma * g[1][0], sigma * g[1][1])]
    # integrate from 0 to s first (state at time s along the teacher ODE)
    x = x0
    def advance(x, j0, j1, acc=None):
        for j in range(j0, j1):
            ta = [tau(i, j / NF, lam) for i in range(2)]
            tb = [tau(i, (j + 1) / NF, lam) for i in range(2)]
            p = denoiser(x, ta, sigma)
            nx = []
            for i in range(2):
                a = (tb[i] - ta[i]) / (1 - ta[i])
                m = meanemb(p[i])
                nx.append((x[i][0] + a * (m[0] - x[i][0]), x[i][1] + a * (m[1] - x[i][1])))
                if acc is not None:
                    dz = 1 / (1 - tb[i]) - 1 / (1 - ta[i])
                    for v in range(5):
                        acc[i][v] += dz * p[i][v]
            x = nx
        return x
    xs = advance(x, 0, js)
    acc = [[0.0] * 5, [0.0] * 5]
    xt = advance(xs, js, jt, acc)
    ts = [tau(i, js / NF, lam) for i in range(2)]
    tt = [tau(i, jt / NF, lam) for i in range(2)]
    tu = [tau(i, ju / NF, lam) for i in range(2)]
    Z = [1 / (1 - tt[i]) - 1 / (1 - ts[i]) for i in range(2)]
    delta = [[acc[i][v] / Z[i] for v in range(5)] for i in range(2)]
    # flow map via Eq. 7 with delta, and one Euler jump with D_s
    Ds = denoiser(xs, ts, sigma)
    res7, errE = [], []
    for i in range(2):
        a = (1 - tt[i]) / (1 - ts[i]); b = (tt[i] - ts[i]) / (1 - ts[i])
        m = meanemb(delta[i]); mE = meanemb(Ds[i])
        X7 = (a * xs[i][0] + b * m[0], a * xs[i][1] + b * m[1])
        XE = (a * xs[i][0] + b * mE[0], a * xs[i][1] + b * mE[1])
        res7.append(math.hypot(X7[0] - xt[i][0], X7[1] - xt[i][1]))
        errE.append(math.hypot(XE[0] - xt[i][0], XE[1] - xt[i][1]))
    gamma = [(1 - tt[i]) * (tu[i] - ts[i]) / ((1 - tu[i]) * (tt[i] - ts[i])) for i in range(2)]
    # semigroup check: delta_{s,t} vs gamma delta_{s,u} + (1-gamma) delta_{u,t}(X_{s,u})
    acc1 = [[0.0] * 5, [0.0] * 5]
    xu = advance(xs, js, ju, acc1)
    acc2 = [[0.0] * 5, [0.0] * 5]
    advance(xu, ju, jt, acc2)
    Zsu = [1 / (1 - tu[i]) - 1 / (1 - ts[i]) for i in range(2)]
    Zut = [1 / (1 - tt[i]) - 1 / (1 - tu[i]) for i in range(2)]
    semi = max(abs(delta[i][v] - (gamma[i] * acc1[i][v] / Zsu[i] + (1 - gamma[i]) * acc2[i][v] / Zut[i]))
               for i in range(2) for v in range(5))
    return dict(Ds=Ds, delta=delta, res7=res7, errE=errE, gamma=gamma, semi=semi, Z=Z, xt=xt)

# ---------- C5 self-speculative ----------
def c5_expected(B, r1, rho):
    r = [r1 * rho ** i for i in range(B)]
    ea, prod = 1.0, r[0]
    for j in range(1, B):
        prod *= r[j]
        ea += prod
    return ea

def c5_sim(B, r1, rho, T=1024, seed=1):
    rng = mulberry32(seed)
    r = [r1 * rho ** i for i in range(B)]
    done, loops, hist, pos = 0, 0, [], 0
    while done < T:
        ok = [rng() < r[i] for i in range(B)]
        if not ok[0]:
            a = 1; corr = True
        else:
            a = 1
            while a < B and ok[a]:
                a += 1
            corr = False
        a = min(a, T - done)
        done += a; loops += 1
        pos += (1 + B) if corr else (B + B * B)
        hist.append(a)
    return dict(T=done, F=loops + 1, TPF=done / (loops + 1), loops=loops, hist=hist, pos_per_fwd=pos / loops)

if __name__ == "__main__":
    print("== C1 flow toy (seed 7, N=500)")
    print("sigma lam K | valid  TV    | hotdog icecream sourcream | other pairs")
    for sigma in (1.0,):
        for lam in (0.0, 0.5, 1.0):
            for K in (1, 2, 4, 8, 16, 64):
                v, tv, f, cnt = c1(sigma, lam, K)
                other = {VOCAB[a] + " " + VOCAB[b]: n for (a, b), n in cnt.items() if (a, b) not in dict(PAIRS)}
                print(f"{sigma:4.2f} {lam:3.1f} {K:3d} | {v:.3f} {tv:.3f} | {f[(0,3)]:.3f} {f[(1,4)]:.3f} {f[(2,4)]:.3f} | {other}")
    for sigma in (0.25, 4.0):
        for K in (1, 2, 4, 64):
            v, tv, f, cnt = c1(sigma, 0.5, K)
            print(f"{sigma:4.2f} 0.5 {K:3d} | {v:.3f} {tv:.3f} | {f[(0,3)]:.3f} {f[(1,4)]:.3f} {f[(2,4)]:.3f}")
    ps = particles(7, 500)
    print("particle0 g =", ps[0])
    for K in (1, 2, 8, 64):
        y1, y2, tr = flow_decode(ps[0], 1.0, 0.5, K)
        print(" p0 K", K, VOCAB[y1], VOCAB[y2], "end", [tuple(round(c, 4) for c in pt) for pt in tr[-1]])
    D0 = denoiser([(0.3, -0.2), (1.0, 2.0)], [0.0, 0.0], 1.0)
    print(" D_0 (any x) pos1", [round(v, 4) for v in D0[0]], "pos2", [round(v, 4) for v in D0[1]])

    print("\n== C2 masks: (rows, allowed cells)")
    for mode, N, B in (("infer", 3, 3), ("infer", 4, 4), ("train", 0, 3), ("fused", 3, 3), ("fused", 2, 4)):
        print(mode, N, B, mask(mode, N, B))

    print("\n== C3 tokenwise time (eps=0.05, M=200)")
    for q1, dq in ((0.5, 0.5), (1.0, 0.0), (2.0, -0.5)):
        qs = [q1 + i * dq for i in range(4)]
        for t in (0.25, 0.5, 0.75):
            row = []
            for q in qs:
                if q <= 0:
                    row.append("q<=0"); continue
                tb, te = tau_binned(q, t), tau_exact(q, t)
                row.append(f"{tb:.4f}/{te:.4f}")
            print(f"q={qs} t={t}: binned/exact", row)
        print("   taudot(0.5) exact", [round(taudot_exact(q, 0.5), 4) for q in qs if q > 0])
    print("   K=8 grid tau for q=0.5:", [round(tau_binned(0.5, k / 8), 4) for k in range(9)])
    print("   K=8 grid tau for q=2.0:", [round(tau_binned(2.0, k / 8), 4) for k in range(9)])
    worst = max(abs(tau_binned(q, k / 100) - tau_exact(q, k / 100)) for q in (0.5, 1, 1.5, 2, 3, 4, 5.5) for k in range(101))
    print("   max |binned-exact| over q grid:", worst)

    print("\n== C4 two-time denoiser (seed 7 particle 0, sigma 1, lam 0.5)")
    for (s, t) in ((0.0, 0.5), (0.0, 0.9), (0.0, 0.99), (0.3, 0.6), (0.5, 0.99)):
        r = c4(ps[0], 1.0, 0.5, s, t)
        print(f"s={s} t={t}")
        for i in range(2):
            print("   pos", i + 1, "D_s", [round(v, 3) for v in r["Ds"][i]], "delta", [round(v, 3) for v in r["delta"][i]],
                  "argmax", VOCAB[argmax(r["Ds"][i])], "->", VOCAB[argmax(r["delta"][i])])
        print("   res(Eq7) ", [f"{v:.1e}" for v in r["res7"]], " Euler-jump err", [round(v, 4) for v in r["errE"]],
              " gamma(u=mid)", [round(v, 4) for v in r["gamma"]], " semigroup resid", f"{r['semi']:.1e}")
    print("C4 stress (lam 1, zeta up to 1e6):")
    for (s, t) in ((0.98, 0.99), (0.0, 0.99)):
        r = c4(ps[0], 1.0, 1.0, s, t)
        print(f"   s={s} t={t} res(Eq7) max {max(r['res7']):.1e} semigroup resid {r['semi']:.1e} gamma", [round(v, 4) for v in r["gamma"]])

    print("\n== C5 self-speculative (T=1024, seed 1)")
    for B in (4, 8, 16):
        for r1, rho in ((0.95, 0.9), (0.9, 0.8), (1.0, 1.0), (0.5, 0.9)):
            e = c5_expected(B, r1, rho)
            s = c5_sim(B, r1, rho)
            print(f"B={B:2d} r1={r1} rho={rho}: E[a]={e:.4f} sim T={s['T']} F={s['F']} TPF={s['TPF']:.4f} pos/fwd={s['pos_per_fwd']:.1f}")
    print("B=8 default first 12 a:", c5_sim(8, 0.95, 0.9)["hist"][:12])
    print("T=256 (sampling-error comparison):", [(B, round(c5_sim(B, 0.95, 0.9, T=256)["TPF"], 4)) for B in (4, 8, 16)])
    print("\n== paper ratios")
    print("TPS 100.2/39.5", 100.2 / 39.5, " 79.9/39.5", 79.9 / 39.5, " 93.8/39.5", 93.8 / 39.5)
    print("500/27", 500 / 27, " B+B^2 B=4,8,16:", [b + b * b for b in (4, 8, 16)])
