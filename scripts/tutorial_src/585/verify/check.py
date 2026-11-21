"""Acceptance numbers for the HiT tutorial widgets (design.md section 5).
Same algorithms as the page. Run: python3 -I check.py
"""
import math

# ---------- shared PRNG (mulberry32 + Box-Muller cos branch), identical to page ----------
def mulberry32(seed):
    s = [seed & 0xFFFFFFFF]
    def r():
        s[0] = (s[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = s[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r

def gauss(r):
    u1, u2 = r(), r()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)

# ---------- C1: affinity, average linkage, soft weights, pooling ----------
def affinity(qa, ka):
    """Eq. 2 with 2-D unit vectors given by angles (deg)."""
    q = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in qa]
    k = [(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in ka]
    n = len(q)
    return [[0.5 * (q[i][0]*k[j][0] + q[i][1]*k[j][1] + q[j][0]*k[i][0] + q[j][1]*k[i][1])
             for j in range(n)] for i in range(n)]

def avg_linkage(S):
    """Full merge tree. Returns list of (score, clusterA, clusterB) in merge order.
    Ties: smallest (min index of A, min index of B) pair wins."""
    clusters = [[i] for i in range(len(S))]
    merges = []
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                U, V = clusters[a], clusters[b]
                s = sum(S[i][j] for i in U for j in V) / (len(U) * len(V))  # Eq. 8
                if best is None or s > best[0] + 1e-12:
                    best = (s, a, b)
        s, a, b = best
        merges.append((s, clusters[a][:], clusters[b][:]))
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]
    return merges

def hard_groups(S, tau):
    clusters = [[i] for i in range(len(S))]
    for s, U, V in avg_linkage(S):
        if s < tau:
            break  # merge scores are non-increasing for average linkage
        clusters = [c for c in clusters if c[0] not in U and c[0] not in V] + [sorted(U + V)]
    return sorted(clusters, key=lambda c: c[0])

def soft(S, groups, Tr):
    """Eq. 9 and P (App. A.1). Max-subtracted softmax."""
    n, K = len(S), len(groups)
    A = []
    for i in range(n):
        e = [sum(S[i][j] for j in g) / len(g) for g in groups]
        m = max(e)
        w = [math.exp((x - m) / Tr) for x in e]
        z = sum(w)
        A.append([x / z for x in w])
    col = [sum(A[i][g] for i in range(n)) for g in range(K)]
    P = [[A[i][g] / col[g] for g in range(K)] for i in range(n)]
    return A, P

def sig(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))

def k_soft(merge_scores, N, tau, Tb):
    """Eq. 11."""
    return N - sum(sig((s - tau) / Tb) for s in merge_scores)

C1_ANGLES = [10, 22, 150, 38, 165, 178]

def c1(angles, tau, Tr, Tb=0.1):
    S = affinity(angles, angles)
    G = hard_groups(S, tau)
    A, P = soft(S, G, Tr)
    ms = [m[0] for m in avg_linkage(S)]
    return S, G, A, P, ms, k_soft(ms, len(angles), tau, Tb)

# ---------- C2: batch compression budget ----------
C2_CENTERS = [0.92, 0.80, 0.68, 0.50]  # per-example "simplicity" (推定 toy values)
C2_N = 256

def c2_scores(b):
    r = mulberry32(101 + b)
    xs = [max(-1.0, min(1.0, C2_CENTERS[b] + 0.18 * gauss(r))) for _ in range(C2_N - 1)]
    out = []
    for x in xs:
        q = round(x * 10000)          # 4-decimal integer
        if q % 50 == 0:               # never on the tau grid i/200 -> no ties
            q += 1 if q < 10000 else -1
        out.append(q / 10000)
    return sorted(out, reverse=True)

C2_SC = [c2_scores(b) for b in range(4)]

def c2(tau, rho, Tb=0.1):
    Kh = [C2_N - sum(1 for s in sc if s >= tau) for sc in C2_SC]
    Ks = [k_soft(sc, C2_N, tau, Tb) for sc in C2_SC]
    fh = sum(Kh) / (4 * C2_N)
    fs = sum(Ks) / (4 * C2_N)
    L4 = 1 + (fh - rho) * (fs - rho) / (rho * (1 - rho))       # Eq. 4
    r = 1 / rho
    L10 = r / (r - 1) * ((r - 1) * fh * fs + (1 - fh) * (1 - fs))  # Eq. 10
    g = (fh - rho) / (rho * (1 - rho))                          # dL/dfs
    return Kh, Ks, fh, fs, L4, L10, g

TAU_I = range(-40, 201)  # slider grid tau = i/200, i.e. [-0.2, 1] step 0.005

def c2_calibrate(rho):
    """Smallest grid tau with fh >= rho (fh is non-decreasing in tau). Page's own rule (推定)."""
    for i in TAU_I:
        if c2(i / 200, rho)[2] >= rho:
            return i
    return TAU_I[-1]

# ---------- C3: FLOPs model fit ----------
def layout_tokens(layout, counts):
    """layout: block counts in execution order; counts: image tokens per level (len = levels+1)."""
    J = len(counts) - 1
    lv = list(range(J + 1)) + list(range(J - 1, -1, -1))
    assert len(layout) == len(lv)
    return [(layout[i], counts[lv[i]]) for i in range(len(layout))]

DATA = [  # (name, layout, image-token counts, GFLOPs) from Tables 4,5,7,12 (all D18, in-context, width 768)
    ("4x", [3, 12, 3], [256, 64], 19.1),
    ("2x4", [3, 3, 6, 3, 3], [256, 128, 32], 20.8),
    ("4x2", [3, 3, 6, 3, 3], [256, 64, 32], 17.8),
    ("4x4", [3, 3, 6, 3, 3], [256, 64, 16], 17.1),
    ("2x2", [3, 3, 6, 3, 3], [256, 128, 64], 22.2),
    ("2/2/10", [2, 2, 10, 2, 2], [256, 128, 64], 18.3),
    ("2/3/8", [2, 3, 8, 3, 2], [256, 128, 64], 19.3),
    ("3/2/8", [3, 2, 8, 2, 3], [256, 128, 64], 21.3),
    ("3st222", [3, 3, 2, 2, 2, 3, 3], [256, 128, 64, 32], 21.9),
    ("3st224", [3, 3, 2, 2, 2, 3, 3], [256, 128, 64, 16], 21.6),
    ("3st222b", [2, 2, 2, 6, 2, 2, 2], [256, 128, 64, 32], 17.1),
]
COND = 9  # 8 class tokens + 1 time token, present at every stage (App. A.4)
D_MODEL = 768

def features(layout, counts):
    lin = quad = 0.0
    for nb, n in layout_tokens(layout, counts):
        m = n + COND
        lin += nb * m
        quad += nb * m * m
    return lin, quad

def fit():
    # theoretical quadratic coefficient: QK^T and AV, 2*m^2*d multiply-adds per block
    beta = 2 * D_MODEL / 1e9
    rows = [(features(l, c), g) for _, l, c, g in DATA]
    # least squares for g - beta*quad = alpha*lin + gamma
    X = [(f[0], 1.0) for f, _ in rows]
    y = [g - beta * f[1] for f, g in rows]
    sxx = sum(a * a for a, _ in X); sx = sum(a for a, _ in X); n = len(X)
    sxy = sum(a * t for (a, _), t in zip(X, y)); sy = sum(y)
    alpha = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    gamma = (sy - alpha * sx) / n
    return alpha, beta, gamma

def c3_pred(layout, counts, coef):
    a, b, g = coef
    lin, quad = features(layout, counts)
    return a * lin + b * quad + g

# ---------- C4: time shift Eq. 16, guidance schedule Eq. 15 ----------
def shift(t, k):
    s = 1 - t
    return 1 - k * s / (1 + (k - 1) * s)

def c4(k, steps=50):
    grid = [shift(i / steps, k) for i in range(steps + 1)]
    starts = [round(t, 9) for t in grid[:-1]]  # page rule: round t' to 1e-9, then compare
    n_noisy = sum(1 for t in starts if t < 0.5)
    n_cfg = sum(1 for t in starts if 0.1 <= t <= 1.0)
    n_ru = sum(1 for t in starts if t >= 0.85)
    return grid, n_noisy, n_cfg, n_ru


if __name__ == "__main__":
    print("== C1 (angles", C1_ANGLES, ") ==")
    for tau, Tr in [(0.8, 0.2), (0.8, 0.05), (0.95, 0.2), (0.5, 0.2), (-1.0, 0.2)]:
        S, G, A, P, ms, ks = c1(C1_ANGLES, tau, Tr)
        print(f"tau={tau} Tr={Tr} K={len(G)} groups={[[i+1 for i in g] for g in G]} Ksoft={ks:.3f}")
        print("  A rows:", [[round(x, 3) for x in row] for row in A])
        print("  P col0:", [round(P[i][0], 3) for i in range(len(A))])
    S = affinity(C1_ANGLES, C1_ANGLES)
    print("  merge scores:", [round(m[0], 4) for m in avg_linkage(S)])
    print("  S12, S13:", round(S[0][1], 4), round(S[0][2], 4))
    # monotone check
    for seed in range(200):
        r = mulberry32(seed)
        ang = [r() * 360 for _ in range(8)]
        ms = [m[0] for m in avg_linkage(affinity(ang, ang))]
        assert all(ms[i] >= ms[i + 1] - 1e-12 for i in range(len(ms) - 1)), seed
    print("  average-linkage merge scores non-increasing on 200 random sets: OK")

    print("== C2 ==")
    import json
    json.dump([[round(x, 4) for x in sc] for sc in C2_SC], open(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "c2_scores.json"), "w"), separators=(",", ":"))
    print("wrote c2_scores.json (embed verbatim in page as C2_SC)")
    for rho in (0.5, 0.25):
        ic = c2_calibrate(rho)
        print(f"rho={rho} calibrated i={ic} tau={ic/200}")
        for tau in ((ic - 1) / 200, ic / 200, 0.6, 0.8, 0.9):
            Kh, Ks, fh, fs, L4, L10, g = c2(tau, rho)
            print(f"  tau={tau:.3f} Kh={Kh} Ks={[round(x,1) for x in Ks]} fh={fh:.4f} fs={fs:.4f} "
                  f"L4={L4:.4f} L10={L10:.4f} dL/dfs={g:+.4f}")
            assert abs(L4 - L10) < 1e-12

    print("== C3 ==")
    coef = fit()
    print("alpha=%.6e beta=%.6e gamma=%.4f" % coef)
    print("alpha theory if 4d^2+3*d*h with h=2048:", (4*768*768 + 3*768*2048)/1e9)
    for name, l, c, g in DATA:
        p = c3_pred(l, c, coef)
        print(f"  {name:8s} table={g:5.1f} model={p:6.2f} err={p-g:+.2f}")
    print("  JiT-like 12 blocks x 256 tokens (no grouping):", round(c3_pred([12], [256], coef), 2))

    print("== C4 ==")
    for k in (1.0, 1.7, 3.0):
        grid, nn, nc, nr = c4(k)
        print(f"k={k} t'[1]={grid[1]:.4f} t'[25]={grid[25]:.4f} t'[49]={grid[49]:.4f} "
              f"steps starting t<0.5: {nn}, in CFG [0.1,1]: {nc}, with ru=(0.5,1) (t>=0.85): {nr}")

    S = affinity([10, 22, 150], [10, 22, 150])
    print("3-token example: S12=%.4f S13=%.4f S23=%.4f s({1,2},{3})=%.4f" % (S[0][1], S[0][2], S[1][2], (S[0][2] + S[1][2]) / 2))
    print("== C1 acceptance sweep (default tau=0.95, Tr=0.2, Tb=0.1; vary angle of token 4) ==")
    for a4 in (38, 60, 90, 120):
        ang = C1_ANGLES[:]; ang[3] = a4
        S, G, A, P, ms, ks = c1(ang, 0.95, 0.2)
        print(f"  a4={a4}: groups={[[i+1 for i in g] for g in G]} Ksoft={ks:.3f} A[4]={[round(x,3) for x in A[3]]} "
              f"merges={[round(m,4) for m in ms]}")
    for tau in (0.9, 0.95, 0.975, 0.98):
        S, G, A, P, ms, ks = c1(C1_ANGLES, tau, 0.2)
        print(f"  tau={tau}: K={len(G)} Ksoft={ks:.3f}")

    print("== precise reference values (for relative tolerances) ==")
    S, G, A, P, ms, ks = c1(C1_ANGLES, 0.95, 0.2)
    print("A[4] =", [f"{x:.5g}" for x in A[3]], " P col{1,2} =", [f"{P[i][0]:.5g}" for i in range(6)])
    for a4 in (60, 90, 120):
        ang = C1_ANGLES[:]; ang[3] = a4
        S, G, A, P, ms, ks = c1(ang, 0.95, 0.2)
        print(f"  a4={a4}: Ksoft={ks:.6f} A[4][{{1,2}}]={A[3][0]:.5g}")
    for k in (1.0, 1.7, 3.0):
        g = c4(k)[0]
        print(f"  k={k}: t'1={g[1]:.6g} t'25={g[25]:.6g} t'49={g[49]:.6g}")
