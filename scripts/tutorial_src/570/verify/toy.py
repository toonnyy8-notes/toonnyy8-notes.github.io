"""C1/C2 toy: same algorithm as the page (mulberry32 + Box-Muller cos, ridge poly fit, grid argmax)."""
import math, sys
M32 = 0xFFFFFFFF
def mulberry32(seed):
    a = seed & M32
    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return r
def gauss(r):
    u1, u2 = r(), r()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)

MU_B, SD_B, N, SD_R, DEG, RIDGE = -0.3, 0.3, 40, 0.05, 2, 1e-3
A_STAR, W_Q, B0 = 0.2, 0.3, 4.0
G = 2001
GRID = [-1 + 2 * j / (G - 1) for j in range(G)]
def q(a): return B0 + math.exp(-(a - A_STAR) ** 2 / (2 * W_Q ** 2))

def data(seed):
    r = mulberry32(seed); A = []; R = []
    for _ in range(N):
        a = min(1, max(-1, MU_B + SD_B * gauss(r)))
        A.append(a); R.append(q(a) + SD_R * gauss(r))
    return A, R

def solve(Mx, b):
    n = len(b); Mx = [row[:] + [b[i]] for i, row in enumerate(Mx)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(Mx[i][c])); Mx[c], Mx[p] = Mx[p], Mx[c]
        for i in range(c + 1, n):
            f = Mx[i][c] / Mx[c][c]
            for j in range(c, n + 1): Mx[i][j] -= f * Mx[c][j]
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = (Mx[i][n] - sum(Mx[i][j] * x[j] for j in range(i + 1, n))) / Mx[i][i]
    return x

def fit(A, R):
    abar = sum(A) / N; va = sum((a - abar) ** 2 for a in A) / N; sd = math.sqrt(va) or 1e-12
    phi = lambda a: [((a - abar) / sd) ** j for j in range(DEG + 1)]
    P = [phi(a) for a in A]; n = DEG + 1
    H = [[sum(p[i] * p[j] for p in P) / N + (RIDGE if i == j and i > 0 else 0) for j in range(n)] for i in range(n)]
    b = [sum(p[i] * y for p, y in zip(P, R)) / N for i in range(n)]
    c = solve(H, b)
    Qh = lambda a: sum(ci * fi for ci, fi in zip(c, phi(a)))
    return Qh, abar, va, c, sd

def policy(Qh, abar, va, A, alpha, scale=1.0, normalize=True):
    QG = [scale * Qh(a) for a in GRID]
    if alpha == math.inf:
        j = max(range(G), key=lambda j: QG[j]); return GRID[j], None
    m = max(sum(abs(scale * Qh(a)) for a in A) / N, 1e-12)
    lam = alpha / m if normalize else alpha
    J = [lam * QG[j] - (GRID[j] - abar) ** 2 - va for j in range(G)]
    j = max(range(G), key=lambda j: J[j])  # first max
    return GRID[j], lam

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    A, R = data(seed); Qh, abar, va, c, sd = fit(A, R)
    print(f"seed={seed} abar={abar:.4f} va={va:.5f} sd={sd:.4f} amin={min(A):.3f} amax={max(A):.3f} coef={[round(x,5) for x in c]}")
    print(f"mean|Qh(a_i)|={sum(abs(Qh(a)) for a in A)/N:.5f}")
    for al in [0, 0.5, 1, 2, 2.5, 3, 4, 5, 10, math.inf]:
        p, lam = policy(Qh, abar, va, A, al)
        print(f"alpha={al:>5}: lam={'-' if lam is None else f'{lam:.4f}'} pi={p:+.3f} q={q(p):.4f} Qh={Qh(p):.4f}")
    print("q(abar)=%.4f" % q(abar))
    # C2: reward scale
    for s in [0.01, 1, 100]:
        for nz in (True, False):
            p, lam = policy(Qh, abar, va, A, 2.5, s, nz)
            print(f"scale={s} norm={nz}: lam={lam:.5g} pi={p:+.3f} q={q(p):.4f}")

def interior(fit, alpha):
    """Closed-form stationary point of J (Q-hat quadratic in u=(a-abar)/sd): J'=0 -> d = lam*c1*sd/(2 sd^2 - 2 lam c2)."""
    Qh, abar, va, c, sd = fit
    m = sum(abs(Qh(a)) for a in data(SEED)[0]) / N
    lam = alpha / m
    den = 2 * sd * sd - 2 * lam * c[2]
    return lam, (abar + lam * c[1] * sd / den) if den > 0 else None, 2 * lam * c[2] / (sd * sd) - 2
SEED = 7
