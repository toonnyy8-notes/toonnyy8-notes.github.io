"""W4 acceptance values: information curves, critical points, recovery probability, window widths.

Recovery model = paper App. C (Poissonized codebook, planted y*). Masked: closed form.
Uniform: deterministic quadrature (design.md 4.4), cross-checked by the App. C Monte Carlo.
Same algorithm as the page. Natural log. Run: python3 -I w4_window.py [--mc]
"""
import math
import sys
import numpy as np

LN2 = math.log(2)
XHI, XLO = math.log(40.0), math.log(1e-8)
VMIN, VMAX, VSTEP = -36.0, 4.0, 0.25
EPS1 = 1e-10
LF = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, 70000)))])  # log n!


def h2(p):
    return -p * math.log(p) - (1 - p) * math.log1p(-p)


def I_unif(t):
    return LN2 - h2((1 - math.exp(-t)) / 2)


def logcosh(x):
    x = abs(x)
    return x + math.log1p(math.exp(-2 * x)) - LN2


W_GRID = np.linspace(-10, 10, 401)
W_SIMP = np.ones(401); W_SIMP[1:-1:2] = 4; W_SIMP[2:-1:2] = 2
W_SIMP *= (20 / 400) / 3 * np.exp(-W_GRID ** 2 / 2) / math.sqrt(2 * math.pi)


def I_gauss(t):
    """I(V; uV+W) = u^2 - E_W log cosh(u^2 + u W), u = e^-t / sigma_t (Lemma 7.2 proof)."""
    u = math.exp(-t) / math.sqrt(-math.expm1(-2 * t))
    return u * u - float(sum(wt * logcosh(u * u + u * w) for w, wt in zip(W_GRID, W_SIMP)))


def bisect(f, lo, hi, it=80):
    flo = f(lo)
    for _ in range(it):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (flo > 0):
            lo, flo = mid, f(mid)
        else:
            hi = mid
    return (lo + hi) / 2


def t_of_I_unif(I):
    beta = bisect(lambda b: (LN2 - h2(b)) - I, 1e-15, 0.5 - 1e-15)
    return -math.log1p(-2 * beta)


def masked_rec(lam):
    return 1.0 if lam < 1e-12 else -math.expm1(-lam) / lam


def masked_width():
    l80 = bisect(lambda x: masked_rec(math.exp(x)) - 0.8, -10, 10)
    l20 = bisect(lambda x: masked_rec(math.exp(x)) - 0.2, -10, 10)
    return math.exp(l80), math.exp(l20), l20 - l80  # last = ln(lam20/lam80)


def lse(a):
    m = max(a)
    return m if m == -math.inf else m + math.log(sum(math.exp(x - m) for x in a))


class UnifRec:
    """P[X0 = y* | Xt] averaged, uniform diffusion, Poissonized codebook (App. C)."""

    def __init__(self, d, kappa):
        self.d, self.kappa = d, kappa
        j = np.arange(d + 1)
        self.loglam = kappa * d + LF[d] - LF[j] - LF[d - j] - d * LN2
        self.v = np.arange(VMIN, VMAX + 1e-9, VSTEP)
        n = len(self.v)
        w = np.ones(n); w[1:-1:2] = 4; w[2:-1:2] = 2
        self.vw = w * VSTEP / 3
        self.base = self.v - np.exp(self.v)  # log of e^{v - e^v}
        self.Z0 = float((self.vw * np.exp(self.base)).sum())

    def setup_t(self, t):
        d = self.d
        self.beta = (1 - math.exp(-t)) / 2
        self.L = math.log(self.beta / (1 - self.beta))
        ll = self.loglam
        # logP0[j] = log sum_{j'<j} lam_j' ; logS1[j] = log sum_{j'>=j} lam_j' ell^j'
        self.logP0 = np.full(d + 2, -math.inf)
        acc = -math.inf
        for j in range(d + 1):
            self.logP0[j] = acc
            acc = np.logaddexp(acc, ll[j])
        self.logP0[d + 1] = acc
        self.logS1 = np.full(d + 2, -math.inf)
        acc = -math.inf
        for j in range(d, -1, -1):
            acc = np.logaddexp(acc, ll[j] + j * self.L)
            self.logS1[j] = acc

    def G(self, v, D):
        d, L = self.d, self.L
        c = v - D * L  # x_j = c + j L, decreasing in j
        ja = min(d + 1, max(0, math.ceil((c - XHI) / (-L))))
        jb = min(d + 1, max(ja, math.floor((c - XLO) / (-L)) + 1))
        parts = [self.logP0[ja], c + self.logS1[jb]]
        for j in range(ja, jb):
            x = c + j * L
            parts.append(self.loglam[j] + math.log(-math.expm1(-math.exp(x))))
        m = max(parts)
        if m > 700:
            return math.inf
        return sum(math.exp(p) for p in parts if p > -745)

    def p_given_D(self, D):
        tot = 0.0
        for v, wv, b in zip(self.v, self.vw, self.base):
            g = self.G(v, D)
            if g < 700:
                tot += wv * math.exp(b - g)
        return min(1.0, tot / self.Z0)

    def rec(self, t):
        self.setup_t(t)
        d, beta = self.d, self.beta
        sd = math.sqrt(d * beta * (1 - beta))
        D0 = max(0, math.floor(d * beta - 7 * sd - 2))
        D1 = min(d, math.ceil(d * beta + 7 * sd + 2))
        Ds = np.arange(D0, D1 + 1)
        logpmf = LF[d] - LF[Ds] - LF[d - Ds] + Ds * math.log(beta) + (d - Ds) * math.log1p(-beta)
        pmf = np.exp(logpmf)
        cache = {}

        def P(D):
            if D not in cache:
                cache[D] = self.p_given_D(D)
            return cache[D]

        def first_below(th):  # first index k in [0, len] with P(Ds[k]) < th (P decreasing in D)
            lo, hi = 0, len(Ds)
            while lo < hi:
                mid = (lo + hi) // 2
                if P(int(Ds[mid])) < th:
                    hi = mid
                else:
                    lo = mid + 1
            return lo

        b1 = first_below(1 - EPS1)
        b2 = first_below(EPS1)
        s = float(pmf[:b1].sum())
        for k in range(b1, b2):
            s += pmf[k] * P(int(Ds[k]))
        self.n_eval = len(cache)
        return s

    def rec_I(self, I):
        return self.rec(t_of_I_unif(I))

    def width(self):
        k = self.kappa
        lo, hi = max(1e-4, k - 0.2), min(LN2 - 1e-4, k + 0.2)
        i80 = bisect(lambda I: self.rec_I(I) - 0.8, lo, hi, it=26)
        i20 = bisect(lambda I: self.rec_I(I) - 0.2, lo, hi, it=26)
        return i20, i80, i80 - i20


def mc_unif(d, kappa, t, n, rng):
    """App. C Monte Carlo: P = 1/(1 + sum_j N_j tanh(t/2)^(j - D))."""
    beta = (1 - math.exp(-t)) / 2
    ell = math.tanh(t / 2)
    j = np.arange(d + 1)
    lam = np.exp(kappa * d + LF[d] - LF[j] - LF[d - j] - d * LN2)
    out = np.empty(n)
    for s in range(n):
        D = rng.binomial(d, beta)
        Nj = np.where(lam < 1e7, rng.poisson(np.minimum(lam, 1e7)), lam + np.sqrt(lam) * rng.standard_normal(d + 1))
        with np.errstate(over="ignore"):
            Z = float((Nj * ell ** (j - D)).sum())
        out[s] = 1 / (1 + Z)
    return out.mean(), out.std() / math.sqrt(n)


if __name__ == "__main__":
    print("== critical points (Eq. 22) and info curves")
    for kappa in (0.1, 0.2, 0.3, 0.5):
        tu = bisect(lambda t: I_unif(t) - kappa, 1e-6, 20)
        tg = bisect(lambda t: I_gauss(t) - kappa, 1e-4, 10)
        bstar = (1 - math.exp(-tu)) / 2
        print(f"kappa={kappa}: beta*={bstar:.4f} t*_unif={tu:.4f} e^-t*={math.exp(-tu):.4f} "
              f"t*_gauss={tg:.4f} lambda_t*={math.exp(-tg)/math.sqrt(-math.expm1(-2*tg)):.4f} m*/d={kappa/LN2:.4f}")
    for t in (0.25, 0.5, 1.0, 2.0):
        print(f"t={t}: I_unif={I_unif(t):.5f} I_gauss={I_gauss(t):.5f} masked E[I]=e^-t ln2={math.exp(-t)*LN2:.5f}")
    l80, l20, lr = masked_width()
    print(f"\n== masked: lam80={l80:.4f} lam20={l20:.4f} ln ratio={lr:.4f}; width_m={lr/LN2:.4f} coords")
    for d in (64, 128, 256, 512, 1024, 2048, 4096):
        print(f"  d={d:5d} masked width_I = {lr/d:.6f}")
    kappa = 0.2
    print(f"\n== uniform recovery (quadrature), kappa={kappa}")
    for d in (64, 256, 1024):
        u = UnifRec(d, kappa)
        row = []
        for I in (0.15, 0.18, 0.20, 0.22, 0.25):
            row.append(f"I={I}:{u.rec_I(I):.4f}")
        print(f"  d={d}: " + "  ".join(row))
    print(f"\n== widths, kappa={kappa}")
    ws = {}
    for d in (64, 128, 256, 512, 1024, 2048, 4096):
        u = UnifRec(d, kappa)
        i20, i80, w = u.width()
        ws[d] = w
        print(f"  d={d:5d} uniform I20={i20:.5f} I80={i80:.5f} width={w:.6f} width*sqrt(d)={w*math.sqrt(d):.4f}  masked={lr/d:.6f}")
    ds = np.array(sorted(ws)); wv = np.array([ws[x] for x in ds])
    print("  log-log slope uniform:", np.polyfit(np.log(ds), np.log(wv), 1)[0])
    for kappa2 in (0.1, 0.4):
        u = UnifRec(256, kappa2)
        print(f"  kappa={kappa2} d=256 uniform width={u.width()[2]:.6f}")
    if "--mc" in sys.argv:
        rng = np.random.default_rng(1)
        print("\n== MC cross-check (App. C), 20000 samples")
        for d in (64, 256):
            u = UnifRec(d, 0.2)
            for I in (0.17, 0.19, 0.20, 0.21, 0.23):
                t = t_of_I_unif(I)
                m, se = mc_unif(d, 0.2, t, 20000, rng)
                print(f"  d={d} I={I}: quad={u.rec(t):.4f} mc={m:.4f}±{se:.4f}")
