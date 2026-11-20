"""Shared math for the 394 tutorial widgets. Pure Python, mirrors the page JS 1:1.
Run `python3 core.py` for self-checks against the paper's formulas."""
import math

# ---------- categorical helpers ----------
def onehot(i, K):
    v = [0.0] * K; v[i] = 1.0; return v

def prior(kind, K):
    """MDM: mask is the LAST index (pi = m). USDM: uniform 1/K."""
    return onehot(K - 1, K) if kind == 'mask' else [1.0 / K] * K

def q_marg(x, a, pi):            # Eq. (1): Cat(a x + (1-a) pi)
    return [a * xi + (1 - a) * p for xi, p in zip(x, pi)]

def post(zt, x, at, as_, pi):
    """Generic Bayes posterior q_{s|t}(.|z_t=e_zt, x), x may be soft.
    vector over z_s: (a_{t|s} e_zt + (1-a_{t|s}) pi[zt] 1) * (a_s x + (1-a_s) pi), normalised.
    Reproduces Eq. (2) and (3) (checked below). a_{t|s}=a_t/a_s.  Guard: a_s>0."""
    ats = at / as_
    lik = [(1 - ats) * pi[zt]] * len(x); lik[zt] += ats
    pr = q_marg(x, as_, pi)
    num = [l * p for l, p in zip(lik, pr)]
    S = sum(num)
    if S < 1e-300:
        return onehot(zt, len(x))
    return [n / S for n in num]

def q0t(zt, x, at, pi):          # posterior with s=0 (a_s = 1)
    return post(zt, x, at, 1.0, pi)

def psi_true(zt, x, at, as_, kap, pi):           # Eq. (11)
    a = post(zt, x, at, as_, pi); b = q_marg(x, as_, pi)
    return [kap * ai + (1 - kap) * bi for ai, bi in zip(a, b)]

def psi_theta(zt, xth, at, as_, kap, pi):        # Eq. (12)/(15)
    a = post(zt, xth, at, as_, pi); c = q0t(zt, xth, at, pi)
    return [kap * ai + (1 - kap) * (as_ * ci + (1 - as_) * p) for ai, ci, p in zip(a, c, pi)]

def marginal_after(x, at, as_, kap, pi):         # sum_zt q_t(zt|x) Psi_{s|t}(.|zt,x)  (Eq. 16)
    qt = q_marg(x, at, pi); K = len(x); out = [0.0] * K
    for z in range(K):
        if qt[z] == 0: continue
        row = psi_true(z, x, at, as_, kap, pi)
        for j in range(K): out[j] += qt[z] * row[j]
    return out

# ---------- paper closed forms for cross-checks ----------
def usdm_eq3(zt, x, at, as_):
    K = len(x); ats = at / as_; z = onehot(zt, K)
    num = [K * at * zi * xi + (ats - at) * zi + (as_ - at) * xi + (1 - ats) * (1 - as_) / K for zi, xi in zip(z, x)]
    den = K * at * sum(zi * xi for zi, xi in zip(z, x)) + 1 - at
    return [n / den for n in num]

def mdm_eq2(zt, x, at, as_):
    K = len(x); m = K - 1
    if zt != m: return list(x)
    return [(as_ - at) / (1 - at) * xi + (1 - as_) / (1 - at) * (1.0 if j == m else 0.0) for j, xi in enumerate(x)]

def remdm_eq21(zt, x, at, as_, sig):
    K = len(x); m = K - 1; e = onehot(m, K)
    if zt != m: return [(1 - sig) * xi + sig * ei for xi, ei in zip(x, e)]
    return [(as_ - (1 - sig) * at) / (1 - at) * xi + (1 - as_ - sig * at) / (1 - at) * ei for xi, ei in zip(x, e)]

# ---------- kappa schedules (Suppl. A.4/A.5) ----------
def sigma_max(at, as_):
    return 1.0 if at <= 0 else min(1.0, (1 - as_) / at)

def kappa_from_sigma(sig, as_):
    return 1.0 if 1 - as_ <= 0 else 1 - sig / (1 - as_)

def kappa_sched(kind, t, at, as_, p):
    """p: dict(eta, kap, ton, toff). 'window': piecewise constant (Sec. 5.1.2)."""
    if kind == 'ancestral': return 1.0
    if kind == 'window': return p['kap'] if p['toff'] <= t <= p['ton'] else 1.0
    if kind == 'rescale': return kappa_from_sigma(p['eta'] * sigma_max(at, as_), as_)
    if kind == 'cap': return kappa_from_sigma(min(p['eta'], sigma_max(at, as_)), as_)
    raise ValueError(kind)

# ---------- PRNG identical to page ----------
class Mulberry32:
    def __init__(self, seed): self.a = seed & 0xFFFFFFFF
    def __call__(self):
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

def cat_sample(r, p):
    u = r(); c = 0.0
    for i, pi in enumerate(p):
        c += pi
        if u < c: return i
    for i in range(len(p) - 1, -1, -1):   # guard: rounding
        if p[i] > 0: return i

def gauss(r):
    u1 = r(); u2 = r()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)

# ---------- normal CDF helpers ----------
def log_phi_cdf(x):
    """log Phi(x), stable for very negative x."""
    if x > -5: return math.log(0.5 * math.erfc(-x / math.sqrt(2)))
    x2 = x * x
    return -x2 / 2 - math.log(-x) - 0.5 * math.log(2 * math.pi) + math.log1p(-1 / x2 + 3 / (x2 * x2) - 15 / (x2 ** 3))

def norm_ppf(p):
    """Acklam + one Halley step. Uses upper tail for p>0.5."""
    if p <= 0: return -math.inf
    if p >= 1: return math.inf
    if p > 0.5: return -norm_ppf_tail(1 - p)
    return norm_ppf_tail(p)

def norm_ppf_tail(p):            # p in (0, 0.5]
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q = p - 0.5; r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    e = 0.5 * math.erfc(-x / math.sqrt(2)) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)

if __name__ == '__main__':
    # (2),(3) and ReMDM (21)/(26) reproduced by the generic posterior
    import itertools
    for K in (3, 4, 6):
        for at, as_ in ((0.3, 0.6), (0.05, 0.9), (0.5, 0.51)):
            piU = prior('unif', K); piM = prior('mask', K)
            for xi in range(K):
                x = onehot(xi, K)
                for z in range(K):
                    assert max(abs(a - b) for a, b in zip(post(z, x, at, as_, piU), usdm_eq3(z, x, at, as_))) < 1e-12
            xs = [0.5, 0.3, 0.2][:K-1] + [0.0] * (K - 1 - min(3, K - 1)) + [0.0]
            xs = [v / sum(xs) for v in xs]
            for z in range(K):
                xx = xs if z == K - 1 else onehot(z, K)
                assert max(abs(a - b) for a, b in zip(post(z, xx, at, as_, piM), mdm_eq2(z, xx, at, as_))) < 1e-12
                for sig in (0.0, 0.1, 0.3):
                    if sig > sigma_max(at, as_): continue
                    k = kappa_from_sigma(sig, as_)
                    r1 = psi_true(z, xx, at, as_, k, piM); r2 = remdm_eq21(z, xx, at, as_, sig)
                    assert max(abs(a - b) for a, b in zip(r1, r2)) < 1e-12, (z, sig, r1, r2)
            for kap in (0, 0.5, 0.95, 1, -0.3):
                for pi in (piU, piM):
                    x = onehot(0, K)
                    mg = marginal_after(x, at, as_, kap, pi); qs = q_marg(x, as_, pi)
                    assert max(abs(a - b) for a, b in zip(mg, qs)) < 1e-12
    print('core self-check OK')
