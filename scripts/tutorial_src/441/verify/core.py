"""Shared math for the No. 441 (Decision Diffuser) tutorial widgets. Mirrors the page JS step by step.
C1: 1-D return toy, exact Gaussian-mixture eps-predictors, classifier-free guidance (Eq. 8) + low temperature (Alg. 1).
C2: 2-D constraint toy, kernel (Gaussian-mixture) datasets, composition (Eq. 9) and NOT (App. J).
All toy values are 推定 (chosen for visualisation), not the paper's."""
import math
import numpy as np

def mulberry32(seed):
    a = seed & 0xFFFFFFFF
    def imul(x, y): return (x * y) & 0xFFFFFFFF
    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return nxt

def gauss(r):
    u1 = r(); u2 = r()
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)

def cosine_schedule(K, s=0.008):
    f = lambda i: math.cos((i / K + s) / (1 + s) * math.pi / 2) ** 2
    beta = [0.0] + [min(1 - f(i) / f(i - 1), 0.999) for i in range(1, K + 1)]
    abar = [1.0] * (K + 1)
    for i in range(1, K + 1): abar[i] = abar[i - 1] * (1 - beta[i])
    bt = [0.0] * (K + 1)
    for i in range(1, K + 1): bt[i] = beta[i] * (1 - abar[i - 1]) / (1 - abar[i])
    return beta, abar, bt

K = 100
SCH = cosine_schedule(K)

# ---------------- C1 ----------------
M1 = [0.25, 0.55, 0.85]; PI1 = [0.50, 0.35, 0.15]; S1 = 0.07; SY = 0.25; TOP = 0.70
NP1 = 2000

def comps_uncond():
    return [(PI1[j], M1[j], S1 * S1) for j in range(3)]

def comps_tilt(y, omega):
    """p(x) * N(y; x, SY^2)^omega, normalised. omega=0 -> p(x). omega=1 -> p(x|y)."""
    if omega <= 0: return comps_uncond()
    s2 = S1 * S1; e2 = SY * SY / omega
    v = s2 * e2 / (s2 + e2)
    lw = [math.log(PI1[j]) - 0.5 * math.log(s2 + e2) - (y - M1[j]) ** 2 / (2 * (s2 + e2)) for j in range(3)]
    mx = max(lw); w = [math.exp(l - mx) for l in lw]; Z = sum(w)
    return [(w[j] / Z, (M1[j] * e2 + y * s2) / (s2 + e2), v) for j in range(3)]

def eps_mix(x, k, comps):
    """exact eps-prediction of a diffused 1-D Gaussian mixture; x: np array."""
    ab = SCH[1][k]; sa = math.sqrt(ab)
    lws = []; terms = []
    for (w, m, v) in comps:
        V = ab * v + 1 - ab
        d = x - sa * m
        lws.append(math.log(w) - 0.5 * math.log(V) - d * d / (2 * V))
        terms.append(-d / V)
    L = np.array(lws); T = np.array(terms)
    L = L - L.max(axis=0); R = np.exp(L); R /= R.sum(axis=0)
    score = (R * T).sum(axis=0)
    return -math.sqrt(1 - ab) * score

def c1_sample(omega, alpha, y, seed=1, n=NP1):
    beta, abar, bt = SCH
    r = mulberry32(seed)
    x = np.array([gauss(r) for _ in range(n)]) * math.sqrt(alpha)
    cu = comps_uncond(); cc = comps_tilt(y, 1.0)
    for k in range(K, 0, -1):
        eu = eps_mix(x, k, cu); ec = eps_mix(x, k, cc)
        e = eu + omega * (ec - eu)
        mu = (x - beta[k] / math.sqrt(1 - abar[k]) * e) / math.sqrt(1 - beta[k])
        if k > 1:
            z = np.array([gauss(r) for _ in range(n)])
            x = mu + math.sqrt(alpha * bt[k]) * z
        else:
            x = mu
    return x

def Phi(z): return 0.5 * (1 + math.erf(z / math.sqrt(2)))

def mix_stats(comps, thr=TOP):
    m = sum(w * mu for w, mu, v in comps)
    var = sum(w * (v + mu * mu) for w, mu, v in comps) - m * m
    fr = sum(w * (1 - Phi((thr - mu) / math.sqrt(v))) for w, mu, v in comps)
    return m, math.sqrt(max(var, 0)), fr

def stats(x, thr=TOP):
    return float(x.mean()), float(x.std()), float((x > thr).mean())

# ---------------- C2 ----------------
R_IN = 1.0; R_OUT2 = 1.5; H2 = 0.05; N_DATA = 200; NP2 = 500; ALPHA2 = 0.5

def disk_pts(seed=11, n=N_DATA, R=R_IN):
    r = mulberry32(seed); P = []
    for _ in range(n):
        u1 = r(); u2 = r(); rho = R * math.sqrt(u1); th = 2 * math.pi * u2
        P.append((rho * math.cos(th), rho * math.sin(th)))
    return np.array(P)

def ann_pts(rin, seed=12, n=N_DATA, ro=R_OUT2):
    r = mulberry32(seed); P = []
    for _ in range(n):
        u1 = r(); u2 = r(); rho = math.sqrt(rin * rin + u1 * (ro * ro - rin * rin)); th = 2 * math.pi * u2
        P.append((rho * math.cos(th), rho * math.sin(th)))
    return np.array(P)

def set_score(X, C, sa, V):
    """score of mixture (1/n) sum N(sa*c, V I) at points X (n_p,2) -> (n_p,2), plus log-density (up to const)."""
    D = X[:, None, :] - sa * C[None, :, :]
    L = -(D ** 2).sum(-1) / (2 * V)
    mx = L.max(axis=1, keepdims=True)
    W = np.exp(L - mx); Z = W.sum(axis=1, keepdims=True)
    mean = (W[:, :, None] * (sa * C[None, :, :])).sum(1) / Z
    return (mean - X) / V, (mx[:, 0] + np.log(Z[:, 0] / len(C)))

def c2_sample(mode, omega, rin, seed=21, n=NP2, alpha=ALPHA2):
    """mode: 'y1','y2','and','not2'."""
    beta, abar, bt = SCH
    C1 = disk_pts(); C2 = ann_pts(rin)
    r = mulberry32(seed)
    X = np.array([[gauss(r), gauss(r)] for _ in range(n)]) * math.sqrt(alpha)
    for k in range(K, 0, -1):
        ab = abar[k]; sa = math.sqrt(ab); V = ab * H2 * H2 + 1 - ab
        s1, l1 = set_score(X, C1, sa, V); s2, l2 = set_score(X, C2, sa, V)
        # unconditional = 0.5 p1 + 0.5 p2 : score = weighted by posterior of set
        m = np.maximum(l1, l2); a1 = np.exp(l1 - m); a2 = np.exp(l2 - m); g1 = (a1 / (a1 + a2))[:, None]
        su = g1 * s1 + (1 - g1) * s2
        f = -math.sqrt(1 - ab)
        eu, e1, e2 = f * su, f * s1, f * s2
        if mode == 'y1': e = eu + omega * (e1 - eu)
        elif mode == 'y2': e = eu + omega * (e2 - eu)
        elif mode == 'and': e = eu + omega * ((e1 - eu) + (e2 - eu))
        elif mode == 'not2': e = eu + omega * ((e1 - eu) - (e2 - eu))
        mu = (X - beta[k] / math.sqrt(1 - ab) * e) / math.sqrt(1 - beta[k])
        if k > 1:
            Z = np.array([[gauss(r), gauss(r)] for _ in range(n)])
            X = mu + math.sqrt(alpha * bt[k]) * Z
        else:
            X = mu
    return X

def c2_rates(X, rin):
    rho = np.sqrt((X ** 2).sum(1))
    return dict(le_R=float((rho <= R_IN).mean()), ge_r=float((rho >= rin).mean()),
                both=float(((rho >= rin) & (rho <= R_IN)).mean()), lt_r=float((rho < rin).mean()),
                rho_mean=float(rho.mean()), rho_max=float(rho.max()))

# ---------------- fast variants for slider sweeps (same draws, same order; asserted equal in sweep.py) ----------------
def draws_1d(seed, n, K=K):
    """z[0] = start draws, z[j] (j=1..K-1) = draws used at step k=K-j+1 (k=K..2). Step-major order as in c1_sample."""
    r = mulberry32(seed)
    return np.array([[gauss(r) for _ in range(n)] for _ in range(K)])

def c1_sample_fast(omega, alpha, y, Z):
    beta, abar, bt = SCH
    x = Z[0] * math.sqrt(alpha)
    cu = comps_uncond(); cc = comps_tilt(y, 1.0)
    j = 1
    for k in range(K, 0, -1):
        eu = eps_mix(x, k, cu); ec = eps_mix(x, k, cc)
        e = eu + omega * (ec - eu)
        mu = (x - beta[k] / math.sqrt(1 - abar[k]) * e) / math.sqrt(1 - beta[k])
        if k > 1: x = mu + math.sqrt(alpha * bt[k]) * Z[j]; j += 1
        else: x = mu
    return x

def draws_2d(seed, n, K=K):
    r = mulberry32(seed)
    return np.array([[[gauss(r), gauss(r)] for _ in range(n)] for _ in range(K)])

def set_logdens(X, C, sa, V):
    """log p_i(x) = logsumexp_n(-|x - sa c_n|^2/(2V)) - log n - log(2 pi V)  (2-D Gaussian kernels)."""
    _, l = set_score(X, C, sa, V)          # l = logsumexp - log n
    return l - math.log(2 * math.pi * V)

def g1_at(X, rin, k):
    ab = SCH[1][k]; sa = math.sqrt(ab); V = ab * H2 * H2 + 1 - ab
    l1 = set_logdens(X, disk_pts(), sa, V); l2 = set_logdens(X, ann_pts(rin), sa, V)
    m = np.maximum(l1, l2); return np.exp(l1 - m) / (np.exp(l1 - m) + np.exp(l2 - m))

def c2_sample_fast(mode, omega, rin, Z, alpha=ALPHA2):
    beta, abar, bt = SCH
    C1 = disk_pts(); C2 = ann_pts(rin)
    X = Z[0] * math.sqrt(alpha); j = 1
    for k in range(K, 0, -1):
        ab = abar[k]; sa = math.sqrt(ab); V = ab * H2 * H2 + 1 - ab
        s1, l1 = set_score(X, C1, sa, V); s2, l2 = set_score(X, C2, sa, V)
        m = np.maximum(l1, l2); a1 = np.exp(l1 - m); a2 = np.exp(l2 - m); g1 = (a1 / (a1 + a2))[:, None]
        su = g1 * s1 + (1 - g1) * s2
        f = -math.sqrt(1 - ab)
        eu, e1, e2 = f * su, f * s1, f * s2
        if mode == 'y1': e = eu + omega * (e1 - eu)
        elif mode == 'y2': e = eu + omega * (e2 - eu)
        elif mode == 'and': e = eu + omega * ((e1 - eu) + (e2 - eu))
        else: e = eu + omega * ((e1 - eu) - (e2 - eu))
        mu = (X - beta[k] / math.sqrt(1 - ab) * e) / math.sqrt(1 - beta[k])
        if k > 1: X = mu + math.sqrt(alpha * bt[k]) * Z[j]; j += 1
        else: X = mu
    return X
