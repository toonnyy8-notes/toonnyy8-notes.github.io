"""Shared math for the No. 447 (Diffuser) tutorial widgets. Mirrors the page JS exactly.
Gaussian trajectory prior per coordinate: x ~ N(0, Q^-1), Q = a I + b L^2, L = Neumann 2nd-difference.
Eigenvectors of Q are DCT-II vectors, so the exact eps-predictor is diagonal in that basis."""
import math
import numpy as np

A_PRIOR = 0.01   # a : precision of the mean level (k=0 mode std = 10)
B_PRIOR = 400.0  # b : smoothness weight

def mulberry32(seed):
    a = seed & 0xFFFFFFFF
    def imul(x, y):
        return (x * y) & 0xFFFFFFFF
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

def basis(H):
    t = np.arange(H) + 0.5
    k = np.arange(H)
    P = np.cos(np.pi * np.outer(t, k) / H) * math.sqrt(2.0 / H)
    P[:, 0] = math.sqrt(1.0 / H)
    return P  # P[t,k]; columns orthonormal

def lam(H, a=A_PRIOR, b=B_PRIOR):
    k = np.arange(H)
    nu = 4.0 * np.sin(np.pi * k / (2.0 * H)) ** 2
    return 1.0 / (a + b * nu ** 2)

def Qmat(H, a=A_PRIOR, b=B_PRIOR):
    L = np.zeros((H, H))
    for t in range(H):
        L[t, t] = -2.0
        if t > 0: L[t, t - 1] = 1.0
        if t < H - 1: L[t, t + 1] = 1.0
    L[0, 0] = -1.0; L[H - 1, H - 1] = -1.0
    return a * np.eye(H) + b * (L @ L)

def cosine_schedule(N, s=0.008):
    f = lambda i: math.cos((i / N + s) / (1 + s) * math.pi / 2) ** 2
    beta = np.array([0.0] + [min(1 - f(i) / f(i - 1), 0.999) for i in range(1, N + 1)])
    abar = np.ones(N + 1)
    for i in range(1, N + 1):
        abar[i] = abar[i - 1] * (1 - beta[i])
    bt = np.zeros(N + 1)
    for i in range(1, N + 1):
        bt[i] = beta[i] * (1 - abar[i - 1]) / (1 - abar[i])
    return beta, abar, bt  # index 0 unused for beta/bt

def reverse_step(X, i, sch, P, lmb, guide=None, z=None):
    """X: (2,H) array tau^i. returns tau^{i-1} (before inpainting)."""
    beta, abar, bt = sch
    Xh = X @ P                                  # DCT coeffs, (2,H)
    eps_h = math.sqrt(1 - abar[i]) * Xh / (abar[i] * lmb + 1 - abar[i])
    eps = eps_h @ P.T
    mu = (X - beta[i] / math.sqrt(1 - abar[i]) * eps) / math.sqrt(1 - beta[i])
    if guide is not None:                       # guide(mu, i) returns alpha * gradient (Alg. 1 line 8: mu + Sigma * g)
        mu = mu + bt[i] * guide(mu, i)
    if z is not None and i > 1:
        mu = mu + math.sqrt(bt[i]) * z
    return mu

def draw(r, H):
    return np.array([[gauss(r) for _ in range(H)] for _ in range(2)])

def run_chain(H, N, seed=None, cond=None, guide=None, noise=True):
    """cond: list of (t, (x,y)); replaced with the clean value after every step (paper Sec 3.3).
    noise=False -> procedure mean (tau^N = 0 and z = 0; the chain is affine, so this is E[tau^0])."""
    P = basis(H); lmb = lam(H); sch = cosine_schedule(N)
    r = mulberry32(seed) if seed is not None else None
    def apply(X):
        for t, c in (cond or []):
            X[:, t] = c
        return X
    X = apply(draw(r, H) if noise else np.zeros((2, H)))
    hist = [X.copy()]
    for i in range(N, 0, -1):
        z = draw(r, H) if (noise and i > 1) else None
        X = apply(reverse_step(X, i, sch, P, lmb, guide, z))
        hist.append(X.copy())
    return X, hist

def exact_mean(H, fixed, extra_prec=0.0, lin=(0.0, 0.0)):
    """Mean of N(0,Q^-1) * exp(-extra/2 |x|^2 + lin.x) conditioned on fixed entries. fixed: list of (t,(x,y))."""
    Qm = Qmat(H) + extra_prec * np.eye(H)
    ft = [t for t, _ in fixed]; free = [t for t in range(H) if t not in ft]
    out = np.zeros((2, H))
    for c in range(2):
        xf = np.array([v[c] for _, v in fixed])
        rhs = lin[c] * np.ones(len(free)) - Qm[np.ix_(free, ft)] @ xf
        L = np.linalg.cholesky(Qm[np.ix_(free, free)])
        out[c, free] = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        out[c, ft] = xf
    return out

def step_sd(H):
    P = basis(H); C = P @ np.diag(lam(H)) @ P.T
    return math.sqrt(np.mean([C[t, t] + C[t + 1, t + 1] - 2 * C[t, t + 1] for t in range(H - 1)]))

def var_ratio(H, N):
    l = lam(H); beta, abar, bt = cosine_schedule(N); v = np.ones(H)
    for i in range(N, 0, -1):
        A = (1 - beta[i] / (abar[i] * l + 1 - abar[i])) / math.sqrt(1 - beta[i])
        v = A * A * v + bt[i]
    return v.sum() / l.sum()

def rms(A, B):
    return math.sqrt(np.mean((A - B) ** 2))

def make_guide(mode, alpha, c, H, N):
    """C3 guidance. 'clean': alpha * grad J(mu), J = -1/2 sum |s_t - c|^2 (Alg. 1 with the true J).
    'noisy': grad of log E[exp(alpha J(tau^0)) | tau^{i-1} = mu] -- closed form for this prior (DCT-diagonal):
    m_k = sqrt(ab) lam/(ab lam + 1 - ab), S_k = lam (1-ab)/(ab lam + 1 - ab), ab = abar_{i-1};
    g = P diag(m) [ -alpha (m * mu_hat - c_hat) / (1 + alpha S) ]."""
    c = np.asarray(c, float)
    if mode == 'clean':
        return lambda mu, i: -alpha * (mu - c[:, None])
    P = basis(H); l = lam(H); beta, abar, bt = cosine_schedule(N)
    ch = np.zeros((2, H)); ch[:, 0] = c * math.sqrt(H)
    def g(mu, i):
        ab = abar[i - 1]; m = math.sqrt(ab) * l / (ab * l + 1 - ab); S = l * (1 - ab) / (ab * l + 1 - ab)
        return ((-alpha * ((mu @ P) * m - ch) / (1 + alpha * S)) * m) @ P.T
    return g
