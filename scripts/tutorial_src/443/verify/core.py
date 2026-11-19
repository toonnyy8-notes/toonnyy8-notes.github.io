"""Shared toy math for the 443 page (same algorithm as the page JS; double precision).
Run with: python3 -I core.py  (prints a few self-checks)."""
import math
import numpy as np

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

# ---- Sec. 3.1 VP schedule (Xiao et al. 2021), beta_min=0.1, beta_max=10
BMIN, BMAX = 0.1, 10.0
def schedule(N):
    i = np.arange(1, N + 1)
    beta = 1 - np.exp(-BMIN / N - 0.5 * (BMAX - BMIN) * (2 * i - 1) / N ** 2)
    alpha = 1 - beta
    abar = np.cumprod(alpha)
    return beta, alpha, abar

# ---- Sec. 4 bandit data: 4 Gaussians, sd 0.05
MU = np.array([[0.0, 0.8], [0.8, 0.0], [0.0, -0.8], [-0.8, 0.0]])   # top, right, bottom, left
RBAR = np.array([3.0, 1.5, 5.0, 0.0])                               # Fig. 1 legend (top 3.0, right 1.5, bottom 5.0, left 0.0)
SD = 0.05

def eps_star(x, i, sch, logits, need_jac=False):
    """Ideal noise prediction E[eps | a^i = x] for data ~ sum_k w_k N(MU_k, SD^2 I), w = softmax(logits).
    x: (B,2). i: diffusion step 1..N (int). Returns eps (B,2) [, Jx (B,2,2), Jl (B,2,4)]."""
    beta, alpha, abar = sch
    ab = abar[i - 1]; sa = math.sqrt(ab); s1 = math.sqrt(1 - ab)
    v = ab * SD ** 2 + 1 - ab
    d = x[:, None, :] - sa * MU[None, :, :]                       # (B,4,2)
    lg = logits[None, :] - (d ** 2).sum(-1) / (2 * v)            # (B,4)
    lg = lg - lg.max(1, keepdims=True)                           # guard: log-sum-exp
    r = np.exp(lg); r /= r.sum(1, keepdims=True)
    mbar = r @ MU                                                # (B,2)
    eps = s1 / v * (x - sa * mbar)
    if not need_jac:
        return eps
    cov = np.einsum('bk,ki,kj->bij', r, MU, MU) - mbar[:, :, None] * mbar[:, None, :]
    Jx = s1 / v * (np.eye(2)[None] - ab / v * cov)
    dm = r[:, None, :] * (MU.T[None, :, :] - mbar[:, :, None])  # (B,2,4) dmbar/dlogit_j
    Jl = -s1 * sa / v * dm
    return eps, Jx, Jl

def sample_chain(aN, Z, sch, logits, tangents=False):
    """Eq. (1): a^{i-1} = a^i/sqrt(alpha_i) - beta_i/sqrt(alpha_i(1-abar_i)) eps + sqrt(beta_i) z, z=0 at i=1.
    aN: (B,2); Z: dict i->(B,2) for i=N..2. Returns a0 [, T0 = d a0/d logits (B,2,4)]."""
    beta, alpha, abar = sch
    N = len(beta)
    a = aN.copy()
    T = np.zeros((a.shape[0], 2, 4))
    for i in range(N, 0, -1):
        c = beta[i - 1] / math.sqrt(1 - abar[i - 1])
        sq = math.sqrt(alpha[i - 1])
        if tangents:
            e, Jx, Jl = eps_star(a, i, sch, logits, True)
            T = (T - c * (np.einsum('bij,bjk->bik', Jx, T) + Jl)) / sq
        else:
            e = eps_star(a, i, sch, logits)
        a = (a - c * e) / sq
        if i > 1:
            a = a + math.sqrt(beta[i - 1]) * Z[i]
    return (a, T) if tangents else a

# ---- toy Q (推定): Nadaraya-Watson smoothing of the 4 reward means, bandwidth ELL
ELL = 0.2
def q_toy(a, need_grad=False):
    d = a[:, None, :] - MU[None]
    lg = -(d ** 2).sum(-1) / (2 * ELL ** 2)
    lg = lg - lg.max(1, keepdims=True)
    g = np.exp(lg); g /= g.sum(1, keepdims=True)
    q = g @ RBAR
    if not need_grad:
        return q
    # dQ/da = sum_k (r_k - Q) g_k (MU_k - a)/ell^2
    grad = np.einsum('bk,bkj->bj', g * (RBAR[None] - q[:, None]), -d) / ELL ** 2
    return q, grad

QNORM = float(np.mean(np.abs(RBAR)))   # E_D|Q| at the 4 centres = 2.375 (not differentiated)

if __name__ == '__main__':
    for N in (2, 5, 10, 20, 50):
        b, a, ab = schedule(N)
        print(N, 'beta1=%.4f betaN=%.4f abarN=%.6f' % (b[0], b[-1], ab[-1]))
    print('exp(-5.05)=%.6f' % math.exp(-5.05), 'QNORM', QNORM)
    # finite-difference check of jacobians
    rng = np.random.default_rng(0)
    sch = schedule(5); lgt = np.array([0.3, -0.2, 0.5, 0.1]); x = rng.normal(size=(3, 2)) * 0.7
    for i in (1, 3, 5):
        e, Jx, Jl = eps_star(x, i, sch, lgt, True)
        h = 1e-6
        for j in range(2):
            xp = x.copy(); xp[:, j] += h; xm = x.copy(); xm[:, j] -= h
            fd = (eps_star(xp, i, sch, lgt) - eps_star(xm, i, sch, lgt)) / (2 * h)
            assert np.abs(fd - Jx[:, :, j]).max() < 1e-6, (i, j)
        for j in range(4):
            lp = lgt.copy(); lp[j] += h; lm = lgt.copy(); lm[j] -= h
            fd = (eps_star(x, i, sch, lp) - eps_star(x, i, sch, lm)) / (2 * h)
            assert np.abs(fd - Jl[:, :, j]).max() < 1e-6, (i, j)
    aN = rng.normal(size=(4, 2)); Z = {i: rng.normal(size=(4, 2)) for i in range(2, 6)}
    a0, T = sample_chain(aN, Z, sch, lgt, True)
    for j in range(4):
        lp = lgt.copy(); lp[j] += 1e-6; lm = lgt.copy(); lm[j] -= 1e-6
        fd = (sample_chain(aN, Z, sch, lp) - sample_chain(aN, Z, sch, lm)) / 2e-6
        assert np.abs(fd - T[:, :, j]).max() < 1e-5, j
    a = rng.normal(size=(5, 2)) * 0.5
    q, gq = q_toy(a, True)
    for j in range(2):
        ap = a.copy(); ap[:, j] += 1e-6; am = a.copy(); am[:, j] -= 1e-6
        assert np.abs((q_toy(ap) - q_toy(am)) / 2e-6 - gq[:, j]).max() < 1e-5
    print('jacobian checks passed')
