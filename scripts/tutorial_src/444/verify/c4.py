"""C4: DDPM reverse sampling (Eq. 5) of the toy mu with the exact (Bayes-optimal) noise predictor.
Schedules copied from the IDQL repo jaxrl5/networks/diffusion.py (推定: the paper only names them).
Sampler follows repo ddpm_sampler: no noise added at the last step (time index 0); no clipping in the toy."""
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def sched(name, T):
    if name == 'vp':
        return [1 - math.exp(-0.1 / T - 0.5 * (10 - 0.1) * (2 * t - 1) / T ** 2) for t in range(1, T + 1)]
    if name == 'linear':
        return [1e-4 + (2e-2 - 1e-4) * i / (T - 1) for i in range(T)] if T > 1 else [1e-4]
    if name == 'cosine':
        s = 0.008
        ac = [math.cos(((i / T) + s) / (1 + s) * math.pi / 2) ** 2 for i in range(T + 1)]
        ac = [a / ac[0] for a in ac]
        return [min(max(1 - ac[i + 1] / ac[i], 0.0), 0.999) for i in range(T)]

def eps_star(x, abar, pi, m=M_K, s=S):
    # p_t = sum pi_k N(sqrt(abar) m_k, abar s^2 + 1 - abar); eps* = -sqrt(1-abar) d/dx log p_t
    sa = math.sqrt(abar); var = abar * s * s + 1 - abar
    lw = [math.log(p) - 0.5 * (x - sa * mk) ** 2 / var for p, mk in zip(pi, m)]
    mx = max(lw); w = [math.exp(l - mx) for l in lw]; z = sum(w)
    score = sum(wi * (-(x - sa * mk) / var) for wi, mk in zip(w, m)) / z
    return -math.sqrt(1 - abar) * score

def arrays(name, T):
    """1-based: beta[1..T], alpha[1..T], abar[0..T] with abar[0] = 1 (index 0 of beta/alpha unused)."""
    b = [None] + sched(name, T)
    a = [None] + [1 - x for x in b[1:]]
    ab = [1.0]
    for k in range(1, T + 1): ab.append(ab[-1] * a[k])
    return b, a, ab

def sample(name, T, K, seed, pi=PI_DEFAULT):
    b, al, ab = arrays(name, T)
    rnd = mulberry32(seed); out = []
    for _ in range(K):
        x = gauss(rnd)                                  # a_T ~ N(0, 1)
        for k in range(T, 0, -1):                       # Eq. 5: k = T..1
            e = eps_star(x, ab[k], pi)
            x = (x - b[k] / math.sqrt(1 - ab[k]) * e) / math.sqrt(al[k])
            z = gauss(rnd)                              # drawn every step; added only if k > 1 (repo: time index k-1 > 0)
            if k > 1: x += math.sqrt(b[k]) * z
        out.append(x)
    return out, ab[T]

pi = PI_DEFAULT
thr = 0.01 * max(mix_pdf(x, pi) for x in grid()[0])
def stats(xs):
    K = len(xs); mean = sum(xs) / K; sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / K)
    fr = [sum(1 for x in xs if min(range(3), key=lambda k: abs(x - M_K[k])) == k) / K for k in range(3)]
    low = sum(1 for x in xs if mix_pdf(x, pi) < thr) / K
    return mean, sd, fr, low
print('target: mean=1.9000 std=0.7159 clusters=[0.3,0.5,0.2]')
for name in ('vp', 'cosine', 'linear'):
    for T in (1, 2, 5, 10, 20, 50):
        if name == 'linear' and T == 1: pass
        xs, abT = sample(name, T, 4000, 1)
        mean, sd, fr, low = stats(xs)
        print(f'{name:6s} T={T:2d} abar_T={abT:.4f} mean={mean:.4f} std={sd:.4f} clusters={[round(f,4) for f in fr]} low={low:.4f}')
