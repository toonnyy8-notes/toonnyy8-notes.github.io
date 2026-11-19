"""Algorithm 2/3 applied to a PERFECT network on Gaussian data x0~N(0,s2) (per dimension), respaced DDPM.
2x2 covariance recursion of (x0, xhat). Also: effect of constant lambda (Epsilon Scaling) in this toy.
Same algorithm as page widget C2 curve (3) (design.md 4.0)."""
import math, sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from check import respace, coefs
def floor(kind, Tp, s2, lam=1.0, e=0.0):
    A = respace(kind, Tp)
    P00, P01, P11 = s2, math.sqrt(A[Tp]) * s2, A[Tp] * s2 + 1 - A[Tp]
    out = {}
    for t in range(Tp, 0, -1):
        c = coefs(A, t)
        g = math.sqrt(1 - A[t]) / (A[t] * s2 + 1 - A[t])           # eps*(x) = g x  (optimal for Gaussian data)
        a = (1 - c['q'] * math.sqrt(1 - A[t]) * g / lam) / math.sqrt(c['alpha'])
        # optional independent x0-prediction error e (Eq. 7 style) added on top: xhat += c1*e*eps0/lam
        P01, P11 = a * P01, a * a * P11 + c['bt'] + (c['c1'] * e / lam) ** 2
        tt = t - 1
        if tt >= 1:
            bh = P11 - 2 * math.sqrt(A[tt]) * P01 + A[tt] * P00
            out[tt] = dict(err=bh - (1 - A[tt]), delta=(math.sqrt(bh) - math.sqrt(1 - A[tt])) ** 2, varx=P11, target=A[tt] * s2 + 1 - A[tt], two_ab_s2=2 * A[tt] * s2)
    return out
if __name__ == '__main__':
    for kind in ['cosine', 'linear']:
        print('==', kind, "T'=20 s2=0.25 perfect network ==")
        o = floor(kind, 20, 0.25)
        for t in [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19]:
            r = o[t]; print(f"t={t:2d} err={r['err']:.4f} 2*abar*s2={r['two_ab_s2']:.4f} delta={r['delta']:.4f}")
    print('== cosine T\'=20 s2=0.25: lambda effect (perfect net) ==')
    for lam in [0.98, 1.0, 1.017, 1.05]:
        o = floor('cosine', 20, 0.25, lam); print(lam, f"Var(xhat_1)={o[1]['varx']:.4f} target={o[1]['target']:.4f} delta1={o[1]['delta']:.4f}")
    print('== cosine T\'=20 s2=0.25: lambda effect (net with extra e=0.1) ==')
    for lam in [0.98, 1.0, 1.017, 1.05]:
        o = floor('cosine', 20, 0.25, lam, 0.1); print(lam, f"Var(xhat_1)={o[1]['varx']:.4f} delta1={o[1]['delta']:.4f}")
def perfect(kind, Tp, s2):
    """Page C2 'perfect network' mode, x0-form: x_theta = k_t x, k_t = sqrt(abar) s2 / (abar s2 + 1 - abar).
    single[t]: Algorithm 1 (ground-truth x_{t+1}); multi[t]: Algorithm 2 (from x_T). Returns variance errors and delta."""
    A = respace(kind, Tp); single, multi, dl = {}, {}, {}
    P01, P11 = math.sqrt(A[Tp]) * s2, A[Tp] * s2 + 1 - A[Tp]
    for t in range(Tp, 0, -1):
        c = coefs(A, t); k = math.sqrt(A[t]) * s2 / (A[t] * s2 + 1 - A[t]); a = c['c2'] + c['c1'] * k
        sv = (a * math.sqrt(A[t]) - math.sqrt(A[t - 1])) ** 2 * s2 + a * a * (1 - A[t]) + c['bt']
        P01, P11 = a * P01, a * a * P11 + c['bt']
        tt = t - 1
        if tt >= 1:
            bb = 1 - A[tt]; bh = P11 - 2 * math.sqrt(A[tt]) * P01 + A[tt] * s2
            single[tt] = sv - bb; multi[tt] = bh - bb; dl[tt] = (math.sqrt(bh) - math.sqrt(bb)) ** 2
    return single, multi, dl
if __name__ == '__main__':
    for kind, Tp, s2 in [('cosine', 20, 0.25), ('cosine', 20, 0.5), ('linear', 20, 0.25), ('cosine', 50, 0.25), ('cosine', 100, 0.25), ('cosine', 10, 0.25)]:
        sg, mu, dl = perfect(kind, Tp, s2); ts = [1, 2, 5, 10, 15, 18] if Tp == 20 else [1, Tp // 4, Tp // 2, Tp - 2]
        print('== perfect', kind, Tp, s2, '==', ' '.join(f"t={t}: single={sg[t]:.4f} multi={mu[t]:.4f} delta={dl[t]:.4f}" for t in ts))
