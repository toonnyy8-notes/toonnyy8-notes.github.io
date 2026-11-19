"""C3: Q-loss (Eq. 11) regression input/target in a toy where the next-state clean V-return is N(m1, s1^2).
Closed-form marginal velocity for the independent-coupling linear path to a Gaussian N(m, s^2):
   sigma(tau)^2 = tau^2 s^2 + (1-tau)^2,   u(z|tau) = m + k(tau) (z - tau m),  k = (tau s^2 - (1-tau)) / sigma^2.
Sample (z0, eps): z_tau = tau m1 + sigma_V z0 (1-D Gaussian flow is this affine map),
   input  x = tau r + gamma z_tau + (1-tau) c eps
   target y = r + gamma u_V(z_tau) - c eps            ('ok')
            = r + gamma u_V(z_tau) + c eps            ('sign' trap)
            = tau r + gamma u_V(z_tau) - c eps        ('taur' trap)
Exact E[y|x] is affine: slope Cov(x,y)/Var(x), intercept E[y] - slope E[x]. Compare with u_Q (clean Q-return N(r+gamma m1, gamma^2 s1^2))."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
from common import mulberry32, gauss

def vel(m, s, tau):
    sig2 = tau * tau * s * s + (1 - tau) ** 2
    k = (tau * s * s - (1 - tau)) / sig2
    return k, m - k * tau * m, math.sqrt(sig2)        # slope, intercept, sigma

def c3(gamma, tau, r, m1, s1, mode='ok'):
    c = math.sqrt(1 - gamma * gamma)
    kV, bV, sV = vel(m1, s1, tau)
    # coefficients: x = ax0 + ax1 z0 + ax2 eps ; y = ay0 + ay1 z0 + ay2 eps
    ax = (tau * r + gamma * tau * m1, gamma * sV, (1 - tau) * c)
    uV0 = kV * (tau * m1) + bV                         # u_V at z0 = 0
    ay = ((tau * r if mode == 'taur' else r) + gamma * uV0, gamma * kV * sV, (c if mode == 'sign' else -c))
    var_x = ax[1] ** 2 + ax[2] ** 2
    cov = ax[1] * ay[1] + ax[2] * ay[2]
    slope = cov / var_x; icpt = ay[0] - slope * ax[0]
    kQ, bQ, sQ = vel(r + gamma * m1, gamma * s1, tau)
    return dict(slope=slope, icpt=icpt, kQ=kQ, bQ=bQ, var_x=var_x, sQ=sQ, ax=ax, ay=ay, kV=kV, bV=bV, sV=sV)

def samples(gamma, tau, r, m1, s1, mode='ok', n=300, seed=3):
    o = c3(gamma, tau, r, m1, s1, mode); rnd = mulberry32(seed); out = []
    for _ in range(n):
        z0 = gauss(rnd); e = gauss(rnd)
        out.append((o['ax'][0] + o['ax'][1] * z0 + o['ax'][2] * e, o['ay'][0] + o['ay'][1] * z0 + o['ay'][2] * e))
    return out

def ols(pts):
    n = len(pts); mx = sum(p[0] for p in pts) / n; my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts); sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    b = sxy / sxx; return b, my - b * mx

if __name__ == '__main__':
    D = dict(r=-1.0, m1=-2.0, s1=0.6)
    for g in [0.9, 0.99]:
        for t in [0.0, 0.25, 0.5, 0.75, 0.999]:
            for mode in ['ok', 'sign', 'taur']:
                o = c3(g, t, mode=mode, **D)
                b, a = ols(samples(g, t, mode=mode, **D))
                print(f'g={g} tau={t} {mode:4s}: E[y|x] slope={o["slope"]:.4f} icpt={o["icpt"]:.4f} | u_Q slope={o["kQ"]:.4f} icpt={o["bQ"]:.4f} | OLS(300,seed3) {b:.4f},{a:.4f} | sd(x)={math.sqrt(o["var_x"]):.4f} sigma_Q={o["sQ"]:.4f}')
                if mode == 'ok':
                    assert abs(o['slope'] - o['kQ']) < 1e-12 and abs(o['icpt'] - o['bQ']) < 1e-12
                    assert abs(math.sqrt(o['var_x']) - o['sQ']) < 1e-12
    # V-flow path endpoints for z0 = +-1 at tau=1: m1 +- s1
    o = c3(0.9, 1.0 - 1e-12, **D); print('sigma_V(1)=', round(o['sV'], 6))
    # tau=0: k=-1 so u(z|0) = m - z  -> z0 + u(z0|0) = m (one-step estimate)
    k, b, _ = vel(-2.8, 0.54, 0.0); print('tau=0 velocity slope', k, 'intercept', b)
    print('c3 ok')
