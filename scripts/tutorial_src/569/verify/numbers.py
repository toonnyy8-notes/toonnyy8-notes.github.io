"""C1, C2, C4 acceptance numbers + Fig. 1 (center) normal expectiles. Same algorithms as the page."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math
from common import expectile, L2tau

print('== Fig.1 center: expectiles of N(0,1) (bisection on tau*E(X-m)+ = (1-tau)E(m-X)+)')
phi = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
def normal_expectile(tau):
    lo, hi = -10, 10
    for _ in range(200):
        m = (lo + hi) / 2
        g = tau * (phi(m) - m * (1 - Phi(m))) - (1 - tau) * (m * Phi(m) + phi(m))
        lo, hi = (m, hi) if g > 0 else (lo, m)
    return m
for tau in [0.01, 0.1, 0.5, 0.9, 0.99]:
    print('  tau=%.2f m=%.4f' % (tau, normal_expectile(tau)))

print('== C1 default: Q=[1,2,3,4,8], mu=[.30,.30,.20,.15,.05], unseen a6 Q=12 mu=0')
Q = [1, 2, 3, 4, 8, 12]; mu = [.30, .30, .20, .15, .05, 0]
print('  mean (tau=.5) =', sum(q * p for q, p in zip(Q, mu)))
for tau in [0.01, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 1.0]:
    m = expectile(Q, mu, tau)
    loss = sum(p * L2tau(q - m, tau) for q, p in zip(Q, mu))
    print('  tau=%.3f m=%.4f loss(m)=%.5f' % (tau, m, loss))
print('  ablation: if a6 had mu=0.01 (and others renormalised x0.99): tau=.9 ->',
      '%.4f' % expectile(Q, [p * 0.99 for p in mu[:5]] + [0.01], 0.9),
      ' tau=.99 -> %.4f' % expectile(Q, [p * 0.99 for p in mu[:5]] + [0.01], 0.99))
print('  L2tau(u) at u=+-0.5: tau=.9 ->', L2tau(0.5, .9), L2tau(-0.5, .9), ' tau=.5 ->', L2tau(.5, .5))

print('== C2 lucky sample: a1 target 5 (deterministic); a2 target 10 w.p. p else 0; behaviour picks a2 w.p. q')
def c2(p, q, tau):
    """Per-action (paper-faithful) single loss: Q_naive(a) = expectile over transitions of that action.
    V = expectile over dataset actions (Eq. 5 applied to each Q). Also the pooled toy value (round-0 design)."""
    qn2 = expectile([10, 0], [p, 1 - p], tau)          # naive Q(a2): expectile over outcomes
    vn = expectile([5, qn2], [1 - q, q], tau)            # V built on naive Q
    vi = expectile([5, 10 * p], [1 - q, q], tau)         # IQL: Q(a2)=10p (MSE, Eq. 6), V by Eq. 5
    pooled = expectile([5, 10, 0], [1 - q, q * p, q * (1 - p)], tau)
    return qn2, 10 * p, vn, vi, max(5, 10 * p), pooled
print('  cols: naiveQ(a2) iqlQ(a2) | naiveV iqlV truth | pooled-toy')
for p, q in [(0.3, 0.5), (0.7, 0.5), (0.3, 0.1), (1.0, 0.5), (0.0, 0.5), (0.05, 0.5)]:
    for tau in [0.5, 0.7, 0.9, 0.99, 1.0]:
        a, b, c, d, e, f = c2(p, q, tau)
        print('  p=%.2f q=%.1f tau=%.2f  Qn=%.4f Qi=%.4f | Vn=%.4f Vi=%.4f truth=%.1f | pooled=%.4f' % (p, q, tau, a, b, c, d, e, f))

print('== C4 AWR on a 1-D action: grid a in [-0.8,0.3] step 0.05 (23 pts), counts n=exp(-(a+0.25)^2/(2*0.25^2)), Q(a)=1-(a-0.6)^2')
A = [round(-0.8 + 0.05 * i, 10) for i in range(23)]
n = [math.exp(-(a + 0.25) ** 2 / (2 * 0.25 ** 2)) for a in A]
Qa = [1 - (a - 0.6) ** 2 for a in A]
def awr(tau, beta, clip):
    V = expectile(Qa, n, tau)
    adv = [q - V for q in Qa]
    if clip:
        w = [min(math.exp(min(beta * x, 700)), 100.0) for x in adv]
    else:
        mx = max(beta * x for x in adv)
        w = [math.exp(beta * x - mx) for x in adv]  # common scale cancels in normalised policy
    c = [ni * wi for ni, wi in zip(n, w)]
    S = sum(c); mean = sum(ci * a for ci, a in zip(c, A)) / S
    std = math.sqrt(sum(ci * (a - mean) ** 2 for ci, a in zip(c, A)) / S)
    ess = S * S / (sum(n) * sum(ni * wi * wi for ni, wi in zip(n, w)))
    nclip = sum(ni for ni, x in zip(n, adv) if clip and beta * x > math.log(100)) / sum(n)
    return V, mean, std, ess, nclip
print('  behaviour mean/std (beta=0):', awr(0.7, 0, True)[1:3])
for tau in [0.5, 0.7, 0.9]:
    for beta in [0, 0.5, 3, 10, 30]:
        for clip in [True, False]:
            V, m, s, e, nc = awr(tau, beta, clip)
            print('  tau=%.1f beta=%4.1f clip=%d  V=%.4f mean=%.4f std=%.4f ESS=%.4f clipped-frac=%.4f' % (tau, beta, clip, V, m, s, e, nc))
