"""C1: generalized IQL critic on the toy bandit (Theorem 4.1, Eq. 9-12)."""
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

xs, h = grid()
def report(pi, tag):
    print(f'## {tag}  pi={tuple(round(p,4) for p in pi)}  E_mu[Q]={mean_mu(pi):.4f}')
    _, _, sd_mu = moments([mix_pdf(x, pi) for x in xs], xs, h); print(f'   std_mu={sd_mu:.4f}')
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        v = v_expectile(tau, pi)
        z, m1, sd = moments([mix_pdf(x, pi) * w_expectile(x, v, tau) for x in xs], xs, h)
        pa = 1 - mix_cdf(v, pi)
        print(f'expectile tau={tau}: V={v:.4f}  E_imp[Q]={m1:.4f}  std_imp={sd:.4f}  P_mu(Q>V)={pa:.4f}  ratio={tau/(1-tau):.4f}')
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        v = v_quantile(tau, pi)
        print(f'quantile  tau={tau}: V={v:.4f}  F(V)={mix_cdf(v,pi):.6f}')
    for al in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        v = v_exp(al, pi)
        z, m1, sd = moments([mix_pdf(x, pi) * w_exp(x, v, al) for x in xs], xs, h)
        ma, sa, wk, _ = awr_gauss(al, pi)
        # pi_awr itself is the tilted mixture: mean = ma (same as its Gaussian fit), KL(mu||pi_awr)=alpha(V - E_mu Q) (App C.2)
        dens_mu = [mix_pdf(x, pi) for x in xs]
        kl = al * (v - mean_mu(pi))
        klr = al * (ma - v)   # KL(pi_awr || mu) = alpha (E_awr[Q] - V)  => V = E_awr[Q] - KL(pi_awr||mu)/alpha
        dawr = [mix_pdf(x, [w_ for w_ in wk], [mk + al * S * S for mk in M_K]) for x in xs]
        klrn = sum(h * p * math.log(p / q) for p, q in zip(dawr, dens_mu) if p > 1e-300 and q > 1e-300)
        # numeric KL check
        dens_mu = [mix_pdf(x, pi) for x in xs]
        kln = sum(h * d * (-al * (x - v)) for d, x in zip(dens_mu, xs))
        print(f'exp alpha={al}: V={v:.4f}  E_imp[Q]={m1:.4f}  std_imp={sd:.4f}  E_awr[Q]={ma:.4f}  KL(mu||awr)={kl:.4f} (num {kln:.4f})  KL(awr||mu)={klr:.4f} (num {klrn:.4f})  awr_w={[round(x,4) for x in wk]}')

report(PI_DEFAULT, 'default p_high=0.2')
report(pis(0.05), 'p_high=0.05')
