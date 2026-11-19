"""C2: AWR unimodal Gaussian vs IDQL resampling (Alg. 2) / argmax (Eq. 13) on the toy bandit."""
import sys, os, math; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

xs, h = grid()
pi = PI_DEFAULT
dmu = [mix_pdf(x, pi) for x in xs]
thr = 0.01 * max(dmu)
low = [d < thr for d in dmu]          # low-density set (資料密度 < 1% 峰值)
wts = [h] * len(xs); wts[0] = wts[-1] = h / 2
def gpdf(x, m, s): return phi((x - m) / s) / s
def low_mass(dens):
    z = sum(w * d for w, d in zip(wts, dens))
    return sum(w * d for w, d, l in zip(wts, dens, low) if l) / z
def tv(p, q):
    zp = sum(w * d for w, d in zip(wts, p)); zq = sum(w * d for w, d in zip(wts, q))
    return 0.5 * sum(w * abs(a / zp - b / zq) for w, a, b in zip(wts, p, q))
def nearest(a): return min(range(3), key=lambda k: abs(a - M_K[k]))
def is_low(a): return mix_pdf(a, pi) < thr

print(f'low-density threshold={thr:.5f}; mu mass in low set={low_mass(dmu):.4f}')
for tau in (0.7, 0.9):
    v = v_expectile(tau, pi)
    dimp = [d * w_expectile(x, v, tau) for d, x in zip(dmu, xs)]
    z, mi, si = moments(dimp, xs, h)
    # cluster masses of pi_imp
    print(f'## tau={tau} V={v:.4f} pi_imp mean={mi:.4f} std={si:.4f} low={low_mass(dimp):.4f}')
    dmm = [gpdf(x, mi, si) for x in xs]   # best unimodal Gaussian by moment matching to pi_imp
    print(f'  Gaussian moment-matched to pi_imp: mean={mi:.4f} std={si:.4f} low_mass={low_mass(dmm):.4f} TV(mm,imp)={tv(dmm, dimp):.4f}')
    for al in (0, 1, 3, 10, 30):
        m, s, wk, _ = awr_gauss(al, pi)
        dg = [gpdf(x, m, s) for x in xs]
        print(f'  AWR alpha={al}: mean={m:.4f} std={s:.4f} low_mass={low_mass(dg):.4f} TV(gauss,imp)={tv(dg, dimp):.4f}')

def run(tau, N, K, seed, mode):
    rnd = mulberry32(seed); v = v_expectile(tau, pi)
    picks = []
    for _ in range(K):
        a = [sample_mu(rnd, pi) for _ in range(N)]
        w = [w_expectile(x, v, tau) for x in a]
        u = rnd()
        if mode == 'imp':
            tot = sum(w); c = 0.0; j = N - 1
            for i, wi in enumerate(w):
                c += wi / tot
                if u < c: j = i; break
            picks.append(a[j])
        else:
            picks.append(max(a))
    mean = sum(picks) / K
    sd = math.sqrt(sum((p - mean) ** 2 for p in picks) / K)
    fr = [sum(1 for p in picks if nearest(p) == k) / K for k in range(3)]
    lowf = sum(1 for p in picks if is_low(p)) / K
    return mean, sd, fr, lowf

K = 2000
for mode in ('imp', 'max'):
    for N in (1, 4, 16, 64, 256):
        mean, sd, fr, lowf = run(0.9, N, K, 1, mode)
        print(f'MC tau=0.9 mode={mode} N={N} K={K} seed=1: mean={mean:.4f} std={sd:.4f} clusters={[round(f,4) for f in fr]} low={lowf:.4f}')
# pi_imp cluster masses (exact) for comparison
v = v_expectile(0.9, pi)
for k in range(3):
    pass
dimp = [d * w_expectile(x, v, 0.9) for d, x in zip(dmu, xs)]
Z = sum(w * d for w, d in zip(wts, dimp))
cm = [sum(w * d for w, d, x in zip(wts, dimp, xs) if nearest(x) == k) / Z for k in range(3)]
print('pi_imp(tau=0.9) cluster masses', [round(c, 4) for c in cm])
