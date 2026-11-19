"""Verification for tutorial 406 (Epsilon Scaling). Same algorithms as the page (see design.md 4.0).
Run: python3 -I check.py   (numpy only used for Monte Carlo cross-checks)"""
import math, sys
T = 1000
def abar_full(kind):
    if kind == 'linear':
        ab, p = [], 1.0
        for i in range(T):
            p *= 1 - (1e-4 + (0.02 - 1e-4) * i / (T - 1)); ab.append(p)
        return ab
    f = lambda t: math.cos((t / T + 0.008) / 1.008 * math.pi / 2) ** 2
    ab, p = [], 1.0
    for i in range(T):
        p *= 1 - min(1 - f(i + 1) / f(i), 0.999); ab.append(p)
    return ab
def respace(kind, Tp):
    ab = abar_full(kind)
    return [1.0] + [ab[round(i * (T - 1) / (Tp - 1))] for i in range(Tp)]  # A[0]=1, A[t], t=1..Tp
def coefs(A, t):  # step x_t -> x_{t-1}
    a = A[t] / A[t - 1]; b = 1 - a
    return dict(alpha=a, beta=b, q=b / (1 - A[t]), c1=math.sqrt(A[t - 1]) * b / (1 - A[t]),
                c2=math.sqrt(a) * (1 - A[t - 1]) / (1 - A[t]), bt=(1 - A[t - 1]) * b / (1 - A[t]))
def e_of(model, val, A, t):
    return val if model == 'x0' else val * math.sqrt((1 - A[t]) / A[t])
# ---------- C1: single step (Table 1, Table 8) ----------
def c1(kind, Tp, t, e, rho=0.0):
    A = respace(kind, Tp); c = coefs(A, t + 1)
    train = 1 - A[t]
    gap_ddpm = (c['c1'] * e) ** 2
    extra = rho * (c['beta'] - c['bt'])
    a1 = c['alpha']
    gap_ddim = (1 - math.sqrt((a1 - A[t + 1]) / (1 - A[t + 1]))) ** 2 * A[t] * e ** 2
    return dict(abar=A[t], train=train, gap=gap_ddpm, samp=train + gap_ddpm + extra, bt=c['bt'], beta=c['beta'], gap_ddim=gap_ddim)
# ---------- C2: multi-step, exact recursion under Eq. 7 ----------
def c2(kind, Tp, model, val):
    A = respace(kind, Tp); V = {Tp: 1 - A[Tp]}; single = {}; dN = {Tp: 1.0}
    for t in range(Tp, 0, -1):
        c = coefs(A, t); et = e_of(model, val, A, t)
        V[t - 1] = c['c2'] ** 2 * V[t] + (c['c1'] * et) ** 2 + c['bt']
        single[t - 1] = (c['c1'] * et) ** 2
        dN[t] = math.sqrt((V[t] + A[t] * et ** 2) / (1 - A[t] + A[t] * et ** 2))
    out = {}
    for t in range(1, Tp):
        bb = 1 - A[t]
        out[t] = dict(single=single[t], multi=V[t] - bb, delta=(math.sqrt(V[t]) - math.sqrt(bb)) ** 2, dN=dN[t])
    return A, out
# ---------- C3: Eq. 10 ----------
def c3(Tp, bstar, k):
    b = bstar - k * (Tp - 1) / 2
    lam = [k * i + b for i in range(Tp)]
    F = [sum((lam[s] - 1) * (s - t) for s in range(t + 1, Tp)) for t in range(Tp)]
    return b, lam, F
def main():
    print('== C1 (linear, T\'=20, e=0.1) ==')
    for t in [1, 2, 5, 10, 15, 18]:
        r = c1('linear', 20, t, 0.1); r2 = c1('linear', 20, t, 0.1, 1.0)
        print(f"t={t:2d} abar={r['abar']:.5f} train={r['train']:.5f} gapDDPM={r['gap']:.3e} gapDDIM={r['gap_ddim']:.3e} bt={r['bt']:.5f} beta={r['beta']:.5f} samp(rho=1)={r2['samp']:.5f}")
    print('== C1 cosine T\'=20 e=0.1 =='); 
    for t in [1, 10, 18]:
        r = c1('cosine', 20, t, 0.1); print(f"t={t} train={r['train']:.5f} gapDDPM={r['gap']:.3e} gapDDIM={r['gap_ddim']:.3e}")
    for kind in ['linear', 'cosine']:
        for model, val in [('x0', 0.1), ('x0', 0.3)]:
            print(f'== C2 {kind} T\'=20 model={model} val={val} ==')
            A, o = c2(kind, 20, model, val)
            for t in [1, 2, 5, 10, 15, 19]:
                r = o[t]; print(f"t={t:2d} single={r['single']:.3e} multi={r['multi']:.3e} delta={r['delta']:.3e} dN={r['dN']:.4f}")
    print('== C2 multi/single ratio, linear x0 0.1, T\'=20,50,100 at t=1,mid ==')
    for Tp in [10, 20, 50, 100]:
        A, o = c2('linear', Tp, 'x0', 0.1); m = Tp // 2
        print(Tp, f"t=1 ratio={o[1]['multi']/o[1]['single']:.4f}", f"t={m} ratio={o[m]['multi']/o[m]['single']:.4f}", f"delta1={o[1]['delta']:.4e} maxdN={max(v['dN'] for v in o.values()):.4f}")
    print('== C3 Eq.10 ==')
    for Tp, bs, k in [(20, 1.017, 0.0), (20, 1.011, 0.0008), (20, 1.017, 0.0025)]:
        b, lam, F = c3(Tp, bs, k)
        sd = [F[t - 1] - 2 * F[t] + F[t + 1] for t in range(1, Tp - 1)]
        print(Tp, bs, k, f"b={b:.5f} lam0={lam[0]:.5f} lamLast={lam[-1]:.5f} F0={F[0]:.4f} F5={F[5]:.4f} F10={F[10]:.4f} 2nd-diff ok={max(abs(sd[i]-(lam[i+1]-1)) for i in range(len(sd)))<1e-12}")
    print('== A.10 rule check (Table 14 ADM-ES*, T\'=20, b*) ==')
    for name, bs, k, b in [('CIFAR', 1.017, 0.0025, 1.0), ('LSUN', 1.011, 0.0008, 1.0034), ('FFHQ', 1.008, 0.0004, 1.0042), ('IN64', 1.006, 0.0002, 1.0041), ('IN128', 1.005, 0.00022, 1.00291)]:
        print(name, 'implied b (t=0..19):', round(bs - k * 9.5, 5), 'table b:', b, 'implied b (t=1..20):', round(bs - k * 10.5, 5))
    print('== C4 claims ==')
    print('FFHQ ES:', round(14.52 - 6.77, 2), round(26.15 - 9.50, 2), round(59.35 - 26.14, 2), 'ES*:', round(59.35 - 24.83, 2))
    print('CelebA DDPM %:', [round(100 * (a - b) / a, 1) for a, b in [(22.61, 11.83), (15.09, 5.57), (11.33, 4.45)]])
    print('speed:', 511 / 100, 2047 / 100)
    try:
        import numpy as np
    except ImportError:
        print('numpy missing; skip MC'); return
    rng = np.random.default_rng(0); N = 400000
    print('== MC Algorithm 2 under Eq.7 (linear, T\'=20, e=0.1), control variate = same chain with e=0 ==')
    A = respace('linear', 20); Tp = 20
    x0 = rng.normal(0, 0.5, N); xT = math.sqrt(A[Tp]) * x0 + math.sqrt(1 - A[Tp]) * rng.normal(size=N); xh = xT.copy(); xe = xT.copy(); n = {}
    for t in range(Tp, 0, -1):
        c = coefs(A, t); z = rng.normal(size=N); ep = rng.normal(size=N)
        xh = c['c1'] * (x0 + 0.1 * ep) + c['c2'] * xh + math.sqrt(c['bt']) * z
        xe = c['c1'] * x0 + c['c2'] * xe + math.sqrt(c['bt']) * z
        m = math.sqrt(A[t - 1]) * x0; n[t - 1] = np.var(xh - m) - np.var(xe - m)
    _, o = c2('linear', 20, 'x0', 0.1)
    for t in [1, 2, 5, 10, 19]: print(t, f"MC multi={n[t]:.4e} closed={o[t]['multi']:.4e}")
    print('== Table 7 / Eq.23 two-step check (x_{t+1} ground truth), linear T\'=20, e=0.3 ==')
    for t in [2, 5, 10]:
        ct, ct1 = coefs(A, t), coefs(A, t + 1); e = 0.3
        f = (ct1['c1'] * e) ** 2
        exact = 1 - A[t - 1] + (ct['c1'] * e) ** 2 + ct['c2'] ** 2 * f
        paper = 1 - A[t - 1] + (ct['c1'] * e) ** 2 + ct['alpha'] * (1 - A[t - 1]) ** 2 / (4 * (1 - A[t]) ** 3) * f ** 2
        x0 = rng.normal(0, 0.5, N); xt1 = math.sqrt(A[t + 1]) * x0 + math.sqrt(1 - A[t + 1]) * rng.normal(size=N)
        xt = ct1['c1'] * (x0 + e * rng.normal(size=N)) + ct1['c2'] * xt1 + math.sqrt(ct1['bt']) * rng.normal(size=N)
        xm = ct['c1'] * (x0 + e * rng.normal(size=N)) + ct['c2'] * xt + math.sqrt(ct['bt']) * rng.normal(size=N)
        print(t, f"MC={np.var(xm-math.sqrt(A[t-1])*x0):.6f} exact={exact:.6f} Eq23={paper:.6f} train={1-A[t-1]:.6f}")
    print('== delta floor: perfect Gaussian sampler, x0~N(0,s2), DDPM 1000 steps, Algorithm 3 ==')
    for s2 in [0.25, 0.5]:
        Af = [1.0] + abar_full('linear'); P = np.array([[s2, math.sqrt(Af[T]) * s2], [math.sqrt(Af[T]) * s2, Af[T] * s2 + 1 - Af[T]]])  # cov(x0, xhat)
        for t in range(T, 0, -1):
            c = coefs(Af, t); g = math.sqrt(1 - Af[t]) / (Af[t] * s2 + 1 - Af[t])  # eps* = g x
            a = (1 - c['q'] * math.sqrt(1 - Af[t]) * g) / math.sqrt(c['alpha'])  # xhat_{t-1} = a xhat_t + sqrt(bt) z
            M = np.array([[1, 0], [0, a]]); P = M @ P @ M.T; P[1, 1] += c['bt']
            if t - 1 in (1, 10, 100):
                tt = t - 1; bh = P[1, 1] - 2 * math.sqrt(Af[tt]) * P[0, 1] + Af[tt] * P[0, 0]
                print(f"s2={s2} t={tt} Var(xhat)={P[1,1]:.4f} target={Af[tt]*s2+1-Af[tt]:.4f} beta_hat={bh:.4f} delta={(math.sqrt(bh)-math.sqrt(1-Af[tt]))**2:.4f}")
if __name__ == '__main__':
    main()
