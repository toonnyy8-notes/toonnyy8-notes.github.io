#!/usr/bin/env python3
"""Acceptance numbers for tutorial 239 (AnisoFlow). Stdlib only, fixed seeds.
Same algorithms as the page (design.md §4). Run: python3 check.py"""
import math

# ---------- PRNG: exact port of the page's mulberry32 + Box-Muller (cos branch) ----------
def imul(a, b):
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) & 0xFFFFFFFF

def mulberry32(seed):
    st = [seed & 0xFFFFFFFF]
    def r():
        a = (st[0] + 0x6D2B79F5) & 0xFFFFFFFF
        st[0] = a
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r

def gauss(r):
    u1 = r(); u2 = r()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)

# ---------- C1: target, recovery, amplification ----------
def c1(t, w, delta, seed):
    r = mulberry32(seed); x = gauss(r); e = gauss(r)
    a, s = 1 - t, t
    z = a * x + s * e
    uW = w * e - x
    uWh = uW + delta
    den = a * w + s
    uh = ((1 - w) * z + uWh) / den
    dc = max(den, 0.04)                      # clamped denominator (App. B.1), enters the recovery itself
    uhc = ((1 - w) * z + uWh) / dc
    u = e - x
    return dict(x=x, eps=e, z=z, uW=uW, uWhat=uWh, uhat=uh, u=u, err=uh - u, amp=1 / den,
                den_c=dc, uhat_c=uhc, err_c=uhc - u, bias_c=u * (den / dc - 1), noise_c=delta / dc,
                pi_eps=a * w / den, pi_x=s / den)

# ---------- C2: frozen-error single timestep ----------
def c2(t, A, C, rho):
    a, s = 1 - t, t
    G = rho * math.sqrt(A * C)
    f = lambda w: (A * w * w - 2 * G * w + C) / (a * w + s) ** 2
    P, Q = A * s + a * G, G * s + a * C
    if P > 0 and Q > 0:
        ws = Q / P; kind = 'interior'; fmin = f(ws)
    elif P > 0:
        ws = 0.0; kind = 'w=0'; fmin = f(0)
    else:
        ws = math.inf; kind = 'w->inf'; fmin = A / a ** 2
    out = dict(P=P, Q=Q, kind=kind, wstar=ws, wstar_eq19=(a / s) * C / A, fmin=fmin,
               f0=C / s ** 2, finf=A / a ** 2, f1=f(1.0))
    if rho == 0:
        out['fstar_ivw'] = 1 / (a * a / A + s * s / C)
    return out

# ---------- C3: separable vs zero-skill; time-averaged constant optimum ----------
M = 1024
TG = [(j + 0.5) / M for j in range(M)]

def dens(sampler, mu):
    if sampler == 'uniform':
        p = [1.0] * M
    else:  # logit-normal, s = 1, in THIS paper's t (t=1 noise)
        p = [math.exp(-(math.log(t / (1 - t)) - mu) ** 2 / 2) / (math.sqrt(2 * math.pi) * t * (1 - t)) for t in TG]
    S = sum(p)
    return [v / S for v in p]  # normalised weights summing to 1

def avg_loss(mode, par, wts, w):
    tot = 0.0
    for t, pj in zip(TG, wts):
        a, s = 1 - t, t
        num = (a * w * w + s * par) if mode == 'sep' else (w * w + par)
        tot += pj * num / (a * w + s) ** 2
    return tot

def c3_opt(mode, par, sampler, mu):
    wts = dens(sampler, mu)
    NG = 4001  # log10 w in [-4, 4], step 0.002
    L = [avg_loss(mode, par, wts, 10 ** (-4 + 0.002 * i)) for i in range(NG)]
    i = min(range(NG), key=lambda k: L[k])
    x = -4 + 0.002 * i
    if i == 0 or i == NG - 1:
        return ('<=1e-4' if i == 0 else '>=1e4')
    if True:  # parabolic refinement in log10 w
        y0, y1, y2 = L[i - 1], L[i], L[i + 1]
        dd = y0 - 2 * y1 + y2
        if dd > 0:
            x += 0.002 * 0.5 * (y0 - y2) / dd
    return 10 ** x

def c3_pert(mode, par):
    ts = [0.1, 0.3, 0.5, 0.7, 0.9]
    return [(t, par if mode == 'sep' else (1 - t) / t * par) for t in ts]

# ---------- C4: Gaussian linear model, w-invariance ----------
def c4(lam, t, w):
    a, s = 1 - t, t
    vz = a * a * lam + s * s
    phi = (s * w - a * lam) / vz
    kap_rec = ((1 - w) + phi) / (a * w + s)
    kap_cf = (s - a * lam) / vz
    vel_min = lam / vz
    tgt_min = (a * w + s) ** 2 * vel_min
    return dict(phi=phi, kappa_recovered=kap_rec, kappa_closed=kap_cf, vel_min=vel_min, tgt_min=tgt_min)

# ---------- C5: first spectral update from W = I (r = x, a_t = b_t = 1) ----------
def c5_draw(r, lam1, lam2, th, n, cross=True):
    c, sn = math.cos(th), math.sin(th)
    g11 = g12 = g22 = 0.0
    for _ in range(n):
        ga, gb = gauss(r), gauss(r)
        e1, e2 = gauss(r), gauss(r)
        p, q = math.sqrt(lam1) * ga, math.sqrt(lam2) * gb
        x1, x2 = c * p - sn * q, sn * p + c * q
        g11 += x1 * x1; g12 += x1 * x2; g22 += x2 * x2
        if cross:
            g11 -= 2 * x1 * e1; g22 -= 2 * x2 * e2; g12 -= x1 * e2 + e1 * x2
    g11 /= n; g12 /= n; g22 /= n
    ang = 0.5 * math.atan2(2 * g12, g11 - g22)  # major-eigvec angle
    err = math.degrees(ang - th)
    err = (err + 90) % 180 - 90
    return err

def c5(lam2, thdeg, n, seed, cross=True, reps=200):
    r = mulberry32(seed)
    first = c5_draw(r, 1.0, lam2, math.radians(thdeg), n, cross)
    r = mulberry32(seed)
    errs = sorted(abs(c5_draw(r, 1.0, lam2, math.radians(thdeg), n, cross)) for _ in range(reps))
    med = 0.5 * (errs[reps // 2 - 1] + errs[reps // 2])
    return first, med

if __name__ == '__main__':
    print('PRNG seed 239 first 4 uniforms:', [round(v, 6) for v in (lambda r: [r(), r(), r(), r()])(mulberry32(239))])
    print('\n== C1')
    for (t, w, d) in [(0.3, 1.0, 0.1), (0.3, 0.0, 0.1), (0.05, 0.0, 0.1), (0.05, 1.0, 0.1), (0.05, 10.0, 0.1), (0.01, 0.0, 0.1)]:
        o = c1(t, w, d, 239)
        print(f't={t} w={w} d={d}: ' + ' '.join(f'{k}={v:.4f}' for k, v in o.items()))
    print('\n== C2')
    for args in [(0.5, 1, 0.1, 0), (0.5, 1, 0.1, 0.5), (0.5, 1, 0.1, -0.5), (0.2, 1, 0.1, 0), (0.8, 1, 0.1, 0),
                 (0.5, 1, 10, -0.9), (0.5, 0.01, 10, 0.9), (0.8, 10, 0.01, -0.9)]:
        o = c2(*args)
        print(args, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()})
    print('\n== C3 per-t minimisers')
    print('sep r=1.5', c3_pert('sep', 1.5))
    print('zs lam=1', [(t, round(w, 4)) for t, w in c3_pert('zs', 1.0)])
    print('\n== C3 time-averaged constant optimum')
    for mode, par in [('sep', 1.5), ('sep', 0.1), ('zs', 1.0), ('zs', 0.1), ('zs', 10.0)]:
        for sampler, mu in [('uniform', 0), ('logitn', 0), ('logitn', -1.0), ('logitn', 1.0)]:
            v = c3_opt(mode, par, sampler, mu)
            print(mode, par, sampler, mu, v if isinstance(v, str) else f'{v:.4g}')
    for mode, par, mu in [('zs', 1.0, -1.5), ('zs', 1.0, -0.5), ('zs', 1.0, 0.5), ('zs', 1.0, 1.5), ('sep', 0.1, 1.5),
                          ('zs', 100.0, -1.5), ('zs', 0.01, 1.5), ('sep', 100.0, -1.5), ('sep', 0.01, 1.5)]:
        v = c3_opt(mode, par, 'logitn', mu)
        print(mode, par, 'logitn', mu, v if isinstance(v, str) else f'{v:.4g}')
    for mode, par in [('zs', 100.0), ('zs', 0.01)]:
        v = c3_opt(mode, par, 'uniform', 0)
        print(mode, par, 'uniform', v if isinstance(v, str) else f'{v:.4g}')
    print('\n== C3 EDM check (t=0.5, lam=1): s, c_skip, w')
    for lam, t in [(1.0, 0.5), (4.0, 0.2)]:
        a, s_ = 1 - t, t; st = s_ / a; cs = lam / (lam + st * st)
        print(lam, t, round(st, 4), round(cs, 4), round(cs * st / (1 - cs), 4), round(a / s_ * lam, 4))
    print('\n== C4')
    for lam, t in [(4.0, 0.5), (0.04, 0.5), (4.0, 0.1)]:
        for w in [0.0, 0.1, 1.0, 10.0]:
            o = c4(lam, t, w)
            print(lam, t, w, {k: round(v, 6) for k, v in o.items()})
    print('\n== C5 (lam1=1, theta=30 deg, seed 239)')
    for lam2 in [0.25, 0.8]:
        for n in [8, 32, 128, 512, 2048]:
            f1, med = c5(lam2, 30, n, 239, True)
            f2, med2 = c5(lam2, 30, n, 239, False)
            print(f'lam2={lam2} n={n}: first_err={f1:.2f} median|err|={med:.2f} | no-cross first={f2:.2f} median={med2:.2f}')
