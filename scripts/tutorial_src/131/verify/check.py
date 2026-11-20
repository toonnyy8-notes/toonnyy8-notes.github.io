"""Acceptance numbers for tutorial 131 (BPO). Same algorithms the page will use (W1-W4)."""
import math

def lse(xs):
    m = max(xs); return m + math.log(sum(math.exp(x - m) for x in xs))

# ---------- W1: single-state PMD ----------
def w1(mu, Q, eta):
    V = sum(m * q for m, q in zip(mu, Q))
    A = [q - V for q in Q]
    lz = lse([math.log(m) + eta * a for m, a in zip(mu, A)])
    pp = [math.exp(math.log(m) + eta * a - lz) for m, a in zip(mu, A)]
    kl_mu_pp = sum(m * math.log(m / p) for m, p in zip(mu, pp))
    kl_pp_mu = sum(p * math.log(p / m) for m, p in zip(mu, pp))
    EA = sum(p * a for p, a in zip(pp, A))
    obj = EA - kl_pp_mu / eta if eta > 0 else 0.0
    lzQ = lz + eta * V
    return dict(V=V, A=A, pi=pp, logZ=lz, KLmu_pi=kl_mu_pp, KLpi_mu=kl_pp_mu, EA=EA, obj=obj, logZQ=lzQ)

# ---------- W2/W3: two-step binary tree ----------
LEAVES = ['aa', 'ab', 'ba', 'bb']
def tree(muA, R, eta):
    """muA: dict state->mu(a|s) for states 'x','a','b'; R: dict leaf->reward."""
    mu = lambda s, y: muA[s] if y == 'a' else 1 - muA[s]
    Vs = {'a': muA['a'] * R['aa'] + (1 - muA['a']) * R['ab'],
          'b': muA['b'] * R['ba'] + (1 - muA['b']) * R['bb']}
    Vx = muA['x'] * Vs['a'] + (1 - muA['x']) * Vs['b']
    A = {('x', 'a'): Vs['a'] - Vx, ('x', 'b'): Vs['b'] - Vx}
    for s, (l1, l2) in (('a', ('aa', 'ab')), ('b', ('ba', 'bb'))):
        A[(s, 'a')] = R[l1] - Vs[s]; A[(s, 'b')] = R[l2] - Vs[s]
    pplus = {}; logZ = {}
    for s in 'xab':
        za = math.log(mu(s, 'a')) + eta * A[(s, 'a')]; zb = math.log(mu(s, 'b')) + eta * A[(s, 'b')]
        l = lse([za, zb]); pplus[s] = math.exp(za - l); logZ[s] = l
    Pmu = {y: mu('x', y[0]) * mu(y[0], y[1]) for y in LEAVES}
    return dict(mu=mu, Vs=Vs, Vx=Vx, A=A, pplus=pplus, logZ=logZ, Pmu=Pmu, R=R, eta=eta, muA=muA)

def kl_bin(q, p):  # KL(mu||pi) on {a,b}, q=mu(a), p=pi(a)
    return q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))

def d_pi(T, piA, s, y, direct=False):
    """Token residual. Default: affine logit form (log-partition cancels; stable for p->0/1).
    direct=True: literal log-ratio + KL form (used only to cross-check)."""
    if direct:
        mu = T['mu']; p = piA[s] if y == 'a' else 1 - piA[s]
        return T['eta'] * T['A'][(s, y)] - (math.log(p) - math.log(mu(s, y)) + kl_bin(T['muA'][s], piA[s]))
    th = piA['_th'][s] if '_th' in piA else logit(piA[s])
    q = T['muA'][s]
    return T['eta'] * T['A'][(s, y)] - ((1 if y == 'a' else 0) - q) * (th - logit(q))

def residuals(T, piA):
    dl = {y: d_pi(T, piA, 'x', y[0]) + d_pi(T, piA, y[0], y[1]) for y in LEAVES}
    L = sum(T['Pmu'][y] * dl[y] ** 2 for y in LEAVES) / (2 * T['eta'])
    J = sum(piA['x'] ** (y[0] == 'a') * (1 - piA['x']) ** (y[0] == 'b') *
            (piA[y[0]] if y[1] == 'a' else 1 - piA[y[0]]) * T['R'][y] for y in LEAVES)
    # martingale checks
    cond = {s: T['muA'][s] * d_pi(T, piA, s, 'a') + (1 - T['muA'][s]) * d_pi(T, piA, s, 'b') for s in 'xab'}
    Ed2 = sum(T['Pmu'][y] * dl[y] ** 2 for y in LEAVES)
    ED1 = sum(T['Pmu'][y] * d_pi(T, piA, 'x', y[0]) ** 2 for y in LEAVES)
    ED2 = sum(T['Pmu'][y] * d_pi(T, piA, y[0], y[1]) ** 2 for y in LEAVES)
    return dict(delta=dl, L=L, J=J, cond=cond, Ed2=Ed2, ED1=ED1, ED2=ED2)

sig = lambda t: 1 / (1 + math.exp(-t))
logit = lambda p: math.log(p / (1 - p))

def g_vec(T, y):
    """d/dtheta of sum_t [log pi + KL(mu||pi)] for trajectory y, theta_s = logit pi(a|s). Independent of pi."""
    g = {'x': 0.0, 'a': 0.0, 'b': 0.0}
    g['x'] += (1 if y[0] == 'a' else 0) - T['muA']['x']
    g[y[0]] += (1 if y[1] == 'a' else 0) - T['muA'][y[0]]
    return g

def grads(T, th):
    piA = {s: sig(th[s]) for s in 'xab'}; piA['_th'] = th
    r = residuals(T, piA)
    ge = {s: 0.0 for s in 'xab'}; gl = {s: 0.0 for s in 'xab'}
    for y in LEAVES:
        g = g_vec(T, y)
        for s in 'xab':
            ge[s] += -T['Pmu'][y] * (r['delta'][y] / T['eta']) * g[s]
            gl[s] += -T['Pmu'][y] * (T['R'][y] - T['Vx']) * g[s]
    return ge, gl, r

def paths(T, steps=200, lr=None):
    lr = T['eta'] if lr is None else lr
    th0 = {s: logit(T['muA'][s]) for s in 'xab'}
    out = {}
    for kind in ('exact', 'lin'):
        th = dict(th0); hist = []
        for k in range(steps + 1):
            ge, gl, r = grads(T, th)
            hist.append((k, sig(th['x']), sig(th['a']), sig(th['b']), r['L'], r['J']))
            g = ge if kind == 'exact' else gl
            for s in 'xab': th[s] -= lr * g[s]
        out[kind] = hist
    return out

# ---------- W5: r vs omega ----------
UNDEF = 'undef'
def ratio(num, den):
    """0/0 -> 'undef'; x/0 (x>0) -> inf; else num/den."""
    if abs(den) < 1e-12:
        return UNDEF if abs(num) < 1e-12 else math.inf
    return num / den

def w4(q, p, eps, C, sgn, mask=None):
    r = ratio(p, q)
    raw = ratio(1 - q, 1 - p)
    om = ratio(1 + eps - q, 1 + eps - p)
    capped = om if om == UNDEF else min(om, C)
    def M(w):
        if mask is None or w == UNDEF: return 1
        lo, hi = mask
        if sgn > 0 and w > 1 + hi: return 0
        if sgn < 0 and w < 1 - lo: return 0
        return 1
    eff = lambda m, w: 0.0 if m == 0 else w   # M=0 -> weight 0 without multiplying (avoids 0*inf)
    return dict(r=r, raw=raw, om=om, capped=capped, Mr=M(r), Mw=M(om),
                wG=eff(M(r), r), wB=eff(M(om), capped))

# ---------- W4: logit-space gradient vectors ----------
def softmax(z):
    l = lse(z); return [math.exp(v - l) for v in z]
def w5(mu, delta, y, eps=0.1):
    zmu = [math.log(m) for m in mu]
    pi = softmax([a + b for a, b in zip(zmu, delta)])
    q, p = mu[y], pi[y]
    e = [1.0 if i == y else 0.0 for i in range(len(mu))]
    pg = [ei - pv for ei, pv in zip(e, pi)]
    full = [ei - m for ei, m in zip(e, mu)]
    bin_ = [(1 - q) / (1 - p) * v for v in pg]
    om = (1 + eps - q) / (1 + eps - p)
    bpo = [om * v for v in pg]
    grpo = [p / q * v for v in pg]
    kl = sum(m * math.log(m / pv) for m, pv in zip(mu, pi))
    klb = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
    cos = sum(a * b for a, b in zip(bin_, full)) / math.sqrt(sum(a * a for a in bin_) * sum(b * b for b in full))
    return dict(pi=pi, pg=pg, full=full, bin=bin_, bpo=bpo, grpo=grpo, om=om, KL=kl, KLbin=klb, cos=cos)

def f(v, n=5):
    if isinstance(v, float):
        return 'inf' if v == math.inf else f'{v:.{n}f}'
    if v == UNDEF: return 'undef'
    if isinstance(v, (list, tuple)): return '(' + ', '.join(f(x, n) for x in v) + ')'
    return str(v)

if __name__ == '__main__':
    print('== W1')
    for mu, Q, eta in [((.4, .3, .2, .1), (.5, 0, -.5, 1.), 2.0), ((.4, .3, .2, .1), (.5, 0, -.5, 1.), 0.0),
                       ((.4, .3, .2, .1), (.5, 0, -.5, 1.), 10.0), ((.25,) * 4, (.5, 0, -.5, 1.), 2.0),
                       ((.85, .05, .05, .05), (.5, 0, -.5, 1.), 2.0)]:
        o = w1(mu, Q, eta)
        print(mu, Q, 'eta', eta, {k: f(v, 4) for k, v in o.items()})
        assert abs(o['logZ'] - o['KLmu_pi']) < 1e-12
    print('== W2')
    base = dict(x=.5, a=.5, b=.5); R0 = dict(aa=1, ab=0, ba=0, bb=0)
    cases = [(base, R0, 1.0), (dict(x=.7, a=.3, b=.6), dict(aa=1, ab=0, ba=0, bb=1), 2.0)]
    for muA, R, eta in cases:
        T = tree(muA, R, eta)
        print('mu', muA, 'R', R, 'eta', eta)
        print('  Vx', f(T['Vx'], 5), 'Va', f(T['Vs']['a'], 5), 'Vb', f(T['Vs']['b'], 5))
        print('  A', {k: f(v, 5) for k, v in T['A'].items()})
        print('  pi+', {k: f(v, 5) for k, v in T['pplus'].items()}, 'logZ', {k: f(v, 5) for k, v in T['logZ'].items()})
        for s in 'xab':
            assert abs(T['logZ'][s] - kl_bin(muA[s], T['pplus'][s])) < 1e-12
        for y in LEAVES:  # telescoping
            assert abs(T['A'][('x', y[0])] + T['A'][(y[0], y[1])] - (R[y] - T['Vx'])) < 1e-12
        for name, piA in [('mu', dict(muA)), ('pi+', dict(T['pplus'])), ('ex', dict(x=.7, a=.6, b=.5))]:
            r = residuals(T, piA)
            print(f'  pi={name}', {k: f(v, 5) for k, v in piA.items()}, 'delta', {k: f(v, 5) for k, v in r['delta'].items()},
                  'L', f(r['L'], 5), 'J', f(r['J'], 4), 'cond', f(list(r['cond'].values()), 2),
                  'E[d^2]', f(r['Ed2'], 5), 'E[D1^2]+E[D2^2]', f(r['ED1'] + r['ED2'], 5), f'({f(r["ED1"],5)} + {f(r["ED2"],5)})')
            assert abs(r['Ed2'] - r['ED1'] - r['ED2']) < 1e-12
            for st in 'xab':
                for yy in 'ab':
                    assert abs(d_pi(T, piA, st, yy) - d_pi(T, piA, st, yy, direct=True)) < 1e-12
            assert all(abs(c) < 1e-12 for c in r['cond'].values())
    print('== W3 (gradient at mu, at pi+, and paths)')
    for muA, R, eta in cases:
        T = tree(muA, R, eta)
        thmu = {s: logit(muA[s]) for s in 'xab'}; thp = {s: logit(T['pplus'][s]) for s in 'xab'}
        ge, gl, _ = grads(T, thmu); print('  at mu exact', f(list(ge.values())), 'lin', f(list(gl.values())))
        ge, gl, _ = grads(T, thp); print('  at pi+ exact', f(list(ge.values())), 'lin', f(list(gl.values())))
        P = paths(T)
        # closed form of the exact path: Hessian of (18) in theta is diagonal, h_s/eta, h_s <= 1/4
        mx, ma, mb = muA['x'], muA['a'], muA['b']
        h = {'x': mx * (1 - mx), 'a': mx * ma * (1 - ma), 'b': (1 - mx) * mb * (1 - mb)}
        assert max(h.values()) <= 0.25 + 1e-15
        print('  Hessian diag * eta (h_x, h_a, h_b) =', f(list(h.values())))
        for k in (10, 50, 200):
            for i, s_ in enumerate('xab'):
                thk = thp[s_] + (1 - h[s_]) ** k * (thmu[s_] - thp[s_])   # alpha = eta
                assert abs(sig(thk) - P['exact'][k][1 + i]) < 1e-12, (k, s_)
        for kind in ('exact', 'lin'):
            for k in (0, 10, 50, 200):
                h = P[kind][k]
                print(f'  {kind:5s} k={k:3d} pi(a|x)={h[1]:.4f} pi(a|a)={h[2]:.4f} pi(a|b)={h[3]:.4f} L={h[4]:.5f} J={h[5]:.4f}')
    print('== W5 (eps=0.1, C=3, A>0, no mask)')
    for q, p in [(.5, .6), (.01, .02), (.98, .99), (.5, .9), (0., 1.), (1., 0.), (.5, .5)]:
        o = w4(q, p, .1, 3., 1)
        print(f'  q={q} p={p}', {k: f(v, 4) for k, v in o.items()})
    print('== W5 mask demo (lo=0.2, hi=0.28, illustration only)')
    for q, p, s in [(.5, .6, 1), (.5, .9, 1), (.5, .9, -1), (.5, .2, -1), (.98, .99, 1), (.01, .02, 1)]:
        o = w4(q, p, .1, 3., s, mask=(.2, .28))
        print(f'  q={q} p={p} sgnA={s}', {k: f(v, 4) for k, v in o.items()})
    print('== W5 undefined / infinite cases')
    for q, p, e in [(1., 1., .1), (1., 1., 0.), (0., 0., .1), (.5, 1., 0.)]:
        o = w4(q, p, e, 3., 1, mask=(.2, .28))
        print(f'  q={q} p={p} eps={e}', {k: (v if isinstance(v, str) else f(float(v), 4)) for k, v in o.items()})
    o = w4(0., .5, .1, 3., 1, mask=(.2, .28)); assert o['r'] == math.inf and o['Mr'] == 0 and o['wG'] == 0.0
    assert w4(1., 1., .1, 3., 1)['raw'] == UNDEF and w4(1., 1., 0., 3., 1)['om'] == UNDEF
    print('== W5 eps=0 check at (.5,.6):', f(w4(.5, .6, 0., 3., 1)['om'], 4))
    print('== W4 mu=(.4,.3,.2,.1)')
    mu = (.4, .3, .2, .1)
    for delta, y in [((0, 0, 0, 0), 0), ((1, 0, 0, 0), 0), ((2, 0, 0, 0), 0), ((0, 1, 0, 0), 0), ((0, 0, 0, 2), 3)]:
        o = w5(mu, delta, y)
        print(f'  delta={delta} y={y}')
        for k in ('pi', 'pg', 'full', 'bin', 'bpo', 'grpo'): print('   ', k, f(o[k], 4))
        print('    omega', f(o['om'], 4), 'KL', f(o['KL'], 5), 'KLbin', f(o['KLbin'], 5), 'cos(bin,full)', f(o['cos'], 4))
        assert o['KLbin'] <= o['KL'] + 1e-12
        assert abs(o['bin'][y] - (1 - mu[y])) < 1e-12
    print('== S5 3-token softmax gradient example (z=(1,0,-1), y=t1, mu=(0.5,0.3,0.2))')
    z = [1., 0., -1.]; mu3 = [.5, .3, .2]; pi3 = softmax(z)
    an = [(1. if i == 0 else 0.) - pi3[i] for i in range(3)]
    def logp1(zz): return zz[0] - lse(zz)
    def klf(zz): pz = softmax(zz); return sum(m * math.log(m / pv) for m, pv in zip(mu3, pz))
    hh = 1e-6
    fd = [(logp1([zz + (hh if j == i else 0) for j, zz in enumerate(z)]) - logp1([zz - (hh if j == i else 0) for j, zz in enumerate(z)])) / (2 * hh) for i in range(3)]
    fdk = [(klf([zz + (hh if j == i else 0) for j, zz in enumerate(z)]) - klf([zz - (hh if j == i else 0) for j, zz in enumerate(z)])) / (2 * hh) for i in range(3)]
    assert all(abs(a - b) < 1e-8 for a, b in zip(an, fd))
    assert all(abs((pv - m) - b) < 1e-8 for pv, m, b in zip(pi3, mu3, fdk))
    print('  pi', f(pi3, 4), 'e_y - pi', f(an, 4), 'pi - mu', f([pv - m for pv, m in zip(pi3, mu3)], 4), 'sum', f([a + pv - m for a, pv, m in zip(an, pi3, mu3)], 4))
    print('== group advantage example (G=16, 4 correct, 0/1 reward, population std)')
    Rg = [1] * 4 + [0] * 12; m = sum(Rg) / 16; sd = math.sqrt(sum((r - m) ** 2 for r in Rg) / 16)
    sds = math.sqrt(sum((r - m) ** 2 for r in Rg) / 15)
    print('  mean', m, 'std(pop)', f(sd, 4), 'A+', f((1 - m) / sd, 4), 'A-', f((0 - m) / sd, 4), '| std(sample)', f(sds, 4), 'A+', f((1 - m) / sds, 4), 'A-', f(-m / sds, 4))
    print('== paper arithmetic')
    t1 = {'GRPO-ClipHigher': (45.6, 34.8, 38.0, 39.5), 'GSPO': (50.3, 35.5, 44.6, 43.5), 'CISPO': (52.7, 39.0, 50.4, 47.4),
          'DPPO': (55.8, 39.2, 44.2, 46.4), 'BPO': (57.4, 41.0, 53.0, 50.5)}
    for k, v in t1.items():
        print(' ', k, 'mean3', round(sum(v[:3]) / 3, 2), 'reported', v[3], 'gap', round(50.5 - v[3], 1))
    print('ALL ASSERTS PASSED')
