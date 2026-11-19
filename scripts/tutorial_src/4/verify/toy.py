"""Toy (L=2, |V|=2, modes new-york / san-diego) used by widgets C1a, C1b, C2.
Same algorithm as the page (design.md 4.0). Run: python3 toy.py"""
import math, random

# ---------- PRNG identical to page: mulberry32 + Box-Muller (cos branch, 1-u1) ----------
def mulberry32(seed):
    a = seed & 0xFFFFFFFF
    def imul(x, y):
        return ((x & 0xFFFFFFFF) * (y & 0xFFFFFFFF)) & 0xFFFFFFFF
    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return nxt

def gauss(r):
    u1, u2 = r(), r()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)

def samples(seed=7, n=200):
    r = mulberry32(seed)
    return [[gauss(r) for _ in range(4)] for _ in range(n)]   # x11 x12 x21 x22

# ---------- exact flow of the toy ----------
H_MAX, SAT, MAX_STEPS, T_CAP = 0.01, 20.0, 20000, 1 - 1e-6

def kfun(t):            # logit of p(NY|x) is t*u/(1-t)^2 ; tanh argument is k*u
    return t / (2 * (1 - t) ** 2)

def flowU(u, s, t):
    """integrate du/dt = (2 tanh(k u) - u)/(1-t) from s to t (s<=t<=1) in r=-ln(1-t)."""
    if u == 0 or t <= s:
        return u if t > s or True else u
    sg = 1.0 if u > 0 else -1.0
    rs = -math.log(1 - s)
    rt = math.inf if t >= 1 else -math.log(1 - t)
    r, steps = rs, 0
    f = lambda r_, u_: 2 * math.tanh(kfun(1 - math.exp(-r_)) * u_) - u_
    while r < rt:
        k = kfun(1 - math.exp(-r))
        if abs(k * u) > SAT:                      # saturated: exact linear tail
            dr = rt - r
            return 2 * sg + (u - 2 * sg) * (0.0 if dr == math.inf else math.exp(-dr))
        if steps >= MAX_STEPS:
            return 0.0                              # stuck at the unstable point -> tie
        h = min(H_MAX, 0.5 / (2 * k + 1), rt - r)
        k1 = f(r, u); k2 = f(r + h / 2, u + h / 2 * k1)
        k3 = f(r + h / 2, u + h / 2 * k2); k4 = f(r + h, u + h * k3)
        u = u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if u * sg <= 0:                             # sign guard (u=0 is invariant)
            u = 0.0; return u
        r += h; steps += 1
    return u

def to_coords(x):
    a1, a2 = x[0] - x[1], x[2] - x[3]
    return a1 + a2, a1 - a2, x[0] + x[1], x[2] + x[3]

def from_coords(u, w, s1, s2):
    a1, a2 = (u + w) / 2, (u - w) / 2
    return [(s1 + a1) / 2, (s1 - a1) / 2, (s2 + a2) / 2, (s2 - a2) / 2]

def X(x, s, t):
    u, w, s1, s2 = to_coords(x)
    ut = flowU(u, s, t)
    c = (1 - t) / (1 - s)
    return from_coords(ut, w * c, 1 - (1 - s1) * c, 1 - (1 - s2) * c)

def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))

def pNY(x, t):
    u = to_coords(x)[0]
    if t >= 1: return 1.0 if u > 0 else (0.0 if u < 0 else 0.5)
    return sigmoid(t * u / (1 - t) ** 2)

def D(x, t):
    p = pNY(x, t); return [p, 1 - p, p, 1 - p]

def delta(x, s, t):
    if t - s < 1e-12: return D(x, s)
    Xt = X(x, s, t)
    return [((1 - s) * Xt[i] - (1 - t) * x[i]) / (t - s) for i in range(4)]

def gamma(s, u, t):
    return 0.0 if t >= 1 else (1 - t) * (u - s) / ((1 - u) * (t - s))

# ---------- samplers ----------
def euler(x, N):
    for n in range(N):
        t0, t1 = n / N, (n + 1) / N
        d = D(x, t0)
        x = [x[i] + (t1 - t0) * (d[i] - x[i]) / (1 - t0) for i in range(4)]
    return x

def flowmap(x, N):
    for n in range(N):
        x = X(x, n / N, (n + 1) / N)
    return x

def classify(x, eps=1e-12):
    d1, d2 = x[0] - x[1], x[2] - x[3]
    if abs(d1) < eps or abs(d2) < eps: return 'tie'
    return {(True, True): 'NY', (False, False): 'SD', (True, False): 'new-diego', (False, True): 'san-york'}[(d1 > 0, d2 > 0)]

def dist_onehot(x):
    o1 = [1, 0] if x[0] >= x[1] else [0, 1]
    o2 = [1, 0] if x[2] >= x[3] else [0, 1]
    o = o1 + o2
    return math.sqrt(sum((x[i] - o[i]) ** 2 for i in range(4)))

# ---------- discrete masked-diffusion toy (C1a) ----------
def discrete_closed(N):
    inv = 1 / (2 * N)
    return {'NY': (1 - inv) / 2, 'SD': (1 - inv) / 2, 'new-diego': inv / 2, 'san-york': inv / 2}

def discrete_mc(N, trials=200000, seed=1):
    rnd = random.Random(seed); cnt = {}
    for _ in range(trials):
        y = [None, None]   # None = mask ; tokens: position1 in {new,san}, position2 in {york,diego}
        for n in range(N):
            s, t = n / N, (n + 1) / N
            unmask = [y[l] is None and rnd.random() < (t - s) / (1 - s) for l in range(2)]
            # factorized: each newly unmasked position samples from its conditional given y_s (perfect model)
            ys = list(y); new = list(y)
            for l in range(2):
                if unmask[l]:
                    other = ys[1 - l]
                    if other is None: mode = rnd.random() < 0.5         # marginal 50/50
                    else: mode = other                                 # conditional is deterministic
                    new[l] = mode
            y = new
        key = {(True, True): 'NY', (False, False): 'SD', (True, False): 'new-diego', (False, True): 'san-york'}[(y[0], y[1])]
        cnt[key] = cnt.get(key, 0) + 1
    return {k: v / trials for k, v in cnt.items()}

if __name__ == '__main__':
    print('== C1a discrete (closed form vs Monte Carlo) ==')
    for N in [1, 2, 4, 8, 16, 32, 64]:
        c = discrete_closed(N); m = discrete_mc(N, 100000)
        print(f'N={N:3d} invalid={c["new-diego"]+c["san-york"]:.4f} MC={m.get("new-diego",0)+m.get("san-york",0):.4f}  NY={c["NY"]:.4f}')
    xs = samples()
    print('first sample', [round(v, 4) for v in xs[0]])
    print('== C1b continuous samplers, seed 7, 200 samples ==')
    ref = [flowmap(x, 1) for x in xs]
    for N in [1, 2, 4, 8, 16, 32]:
        for name, fn in [('euler', euler), ('flowmap', flowmap)]:
            ends = [fn(x, N) for x in xs]
            cnt = {}
            for e in ends: cnt[classify(e)] = cnt.get(classify(e), 0) + 1
            md = sum(dist_onehot(e) for e in ends) / len(ends)
            dev = max(max(abs(e[i] - r[i]) for i in range(4)) for e, r in zip(ends, ref))
            print(f'N={N:3d} {name:8s} counts={dict(sorted(cnt.items()))} mean|x-onehot|={md:.4f} maxdev_vs_flowmap1={dev:.2e}')
    # convergence check of flowU: halve H_MAX
    import toy as T
    us = [to_coords(x)[0] for x in xs]
    e1 = [flowU(u, 0, 0.9) for u in us]
    T_H = H_MAX
    globals()['H_MAX'] = 0.001
    e2 = [flowU(u, 0, 0.9) for u in us]
    globals()['H_MAX'] = T_H
    print('flowU(0->0.9) max diff H=0.01 vs 0.001:', max(abs(a - b) for a, b in zip(e1, e2)))

def highlight_index(xs):
    for i, x in enumerate(xs):
        u, w, _, _ = to_coords(x); a1, a2 = (u + w) / 2, (u - w) / 2
        if abs(u) > 0.5 and a1 * a2 < 0: return i
    return 0
