"""C1-C3 + text numbers. Same algorithms as the page. Run: python3 -I check.py"""
import math

def mulberry32(seed):
    a = seed & 0xFFFFFFFF
    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r

TOK = ["James","buys","10","suits",",","each","costing","$","750",".","Total",":","$","7500","<EOS>","<EoT>"]
L = 16

# ---- C1: coupled forward masking, u_k fixed per seed; masked iff u_k < t
def c1_u(seed):
    r = mulberry32(seed); return [r() for _ in range(L)]
print('== C1')
for seed in (7, 8):
    u = c1_u(seed)
    print(f' seed={seed} u=', [round(x, 4) for x in u])
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        masked = [k+1 for k in range(L) if u[k] < t]
        print(f'   t={t}: #masked={len(masked)} (E={t*L:.1f}) positions={masked}')
# reverse with random-K: NFE = ceil(L/K)
for K in (1, 2, 4, 8, 16): print(f'  K={K}: NFE={math.ceil(L/K)}')

# ---- C2
A = [0.62,0.55,0.48,0.41,0.93,0.36,0.30,0.88,0.27,0.95,0.33,0.71,0.85,0.22,0.58,0.64]
B = [0.58,0.51,0.44,0.39,0.90,0.34,0.29,0.84,0.25,0.92,0.97,0.995,0.995,0.995,0.995,0.995]
sig = lambda x: 1/(1+math.exp(-x))
def policy(c, M, tau, beta):
    s = {k: min(1.0, beta*sig(8*(c[k]-0.7)/tau)) for k in M}
    EU = sum(s.values()); P0 = math.prod(1-v for v in s.values())
    return s, EU, P0, EU + P0
def topk(c, M, K): return sorted(sorted(M, key=lambda k: (-c[k], k))[:K])
def thr(c, M, lam):
    U = [k for k in M if c[k] > lam]
    return U if U else [min(M, key=lambda k: (-c[k], k))]
def loglik(s, u):  # u: dict k->0/1 over masked k
    tot = 0.0
    for k, v in s.items():
        if u[k] == 1: tot += math.log(v) if v > 0 else -math.inf
        else: tot += math.log1p(-v) if v < 1 else -math.inf
    return tot
print('== C2')
for name, c in (('A', A), ('B', B)):
    M = list(range(L))
    print(f' preset {name}: top-4={[k+1 for k in topk(c,M,4)]} thr0.9={[k+1 for k in thr(c,M,0.9)]} thr0.7={[k+1 for k in thr(c,M,0.7)]} thr0.99={[k+1 for k in thr(c,M,0.99)]} thr1.0={[k+1 for k in thr(c,M,1.0)]}')
    for tau, beta in ((1,1),(0.5,1),(1.5,1),(1,0.5),(1,2),(0.5,0.25)):
        s, EU, P0, EUf = policy(c, M, tau, beta)
        print(f'   tau={tau} beta={beta}: E|U|={EU:.3f} P(all0)={P0:.3e} E|U|+fallback={EUf:.3f} s1={s[0]:.4f} s10={s[9]:.4f}')
    s, *_ = policy(c, M, 1, 1)
    u = {k: 1 if c[k] > 0.7 else 0 for k in M}
    print(f'   loglik(u = 1[c>0.7]) tau=1 beta=1: {loglik(s,u):.4f}')
# partially unmasked state for A: positions 5,8,10 already unmasked
M = [k for k in range(L) if k not in (4, 7, 9)]
s, EU, P0, EUf = policy(A, M, 1, 1)
print(f' preset A, unmasked {{5,8,10}}: |M|={len(M)} top-4={[k+1 for k in topk(A,M,4)]} thr0.9={[k+1 for k in thr(A,M,0.9)]} E|U|={EU:.3f} P0={P0:.4f}')
print(' random-K per-position prob K/|M|: K=4 ->', round(4/16, 4), round(4/13, 4))

# ---- C3
def adv(r, n, alpha, T, kind):
    if kind == 'mult': R = [ri*((1-ni/T)**alpha if alpha else 1.0) for ri, ni in zip(r, n)]
    else: R = [ri - alpha*ni/T for ri, ni in zip(r, n)]
    m = sum(R)/len(R); return R, [x-m for x in R]
PRE = {'P1': ([0]*8, [20,40,60,80,100,120,160,200]),
       'P2': ([1,1,0,1,0,0,1,0], [40,120,20,200,60,250,80,30]),
       'P3': ([1]*8, [20,40,60,80,100,120,160,200])}
print('== C3 (T=256)')
for name, (r, n) in PRE.items():
    for alpha in (1, 3, 0.3):
        for kind in ('mult', 'add'):
            R, A_ = adv(r, n, alpha, 256, kind)
            print(f' {name} a={alpha} {kind}: R=' + ' '.join(f'{x:+.3f}' for x in R) + ' | A=' + ' '.join(f'{x:+.3f}' for x in A_))
def surr(rho, A, eps): return min(rho*A, max(1-eps, min(1+eps, rho))*A)
for eps in (0.5, 0.2):
    for A_ in (0.25, -0.25):
        print(f' surrogate eps={eps} A={A_}: ' + ' '.join(f'rho={p}:{surr(p,A_,eps):+.4f}' for p in (0.3, 0.8, 1.0, 1.3, 1.8)))
# speed factor table
for a in (0, 0.3, 1, 3, 10):
    print(f' (1-n/256)^{a}: ' + ' '.join(f'n={n}:{(1-n/256)**a if a else 1.0:.4f}' for n in (12, 32, 64, 128, 256)))

# ---- text numbers
print('== misc')
print(' log10(256!) =', round(math.lgamma(257)/math.log(10), 2), ' 8! =', math.factorial(8), ' 16! =', math.factorial(16))
print(' policy share 300K/8B =', 300e3/8e9*100, '%')
print(' B.9 avg delta@1 =', (1.15+0.80)/2, ' avg delta@32 =', (1.46+3.58)/2)

# ---- C1 generation: random-K, PRNG mulberry32(seed+1), partial Fisher-Yates on ascending masked list
print('== C1 generation traces')
def gen(seed, K):
    r = mulberry32(seed + 1); M = list(range(L)); steps = []
    while M:
        M.sort(); n = min(K, len(M))
        for i in range(n):
            j = i + int(r() * (len(M) - i)); M[i], M[j] = M[j], M[i]
        U = sorted(M[:n]); steps.append([k+1 for k in U]); M = M[n:]
    return steps
for K in (4, 3):
    tr = gen(7, K); print(f' seed=7 K={K}: NFE={len(tr)} steps={tr}')
