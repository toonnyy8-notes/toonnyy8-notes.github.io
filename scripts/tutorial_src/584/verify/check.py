"""Verification for tutorial 584 widgets. Same algorithms the page will use. Run: python3 check.py"""
import math

def sig(u):
    return 1/(1+math.exp(-u)) if u >= 0 else math.exp(u)/(1+math.exp(u))

def lse(zs):
    m = max(zs); return m + math.log(sum(math.exp(z-m) for z in zs))

# ---------------- C1: implicit gate in one head ----------------
ZC = [1.0, 0.5, 2.0, -0.5, 0.0]                      # content logits z_1..z_5 (before shift s)
V = [(1,0),(0,1),(-1,0.5),(0.5,-1),(0.8,0.8)]        # v_1..v_5
V0DIR = (1/math.sqrt(2), 1/math.sqrt(2))             # direction of v_0, scaled by rho

def c1(mode, z0, s, sink=None, g=None, rho=0.0):
    z = [z0] + [c+s for c in ZC]
    v = [(rho*V0DIR[0], rho*V0DIR[1])] + V
    if mode == 'vanilla':
        L = lse(z); A = [math.exp(x-L) for x in z]
        Lnot0 = lse(z[1:])
        G_sub = 1-A[0]; G_sig = sig(Lnot0 - z0)
        At = [math.exp(x-Lnot0) for x in z[1:]]
        Od = [sum(A[j]*v[j][i] for j in range(6)) for i in range(2)]
        Og = [G_sig*sum(At[j]*v[j+1][i] for j in range(5)) for i in range(2)]
        return dict(A=A, Asink=A[0], G_sub=G_sub, G_sig=G_sig, LSE=Lnot0, Od=Od, Og=Og,
                    err=math.hypot(Od[0]-Og[0], Od[1]-Og[1]))
    if mode == 'sink':
        Lt = lse(z); Lall = lse(z+[sink])
        A = [math.exp(x-Lall) for x in z]; As = math.exp(sink-Lall)
        G_sig = sig(Lt - sink)
        At = [math.exp(x-Lt) for x in z]
        Od = [sum(A[j]*v[j][i] for j in range(6)) for i in range(2)]
        Og = [G_sig*sum(At[j]*v[j][i] for j in range(6)) for i in range(2)]
        return dict(A=A, Asink=As, G_sub=1-As, G_sig=G_sig, LSE=Lt, Od=Od, Og=Og,
                    err=math.hypot(Od[0]-Og[0], Od[1]-Og[1]))
    if mode == 'gated':
        L = lse(z); A = [math.exp(x-L) for x in z]; G = sig(g)
        Od = [G*sum(A[j]*v[j][i] for j in range(6)) for i in range(2)]
        return dict(A=A, G=G, implicit=1-A[0], Od=Od)

def fmt(xs, n=4): return '[' + ', '.join(f'{x:.{n}f}' for x in xs) + ']'

print('== C1 ==')
for args in [('vanilla',4.0,0.0,None,None,0.0), ('vanilla',4.0,0.0,None,None,1.0),
             ('vanilla',8.0,0.0,None,None,0.0), ('vanilla',-2.0,0.0,None,None,0.0),
             ('vanilla',4.0,2.0,None,None,0.0),
             ('sink',0.5,0.0,4.0,None,0.0), ('sink',0.5,0.0,-4.0,None,0.0), ('sink',4.0,0.0,4.0,None,0.0), ('sink',0.5,0.0,4.0,None,1.0)]:
    r = c1(*args)
    print(args, {k:(fmt(v) if isinstance(v,list) else round(v,4)) for k,v in r.items()})
# cancellation demo: vanilla z0=40 (A0 -> 1): naive 1-A0 vs sigmoid form
r = c1('vanilla', 40.0, 0.0)
print('z0=40: G_sub', r['G_sub'], 'G_sig', r['G_sig'])
r = c1('vanilla', 12.0, -4.0)
print('extreme z0=12,s=-4: G_sub %.6e G_sig %.6e' % (r['G_sub'], r['G_sig']))

# ---------------- C2: CV / Laux calculator ----------------
def profile(N, beta):
    return [0.05 + 0.9*math.exp(-beta*((7*h+3) % N)/(N-1)) for h in range(N)]

def cvstats(x):
    n = len(x)
    if n <= 1: return dict(mu=x[0] if x else 0, sd=0.0, cv=0.0, ncv2=0.0)
    mu = sum(x)/n; S = sum((a-mu)**2 for a in x); sd = math.sqrt(S/n)
    return dict(mu=mu, sd=sd, cv=sd/mu, ncv2=S/mu**2, S=S)

def grad_ncv2(x):   # d(N*CV^2)/dx_h = 2(x_h-mu)/mu^2 - 2S/(N mu^3)
    n = len(x); mu = sum(x)/n; S = sum((a-mu)**2 for a in x)
    return [2*(a-mu)/mu**2 - 2*S/(n*mu**3) for a in x]

def topm(x, m):  # indices of top-m, ties -> lower index first
    return set(sorted(range(len(x)), key=lambda h: (-x[h], h))[:m])

print('\n== C2 ==')
for N, beta, m in [(16,4,3),(16,0,3),(16,8,3),(10,4,2),(13,4,3),(16,4,0)]:
    x = profile(N, beta); st = cvstats(x)
    T = topm(x, m); xr = [x[h] for h in range(N) if h not in T]; sr = cvstats(xr)
    print(f'N={N} beta={beta} m={m}: Imp={fmt(x,3)}')
    print(f'   mu={st["mu"]:.4f} sd={st["sd"]:.4f} CV={st["cv"]:.4f} N*CV^2={st["ncv2"]:.4f} '
          f'| shared={sorted(T)} routed mu={sr["mu"]:.4f} CV={sr["cv"]:.4f} (N-m)CV^2={(N-m)*sr["cv"]**2:.4f}')
    print(f'   Laux(Eq7, lambda=1e-4)={1e-4*N*st["cv"]**2:.6f}  Laux(Eq8, lambda=1e-2)={1e-2*(N-m)*sr["cv"]**2:.6f}')
    g = grad_ncv2(x); print('   grad=', fmt(g,3), ' sum(grad*x)=%.2e (scale invariance -> 0)' % sum(a*b for a,b in zip(g,x)))
# finite-difference check of analytic gradient
x = profile(16,4); g = grad_ncv2(x); eps=1e-6
fd = []
for h in range(16):
    xp = x[:]; xp[h]+=eps; xm = x[:]; xm[h]-=eps
    fd.append((cvstats(xp)['ncv2']-cvstats(xm)['ncv2'])/(2*eps))
print('max |analytic-fd| =', max(abs(a-b) for a,b in zip(g,fd)))

# ---------------- C3: head pinning toy dynamics ----------------
def run_c3(mode, pin, m, N=16, beta=4, eta=0.05, steps=3000, record=(0,100,300,1000,3000)):
    x0 = profile(N, beta); th = [math.log(a/(1-a)) for a in x0]
    T = topm(x0, m)                       # shared / pinned set fixed from initial Imp
    out = {}
    for s in range(steps+1):
        x = [sig(t) for t in th]
        if s in record:
            if mode == 'eq7': st = cvstats(x); Lv = N*st['cv']**2
            else:
                xr = [x[h] for h in range(N) if h not in T]; Lv = (N-m)*cvstats(xr)['cv']**2
            out[s] = (x[:], Lv, cvstats(x)['cv'])
        if s == steps: break
        if mode == 'eq7':
            g = grad_ncv2(x); scale = 1.0
            gi = {h: g[h] for h in range(N)}
        else:
            idx = [h for h in range(N) if h not in T]; xr = [x[h] for h in idx]
            g = grad_ncv2(xr); gi = {h: g[i] for i,h in enumerate(idx)}
            # (N-m)CV^2 over routed == len(routed)*CV^2 -> same function as N*CV^2 on the subset
        for h in range(N):
            if h in gi and not (pin and h in T):
                th[h] -= eta * gi[h] * x[h]*(1-x[h])
    return x0, T, out

print('\n== C3 ==')
for mode, pin, m in [('eq7',True,3),('eq7',False,3),('eq8',True,3),('eq8',True,1),('eq7',True,1)]:
    x0, T, out = run_c3(mode, pin, m)
    print(f'mode={mode} pin={pin} m={m} shared={sorted(T)}')
    for s,(x,Lv,cv) in out.items():
        xr = [x[h] for h in range(16) if h not in T]
        print(f'  step {s:5d}: L={Lv:.4f} CV(all)={cv:.4f} routed mean={sum(xr)/len(xr):.4f} min={min(xr):.4f} max={max(xr):.4f}')
    print('  final Imp=', fmt(out[3000][0],3))

# limit of routed common value for eq7+pin, m=3 (grid search, step 1e-5)
x0 = profile(16, 4); T = topm(x0, 3); fixed = [x0[h] for h in sorted(T)]
best = min((cvstats(fixed + [i/1e5]*13)['ncv2'], i/1e5) for i in range(1, 100000))
print(f'eq7+pin m=3 routed limit (grid 1e-5): y*={best[1]:.5f}  N*CV^2={best[0]:.4f}')

# ---------------- C4: block-wise (online) LSE ----------------
Z4 = [5.0, 1.2, -0.3, 2.1, 0.7, -1.5, 3.0, 0.2, 1.8, -0.8, 2.6, 0.9]
def online(z, B):
    m = -math.inf; l = 0.0; trace = []
    for i in range(0, len(z), B):
        blk = z[i:i+B]; mn = max(m, max(blk))
        l = l*math.exp(m-mn) + sum(math.exp(a-mn) for a in blk) if m != -math.inf else sum(math.exp(a-mn) for a in blk)
        m = mn; trace.append((m, l, m+math.log(l)))
    return trace

print('\n== C4 ==')
for scale in [1, 10, 200]:
    z = [scale*a for a in Z4]
    for B in [1,4,12]:
        tr = online(z, B)
        L = tr[-1][2]
        print(f'scale={scale} B={B}: trace(m,l,LSE)=', [(round(a,4),round(b,4),round(c,4)) for a,b,c in tr])
    L = online(z, 4)[-1][2]
    try: naive = math.log(sum(math.exp(a) for a in z))
    except OverflowError: naive = float('inf')
    sink = 3.0*scale
    Gs = sig(L - sink); Gv = -math.expm1(z[0]-L); Gv_sub = 1-math.exp(z[0]-L)
    Lnot0 = lse(z[1:])
    em = -math.expm1(z[0]-L)
    Lnot0_from_lse = L + math.log(em) if em > 0 else -math.inf
    print(f'  paper-route LSE_not0 = LSE + log(-expm1(z0-LSE)) = {Lnot0_from_lse} vs direct {Lnot0:.6f}')
    print(f'  LSE={L:.6f} direct={lse(z):.6f} naive={naive}  G_sink(sink={sink})={Gs:.6f}  '
          f'G_vanilla(-expm1)={Gv:.6e} (1-exp)={Gv_sub:.6e} sigma(LSE_not0 - z0)={sig(Lnot0-z[0]):.6e}')
