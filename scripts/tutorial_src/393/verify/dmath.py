"""Shared math mirroring the page's JS (W1-W5). Pure Python, no deps."""
import math
SQ2PI = math.sqrt(2*math.pi)

def erfc_nr(x):
    # Numerical Recipes erfcc, fractional error < 1.2e-7 everywhere
    z = abs(x); t = 1/(1+0.5*z)
    r = t*math.exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))))
    return r if x >= 0 else 2-r

def Phi(z): return 0.5*erfc_nr(-z/math.sqrt(2))
def phi(z): return math.exp(-0.5*z*z)/SQ2PI
def logPhi(z):
    if z > 0: return math.log1p(-0.5*erfc_nr(z/math.sqrt(2)))
    return math.log(0.5*erfc_nr(-z/math.sqrt(2)))

ZLO, ZHI, H = -10.0, 12.0, 0.005
NZ = int(round((ZHI-ZLO)/H)) + 1
ZG = [ZLO + i*H for i in range(NZ)]

_Gcache = {}
def Gtab(K):
    if K not in _Gcache:
        _Gcache[K] = [math.exp((K-1)*logPhi(z)) for z in ZG]
    return _Gcache[K]

def T_op(at, K):
    """Eq. (11) via I - 1/K = int [phi(z-m)-phi(z)] Phi^{K-1}(z) dz (cancellation-free),
    trapezoid on [ZLO,ZHI], analytic tail beyond ZHI where Phi^{K-1}=1."""
    if at <= 0: return 0.0
    if at >= 1: return 1.0
    m = at/math.sqrt(1-at*at)
    G = Gtab(K); s = 0.0
    for i, z in enumerate(ZG):
        wgt = 0.5 if i in (0, NZ-1) else 1.0
        s += wgt*(phi(z-m)-phi(z))*G[i]
    s *= H
    s += Phi(ZHI) - Phi(ZHI-m)      # tail
    return max(0.0, min(1.0, K/(K-1)*s))

def mulberry32(a):
    a &= 0xFFFFFFFF
    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r

def gauss(r):
    u1 = r(); u2 = r()
    return math.sqrt(-2*math.log(1-u1))*math.cos(2*math.pi*u2)

def bisect_T(target, K, lo=0.0, hi=0.999999):
    for _ in range(60):
        mid = (lo+hi)/2
        if T_op(mid, K) < target: lo = mid
        else: hi = mid
    return (lo+hi)/2

# ---------- W3: kernels ----------
UL, UH, HU = -9.0, 9.0, 0.01
NU = int(round((UH-UL)/HU)) + 1
def push(ws, ats, att):
    """(39) after substitution u=(l-c_i)/s. Returns P list."""
    a = att/ats; s = math.sqrt(1-a*a); c = [a*v for v in ws]; K = len(ws); P = []
    for i in range(K):
        acc = 0.0
        for q in range(NU):
            u = UL+q*HU; wgt = 0.5 if q in (0, NU-1) else 1.0
            pr = phi(u)
            for j in range(K):
                if j != i: pr *= Phi(u + (c[i]-c[j])/s)
            acc += wgt*pr
        P.append(acc*HU)
    return P
def argmax_first(v):
    b = 0
    for i in range(1, len(v)):
        if v[i] > v[b]: b = i
    return b
def dkern(ws, ats, att):
    K = len(ws); zs = argmax_first(ws)
    a = T_op(att, K)/T_op(ats, K)
    return [a*(i == zs)+(1-a)/K for i in range(K)], a, zs

# ---------- W4: sequences ----------
VOCAB = ['Really','Through','Little','Good','Always','Number','Building','Question',
         'Other','Think','Date','Want','Dual','Diffusion','For','Discrete','Data']
TARGET = [VOCAB.index(w) for w in ['Dual','Diffusion','For','Discrete','Data']]
def posterior(zt, x, at, as_, K):
    """Eq. (4), s<t; at=alpha_t, as_=alpha_s."""
    ats = at/as_
    den = K*at*(zt == x) + 1 - at
    return [(K*at*(i == zt)*(i == x) + (ats-at)*(i == zt) + (as_-at)*(i == x) + (1-ats)*(1-as_)/K)/den for i in range(K)]
def run_usdm(Tst, seed, K=len(VOCAB), target=TARGET):
    r = mulberry32(seed); L = len(target)
    z = [min(K-1, int(r()*K)) for _ in range(L)]
    hist = [z[:]]                      # hist[n] = tokens after n steps, n=0..Tst (t = 1 - n/Tst)
    for i in range(Tst, 0, -1):
        at = 1-i/Tst; as_ = 1-(i-1)/Tst
        nz = []
        for l in range(L):
            p = posterior(z[l], target[l], at, as_, K)
            u = r(); c = 0.0; pick = K-1
            for k in range(K):
                c += p[k]
                if u < c: pick = k; break
            nz.append(pick)
        z = nz; hist.append(z[:])
    return hist
def run_ddt(Tst, seed, K=len(VOCAB), target=TARGET):
    r = mulberry32(seed+1000); L = len(target)
    eps = [[gauss(r) for _ in range(K)] for _ in range(L)]
    hist = []
    for n in range(Tst+1):
        a = n/Tst; s = math.sqrt(max(0.0, 1-a*a))   # alpha~ = 1 - t, t = 1 - n/Tst
        hist.append([argmax_first([(a if k == target[l] else 0.0)+s*eps[l][k] for k in range(K)]) for l in range(L)])
    flips = []
    for l in range(L):
        k = target[l]; j = max((q for q in range(K) if q != k), key=lambda q: eps[l][q])
        D = eps[l][j]-eps[l][k]
        flips.append((VOCAB[j], D, D/math.sqrt(1+D*D) if D > 0 else None))
    return hist, flips
def changes(hist):
    """per column: number of n in 1..Tst with hist[n][l] != hist[n-1][l]"""
    return [sum(hist[n][l] != hist[n-1][l] for n in range(1, len(hist))) for l in range(len(hist[0]))]

# ---------- W5 ----------
def w5(K, at, tau, seed):
    r = mulberry32(seed); s = math.sqrt(1-at*at)
    w = [(at if k == 0 else 0.0)+s*gauss(r) for k in range(K)]
    m = max(w); e = [math.exp((v-m)/tau) for v in w]; Z = sum(e); p = [v/Z for v in e]
    jmax = argmax_first(w[1:])+1
    margin = (w[0]-w[jmax])/tau
    H = -sum(q*math.log(q) for q in p if q > 0)
    return dict(argmax=argmax_first(w), pmax=max(p), px=p[0], H=H, margin=margin, gap=w[0]-w[jmax])

# ---------- S8 table ----------
def _bars(m, xth, a, K):
    return [K*a*(j == m)+(1-a) for j in range(K)], [K*a*q+(1-a) for q in xth]
def f49(i, m, xth, a, ad, K):
    xb, xbt = _bars(m, xth, a, K)
    s = sum(xb[j]/xb[i]*math.log(xbt[i]*xb[j]/(xbt[j]*xb[i])) for j in range(K))
    return -ad/(K*a)*(K/xb[i]-K/xbt[i]-s)
def f17(i, m, xth, a, ad, K):
    xb, xbt = _bars(m, xth, a, K)
    kap = (1-a)/(K*a+1-a); eq = (i == m)
    S = sum(math.log(xbt[i]/xbt[j]) for j in range(K))
    val = K/xb[i]-K/xbt[i]-(kap if eq else 1.0)*S
    if not eq: val -= K*a/(1-a)*math.log(xbt[i]/xbt[m])
    val -= ((K-1)*kap if eq else -1/kap)*math.log(kap)
    return -ad/(K*a)*val
