"""Flow-GRPO tutorial toy math. Same algorithm as the page (build/index.html). Pure Python floats."""
import math

def mulberry32(seed):
    a = seed & 0xFFFFFFFF
    def imul(x, y): return (x * y) & 0xFFFFFFFF
    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(a ^ (a >> 15), 1 | a)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return nxt

def gauss(r):
    u1 = r(); u2 = r()
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)

# ---------- toy data distribution (推定) ----------
W = (0.4, 0.6); MU = (-1.2, 1.2); S = (0.3, 0.3)

def comp(t):
    """components of p_t: means (1-t)mu_k, variances (1-t)^2 s_k^2 + t^2"""
    return [((1 - t) * m, (1 - t) ** 2 * s * s + t * t) for m, s in zip(MU, S)]

def density(x, t):
    return sum(w * math.exp(-(x - m) ** 2 / (2 * v)) / math.sqrt(2 * math.pi * v)
               for w, (m, v) in zip(W, comp(t)))

def velocity(x, t):
    """exact v_t(x) = (x - E[x0|xt=x]) / t, t>0. log-sum-exp responsibilities."""
    cs = comp(t)
    logs = [math.log(w) - (x - m) ** 2 / (2 * v) - 0.5 * math.log(v) for w, (m, v) in zip(W, cs)]
    mx = max(logs); ps = [math.exp(l - mx) for l in logs]; z = sum(ps)
    e0 = 0.0
    for p, mu, s, (m, v) in zip(ps, MU, S, cs):
        e0 += p / z * (mu + (1 - t) * s * s / v * (x - m))
    return (x - e0) / t

def sigma(t, a, T):
    """sigma_t = a sqrt(t/(1-t)); at t=1 use t' = 1 - 1/T (推定 guard)"""
    tt = t if t < 1 - 1e-12 else 1 - 1.0 / T
    return a * math.sqrt(tt / (1 - tt))

def simulate(a, T, seed, M=4000, sde=True):
    """returns snapshots[j][m] at t_j = 1 - j/T, j=0..T. a=0 or sde=False -> ODE.
    RNG order: M initial gauss draws, then per step j, per particle m one gauss (SDE only)."""
    r = mulberry32(seed)
    x = [gauss(r) for _ in range(M)]
    snaps = [x[:]]
    dt = -1.0 / T
    for j in range(T):
        t = 1 - j / T
        sg = sigma(t, a, T) if sde else 0.0
        for m in range(M):
            v = velocity(x[m], t)
            drift = v + sg * sg / (2 * t) * (x[m] + (1 - t) * v)
            nz = gauss(r) if sde else 0.0
            x[m] = x[m] + drift * dt + sg * math.sqrt(-dt) * nz
        snaps.append(x[:])
    return snaps

def erf(x):
    # Abramowitz-Stegun 7.1.26, |err| < 1.5e-7
    s = 1 if x >= 0 else -1; x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return s * y

def cdf(x, t):
    return sum(w * 0.5 * (1 + erf((x - m) / math.sqrt(2 * v))) for w, (m, v) in zip(W, comp(t)))

def w1(samples, t, lo=-4.0, hi=4.0, n=801):
    """W1 ~= trapezoid of |F_emp - F| on grid [lo,hi]"""
    xs = sorted(samples); M = len(xs); k = 0; h = (hi - lo) / (n - 1); acc = 0.0; prev = None
    for i in range(n):
        g = lo + i * h
        while k < M and xs[k] <= g: k += 1
        d = abs(k / M - cdf(g, t))
        if prev is not None: acc += 0.5 * (d + prev) * h
        prev = d
    return acc

def stats(xs):
    M = len(xs); m = sum(xs) / M; v = sum((x - m) ** 2 for x in xs) / M
    return m, math.sqrt(v), sum(1 for x in xs if x > 0) / M

def exact_stats(t=0.0):
    cs = comp(t); m = sum(w * mm for w, (mm, v) in zip(W, cs))
    v = sum(w * (vv + mm * mm) for w, (mm, vv) in zip(W, cs)) - m * m
    return m, math.sqrt(v), 1 - cdf(0.0, t)

# ---------- GRPO advantage ----------
def advantages(R):
    """Eq. 4 with population std; std < 1e-12 -> all zero (推定 guard)"""
    G = len(R); m = sum(R) / G; s = math.sqrt(sum((x - m) ** 2 for x in R) / G)
    if s < 1e-12: return [0.0] * G, m, s
    return [(x - m) / s for x in R], m, s

def draw_reward(r, kind, p):
    if kind == 'binary': return 1.0 if r() < p else 0.0
    return 0.5 + 0.15 * gauss(r)   # continuous toy N(0.5, 0.15^2)

def true_z(R, kind, p):
    if kind == 'binary': return (R - p) / math.sqrt(p * (1 - p))
    return (R - 0.5) / 0.15

def adv_mc(G, kind, p, seed, K=2000):
    """RMSE of Â vs true z over K groups, and MC fraction of degenerate groups.
    RNG: one stream mulberry32(seed); groups drawn sequentially."""
    r = mulberry32(seed); se = 0.0; deg = 0
    for _ in range(K):
        R = [draw_reward(r, kind, p) for _ in range(G)]
        A, m, s = advantages(R)
        if s < 1e-12: deg += 1
        se += sum((a - true_z(x, kind, p)) ** 2 for a, x in zip(A, R))
    return math.sqrt(se / (K * G)), deg / K

def p_degenerate(G, kind, p):
    return p ** G + (1 - p) ** G if kind == 'binary' else 0.0

# ---------- clipped surrogate ----------
def clip_obj(rho, A, eps):
    c = min(max(rho, 1 - eps), 1 + eps)
    u, k = rho * A, c * A
    val = min(u, k)
    grad = A if u <= k else 0.0   # d/drho; tie -> unclipped branch
    return val, grad

# ---------- Gaussian step policy & KL ----------
def step_policy(j, T, a, dv, z):
    """step from t=1-j/T to t-1/T. Returns sigma, step std, mean shift, KL(paper form1), KL(form2), log ratio at x=m'+std*z"""
    t = 1 - j / T; adt = 1.0 / T
    sg = sigma(t, a, T)
    sd = sg * math.sqrt(adt)
    shift = (1 + sg * sg * (1 - t) / (2 * t)) * dv * (-adt)  # signed m_theta - m', dt = -1/T
    kl1 = shift ** 2 / (2 * sg * sg * adt)
    kl2 = adt / 2 * (sg * (1 - t) / (2 * t) + 1 / sg) ** 2 * dv * dv
    # x = m' + sd*z ; m_theta = m' + shift
    logr = ((sd * z) ** 2 - (sd * z - shift) ** 2) / (2 * sd * sd)
    return sg, sd, shift, kl1, kl2, logr

def kl_coeff(j, T, a):
    """closed form c_j = (1/(2T)) (sigma(1-t)/(2t) + 1/sigma)^2; sigma uses guard t' at t=1, (1-t)/(2t) uses true t"""
    t = 1 - j / T; sg = sigma(t, a, T)
    return (sg * (1 - t) / (2 * t) + 1 / sg) ** 2 / (2 * T)
