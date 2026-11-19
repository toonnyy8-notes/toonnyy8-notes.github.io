"""Shared math for the IDQL page (same algorithms as the page JS). Run directly for self-check."""
import math

# Toy bandit (推定): 1-D action a, Q(a) = a; behavior mu = 3-component Gaussian mixture.
M_K = (1.0, 2.0, 3.0)          # low / medium / high reward cluster centres
S = 0.15                       # common std
PI_DEFAULT = (0.3, 0.5, 0.2)

def pis(p_high):
    """weights with high-cluster weight p_high, other two keep ratio 3:5."""
    r = 1.0 - p_high
    return (r * 3 / 8, r * 5 / 8, p_high)

def erfc(x):
    # Numerical Recipes erfcc, fractional error < 1.2e-7
    z = abs(x); t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(-z*z - 1.26551223 + t*(1.00002368 + t*(0.37409196 + t*(0.09678418 +
        t*(-0.18628806 + t*(0.27886807 + t*(-1.13520398 + t*(1.48851587 +
        t*(-0.82215223 + t*0.17087277)))))))))
    return r if x >= 0 else 2.0 - r

def Phi(z): return 0.5 * erfc(-z / math.sqrt(2))
def phi(z): return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

def mix_pdf(x, pi, m=M_K, s=S):
    return sum(p * phi((x - mk) / s) / s for p, mk in zip(pi, m))

def mix_cdf(x, pi, m=M_K, s=S):
    return sum(p * Phi((x - mk) / s) for p, mk in zip(pi, m))

def upper_pm(v, pi, m=M_K, s=S):
    """E[(Q-v)+] under the mixture."""
    tot = 0.0
    for p, mk in zip(pi, m):
        z = (v - mk) / s
        tot += p * (s * phi(z) + (mk - v) * (1 - Phi(z)))
    return tot

def mean_mu(pi, m=M_K): return sum(p * mk for p, mk in zip(pi, m))

def bisect(g, lo, hi, it=200):
    """g decreasing; returns root."""
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0: lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

LO, HI = M_K[0] - 10 * S, M_K[-1] + 10 * S

def v_expectile(tau, pi):
    # first-order condition: tau E[(Q-v)+] - (1-tau) E[(v-Q)+] = 0 ; E[(v-Q)+] = E[(Q-v)+] + v - mean
    mu = mean_mu(pi)
    return bisect(lambda v: tau * upper_pm(v, pi) - (1 - tau) * (upper_pm(v, pi) + v - mu), LO, HI)

def v_quantile(tau, pi):
    return bisect(lambda v: tau - mix_cdf(v, pi), LO, HI)

def v_exp(alpha, pi, m=M_K, s=S):
    # Eq. 11 for a Gaussian mixture: (1/alpha) log sum pi_k exp(alpha m_k + alpha^2 s^2/2)  (log-sum-exp)
    xs = [math.log(p) + alpha * mk + 0.5 * alpha * alpha * s * s for p, mk in zip(pi, m) if p > 0]
    mx = max(xs)
    return (mx + math.log(sum(math.exp(x - mx) for x in xs))) / alpha

# grid for densities / moments
G_N = 2001
def grid():
    h = (HI - LO) / (G_N - 1)
    return [LO + i * h for i in range(G_N)], h

def w_expectile(q, v, tau): return tau if q >= v else 1 - tau          # Eq. 9 (factor 2 dropped)
def w_exp(q, v, alpha):                                                 # Eq. 12
    u = q - v
    if abs(alpha * u) < 1e-12: return alpha * alpha
    return alpha * abs(math.expm1(alpha * u)) / abs(u)

def moments(dens, xs, h):
    """trapezoid: mass, mean, std of an unnormalized density on grid."""
    wts = [h] * len(xs); wts[0] = wts[-1] = h / 2
    z = sum(w * d for w, d in zip(wts, dens))
    m1 = sum(w * d * x for w, d, x in zip(wts, dens, xs)) / z
    m2 = sum(w * d * (x - m1) ** 2 for w, d, x in zip(wts, dens, xs)) / z
    return z, m1, math.sqrt(m2)

def awr_gauss(alpha, pi, m=M_K, s=S):
    """Gaussian weighted-MLE fit to mu * exp(alpha Q): tilted mixture, closed form."""
    lw = [math.log(p) + alpha * mk + 0.5 * alpha * alpha * s * s for p, mk in zip(pi, m)]
    mx = max(lw); w = [math.exp(x - mx) for x in lw]; z = sum(w); w = [x / z for x in w]
    mus = [mk + alpha * s * s for mk in m]
    mean = sum(a * b for a, b in zip(w, mus))
    var = s * s + sum(a * (b - mean) ** 2 for a, b in zip(w, mus))
    return mean, math.sqrt(var), w, mus

def mulberry32(seed):
    st = [seed & 0xFFFFFFFF]
    def rnd():
        st[0] = (st[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = st[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd

def gauss(rnd):
    u1 = rnd(); u2 = rnd()
    return math.sqrt(-2 * math.log(1 - u1)) * math.cos(2 * math.pi * u2)

def sample_mu(rnd, pi, m=M_K, s=S):
    u = rnd(); c = 0.0; k = len(pi) - 1
    for i, p in enumerate(pi):
        c += p
        if u < c: k = i; break
    return m[k] + s * gauss(rnd)

if __name__ == '__main__':
    for x in (-3, -1, -0.2, 0, 0.5, 2, 4):
        assert abs(erfc(x) - math.erfc(x)) < 2e-7 * max(1, math.erfc(x)), x
    pi = PI_DEFAULT
    v = v_expectile(0.5, pi); assert abs(v - mean_mu(pi)) < 1e-9
    v = v_quantile(0.5, pi); assert abs(mix_cdf(v, pi) - 0.5) < 1e-9
    r = mulberry32(1); a, b = r(), r()
    assert abs(a - 0.6270739405881613) < 1e-15 and abs(b - 0.002735721180215478) < 1e-15
    print('common ok')
