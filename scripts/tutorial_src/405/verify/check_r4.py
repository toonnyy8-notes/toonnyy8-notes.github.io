"""Round-1 review checks (R4, item 5/7/13): run `python3 -I check_r4.py` from this dir."""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check import b, ab, T, c1_steps, c1_ratio_path

def slopes(t, g, s):
    a_ = ab(t); s2 = s * s; den = a_ * s2 + (1 - a_) * (1 + g * g)
    c = math.sqrt(1 - a_) / (a_ * s2 + 1 - a_)
    cip = math.sqrt(1 - a_) / den                       # target eps
    cy = math.sqrt(1 - a_) * math.sqrt(1 + g * g) / den # target eps' = (eps+g xi)/sqrt(1+g^2)
    return c, cip, cy

def coefs(t):
    be, a_ = b(t), ab(t); ap = ab(t - 1) if t > 1 else 1.0
    return math.sqrt(ap) * be / (1 - a_), math.sqrt(1 - be) * (1 - ap) / (1 - a_), (1 - ap) / (1 - a_) * be

def fd_additive(s, nu, t0):
    """zero-mean independent x0-prediction error N(0,nu^2) at every call; Tab.1 protocol."""
    s2 = s * s; a_ = ab(t0); v = a_ * s2 + 1 - a_
    for t in range(t0, 0, -1):
        c1, c2, bt = coefs(t); at = ab(t); g = math.sqrt(at) * s2 / (at * s2 + 1 - at)
        A = c1 * g + c2; v = A * A * v + c1 * c1 * nu * nu + (bt if t > 1 else 0.0)
    return (math.sqrt(v) - s) ** 2

def fd_full_gen(s, gamma):
    """generate from x_T with the Bayes network trained clean (gamma=0) or with IP (gamma)."""
    s2 = s * s; a_ = ab(T); v = a_ * s2 + 1 - a_
    for t in range(T, 0, -1):
        c1, c2, bt = coefs(t); at = ab(t)
        sb = s2 + gamma * gamma * (1 - at) / at       # IP == clean DDPM on x0 + gamma*sqrt((1-ab)/ab) xi
        g = math.sqrt(at) * sb / (at * sb + 1 - at)
        A = c1 * g + c2; v = A * A * v + (bt if t > 1 else 0.0)
    return (math.sqrt(v) - s) ** 2, math.sqrt(v)

if __name__ == '__main__':
    for t in (100, 500, 900):
        c, cip, cy = slopes(t, 0.1, 0.5)
        print('slopes g=0.1 s=0.5 t=%d  IP/c=%.5f  DDPM-y/c=%.5f' % (t, cip / c, cy / c))
    for nu in (0.0, 0.05, 0.1, 0.2):
        print('additive nu=%.2f FD' % nu, ' '.join('%d:%.3e' % (t, fd_additive(0.5, nu, t)) for t in (10, 100, 300, 1000)))
    for g in (0.0, 0.1):
        fd, sd = fd_full_gen(0.5, g); print('full generation, Bayes net trained with gamma=%.1f: FD=%.3e std=%.5f' % (g, fd, sd))
    # C1 axis sweep over the full slider grid (item 14/13)
    lo, hi = 9, 0
    for si in range(2, 21):
        s = si * 0.05
        for di in range(-10, 11):
            d = di * 0.005
            for t0 in (1, 2, 10, 50, 100, 200, 300, 500, 700, 1000):
                r = c1_ratio_path(s, d, t0); lo = min(lo, min(r.values())); hi = max(hi, max(r.values()))
    print('C1 ratio range over slider grid (t0 sampled): %.4f .. %.4f' % (lo, hi))
    for d in (-0.05, 0.05):
        r = c1_ratio_path(1.0, d, 1000); print('worst corner s=1 delta=%+.2f t0=1000 r0=%.4f' % (d, r[0]))
