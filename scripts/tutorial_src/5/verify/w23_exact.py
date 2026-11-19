"""W2/W3 acceptance values: exact enumeration on {0,1}^6 under binary uniform diffusion.

Same algorithm as the page (design.md 4.2/4.3). Natural log (nats).
Run: python3 -I w23_exact.py
"""
import math
import numpy as np

D = 6
N = 1 << D
LN2 = math.log(2)
POP = np.array([bin(x).count("1") for x in range(N)])
HAM = np.array([[POP[x ^ y] for y in range(N)] for x in range(N)])  # Hamming distance table
BIT = np.array([[(x >> i) & 1 for i in range(D)] for x in range(N)])  # BIT[x, i]


def preset(name):
    q = np.zeros(N)
    if name == "copy":
        q[0] = q[N - 1] = 0.5
    elif name == "parity":
        q[POP % 2 == 0] = 1.0
    elif name == "codebook":
        for c in (5, 10, 19, 28, 38, 45, 51, 60):
            q[c] = 1.0
    elif name == "indep":  # each bit = 1 w.p. 0.8, independent
        q = 0.8 ** POP * 0.2 ** (D - POP)
    return q / q.sum()


def kernel(t):
    """(K_t^U)^{⊗d}(y|x) for S=2: per coordinate stay a+b, flip b; a=e^-t, b=(1-a)/2."""
    a = math.exp(-t)
    b = (1 - a) / 2
    return (a + b) ** (D - HAM) * b ** HAM  # symmetric matrix


def H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def dtc(p):
    """DTC(p) = H(X) - sum_i H(X_i | X_-i), H(X_i|X_-i) = H(X) - H(X_-i)."""
    hx = H(p)
    s = 0.0
    for i in range(D):
        m = 1 << i
        lo = np.array([x for x in range(N) if not x & m])
        marg = p[lo] + p[lo | m]  # law of X_-i
        s += hx - H(marg)
    return hx - s


def tc(p):
    s = 0.0
    for i in range(D):
        p1 = p[BIT[:, i] == 1].sum()
        s += H(np.array([p1, 1 - p1]))
    return s - H(p)


def tc_cond(q, s, h):
    """TC(X_s | X_t), t = s+h, exact. J[x_s, y] = q_s(x_s) K_h(y|x_s)."""
    qs = q @ kernel(s)
    J = qs[:, None] * kernel(h)
    qt = J.sum(0)
    h_s_given_t = H(J.ravel()) - H(qt)
    tot = 0.0
    for i in range(D):
        j1 = J[BIT[:, i] == 1].sum(0)  # P((X_s)_i=1, X_t=y)
        j0 = qt - j1
        tot += H(np.concatenate([j0, j1])) - H(qt)
    return tot - h_s_given_t, qs, qt


def c_s(s, h):
    return (math.exp(2 * h) - 1) / (1 - math.exp(-s))


def schedule(a, delta=0.001, T=6.0):
    u0, U = math.exp(delta) - 1, math.exp(T) - 1
    us = [u0]
    while us[-1] < U:
        us.append(min((1 + a) * us[-1], U))
    return [math.log1p(u) for u in us]


def run_schedule(q, a, delta=0.001, T=6.0):
    ts = schedule(a, delta, T)
    n = len(ts) - 1
    sum_tc = sum_bound = 0.0
    max_c = 0.0
    # exact sampler: start Unif at t_N, product reverse kernels with exact posterior marginals, final forward kernel
    p = np.full(N, 1.0 / N)
    for j in range(n - 1, -1, -1):
        s, t = ts[j], ts[j + 1]
        h = t - s
        tcv, qs, qt = tc_cond(q, s, h)
        c = c_s(s, h)
        max_c = max(max_c, c)
        sum_tc += tcv
        sum_bound += c * (dtc(qs) - dtc(qt))
        # posterior marginals m[y, i] = P((X_s)_i = 1 | X_t = y)
        J = qs[:, None] * kernel(h)
        m = np.stack([J[BIT[:, i] == 1].sum(0) for i in range(D)], 1) / qt[:, None]
        R = np.prod(np.where(BIT[None, :, :] == 1, m[:, None, :], 1 - m[:, None, :]), axis=2)  # R[y, x]
        p = p @ R
    qhat = p @ kernel(ts[0])
    qT = q @ kernel(ts[-1])
    init = float((qT * np.log(qT * N)).sum())
    end = H(q @ kernel(ts[0])) - H(q)
    mask = q > 0
    kl = float((q[mask] * np.log(q[mask] / qhat[mask])).sum())
    return dict(N=n, max_c=max_c, sum_tc=sum_tc, sum_bound=sum_bound, three_a_dtc=3 * a * dtc(q),
                init=init, init_bound=math.exp(-ts[-1]) * D * LN2, end=end, path=init + sum_tc + end, kl=kl)


if __name__ == "__main__":
    names = ["copy", "parity", "codebook", "indep"]
    print("== preset TC / DTC (nats); /ln2 in brackets")
    for nm in names:
        q = preset(nm)
        print(f"{nm:9s} H={H(q):.4f} TC={tc(q):.4f} [{tc(q)/LN2:.3f}] DTC={dtc(q):.4f} [{dtc(q)/LN2:.3f}]")
    print("\n== W2 one step: TC(X_s|X_t) vs c_s(h)*(DTC(q_s)-DTC(q_t))")
    for nm in names:
        q = preset(nm)
        for s, h in [(0.5, 0.2), (0.1, 0.05), (1.0, 0.5), (0.05, 0.5), (2.0, 1.0)]:
            tcv, qs, qt = tc_cond(q, s, h)
            dd = dtc(qs) - dtc(qt)
            print(f"{nm:9s} s={s:<4} h={h:<4} c={c_s(s,h):9.4f} TC={tcv:.6f} dDTC={dd:.6f} "
                  f"bound={c_s(s,h)*dd:.6f} DTC(qs)={dtc(qs):.4f} DTC(qt)={dtc(qt):.4f} ratio={(tcv/(c_s(s,h)*dd) if abs(dd)>1e-12 else float("nan")):.4f}")
    print("\n== W3 schedule (delta=0.001, T=6)")
    for nm in names:
        q = preset(nm)
        for a in (1.0, 0.3, 0.1, 0.03):
            r = run_schedule(q, a)
            print(f"{nm:9s} a={a:<4} N={r['N']:4d} maxc={r['max_c']:.4f} sumTC={r['sum_tc']:.6f} "
                  f"sumBound={r['sum_bound']:.6f} 3aDTC={r['three_a_dtc']:.6f} init={r['init']:.2e}"
                  f"(<= {r['init_bound']:.2e}) end={r['end']:.6f} path={r['path']:.6f} KL={r['kl']:.6f}")
