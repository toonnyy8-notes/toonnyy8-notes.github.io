"""C3: Hutchinson (Eq. 4) and the L1 surrogate bounds (Thm D.1). Exact enumeration + seeded MC.
Mirrors page algorithm: xi_k = (r() < 0.5 ? -1 : +1), consumed sample by sample, k = 0..d-1."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import mulberry32

PRESETS = {
    "single (3)": [3.0],
    "(1,1)": [1.0, 1.0],
    "(3,4)": [3.0, 4.0],
    "ones x8": [1.0] * 8,
    "one big (3, 0.5 x7)": [3.0] + [0.5] * 7,
    "(1.05)": [1.05],
}


def exact(a):
    d = len(a)
    s2 = s1 = 0.0
    for bits in range(1 << d):
        x = sum((-1 if bits >> k & 1 else 1) * a[k] for k in range(d))
        s2 += x * x
        s1 += abs(x)
    n = 1 << d
    na = math.sqrt(sum(v * v for v in a))
    var = 2 * (na ** 4 - sum(v ** 4 for v in a))
    return s2 / n, s1 / n, na, var


def mc(a, R, seed):
    r = mulberry32(seed)
    xs = []
    for _ in range(R):
        x = 0.0
        for v in a:
            x += (-1 if r() < 0.5 else 1) * v
        xs.append(x)
    l2 = math.sqrt(sum(x * x for x in xs) / R)
    l1 = sum(abs(x) for x in xs) / R
    return l2, l1


def repeat(a, R, runs=200):
    l2s, l1s = [], []
    for s in range(1, runs + 1):
        l2, l1 = mc(a, R, s)
        l2s.append(l2)
        l1s.append(l1)
    m = lambda v: sum(v) / len(v)
    sd = lambda v: math.sqrt(sum((x - m(v)) ** 2 for x in v) / (len(v) - 1))
    return m(l2s), sd(l2s), m(l1s), sd(l1s)


if __name__ == "__main__":
    print("| 向量 a | E[X²] | ‖a‖² | E|X| | ‖a‖ | E|X|/‖a‖ | Var[X²] |")
    for name, a in PRESETS.items():
        e2, e1, na, var = exact(a)
        assert abs(e2 - na * na) < 1e-9
        assert math.sqrt(0.5) * na - 1e-12 <= e1 <= na + 1e-12
        print(f"| {name} | {e2:.4f} | {na*na:.4f} | {e1:.4f} | {na:.4f} | {e1/na:.4f} | {var:.4f} |")
    print("\nMC, seed=1:")
    for name in ("(3,4)", "ones x8", "one big (3, 0.5 x7)"):
        a = PRESETS[name]
        for R in (1, 2, 8, 64):
            l2, l1 = mc(a, R, 1)
            print(f"| {name} | R={R} | sqrt(mean X²)={l2:.4f} | mean|X|={l1:.4f} |")
    print("\n200 repeats (seeds 1..200): mean±sd of sqrt(mean X²), mean|X|")
    for name in ("(3,4)", "ones x8"):
        a = PRESETS[name]
        e2, e1, na, _ = exact(a)
        for R in (1, 8, 64):
            m2, s2, m1, s1 = repeat(a, R)
            print(f"| {name} | R={R} | L2 {m2:.4f}±{s2:.4f} (‖a‖={na:.4f}) | L1 {m1:.4f}±{s1:.4f} (E|X|={e1:.4f}) |")
