"""C2: three importance scores on a 4x4 toy layer, masks, and pruning-style output shift.
Same algorithm the page will run. Usage: python3 c2_scores.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import forward, jac_cols, norm, dgelu

H = [1.0, -0.2, 1.5, 0.6]                     # input activation h^(l-1)
W = [[0.8, -1.5, 0.3, 0.6],
     [-0.4, 2.0, -0.9, 0.2],
     [1.2, 0.5, -0.3, -1.0],
     [0.1, -0.7, 0.6, 1.4]]
V = [[1.0, -0.5, 0.8, 0.3],
     [0.4, 1.2, -0.6, 0.9],
     [-0.7, 0.3, 1.1, -0.4]]
N = 16
EPS = 1e-9   # sort key = round(score / EPS); ties broken by index


def scores(b, h=H):
    z, _ = forward(W, b, V, h)
    Jn = [norm(c) for c in jac_cols(V, z)]
    mag = [abs(W[i][j]) for i in range(4) for j in range(4)]
    wanda = [abs(W[i][j]) * abs(h[j]) for i in range(4) for j in range(4)]
    full = [Jn[i] * abs(W[i][j]) * abs(h[j]) for i in range(4) for j in range(4)]
    return z, Jn, {"mag": mag, "wanda": wanda, "full": full}


def mask_lowest(s, m):
    # M=1 (updated) for the m lowest scores; stable tie-break by index (row-major)
    order = sorted(range(N), key=lambda k: (round(s[k] / EPS), k))
    return set(order[:m])


def shift_if_zeroed(b, upd, h=H):
    _, f0 = forward(W, b, V, h)
    W2 = [row[:] for row in W]
    for k in upd:
        W2[k // 4][k % 4] = 0.0
    _, f1 = forward(W2, b, V, h)
    return norm([p - q for p, q in zip(f1, f0)])


def random_mean_by_m(b, h=H):
    acc = [0.0] * (N + 1)
    cnt = [0] * (N + 1)
    for bits in range(1 << N):
        upd = [k for k in range(N) if bits >> k & 1]
        acc[len(upd)] += shift_if_zeroed(b, upd, h)
        cnt[len(upd)] += 1
    return [a / c for a, c in zip(acc, cnt)]


def random_mean_gray(b, h=H):
    """Page algorithm: Gray-code enumeration, toggling one weight per step and updating z incrementally."""
    z0, f0 = forward(W, b, V, h)
    from common import gelu, matvec
    z = z0[:]
    acc = [0.0] * (N + 1); cnt = [0] * (N + 1); size = 0
    cnt[0] = 1
    on = [False] * N
    for step in range(1, 1 << N):
        k = (step & -step).bit_length() - 1          # bit flipped between gray(step-1) and gray(step)
        i, j = divmod(k, 4)
        on[k] = not on[k]
        z[i] += (-W[i][j] if on[k] else W[i][j]) * h[j]
        size += 1 if on[k] else -1
        f1 = matvec(V, [gelu(v) for v in z])
        acc[size] += norm([p - q for p, q in zip(f1, f0)]); cnt[size] += 1
    return [a / c for a, c in zip(acc, cnt)]


def ranks(s):
    # average ranks for ties (Spearman)
    order = sorted(range(N), key=lambda k: s[k])
    r = [0.0] * N
    i = 0
    while i < N:
        j = i
        while j + 1 < N and s[order[j + 1]] == s[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / N, sum(rb) / N
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den > 0 else float('nan')


def report(b3, ms=(2, 4, 8, 15)):
    b = [0.0, 0.0, b3, 0.0]
    z, Jn, S = scores(b)
    print(f"\n### b3 = {b3}")
    print("z =", [round(v, 4) for v in z], " GELU'(z) =", [round(dgelu(v), 4) for v in z])
    print("||J_i|| =", [round(v, 4) for v in Jn])
    for name in ("mag", "wanda", "full"):
        print(f"{name:5s}", [round(v, 3) for v in S[name]])
    print("Spearman(full, mag) =", round(spearman(S["full"], S["mag"]), 3),
          " Spearman(full, wanda) =", round(spearman(S["full"], S["wanda"]), 3))
    rnd = random_mean_by_m(b)
    g = random_mean_gray(b)
    assert max(abs(x - y) for x, y in zip(rnd, g)) < 1e-9
    for m in ms:
        row = []
        for name in ("mag", "wanda", "full"):
            upd = mask_lowest(S[name], m)
            fo = sum(S["full"][k] for k in upd)
            row.append(f"{name}: upd={sorted(k+1 for k in upd)} real={shift_if_zeroed(b, upd):.4f} sumS={fo:.4f}")
        print(f"m={m} (rho={m/16:.4f}) | " + " | ".join(row) + f" | random mean real={rnd[m]:.4f}")


if __name__ == "__main__":
    for b3 in (0.0, -0.75, -2.0, -4.0):
        report(b3)
