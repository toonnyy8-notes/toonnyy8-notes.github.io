"""C1: Theorem 3.1 on f = V * GELU(W x). Prints the acceptance table."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import forward, jac_cols, norm, dgelu

W = [[1.0, -2.0], [0.5, 3.0]]
x = [1.0, 0.5]
V = [[1.0, 2.0], [0.0, -1.0], [2.0, 0.0]]
b = [0.0, 0.0]

z, f0 = forward(W, b, V, x)
J = jac_cols(V, z)
print("z =", z, "GELU'(z) =", [round(dgelu(v), 5) for v in z])
print("||J_i|| =", [round(norm(c), 5) for c in J])
S = [[norm(J[i]) * abs(W[i][j]) * abs(x[j]) for j in range(2)] for i in range(2)]
print("S =", [[round(s, 5) for s in row] for row in S])


def true_shift(i, j, dw):
    W2 = [row[:] for row in W]
    W2[i][j] += dw
    _, f1 = forward(W2, b, V, x)
    return norm([p - q for p, q in zip(f1, f0)])


def pred(i, j, dw):
    return norm(J[i]) * abs(dw) * abs(x[j])


print("| (i,j) | ΔW | 一階預測 | 真實 ||Δf|| | 相對誤差 |")
for i in range(2):
    for j in range(2):
        for dw in dict.fromkeys((0.01, 0.1, 0.5, 1.0, -1.0, -W[i][j], 2.0, -2.0)):  # de-duplicated
            p, t = pred(i, j, dw), true_shift(i, j, dw)
            rel = (p - t) / t if t > 1e-12 else float('nan')
            print(f"| ({i+1},{j+1}) | {dw:+.2f} | {p:.5f} | {t:.5f} | {rel:+.3f} |")

# self-check against advisor numbers
assert abs(true_shift(0, 0, 0.01) - 0.01127) < 5e-6
assert abs(true_shift(0, 0, -1.0) - 0.35476) < 5e-6
assert abs(S[1][1] - 3.63998) < 5e-6

# axis max per chip: max over 601 points of true and predicted, x1.08 (page spec)
for i in range(2):
    for j in range(2):
        pts = [k / 100 for k in range(-300, 301)]
        mx = max(max(true_shift(i, j, d), pred(i, j, d)) for d in pts)
        edge = (pred(i, j, -3.0), pred(i, j, 3.0), true_shift(i, j, -3.0), true_shift(i, j, 3.0))
        print(f"axis ({i+1},{j+1}): ymax={mx*1.08:.4f}  pred(-3,+3)={edge[0]:.4f},{edge[1]:.4f}  true(-3,+3)={edge[2]:.4f},{edge[3]:.4f}")

# S2a homogeneity example: GELU(2) vs 2*GELU(1)
from common import gelu
print(f"GELU(2)={gelu(2.0):.4f}  2*GELU(1)={2*gelu(1.0):.4f}")
