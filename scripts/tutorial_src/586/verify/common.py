"""Shared math for the 586 verify scripts. Mirrors the page JS exactly (stdlib only)."""
import math

SQRT1_2 = math.sqrt(0.5)


def erf_nr(x):
    # Numerical Recipes erfc Chebyshev fit (fractional error < 1.2e-7); page uses the same.
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return 1 - r if x >= 0 else r - 1


def Phi(x):
    return 0.5 * (1 + erf_nr(x * SQRT1_2))


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def gelu(x):
    return x * Phi(x)


def dgelu(x):
    return Phi(x) + x * phi(x)


def matvec(W, x):
    return [sum(w * v for w, v in zip(row, x)) for row in W]


def norm(v):
    return math.sqrt(sum(a * a for a in v))


def forward(W, b, V, h):
    z = [zi + bi for zi, bi in zip(matvec(W, h), b)]
    a = [gelu(v) for v in z]
    return z, matvec(V, a)


def jac_cols(V, z):
    # J_i = V[:, i] * GELU'(z_i)
    return [[V[r][i] * dgelu(z[i]) for r in range(len(V))] for i in range(len(z))]


def mulberry32(seed):
    s = [seed & 0xFFFFFFFF]

    def r():
        s[0] = (s[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = s[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r


if __name__ == "__main__":
    worst = max(abs(erf_nr(x / 100) - math.erf(x / 100)) for x in range(-600, 601))
    assert worst < 2e-7, worst
    print("erf_nr max abs err on [-6,6]:", worst)
