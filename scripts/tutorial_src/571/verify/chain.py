# C2: synchronous tabular TD backups on an H-state chain with reward 1 on leaving s_H, vs return-to-go.
# Page algorithm: V^0=0; V_i <- (1-a)V_i + a(r_i + g V_{i+1}), V_{H+1}=0, all i updated from the old vector.
from math import comb
def run(H, g, a, n):
    V = [0.0] * (H + 2)            # 1-based, V[H+1]=0 terminal
    for _ in range(n):
        old = V[:]
        for i in range(1, H + 1):
            r = 1.0 if i == H else 0.0
            V[i] = (1 - a) * old[i] + a * (r + g * old[i + 1])
    return V[1:H + 1]
def closed(H, g, a, n, i):          # 〔本頁推導〕 V_i^n = g^(H-i) P(Bin(n,a) >= H-i+1)
    m = H - i + 1
    p = sum(comb(n, j) * a**j * (1 - a)**(n - j) for j in range(m, n + 1))
    return g**(H - i) * p
def first_nonzero(H, a):            # iterations until V_1>0: H (any a>0)
    return H
cases = [(30, 0.99, 1.0, 0), (30, 0.99, 1.0, 10), (30, 0.99, 1.0, 29), (30, 0.99, 1.0, 30), (30, 0.99, 1.0, 60),
         (30, 0.99, 0.5, 30), (30, 0.99, 0.5, 60), (30, 0.99, 0.5, 100), (100, 0.99, 1.0, 100), (100, 0.9, 1.0, 100),
         (30, 1.0, 0.1, 200), (50, 0.95, 0.3, 200)]
maxerr = 0
for H, g, a, n in cases:
    V = run(H, g, a, n)
    for i in range(1, H + 1):
        maxerr = max(maxerr, abs(V[i - 1] - closed(H, g, a, n, i)))
    front = sum(1 for v in V if v > 0)
    print(f"H={H} g={g} a={a} n={n}: V1={V[0]:.6f} V_mid(i={H//2})={V[H//2-1]:.6f} VH={V[-1]:.6f} "
          f"nonzero states={front} g^(H-1)={g**(H-1):.6f}")
print('max |iter - closed form| =', maxerr)
# n needed for V1 >= 0.5 * g^(H-1) (median of Bin) for a in {1,0.5,0.1}, H=30
for a in (1.0, 0.5, 0.25, 0.1):
    n = 0
    while closed(30, 1.0, a, n, 1) < 0.5: n += 1
    print(f"H=30 a={a}: smallest n with P(Bin(n,a)>=30)>=0.5 -> {n}")
print('reached count = #{i: V_i >= 0.5*g^(H-i)} (page readout)')
for H, g, a, n in [(30,0.99,1.0,10),(30,0.99,1.0,30),(30,0.99,0.5,30),(30,0.99,0.5,60),(30,0.99,0.1,100),(30,0.99,0.1,297),(100,0.99,1.0,50)]:
    V = run(H, g, a, n)
    c = sum(1 for i in range(1, H+1) if V[i-1] >= 0.5 * g**(H-i) * (1 - 1e-9))
    print(f"H={H} g={g} a={a} n={n}: reached={c} V1={V[0]:.6g}")
# Reduced-motion jump target: smallest n<=400 with reached==H, else 400 (fallback).
def jump_target(H, g, a, NMAX=400):
    for n in range(NMAX + 1):
        V = run(H, g, a, n)
        if sum(1 for i in range(1, H+1) if V[i-1] >= 0.5 * g**(H-i) * (1 - 1e-9)) == H: return n, True
    return NMAX, False
print('reduced-motion jump target (n, all reached?)')
for H, a in [(30,1.0),(30,0.5),(30,0.05),(100,1.0),(100,0.5),(100,0.25),(100,0.05)]:
    n, ok = jump_target(H, 0.99, a)
    V = run(H, 0.99, a, n)
    c = sum(1 for i in range(1, H+1) if V[i-1] >= 0.5 * 0.99**(H-i) * (1 - 1e-9))
    print(f"H={H} a={a}: n={n} all={ok} reached={c}")
print('normalized view V_i/g^(H-i) at H=100,g=0.9,a=1,n=50: i=51 ->', run(100,0.9,1.0,50)[50]/0.9**49, ' i=50 ->', run(100,0.9,1.0,50)[49]/0.9**50)
