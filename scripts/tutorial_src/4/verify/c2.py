"""C2: two-time denoiser on the simplex and semigroup identity (Prop. 3.3), default state."""
import math
from toy import X, D, delta, gamma, pNY, to_coords, flowU

def fmt(v): return '[' + ', '.join(f'{a:.4f}' for a in v) + ']'

def eq82(x, s, t, n=4000):
    # q of delta via (1-s)(1-t)/(t-s) * int_s^t p_tau / (1-tau)^2, Simpson, trajectory X_{s,tau}(x)
    h = (t - s) / n; acc = 0.0
    for i in range(n + 1):
        tau = s + i * h
        xt = X(x, s, tau) if i else x
        f = pNY(xt, tau) / (1 - tau) ** 2
        acc += f * (1 if i in (0, n) else (4 if i % 2 else 2))
    return (1 - s) * (1 - t) / (t - s) * acc * h / 3

def report(x0, s, u, t):
    x = X(x0, 0, s)
    Xsu, Xst = X(x, s, u), X(x, s, t)
    dsu, dst = delta(x, s, u), delta(x, s, t)
    dut = delta(Xsu, u, t)
    g = gamma(s, u, t)
    teacher = [g * dsu[i] + (1 - g) * dut[i] for i in range(4)]
    print(f's={s} u={u} t={t}  gamma={g:.4f}')
    print(' x_s        ', fmt(x), ' u-coord', round(to_coords(x)[0], 4))
    print(' D_s(x)     ', fmt(D(x, s)))
    print(' X_su       ', fmt(Xsu), ' sums', round(Xsu[0]+Xsu[1],4), round(Xsu[2]+Xsu[3],4))
    print(' X_st       ', fmt(Xst), ' sums', round(Xst[0]+Xst[1],4), round(Xst[2]+Xst[3],4))
    print(' delta_su   ', fmt(dsu))
    print(' delta_ut(Xsu)', fmt(dut))
    print(' delta_st   ', fmt(dst), ' sums', round(dst[0]+dst[1],6))
    print(' teacher    ', fmt(teacher), ' max|delta_st-teacher|', f'{max(abs(dst[i]-teacher[i]) for i in range(4)):.2e}')
    if t < 1:
        q = eq82(x, s, t)
        print(' Eq82 quad q', f'{q:.6f}', ' vs inversion', f'{dst[0]:.6f}', ' diff', f'{abs(q-dst[0]):.1e}')
    # Eq 19 recover
    rec = [(1-t)/(1-s)*x[i] + (t-s)/(1-s)*dst[i] for i in range(4)]
    print(' Eq19 max|X-rec|', f'{max(abs(rec[i]-Xst[i]) for i in range(4)):.1e}')

x0 = [0.9, -0.2, -0.4, 0.3]
print('x0', x0, 'u0', to_coords(x0)[0])
for (s, u, t) in [(0.1, 0.5, 0.9), (0.0, 0.5, 1.0), (0.3, 0.6, 0.95), (0.0, 0.25, 0.5)]:
    report(x0, s, u, t); print()
# endpoint
print('X_{0,1}(x0)', [round(v, 6) for v in X(x0, 0, 1)])
