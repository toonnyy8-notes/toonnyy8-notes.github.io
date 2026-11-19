import math, time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import *
out = []
P = lambda *a: out.append(' '.join(str(x) for x in a))
f4 = lambda x: f"{x:.4f}"

r = mulberry32(1); P("PRNG seed1 first3", [f"{r():.6f}" for _ in range(3)])
r = mulberry32(1); P("gauss seed1 first2", [f4(gauss(r)) for _ in range(2)])
P("velocity(0.5,0.5)", f4(velocity(0.5, 0.5)), "velocity(-1,1)", f4(velocity(-1.0, 1.0)), "velocity(1.0,0.1)", f4(velocity(1.0, 0.1)))
P("density(0,0)", f4(density(0, 0)), "density(1.2,0)", f4(density(1.2, 0)), "density(0,1)", f4(density(0, 1)))
P("exact t=0 mean,std,P(x>0)", [f4(v) for v in exact_stats(0)])
P("sigma t=0.5 a=0.7", f4(sigma(0.5, .7, 10)), "t=1 T=10", f4(sigma(1, .7, 10)), "t=1 T=40", f4(sigma(1, .7, 40)))

P("\n# C3: ODE/SDE final (t=0) stats, M=4000, seed=1: mean std P(x>0) W1")
for T in (5, 10, 40):
    s = simulate(0, T, 1, sde=False)[-1]
    P(f"ODE T={T}", *map(f4, stats(s)), f4(w1(s, 0)))
for T in (5, 10, 40):
    for a in (0.1, 0.4, 0.7, 1.0, 1.5, 2.0, 2.5):
        s = simulate(a, T, 1)[-1]
        P(f"SDE T={T} a={a}", *map(f4, stats(s)), f4(w1(s, 0)))
# a=0 SDE equals ODE
s0 = simulate(0, 10, 1)[-1]; so = simulate(0, 10, 1, sde=False)[-1]
P("a=0 SDE vs ODE first particle", f4(s0[0]), f4(so[0]), "maxdiff", max(abs(x-y) for x,y in zip(s0,so)))
# intermediate t snapshot
sn = simulate(0.7, 10, 1)
P("SDE a=0.7 T=10 snapshot j=5 (t=0.5) W1", f4(w1(sn[5], 0.5)), "first particle j=0..3", [f4(sn[j][0]) for j in range(4)])
sn = simulate(0, 40, 1, sde=False)
P("ODE T=40 j=20 (t=0.5) W1", f4(w1(sn[20], 0.5)))
# MC floor: exact samples W1
r = mulberry32(99); ex = []
for _ in range(4000):
    k = 0 if r() < W[0] else 1
    ex.append(MU[k] + S[k] * gauss(r))
P("W1 of 4000 exact samples (seed 99)", f4(w1(ex, 0)))
fl = []
for sd in range(1, 11):
    r = mulberry32(100 + sd); ex = []
    for _ in range(4000):
        k = 0 if r() < W[0] else 1
        ex.append(MU[k] + S[k] * gauss(r))
    fl.append(w1(ex, 0))
P("W1 floor exact samples seeds 101..110: mean", f4(sum(fl)/10), "min", f4(min(fl)), "max", f4(max(fl)))
for j in (0, 1):
    for a in (0.7, 2.5):
        for T in (5, 10, 40):
            t = 1 - j / T; sg = sigma(t, a, T)
            P(f"x-coeff j={j} a={a} T={T}:", f4(1 - sg * sg / (2 * t) / T))
# seed sensitivity
for sd in (2, 3):
    P(f"seed {sd}: SDE T=10 a=0.7 W1", f4(w1(simulate(.7, 10, sd)[-1], 0)), "ODE T=10 W1", f4(w1(simulate(0, 10, sd, sde=False)[-1], 0)))
t0 = time.time(); simulate(.7, 40, 1); P("python sim time T=40 (s)", f"{time.time()-t0:.2f}")

P("\n# C1: advantage. group seed=1 cont G=24")
r = mulberry32(1); R = [draw_reward(r, 'continuous', .3) for _ in range(24)]
A, m, s = advantages(R)
P("R[0:3]", [f4(x) for x in R[:3]], "mean", f4(m), "std", f4(s), "A[0:3]", [f4(x) for x in A[:3]], "sumA", f"{sum(A):.2e}", "sumA2", f4(sum(a*a for a in A)))
r = mulberry32(1); R = [draw_reward(r, 'binary', .3) for _ in range(24)]
A, m, s = advantages(R)
P("binary p=.3 R", ''.join(str(int(x)) for x in R), "mean", f4(m), "std", f4(s), "A(1)", f4(max(A)), "A(0)", f4(min(A)))
P("degenerate example: G=2 all ones ->", advantages([1.0, 1.0])[0])
Gs = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
for kind, p in (('continuous', .3), ('binary', .3), ('binary', .05), ('binary', .5)):
    row = []
    for G in Gs:
        rm, dg = adv_mc(G, kind, p, 7)
        row.append(f"G={G}: rmse {f4(rm)} degMC {f4(dg)} degExact {f4(p_degenerate(G, kind, p))}")
    P(f"{kind} p={p}"); [P("  ", x) for x in row]

P("\n# C2: clip")
for A in (1.0, -1.0):
    for rho in (0.7, 0.9, 1.0, 1.1, 1.3):
        v, g = clip_obj(rho, A, 0.2)
        P(f"A={A} rho={rho} eps=.2 -> obj {f4(v)} grad {g}")

P("\n# C4: step policy (j,T,a,dv,z) -> sigma sd shift kl1 kl2 logr")
for args in ((5, 10, .7, .2, 1.0), (0, 10, .7, .2, 1.0), (9, 10, .7, .2, 1.0), (9, 10, .1, .2, 1.0), (9, 10, 1.4, .2, 1.0), (20, 40, .7, .2, 1.0), (39, 40, .7, .2, 1.0)):
    P(args, [f"{x:.6g}" for x in step_policy(*args)])
# per-step KL coefficient c_j = kl2/dv^2 and sum
for T in (10, 40):
    for a in (0.1, 0.7, 1.4, 2.0):
        cs = [step_policy(j, T, a, 1.0, 0)[4] for j in range(T)]
        P(f"T={T} a={a} coeff j=0,T/2,T-1:", f"{cs[0]:.6g}", f"{cs[T//2]:.6g}", f"{cs[-1]:.6g}", "sum", f"{sum(cs):.6g}", "argmax j", cs.index(max(cs)))
P('asymptotic small-t coeff factor (1/a+a/2)^2: a=.7', f4((1/.7+.35)**2), 'a=sqrt2', f4(2.0))
mx = 0.0
for T in (10, 40):
    for a in (0.1, 0.35, 0.7, 1.4, 2.0):
        for j in range(T):
            for dv in (-1.0, 0.2, 0.73):
                q = step_policy(j, T, a, dv, 1.0); mx = max(mx, abs(q[3] - q[4]) / q[4])
P('c_j closed vs KL2/dv^2 max abs diff', f"{max(abs(kl_coeff(j,T,a)-step_policy(j,T,a,1.0,0)[4]) for T in (10,40) for a in (0.1,0.7,2.0) for j in range(T)):.2e}", 'c_j at dv=0 (closed form, no division):', f"{kl_coeff(5,10,.7):.6g}")
# multi-seed: ODE vs SDE W1 at t=0, seeds 1..10
for T in (5, 10, 40):
    for a in (0.0, 0.7, 2.5):
        ws = [w1(simulate(a, T, sd, sde=(a > 0))[-1], 0) for sd in range(1, 11)]
        mu = sum(ws) / 10; sdv = math.sqrt(sum((x - mu) ** 2 for x in ws) / 9)
        P(f"seeds1-10 T={T} a={a} W1 mean {f4(mu)} sd {f4(sdv)} min {f4(min(ws))} max {f4(max(ws))}")
P('KL form1 vs form2 max rel diff', f"{mx:.2e}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check_out.txt'), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
