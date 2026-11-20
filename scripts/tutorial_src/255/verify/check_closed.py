"""Closed-form checks: Example 1 (C1), Prop. 3 (C2), Eq. 12 cost (C4), Table 1 deltas."""
import math
# ---- C1 Example 1 ----
def ex1(p, a):
    paa, pab, pba, pbb = p
    pow_a, pow_b = paa**a + pab**a, pba**a + pbb**a
    tmp_a, tmp_b = (paa+pab)**a, (pba+pbb)**a
    Fa = ((paa/(paa+pab))**a + (pab/(paa+pab))**a)
    Fb = ((pba/(pba+pbb))**a + (pbb/(pba+pbb))**a)
    Ppow_a = pow_a/(pow_a+pow_b); Ptmp_a = tmp_a/(tmp_a+tmp_b)
    S = paa**a+pab**a+pba**a+pbb**a
    seq_pow = [x**a/S for x in p]
    def cond(x, y):  # low-temp second token
        zx, zy = x**a, y**a
        return zx/(zx+zy), zy/(zx+zy)
    ca, cb = cond(paa, pab), cond(pba, pbb)
    seq_tmp = [Ptmp_a*ca[0], Ptmp_a*ca[1], (1-Ptmp_a)*cb[0], (1-Ptmp_a)*cb[1]]
    return dict(pow_a=pow_a, pow_b=pow_b, tmp_a=tmp_a, tmp_b=tmp_b, Fa=Fa, Fb=Fb,
                Ppow_a=Ppow_a, Ptmp_a=Ptmp_a, seq_pow=seq_pow, seq_tmp=seq_tmp)
for name, p in [("orig(sum .90)", (0.0, .40, .25, .25)), ("sum1", (0.0, .45, .275, .275))]:
    print("Example1", name, "sum", sum(p), "alpha*", math.log(2)/math.log(p[1]/p[2]))
    for a in (1, 1.5, 2, 4, 8):
        r = ex1(p, a)
        print(f"  a={a}: pow {r['pow_a']:.4f}/{r['pow_b']:.4f} temp {r['tmp_a']:.4f}/{r['tmp_b']:.4f} "
              f"F {r['Fa']:.4f}/{r['Fb']:.4f} P(x0=a) pow {r['Ppow_a']:.4f} temp {r['Ptmp_a']:.4f} "
              f"seqpow {[round(x,4) for x in r['seq_pow']]} seqtmp {[round(x,4) for x in r['seq_tmp']]}")
# ---- C2 Prop 3 ----
def prop3(e, ep, N, a):
    lo = ep / N**(1-1/a)
    lw_pos = a*math.log(e); lw_neg_pow = a*math.log(ep)-(a-1)*math.log(N); lw_neg_tmp = a*math.log(ep)
    sig = lambda z: 1/(1+math.exp(-z))
    return lo, sig(lw_pos-lw_neg_pow), sig(lw_pos-lw_neg_tmp), e > ep/N
for args in [(.4,.5,2,2),(.4,.5,2,1),(.3,.5,2,2),(.3,.5,4,2),(.3,.6,8,4),(.4,.5,1,4),(.2,.7,64,8)]:
    lo, ppow, ptmp, fut = prop3(*args)
    print("Prop3 e,e',N,a=", args, f"lower={lo:.4f} P_pow(x_t|pair)={ppow:.4f} P_temp={ptmp:.4f} future_x_t_higher={fut}")
# ---- C4 cost ----
def cost(N, T, B):
    K = math.ceil(T/B)
    approx = N*T/(4*B)
    exact = N*B*K*(K+1)/4
    trunc = N*sum(min(k*B, T) for k in range(1, K+1))/2
    return approx, exact/T, trunc/T, (trunc+T)/T, K
for args in [(10,679,192),(10,3072,192),(2,679,192),(10,679,679),(10,192,192),(0,679,192),(10,679,1),(50,3072,1)]:
    a,e,t,ti,K = cost(*args); print("Cost N,T,B=",args,f"K={K} approx={a:.3f} eq12sum={e:.3f} trunc={t:.3f} trunc+init={ti:.3f}")
# ---- Table 1 ----
T1 = {"Qwen2.5-Math-7B":[(.496,.690,.748,.785),(.329,.512,.573,.537),(.278,.353,.389,.399),(1.61,2.09,2.88,2.38)],
      "Qwen2.5-7B":[(.498,.628,.706,.740),(.329,.524,.622,.561),(.278,.303,.318,.354),(7.05,5.29,8.59,7.62)],
      "Phi-3.5-mini":[(.400,.478,.508,.406),(.213,.585,.732,.134),(.273,.293,.364,.359),(14.82,18.15,17.65,16.74)]}
cols = ["MATH500","HumanEval","GPQA","AlpacaEval"]
for m, rows in T1.items():
    for c,(b,lt,ps,g) in zip(cols, rows):
        share = (lt-b)/(ps-b)
        print(f"T1 {m:16s} {c:10s} PS-base={ps-b:+.3f} LT-base={lt-b:+.3f} PS-LT={ps-lt:+.3f} PS-GRPO={ps-g:+.3f} LTshare={share:.2f}")
# ---- GPQA question counts (198 questions) ----
for m, rows in T1.items():
    b, lt, ps, g = rows[2]
    print(f"GPQA counts {m:16s} base={b*198:.1f} LT={lt*198:.1f} PS={ps*198:.1f} GRPO={g*198:.1f}")
# ---- C1 guard: 0^a and empty row ----
assert 0.0**1.0 == 0 and math.pow(0, 8) == 0
