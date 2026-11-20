# C1 (group-relative advantage, Eq. 3) and C2 (clipped surrogate + KL estimator, Eq. 1-2): closed forms the page uses.
import math
def adv(rewards, ddof=0):
    G=len(rewards); m=sum(rewards)/G
    var=sum((r-m)**2 for r in rewards)/(G-ddof)
    s=math.sqrt(var)
    if s<1e-12: return [0.0]*G, m, s          # zero-signal guard
    return [(r-m)/s for r in rewards], m, s
print('== C1: binary rewards, k correct of G (population std / sample std)')
for G,k in [(16,4),(16,1),(16,8),(16,15),(16,0),(16,16),(4,1),(64,8)]:
    r=[1.0]*k+[0.0]*(G-k)
    a0,m,s0=adv(r,0); a1,_,s1=adv(r,1)
    ac0=a0[0] if k else float('nan'); aw0=a0[-1] if k<G else float('nan')
    ac1=a1[0] if k else float('nan'); aw1=a1[-1] if k<G else float('nan')
    # closed forms (page derivation): A+ = sqrt((G-k)/k), A- = -sqrt(k/(G-k)); ddof=1 multiplies by sqrt((G-1)/G)
    if 0<k<G:
        assert abs(ac0-math.sqrt((G-k)/k))<1e-12 and abs(aw0+math.sqrt(k/(G-k)))<1e-12
        assert abs(ac1-ac0*math.sqrt((G-1)/G))<1e-12
        assert abs(sum(a0))<1e-9
    print(f'G={G:2d} k={k:2d} mean={m:.4f} std0={s0:.4f} A+={ac0:.4f} A-={aw0:.4f} | std1={s1:.4f} A+={ac1:.4f} A-={aw1:.4f}')
print('== C1: adding a constant (format reward = 1 for all) leaves A unchanged')
r=[1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1.0]; a,_,_=adv(r); b,_,_=adv([x+1 for x in r]); assert max(abs(x-y) for x,y in zip(a,b))<1e-12; print('ok')
print('== C1: mixed reward acc+fmt example: acc=[1,1,0,0], fmt=[1,0,1,1]')
a,m,s=adv([2,1,1,1]); print([round(x,4) for x in a], round(m,4), round(s,4))
print('== C1: P(zero signal) = p^G + (1-p)^G')
for p in [0.05,0.25,0.5,0.9,0.99]:
    print(p, ' '.join(f'G={G}:{p**G+(1-p)**G:.4f}' for G in [4,16,64]))
print('== C1 round1: P(all r_i equal), r=acc+fmt in {0,1,2}: q0^G+q1^G+q2^G, q0=(1-p)(1-f), q1=p(1-f)+(1-p)f, q2=pf')
def p_equal(p,f,G):
    q0=(1-p)*(1-f); q1=p*(1-f)+(1-p)*f; q2=p*f
    return q0**G+q1**G+q2**G
for p,f in [(0.25,1.0),(0.25,0.9),(0.25,0.5),(0.05,1.0),(0.05,0.9),(0.9,0.9)]:
    print(f'p={p} f={f}', ' '.join(f'G={G}:{p_equal(p,f,G):.4e}' for G in [4,16,64]))
for p in [0.05,0.25,0.9]:
    for G in [4,16,64]: assert abs(p_equal(p,1.0,G)-(p**G+(1-p)**G))<1e-15
print('min over p of P0 at f=1, G=64:', '%.3e'%min(p_equal(i/200,1.0,64) for i in range(201)))
print('== C2: J(x)=min(rho*A, clip(rho,1-e,1+e)*A) - beta*(1/rho + log rho - 1), rho=e^x (pi_ref=pi_old)')
def J(x,A,e,beta):
    rho=math.exp(x); c=min(max(rho,1-e),1+e)
    sur=min(rho*A,c*A); D=1/rho+x-1
    return sur-beta*D
def dJ(x,A,e,beta):  # derivative wrt x = log rho (analytic)
    rho=math.exp(x)
    active = (rho<=1+e) if A>0 else ((rho>=1-e) if A<0 else False)
    return (rho*A if active else 0.0) - beta*(1-1/rho)
for (x,A,e,b) in [(0,1,0.2,0.001),(0.5,1,0.2,0.001),(0.5,1,10,0.001),(-0.5,-1,0.2,0.001),(-0.5,-1,10,0.001),(0.5,-1,0.2,0.001),(1,1,0.2,0.1),(-1,1,0.2,0.1),(2.5,1,10,0.001)]:
    h=1e-6; num=(J(x+h,A,e,b)-J(x-h,A,e,b))/(2*h)
    print(f'x={x:+.2f} A={A:+d} eps={e:<4} beta={b:<6} J={J(x,A,e,b):+.5f} dJ/dx={dJ(x,A,e,b):+.5f} (num {num:+.5f})')
print('clip boundaries in x: eps=0.2 ->', round(math.log(1.2),4), round(math.log(0.8),4), '; eps=10 -> upper', round(math.log(11),4), ', lower none (1-eps<0)')
