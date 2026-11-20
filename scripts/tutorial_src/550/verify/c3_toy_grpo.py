# C3 toy GRPO, round 1 (page-constructed toy, NOT paper data). Same algorithm / PRNG draw order as the page.
# Policy over output length: log L ~ N(mu, S^2), S fixed; only mu is trained.
# Built-in assumption (stated on the plate): accuracy c(L) = CMAX*(1-exp(-L/L0)) rises with length.
# Output longer than the cap is truncated -> no answer -> reward 0.
import math
CMAX, L0, S = 0.9, 10000.0, 0.5
MU0 = math.log(2000.0)
CAP0, CAP1 = 32768, 65536
J = 96                                   # midpoint quadrature over z in [-6, 6] for the expected curves
ZQ = [-6 + 12*(j+0.5)/J for j in range(J)]
WQ = [math.exp(-z*z/2)/math.sqrt(2*math.pi)*12/J for z in ZQ]
def mulberry32(seed):
    a=seed & 0xFFFFFFFF
    def imul(x,y): return (x*y) & 0xFFFFFFFF
    def r():
        nonlocal a
        a=(a+0x6D2B79F5) & 0xFFFFFFFF
        t=a
        t=imul(t ^ (t>>15), t|1)
        t=(t ^ ((t + imul(t ^ (t>>7), t|61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t>>14)) & 0xFFFFFFFF)/4294967296
    return r
def acc_of(L): return CMAX*(1-math.exp(-L/L0))
def expected(mu, cap):                   # (E[c(L) 1(L<=cap)], E[min(L,cap)], P(L>cap))
    a=l=tr=0.0
    for z,w in zip(ZQ,WQ):
        L=math.exp(mu+S*z)
        if L<=cap: a+=w*acc_of(L); l+=w*L
        else: l+=w*cap; tr+=w
    return a,l,tr
def capAt(step,Sraise): return CAP0 if (Sraise is None or step<Sraise) else CAP1
def run(seed, G=16, lr=0.02, beta=0.001, Sraise=1600, steps=2000, refEvery=400):
    rng=mulberry32(seed); mu=MU0; muref=mu; out=[]; zero=0
    for step in range(steps):
        if step%refEvery==0: muref=mu
        muold=mu; cap=capAt(step,Sraise)
        ys=[]; rs=[]
        for i in range(G):
            u1=rng(); u2=rng(); z=math.sqrt(-2*math.log(1-u1))*math.cos(2*math.pi*u2)
            y=muold+S*z; L=math.exp(y); u3=rng()
            ys.append(y); rs.append(1.0 if (L<=cap and u3<acc_of(L)) else 0.0)
        m=sum(rs)/G; sd=math.sqrt(sum((r-m)**2 for r in rs)/G)
        if sd>1e-12: A=[(r-m)/sd for r in rs]
        else: A=[0.0]*G; zero+=1
        g=0.0
        for y,a in zip(ys,A):
            # one update per rollout -> rho = 1, clip inactive (eps = 10)
            rref=math.exp(((y-mu)**2-(y-muref)**2)/(2*S*S))     # pi_ref/pi_theta at this sample
            g+=(a - beta*(1-rref))*(y-mu)/(S*S)/G
        mu+=lr*g
        a_,l_,t_=expected(mu,capAt(step+1,Sraise))
        out.append((mu,a_,l_,t_,zero))
    return out
CHK=[1,400,800,1200,1600,1700,2000]
def summary(R=8, base=1, **kw):
    runs=[run(base+r,**kw) for r in range(R)]
    res={s:tuple(sum(x[s-1][j] for x in runs)/R for j in range(4)) for s in CHK if s<=len(runs[0])}
    return res, sum(x[-1][4] for x in runs)/R/len(runs[0])
if __name__=='__main__':
    print('optimal median under each cap (grid over mu, step 0.01):')
    for cap in (CAP0,CAP1):
        best=max((expected(math.log(2000)+0.01*i,cap)[0],i) for i in range(500))
        mu=math.log(2000)+0.01*best[1]; a,l,t=expected(mu,cap)
        print(f'  cap={cap}: median={math.exp(mu):.0f} acc={a:.4f} E[min(L,cap)]={l:.0f} P(trunc)={t:.4f}')
    a,l,t=expected(MU0,CAP0); print(f'initial: median=2000 acc={a:.4f} len={l:.0f} trunc={t:.6f}')
    for name,kw in [('default (S=1600)',{}),('S=400',{'Sraise':400}),('S=1000',{'Sraise':1000}),('cap never raised',{'Sraise':None}),('cap from start',{'Sraise':0}),('G=4',{'G':4}),('G=32',{'G':32})]:
        res,zf=summary(**kw)
        print('==',name,'zero-signal frac %.3f'%zf)
        for s,(mu,acc,ln,tr) in res.items(): print(f'  step {s:4d} median={math.exp(mu):7.0f} acc={acc:.3f} len={ln:7.0f} trunc={tr:.3f}')
    print('== jump after cap raise: steps S, S+50, S+100 (median/acc)')
    for Sr in (400,1000,1600):
        CHK=[Sr,Sr+50,Sr+100]
        res,_=summary(Sraise=Sr)
        print(f'  S={Sr}: '+' | '.join(f'step {s}: median={math.exp(m):.0f} acc={a:.3f}' for s,(m,a,l,t) in res.items()))
    r=mulberry32(7); print('mulberry32(7) first 3:',[round(r(),10) for _ in range(3)])
