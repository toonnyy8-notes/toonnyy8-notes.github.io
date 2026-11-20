import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, random
from dmath import *
random.seed(0)
for _ in range(1000):
    K=random.randint(2,17); at=random.random()*0.99; as_=at+random.random()*(1-at)
    p=posterior(random.randrange(K),random.randrange(K),at,as_,K); assert abs(sum(p)-1)<1e-12 and min(p)>-1e-15
for _ in range(2000):
    K=random.randint(2,9); at=random.uniform(0.01,0.9); as_=random.uniform(at,0.999)
    zt=random.randrange(K); x=random.randrange(K); ats=at/as_
    b=[(ats*(zt==j)+(1-ats)/K)*(as_*(x==j)+(1-as_)/K) for j in range(K)]; Z=sum(b)
    assert max(abs(b[j]/Z-pj) for j,pj in enumerate(posterior(zt,x,at,as_,K)))<1e-12
print('posterior (4): sums=1 and equals Bayes (random 1000/2000 cases)')
for Tst,seed in [(8,7),(16,7),(64,7),(16,8),(4,7)]:
    h=run_usdm(Tst,seed); hd,fl=run_ddt(Tst,seed)
    print('T=%d seed=%d USDM changes=%s total=%d final_ok=%s row0=%s' % (Tst,seed,changes(h),sum(changes(h)),h[-1]==TARGET,' '.join(VOCAB[v] for v in h[0])))
    print('      DDT changes=%s total=%d row0=%s flips=%s' % (changes(hd),sum(changes(hd)),' '.join(VOCAB[v] for v in hd[0]),[(f[0],round(f[1],4),None if f[2] is None else round(f[2],4)) for f in fl]))
