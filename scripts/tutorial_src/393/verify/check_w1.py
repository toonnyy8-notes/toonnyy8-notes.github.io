import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, random
from dmath import *
def w1(K, at, N, seed):
    r = mulberry32(seed); s = math.sqrt(1-at*at); cnt = [0]*K
    for n in range(N):
        cnt[argmax_first([(at if k==0 else 0.0)+s*gauss(r) for k in range(K)])] += 1
    return [c/N for c in cnt]
for K,at in [(5,0.6),(5,0.0),(5,0.9),(2,0.6),(10,0.8),(3,0.3)]:
    f=w1(K,at,20000,7); T=T_op(at,K); px=T+(1-T)/K; po=(1-T)/K
    se=math.sqrt(px*(1-px)/20000)
    print('K=%d at=%.1f T=%.4f px=%.4f po=%.4f | MC px=%.4f oth mean=%.4f min=%.4f max=%.4f | (MC-px)/SE=%.2f' % (K,at,T,px,po,f[0],sum(f[1:])/(K-1),min(f[1:]),max(f[1:]),(f[0]-px)/se))
for seed in [7,8,9]:
    print('bias check K=3 at=.3 N=200000 seed',seed,'%.4f' % w1(3,0.3,200000,seed)[0])
