import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, random
from dmath import *
K=7; a=0.37; ad=-1.0; r=mulberry32(5); raw=[r() for _ in range(K)]; Z=sum(raw); xth=[v/Z for v in raw]
print('x_theta', [round(v,4) for v in xth])
for (i,m) in [(2,2),(4,2),(0,2)]:
    A=f49(i,m,xth,a,ad,K); B=f17(i,m,xth,a,ad,K); print('z=%d x=%d f=%.10f fDuo=%.10f diff=%.1e' % (i,m,A,B,abs(A-B)))
xo=[1.0 if j==2 else 0.0 for j in range(K)]; print('oracle', f49(2,2,xo,a,ad,K), f49(4,2,xo,a,ad,K))
random.seed(0); mx=-1e9; n=0
for _ in range(20000):
    K=random.randint(2,12); a=random.uniform(0.01,0.99); raw=[random.random()+1e-9 for _ in range(K)]; Z=sum(raw)
    v=f49(random.randrange(K),random.randrange(K),[q/Z for q in raw],a,-1.0,K); mx=max(mx,v); n+= v<=1e-12
print('sign: %d/20000 cases f49<=0 with alpha\'=-1 (max %.2e)' % (n,mx))
