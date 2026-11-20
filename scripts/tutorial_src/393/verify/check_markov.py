import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, random
from dmath import *
# F1: z-level test on the Gaussian chain (37); x index 0, K=3, alpha~ s,t,u = 0.9, 0.6, 0.3
K=3; A=[0.9,0.6,0.3]; r=mulberry32(2024); N=400000
same_x=[0,0]; same_o=[0,0]; mk={True:[0,0],False:[0,0]}
for n in range(N):
    s0=math.sqrt(1-A[0]**2); w=[(A[0] if k==0 else 0)+s0*gauss(r) for k in range(K)]; zs=argmax_first(w)
    a=A[1]/A[0]; s=math.sqrt(1-a*a); w=[a*v+s*gauss(r) for v in w]; zt=argmax_first(w)
    a=A[2]/A[1]; s=math.sqrt(1-a*a); w=[a*v+s*gauss(r) for v in w]; zu=argmax_first(w)
    (same_x if zs==0 else same_o)[0]+=1; (same_x if zs==0 else same_o)[1]+= (zt==zs)
    if zt!=0: mk[zs==zt][0]+=1; mk[zs==zt][1]+=(zu==zt)
ats=T_op(.6,K)/T_op(.9,K); atu=T_op(.3,K)/T_op(.6,K)
print('P(z_t=z_s|z_s=x)=%.4f  P(z_t=z_s|z_s!=x)=%.4f  USDM (38)=%.4f' % (same_x[1]/same_x[0], same_o[1]/same_o[0], ats+(1-ats)/K))
print('P(z_u=z_t|z_t!=x, z_s=z_t)=%.4f  |z_s!=z_t=%.4f  USDM=%.4f  (counts %d, %d)' % (mk[True][1]/mk[True][0], mk[False][1]/mk[False][0], atu+(1-atu)/K, mk[True][0], mk[False][0]))
