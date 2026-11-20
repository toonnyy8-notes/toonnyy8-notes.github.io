import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, random
from dmath import *
K=5
for t in [0.2,0.4,0.6]:
    h=1e-4; Tf=lambda tt: T_op(1-tt,K); T0=Tf(t); dT=(Tf(t+h)-Tf(t-h))/(2*h)
    P=[T0*(k==0)+(1-T0)/K for k in range(K)]
    print(t, 'dP/dt[x]=%.6f rhs[x]=%.6f' % (dT*(1-1/K), -dT/(K*T0)*(sum(P)-K*P[0])))
