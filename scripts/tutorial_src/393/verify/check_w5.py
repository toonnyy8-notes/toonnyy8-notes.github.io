import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, random
from dmath import *
r=mulberry32(3); print('eps K=8 seed 3:', [round(gauss(r),4) for _ in range(8)])
for K in [8,1000,30522]:
    for at in [0.9,0.97,0.99]:
        row=[]
        for tau in [10,1,0.1,0.01,0.001]:
            d=w5(K,at,tau,3); row.append('tau=%g px=%.4f pmax=%.4f H=%.4f' % (tau,d['px'],d['pmax'],d['H']))
        d=w5(K,at,1,3)
        print('K=%d at=%.2f argmax=%d gap(w_x-max_other)=%.4f ;' % (K,at,d['argmax'],d['gap']), ' | '.join(row))
    print()
