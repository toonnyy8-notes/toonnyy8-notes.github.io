import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, random
from dmath import *
mx = max(abs(erfc_nr(x)-math.erfc(x))/math.erfc(x) for x in [i*0.01 for i in range(-600,900)])
print('erfc rel err max %.3g' % mx)
for at in [0.1,0.3,0.6,0.9]:
    s=math.sqrt(1-at*at); print('K2 closed', at, '%.6f %.6f' % (T_op(at,2), math.erf(at/(2*s))))
AL=[0.3,0.5,0.6,0.8,0.85,0.9,0.95,0.97,0.985,0.99]
for K in [2,3,5,10,100,1000,30522,50257]:
    print('T K=%d' % K, ' '.join('%.6f' % T_op(a,K) for a in AL))
print('band A (alpha~=1-t): [0.85, 0.97];  band A2 (alpha~^2=1-t): [%.4f, %.4f]' % (math.sqrt(.85), math.sqrt(.97)))
for K in [2,3,10,100,1000,30522,50257]:
    b0=bisect_T(0.05,K); b1=bisect_T(0.95,K)
    print('K=%d bandB [%.4f, %.4f]  T(A ends)=%.4f %.4f  T(A2 ends)=%.4f %.4f' % (K,b0,b1,T_op(.85,K),T_op(.97,K),T_op(math.sqrt(.85),K),T_op(math.sqrt(.97),K)))
