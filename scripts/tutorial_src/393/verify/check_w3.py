import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import math, random
from dmath import *
for ws in [(1.2,0.6,0.0),(1.2,0.0,0.6),(1.2,0.3,0.3),(0.5,0.4,-1.0),(2.0,-0.5,1.0)]:
    for ats,att in [(0.9,0.6),(0.95,0.3)]:
        P=push(ws,ats,att); Q,a,zs=dkern(ws,ats,att)
        tv=0.5*sum(abs(p-q) for p,q in zip(P,Q))
        print(ws,ats,att,'a~=%.4f P=(%s) sum=%.6f a=%.4f Q=(%s) TV=%.4f'%(att/ats,', '.join('%.4f'%p for p in P),sum(P),a,', '.join('%.4f'%q for q in Q),tv))
