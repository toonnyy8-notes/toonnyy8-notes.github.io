from c3 import run
import sys
for scale in ['multi','0.1','0.3','1.0']:
    for eta in (0.1, 0.25, 0.5):
        out,_=run(scale, eta, 200, report=(0,10,25,50,100,200))
        print(scale, eta, [(s, round(m,4), c, n) for s,m,c,n in out], flush=True)
