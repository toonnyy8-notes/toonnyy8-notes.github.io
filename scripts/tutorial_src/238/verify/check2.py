import numpy as np, sys
sys.argv
exec(open('check.py').read().split("D=128")[0])
def densLN(mu,s):
    lt=np.log(t/(1-t)); return np.exp(-(lt-mu)**2/(2*s*s))/np.sqrt(2*np.pi)/s/(t*(1-t))
import builtins
for mu,s in [(-0.8,0.8),(0.8,0.8),(-0.8,1.0),(0.8,1.0)]:
    globals()['dens']=lambda kind,mu=mu,s=s: np.ones_like(t) if kind=='uniform' else densLN(mu,s)
    out=[]
    for d in [4,32,48,64,96,128]:
        out.append((d,kstar(lambda k:delta(k,128,d,'logit','u')),kstar(lambda k:delta(k,128,d,'logit','v')),kstar(lambda k:delta(k,128,d,'logit','v',bins=8))))
    print(mu,s,' '.join('%d:%.3f/%.3f/%.3f'%o for o in out))
