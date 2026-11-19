import numpy as np
src=open('check.py').read().split("D=128")[0].replace("M=20000","M=1000")
exec(src)
def densLN(mu,s):
    lt=np.log(t/(1-t)); return np.exp(-(lt-mu)**2/(2*s*s))/np.sqrt(2*np.pi)/s/(t*(1-t))
dens=lambda kind: np.ones_like(t) if kind=='uniform' else densLN(-0.8,0.8)
ks=np.linspace(0,1,201)
print('M=1000 LN(-.8,.8): u d48', kstar(lambda k:delta(k,128,48,'logit','u')), 'v8bin d128', kstar(lambda k:delta(k,128,128,'logit','v',bins=8)))
print('uniform v-MSE clamp .05:', [ (d,kstar(lambda k:delta(k,128,d,'uniform','v',clamp=0.05))) for d in [4,32,64,128]])
print('bins sweep uniform u d=32:', [(N,kstar(lambda k:delta(k,128,32,'uniform','u',bins=N))) for N in [1,2,4,8,16,32,64]])
# reduced gradient flow on (a,b,w), uniform, u-loss, D=16 d=4
def L(a,b,w,D,d):
    k=1/(1+np.exp(-w)); al=t; s=1-t; ph=k; ps=-(1-k)
    return 0.5*np.mean(d*((a*al-ph)**2+(a*s-ps)**2)+(D-d)*(b*s-ps)**2)
for D,d in [(16,4),(128,32),(2,1)]:
    p=np.zeros(3); lr=0.5/D; h=1e-5
    for it in range(1,20001):
        g=np.array([(L(*(p+h*e),D,d)-L(*(p-h*e),D,d))/(2*h) for e in np.eye(3)])
        p-=lr*g
        if it in (100,1000,5000,20000): print(D,d,it,'a=%.3f b=%.3f k=%.4f target=%.4f'%(p[0],p[1],1/(1+np.exp(-p[2])),D/(D+d)))
