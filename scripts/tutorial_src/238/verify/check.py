# Population-loss checks for design.md (Eq.12, Eq.54, Fig.1 numbers). Not page code.
import numpy as np
M=20000
t=(np.arange(M)+0.5)/M
def dens(kind):
    if kind=='uniform': return np.ones_like(t)
    lt=np.log(t/(1-t)); return np.exp(-lt**2/2)/np.sqrt(2*np.pi)/(t*(1-t))
def kappa2(k,loss,clamp):
    if loss=='u': return np.ones_like(t)
    den=k*(1-t)+(1-k)*t
    if clamp: den=np.maximum(den,clamp)
    return 1/den**2
def delta(k,D,d,samp,loss,bins=1,clamp=0.0):
    a=t; s=1-t; ph=k; ps=-(1-k)
    w=dens(samp)*kappa2(k,loss,clamp)/M
    tot=0
    for b in range(bins):
        m=(t>=b/bins)&(t<(b+1)/bins); ww=w[m]; aa=a[m]; ss=s[m]
        par=np.sum(ww*(ph**2+ps**2))-np.sum(ww*(ph*aa+ps*ss))**2/np.sum(ww*(aa**2+ss**2))
        per=np.sum(ww*ps**2)-np.sum(ww*ps*ss)**2/np.sum(ww*ss**2)
        tot+=0.5*(d*par+(D-d)*per)
    return tot
def dep(k,D,d,samp,loss,clamp=0.0):
    w=dens(samp)*kappa2(k,loss,clamp)/M
    return 0.5*d*np.sum(w*(k*(1-t)+(1-k)*t)**2/(t**2+(1-t)**2))
ks=np.linspace(0,1,1001)
def kstar(f): 
    v=[f(k) for k in ks]; return ks[int(np.argmin(v))]
D=128
print('closed form check d=32', delta(0.7,128,32,'uniform','u'), (2*(D+32)*.49-4*D*.7+(2*D+3*32))/16)
for d in [4,16,32,48,64,96,128]:
    r=[kstar(lambda k:delta(k,D,d,'uniform','u'))]
    r.append(kstar(lambda k:delta(k,D,d,'uniform','u',bins=8)))
    r.append(kstar(lambda k:dep(k,D,d,'uniform','u')))
    r.append(kstar(lambda k:delta(k,D,d,'logit','u')))
    r.append(kstar(lambda k:delta(k,D,d,'logit','v')))
    r.append(kstar(lambda k:delta(k,D,d,'logit','v',clamp=0.05)))
    r.append(kstar(lambda k:delta(k,D,d,'logit','v',bins=8)))
    r.append(kstar(lambda k:delta(k,D,d,'logit','v',bins=8,clamp=0.05)))
    print(d, 'D/(D+d)=%.3f'%(D/(D+d)), ' '.join('%.3f'%x for x in r))
