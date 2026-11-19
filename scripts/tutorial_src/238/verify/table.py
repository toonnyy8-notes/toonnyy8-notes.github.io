# Reference table for design.md C3 (M=1000 midpoints, k grid step 0.001)
import numpy as np
M=1000; t=(np.arange(M)+0.5)/M
lt=np.log(t/(1-t)); LN=np.exp(-(lt+0.8)**2/(2*.64))/np.sqrt(2*np.pi)/.8/(t*(1-t))
ks=np.linspace(0,1,1001)
def parts(k,samp,loss,N):
    p=np.ones(M) if samp=='U' else LN
    w=p*(np.ones(M) if loss=='u' else 1/(k*(1-t)+(1-k)*t)**2)/M
    ph,ps,a,s=k,-(1-k),t,1-t
    if N==0: return 0.5*np.sum(w*(ph*s-ps*a)**2/(a*a+s*s)),0.0
    P=Q=0
    for b in range(N):
        m=(t>=b/N)&(t<(b+1)/N); ww,aa,ss=w[m],a[m],s[m]
        P+=0.5*(np.sum(ww*(ph**2+ps**2))-np.sum(ww*(ph*aa+ps*ss))**2/np.sum(ww*(aa*aa+ss*ss)))
        Q+=0.5*(np.sum(ww*ps**2)-np.sum(ww*ps*ss)**2/np.sum(ww*ss*ss))
    return P,Q
D=128; ds=[1,4,16,32,48,64,96,128]
for samp,loss in [('U','u'),('L','u'),('L','v')]:
    for N in [1,8,0]:
        PQ=np.array([parts(k,samp,loss,N) for k in ks])
        row=[ks[np.argmin(d*PQ[:,0]+(D-d)*PQ[:,1])] for d in ds]
        print(samp+loss,'N=%s'%(N if N else 'inf'),' '.join('%d:%.3f'%(d,r) for d,r in zip(ds,row)))
PQ=np.array([parts(k,'U','u',0) for k in ks]); print('check par_dep symmetric', PQ[100,0],PQ[900,0])
