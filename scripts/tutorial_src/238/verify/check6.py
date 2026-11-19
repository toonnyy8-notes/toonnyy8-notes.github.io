# Round-1 verification: M=1024, index binning, factored stable P/Q; C4 analytic Adam; clamp effect.
import numpy as np
M=1024; t=(np.arange(M)+0.5)/M
lt=np.log(t/(1-t)); LN=np.exp(-(lt+0.8)**2/(2*.64))/np.sqrt(2*np.pi)/.8/(t*(1-t))
P_={'uniform':np.ones(M),'logitn':LN}
SET={'A':('uniform','u'),'B':('logitn','u'),'C':('logitn','v')}
def w_of(s,k,clamp=0.0):
    samp,loss=SET[s]; p=P_[samp]
    if loss=='u': return p/M
    den=np.maximum(k*(1-t)+(1-k)*t,clamp) if clamp else k*(1-t)+(1-k)*t
    return p/den**2/M
def g(x): return x if x>1e-300 else 0.0
def parts(s,k,N,clamp=0.0):
    w=w_of(s,k,clamp); ph,ps=k,-(1-k); a,sg=t,1-t
    if N==0: return 0.5*np.sum(w*(ph*sg-ps*a)**2/(a*a+sg*sg)),0.0
    P=Q=0.0
    for b in range(N):
        idx=slice(b*M//N,(b+1)*M//N); ww,aa,ss=w[idx],a[idx],sg[idx]
        S0=ww.sum();Sa=(ww*aa).sum();Ss=(ww*ss).sum();Saa=(ww*aa*aa).sum();Sss=(ww*ss*ss).sum()
        den=Saa+Sss; P+=max(0,0.5*((ph*ph+ps*ps)*S0-((ph*Sa+ps*Ss)**2/den if den>1e-300 else 0)))
        Q+=max(0,0.5*ps*ps*(S0-(Ss*Ss/Sss if Sss>1e-300 else 0)))
    return P,Q
ks=np.linspace(0,1,201); D=128; ds=[1,4,16,32,48,64,96,128]
for s in 'ABC':
    for N in [1,8,0]:
        PQ=np.array([parts(s,k,N) for k in ks]); tot=np.array([[d*P+(D-d)*Q for P,Q in PQ] for d in ds])
        flat=[(r.max()-r.min())<1e-9*abs(r).max() for r in tot]
        print(s,N,' '.join('%d:%s'%(d,'flat' if f else '%.3f'%ks[r.argmin()]) for d,r,f in zip(ds,tot,flat)))
print('A d=32 N sweep',[ (N,ks[np.argmin([32*P+96*Q for P,Q in [parts('A',k,N) for k in ks]])]) for N in [1,2,4,8,16,32,64]])
print('C shared with clamp .05:',[ (d,ks[np.argmin([d*P+(D-d)*Q for P,Q in [parts('C',k,1,0.05) for k in ks]])]) for d in ds])
# C4 analytic Adam
def grad(s,d,a,b,wk):
    k=1/(1+np.exp(-wk)); samp,loss=SET[s]; p=P_[samp]/M; al,sg=t,1-t
    e1=a*al-k; e2=a*sg+1-k; e3=b*sg+1-k
    l=d*(e1*e1+e2*e2)+(D-d)*e3*e3
    if loss=='u': w=p; dw=0*p
    else:
        den=k*(1-t)+(1-k)*t; w=p/den**2; dw=-2*p*(1-2*t)/den**3
    ga=np.sum(w*2*d*(e1*al+e2*sg))/(2*D); gb=np.sum(w*2*(D-d)*e3*sg)/(2*D)
    gk=(np.sum(w*(2*d*(-e1-e2)-2*(D-d)*e3))+np.sum(dw*l))/(2*D)
    return np.array([ga,gb,gk*k*(1-k)]), np.sum(w*l)/(2*D)
for s,d in [('A',4),('A',32),('A',64),('A',128),('B',4),('B',32),('B',64),('C',4),('C',32),('C',64),('C',128)]:
    p=np.zeros(3);m=np.zeros(3);v=np.zeros(3);h=[]
    for it in range(1,3001):
        gr,L=grad(s,d,*p); m=.9*m+.1*gr; v=.95*v+.05*gr*gr
        p-=0.02*(m/(1-.9**it))/(np.sqrt(v/(1-.95**it))+1e-8)
        if it in(100,300,1000,3000): h.append('%.3f'%(1/(1+np.exp(-p[2]))))
    print(s,d,h,'a=%.3f b=%.3f L=%.4f'%(p[0],p[1],L))
