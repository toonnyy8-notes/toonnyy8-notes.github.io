import numpy as np
M=1000; t=(np.arange(M)+0.5)/M
lt=np.log(t/(1-t)); LN=np.exp(-(lt+0.8)**2/(2*.64))/np.sqrt(2*np.pi)/.8/(t*(1-t))
def L(p,D,d,samp,loss):
    a,b,w=p; k=1/(1+np.exp(-w)); ph=k; ps=-(1-k); al=t; s=1-t
    pd=np.ones(M) if samp=='u' else LN
    k2=np.ones(M) if loss=='u' else 1/(k*(1-t)+(1-k)*t)**2
    return 0.5*np.sum(pd*k2*(d*((a*al-ph)**2+(a*s-ps)**2)+(D-d)*(b*s-ps)**2))/M/D
for samp,loss in [('u','u'),('ln','u'),('ln','v')]:
  for D,d in [(128,4),(128,32),(128,64),(128,128)]:
    p=np.zeros(3); h=1e-5; lr=0.5 if loss=='u' else 0.05
    hist=[]
    for it in range(1,8001):
        g=np.array([(L(p+h*e,D,d,samp,loss)-L(p-h*e,D,d,samp,loss))/(2*h) for e in np.eye(3)])
        p-=lr*g
        if it in (200,1000,4000,8000): hist.append(round(1/(1+np.exp(-p[2])),3))
    print(samp,loss,D,d,hist, 'a=%.3f b=%.3f'%(p[0],p[1]))
