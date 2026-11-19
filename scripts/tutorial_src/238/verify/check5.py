import numpy as np
exec(open('check4.py').read().split('for samp')[0])
tgt={('u','u'):lambda D,d:D/(D+d)}
for samp,loss in [('u','u'),('ln','u'),('ln','v')]:
  for D,d in [(128,4),(128,32),(128,64),(128,128)]:
    p=np.zeros(3); h=1e-5; lr=0.02; m=np.zeros(3); v=np.zeros(3); b1,b2,eps=0.9,0.95,1e-8
    hist=[]
    for it in range(1,3001):
        g=np.array([(L(p+h*e,D,d,samp,loss)-L(p-h*e,D,d,samp,loss))/(2*h) for e in np.eye(3)])
        m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
        p-=lr*(m/(1-b1**it))/(np.sqrt(v/(1-b2**it))+eps)
        if it in (100,300,1000,3000): hist.append(round(float(1/(1+np.exp(-p[2]))),3))
    print(samp,loss,D,d,hist,'a=%.3f b=%.3f'%(p[0],p[1]))
