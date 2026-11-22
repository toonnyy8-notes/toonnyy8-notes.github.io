# W2: per-token truncated-IS weight, variance contribution v = po*w^2 (Eq. 13-14), lost mass pt - po*w (A == 1)
import math
def caps(po, eps, delta, dual):
    c={'IS':math.inf,'PPO':1+eps,'RIC':1+math.sqrt(delta/po)}
    if dual: c={k:min(v,10.0) for k,v in c.items()}
    return c
def row(po, pt, eps=0.2, delta=0.05, dual=False):
    r=pt/po; out={}
    for k,cap in caps(po,eps,delta,dual).items():
        w=min(r,cap); out[k]=(w, po*w*w, pt-po*w)
    return r,out
for po,pt,dual in [(0.001,0.01,False),(0.001,0.01,True),(0.0001,0.01,False),(0.0001,0.01,True),(0.01,0.02,False),(0.3,0.4,False),(0.001,0.0011,False)]:
    r,o=row(po,pt,dual=dual)
    print(f"po={po} pt={pt} r={r:.3f} dual={dual}")
    for k,(w,v,b) in o.items(): print(f"   {k:3s} w={w:.4f} v={v:.6f} lost={b:.6f}")
# boundary envelopes
def vmax(po,eps=0.2,delta=0.05):
    return po*(1+eps)**2, po*(1+math.sqrt(delta/po))**2
for po in [1e-5,1e-4,1e-3,1e-2,0.1,0.5]:
    a,b=vmax(po); print(f"po={po:g} vPPO={a:.6f} vRIC={b:.6f} (sqrt form {(math.sqrt(po)+math.sqrt(0.05))**2:.6f})")
    assert abs(b-(math.sqrt(po)+math.sqrt(0.05))**2)<1e-12
print('W2 OK')
