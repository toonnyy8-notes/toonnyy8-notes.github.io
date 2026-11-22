# W1: clip boundaries and consumed budgets (same algorithm as page)
import math
def bounds(method, po, eps=0.2, epsh=0.28, delta=0.02):
    if method=='PPO': lo,hi=eps,eps
    elif method=='DAPO': lo,hi=eps,epsh
    else: lo=hi=math.sqrt(delta/po)
    rhi=min(1+hi, 1/po); rlo=max(0.0,1-lo)
    return rlo, rhi
def measures(po, r):
    p=po*r; dp=p-po
    atom=0.5*po*(r-1)**2                      # Eq. 9 (with 1/2)
    eucl=(r-1)**2                             # Eq. 3
    binchi=0.5*dp*dp/(po*(1-po))              # [page derivation] binary chi^2/2
    if p>=1-1e-12: kl=float('inf')        # complement mass -> 0
    else: kl=-po*math.log1p(dp/po)-(1-po)*math.log1p(-dp/(1-po))   # log1p form (page uses the same)
    return p,dp,eucl,atom,binchi,kl
rows=[]
for delta in [0.02,0.05]:
  for po in [0.8,0.5,0.1,0.01,0.001]:
    for m in ['PPO','DAPO','RIC']:
        rlo,rhi=bounds(m,po,delta=delta)
        p,dp,e,a,b,k=measures(po,rhi)
        print(f"delta={delta} po={po} {m:4s} rlo={rlo:.4f} rhi={rhi:.4f} pmax={p:.4f} dp={dp:.5f} eucl={e:.4f} atom={a:.6f} binchi={b:.6f} binKL={k:.6f}")
# assertions vs paper Sec 3.1 / 3.2 / 4.1
assert abs(bounds('PPO',0.8)[1]*0.8-0.96)<1e-12
assert abs(bounds('PPO',0.01)[1]*0.01-0.012)<1e-12
assert abs(bounds('DAPO',0.01)[1]*0.01-0.0128)<1e-12
assert abs(bounds('DAPO',0.8)[1]*0.8-1.0)<1e-12          # capped at 1 (raw 1.024)
assert abs(0.5*0.8*0.2**2-0.016)<1e-12 and abs(0.5*0.01*0.2**2-0.0002)<1e-12
r=bounds('RIC',0.8,delta=0.02)[1]; print('RIC 0.8 ->',0.8*r)   # paper says 0.92 (exact 0.9265)
r=bounds('RIC',0.01,delta=0.02)[1]; print('RIC 0.01 ->',0.01*r) # paper 0.024
assert abs(0.5*0.8*(math.sqrt(0.02/0.8))**2-0.01)<1e-12
# crossover pi_old where RIC eps == PPO eps : delta/eps^2
print('crossover delta=0.02:',0.02/0.04,' delta=0.05:',0.05/0.04)
print('min RIC eps at delta=0.05 over po<=1:',math.sqrt(0.05))
# RIC upper clamp by pi<=1 begins at po + sqrt(delta po) = 1
for d in [0.02,0.05]:
    s=(-math.sqrt(d)+math.sqrt(d+4))/2; print('clamp start delta',d,'po>',s*s)
# lower side: RIC lower bound <=0 when po<=delta ; <0.5 (dual clip) when po<4 delta
print('W1 OK')
# edge cases (round 1 review): tiny pi_old, axis clamp
p,dp,e,a,b,k=measures(1e-4,bounds('RIC',1e-4,delta=0.02)[1])
print(f"edge po=1e-4 RIC d=0.02 atom={a:.6f} binchi={b:.6f} binKL={k:.6f}")
print("edge po=1e-4 PPO eps=0.05 atom=%.3e (below 1e-6 axis floor -> clamp marker)"%(0.5*1e-4*0.05**2))
