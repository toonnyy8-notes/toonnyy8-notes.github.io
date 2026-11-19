# ref.py — independent reference (mpmath) for the closed forms used by core.js
import mpmath as mp, json, subprocess, sys
mp.mp.dps = 30
K, T0 = 4, mp.pi
def logI0(x): return mp.log(mp.besseli(0, x))
def ratio(x): return mp.besseli(1, x)/mp.besseli(0, x)
# brute-force p_sigma on circle by direct quadrature over u (independent of the Bessel closed form)
def logp_brute(rho, phi, s):
    x1, x2 = rho*mp.cos(phi), rho*mp.sin(phi)
    Z = 2*mp.pi*mp.besseli(0, K)
    f = lambda u: mp.exp(-((x1-mp.cos(u))**2+(x2-mp.sin(u))**2)/(2*s*s) + K*mp.cos(u-T0))/Z
    us = mp.atan2(x2, x1)
    val = mp.quad(f, [us-mp.pi, us-0.3, us-0.05, us, us+0.05, us+0.3, us+mp.pi])
    return mp.log(val/(2*mp.pi*s*s))
pts = [0.5, 2, 3.7, 3.8, 10, 49, 51, 200, 1e4, 4e4]
js = json.loads(subprocess.check_output(['node', '-e', '''
const c=require("./core.js");const p=%s;
console.log(JSON.stringify({l:p.map(c.logI0),r:p.map(x=>c.ratioOverX(x)*x),
 lp:[[1.05,1.3,0.1],[1.1,1.5708,0.01],[0.9,0.3,0.05],[1.2,3.0,0.2]].map(a=>c.circleLogP(a[0]*Math.cos(a[1]),a[0]*Math.sin(a[1]),a[2]))}))''' % json.dumps(pts)]))
print('x, logI0 abs err, ratio abs err')
for x, l, r in zip(pts, js['l'], js['r']):
    print(f'{x:8g}  {float(logI0(x)-l):+.2e}  {float(ratio(x)-r):+.2e}')
print('circle log p_sigma: closed form (core.js) vs brute quadrature')
for (rho, phi, s), v in zip([[1.05,1.3,0.1],[1.1,1.5708,0.01],[0.9,0.3,0.05],[1.2,3.0,0.2]], js['lp']):
    b = logp_brute(rho, phi, s); print(rho, phi, s, float(b), v, f'{float(b)-v:+.2e}')
