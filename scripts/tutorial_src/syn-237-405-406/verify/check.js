// check.js — acceptance numbers for the synthesis page. Run: node check.js > check_out.txt
'use strict';
const fs = require('fs'), vm = require('vm'), path = require('path');
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '../../237/verify/core.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(__dirname, 'core.js'), 'utf8'));
const f = (x, d = 4) => x.toFixed(d), e3 = x => x.toExponential(3);

console.log('== W1: multipliers on score / eps ==');
console.log('mES(1.017) =', f(mES(1.017), 5), ' mIPman(0.1) =', f(mIPman(0.1), 5));
for (const s of [0.9, 0.5, 0.3, 0.1, 0.05, 0.01]) {
  console.log(`sigma=${s}: TS a=0.5 ${f(mTS(s, 0.5))} a=1 ${f(mTS(s, 1))} a=1.5 ${f(mTS(s, 1.5))} | ES ${f(mES(1.017))} | IPgauss(s2=.25) ${f(mIPgauss(s, 0.1, 0.25), 5)} | alphaEq(ES) ${f(alphaEq(1.017, s), 5)}`);
}
for (const a of [0.1, 0.5, 1, 1.5]) console.log(`sigmaCross(1.017, ${a}) = ${f(sigmaCross(1.017, a))}`);
const A = respace('cosine', 1000);
for (const t of [1, 10, 50, 100, 200, 500, 900, 999]) console.log(`cosine T=1000 t=${t}: sigma_t=${f(sigmaOfT(A, t), 5)} TS(a=1)=${f(mTS(sigmaOfT(A, t), 1), 5)}`);
// first t (cosine, 1000) where sigma_t^1 >= 1/1.017
{ let tc = -1; for (let t = 1; t <= 1000; t++) if (mTS(sigmaOfT(A, t), 1) >= mES(1.017)) { tc = t; break; } console.log('cosine T=1000: first t with sigma_t >= 1/1.017:', tc); }

console.log('== IP identity: optimal IP network -> score of sigma\' = sigma*sqrt(1+g^2) (Monte Carlo, x0 in {-1,1}) ==');
{
  const rnd = mulberry32(405); const gauss = () => { const u1 = rnd(), u2 = rnd(); return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2); };
  const sig = 0.6, a = Math.sqrt(1 - sig * sig), g = 0.1, sp = sig * Math.sqrt(1 + g * g), N = 3e7, bins = [-1.2, -0.4, 0.5, 1.0], hw = 0.005;
  const acc = bins.map(() => [0, 0, 0]);
  for (let i = 0; i < N; i++) {
    const x0 = rnd() < 0.5 ? -1 : 1, ep = gauss(), xi = gauss(), y = a * x0 + sig * (ep + g * xi);
    for (let k = 0; k < bins.length; k++) if (Math.abs(y - bins[k]) < hw) { acc[k][0] += ep; acc[k][1]++; acc[k][2] += ep * ep; }
  }
  for (let k = 0; k < bins.length; k++) {
    const y = bins[k], post = Math.tanh(a * y / (sp * sp));            // E[x0 | y] under sigma'
    const scoreSp = -(y - a * post) / (sp * sp);                       // d/dy log p_{sigma'}(y)
    const scoreS = -(y - a * Math.tanh(a * y / (sig * sig))) / (sig * sig);
    const m = acc[k][0] / acc[k][1], se = Math.sqrt((acc[k][2] / acc[k][1] - m * m) / acc[k][1]);
    console.log(`y=${y}: MC E[eps|y]=${f(m)} ± ${f(se)} (n=${acc[k][1]})  -sigma*score_{sigma'}=${f(-sig * scoreSp)}  -sigma*score_sigma=${f(-sig * scoreS)}`);
  }
}

console.log('== circle: radial score ratio s_{sigma\'}/s_sigma (gamma=0.1), phi=90deg ==');
for (const s of [0.1, 0.03, 0.01]) {
  const sp = s * Math.sqrt(1.01);
  for (const rho of [1.1, 1 + s]) {
    const r0 = circleScorePolar(rho, Math.PI / 2, s), r1 = circleScorePolar(rho, Math.PI / 2, sp);
    console.log(`sigma=${s} rho=${f(rho, 3)}: rad ratio=${f(r1.rad / r0.rad, 5)} tan ratio=${f(r1.tan / r0.tan, 5)}`);
  }
}

console.log('== W2: on-circle marginal of the Langevin stationary law (gradient field, VE, kappa=4) ==');
const pd = refCurve('pdata', 1, 1), un = refCurve('unif', 1, 1);
for (const s of [0.2, 0.05, 0.01]) {
  const ex = c2Marginal({ a: 1, b: 1, sigma: s, alpha: 0 }).dens;
  const es = c2Marginal({ a: 1, b: 1, sigma: s, alpha: alphaEq(1.017, s) }).dens;
  const ts = c2Marginal({ a: 1, b: 1, sigma: s, alpha: 1 }).dens;
  const ip = c2Marginal({ a: 1, b: 1, sigma: s * Math.sqrt(1.01), alpha: 0 }).dens;
  const vm = vmPow(1 / 1.017);
  console.log(`sigma=${s}: TV(p_data) exact ${f(tv(ex, pd))} ES ${f(tv(es, pd))} TS1 ${f(tv(ts, pd))} IP ${f(tv(ip, pd))} | TV(unif) ES ${f(tv(es, un))} TS1 ${f(tv(ts, un))} | TV(ES, vm^(1/lambda)) ${f(tv(es, vm))} | max/min ES ${f(maxOverMin(es), 1)} exact ${f(maxOverMin(ex), 1)}`);
}
console.log('vm^(1/1.017) max/min =', f(maxOverMin(vmPow(1 / 1.017)), 1), ' p_data max/min =', f(maxOverMin(pd), 1), ' TV(vm^(1/1.017), p_data) =', f(tv(vmPow(1 / 1.017), pd)));
for (const lam of [0.965, 1.06]) console.log(`lambda=${lam}: TV(vm^(1/lambda), p_data) = ${f(tv(vmPow(1 / lam), pd))}`);

console.log('== W3: gamma matching (406 Eq. 7 model, constant e) ==');
for (const [kind, Tp] of [['cosine', 20], ['cosine', 100], ['cosine', 1000], ['linear', 20]]) for (const [e, model] of [[0.1, 'x0'], [0.2, 'x0'], [0.1, 'eps']]) {
  const R = gammaMatch(kind, Tp, e, model), pick = Tp === 1000 ? [1, 2, 10, 100, 500, 999] : Tp === 100 ? [1, 2, 5, 10, 50, 99] : [1, 2, 5, 10, 15, 19];
  const s = pick.map(t => { const r = R[t - 1]; return `t=${t}:s=${f(r.sigma, 3)} g1=${e3(r.g1)} gM=${e3(r.gM)} d=${e3(r.delta)}`; }).join(' | ');
  const maxM = Math.max(...R.map(r => r.gM)), argM = R.find(r => r.gM === maxM).t;
  console.log(`${kind} T'=${Tp} ${model}=${e}: ${s} || max gM=${f(maxM)} at t=${argM}`);
}
{ for (const [Tp, e, model] of [[20, 0.2, 'x0'], [100, 0.2, 'x0'], [1000, 0.2, 'x0'], [100, 0.1, 'eps']]) { const R = gammaMatch('cosine', Tp, e, model); const n = R.filter(r => r.gM >= 0.1); const nb = R.filter(r => r.gM >= 0.05 && r.gM <= 0.2); console.log(`cosine ${Tp} ${model}=${e}: gM>=0.1 for t<=${n.length ? n[n.length - 1].t : '-'} (sigma<=${n.length ? f(n[n.length - 1].sigma, 3) : '-'}); gM in [0.05,0.2] for t=${nb.length ? nb[0].t + '..' + nb[nb.length - 1].t : '-'}`); } }

console.log('== W2 guard: lambda=0.95 (alphaEq<0) ==');
for (const s of [0.2, 0.01]) { const a = alphaEq(0.95, s), r = c2Marginal({ a: 1, b: 1, sigma: s, alpha: a }); let Z = 0, fin = true; for (const v of r.dens) { Z += v; if (!isFinite(v)) fin = false; }
  console.log(`sigma=${s}: alphaEq=${f(a, 5)} w=${f(r.w, 4)} finite=${fin} int=${f(Z * 2 * Math.PI / r.dens.length, 9)} TV(p_data)=${f(tv(r.dens, pd))} TV(vm^(1/0.95))=${f(tv(r.dens, vmPow(1 / 0.95)))}`); }
console.log('W1 guard: alphaEq(1.017, 1) =', alphaEq(1.017, 1), ' mIPgauss(1) =', mIPgauss(1, 0.1, 0.25), ' alphaEq(1.017, 0.99) =', f(alphaEq(1.017, 0.99), 4));
console.log('== W3 small-e precision: e=0.01, cosine 100: D-recursion vs old V-recursion ==');
{ const A = respace('cosine', 100), e = 0.01; const V = new Float64Array(101); V[100] = 1 - A[100];
  for (let s = 100; s >= 1; s--) { const c = coefs(A, s); V[s - 1] = c.c2 * c.c2 * V[s] + (c.c1 * e) ** 2 + c.bt; }
  const R = gammaMatch('cosine', 100, e, 'x0');
  for (const t of [1, 10, 50, 90]) { const bb = 1 - A[t], dOld = (Math.sqrt(V[t]) - Math.sqrt(bb)) ** 2; console.log(`t=${t}: delta new=${e3(R[t - 1].delta)} old=${e3(dOld)} rel.diff=${e3(Math.abs(R[t - 1].delta - dOld) / R[t - 1].delta)} gM=${e3(R[t - 1].gM)}`); } }
console.log('== ES with perfect network, Gaussian data s2=0.25: Var(x0_hat) ==');
for (const [kind, Tp] of [['cosine', 20], ['cosine', 100], ['cosine', 1000]]) {
  console.log(`${kind} T'=${Tp}: lambda=1 ${f(esGaussVar(kind, Tp, 0.25, 1), 5)} lambda=1.017 ${f(esGaussVar(kind, Tp, 0.25, 1.017), 5)} lambda=0.965 ${f(esGaussVar(kind, Tp, 0.25, 0.965), 5)}`);
}
console.log('== order translation (no numbers needed; printed for the table) ==');
for (const s of [0.1, 0.01]) console.log(`sigma=${s}: x0-error 0.2 -> score error ~ 0.2/sigma^2 = ${f(0.2 / s / s, 1)}; eps-error 0.2 -> score error 0.2/sigma = ${f(0.2 / s, 1)}`);
console.log('== W0 converter (exact VP factor sqrt(1-sigma^2)) ==');
for (const s of [0.1, 0.01]) console.log(`sigma=${s}: x0-error 0.2 -> ${(Math.sqrt(1 - s * s) * 0.2 / (s * s)).toPrecision(4)}; eps-error 0.2 -> ${(0.2 / s).toPrecision(3)}`);
