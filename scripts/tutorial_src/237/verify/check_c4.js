const c = require('./core.js');
const meanCos = d => { let C = 0, S = 0; const n = d.length; for (let j = 0; j < n; j++) { const t = -Math.PI + (j + 0.5) * 2 * Math.PI / n; C += d[j] * Math.cos(t); S += d[j] * Math.sin(t); } return [C * 2 * Math.PI / n, S * 2 * Math.PI / n]; };
const N = 1000, steps = +(process.argv[2] || 10000), seed = 237;
for (const sigma of [0.1, 0.05, 0.2]) for (const err of ['none', 'c3', 'rot']) {
  const aTS = 1, o = { sigma, alphas: [0, aTS], err, dt: 0.1 };
  const st = c.c4Init(N, seed), t0 = Date.now(); const marks = {};
  for (const m of [1000, 3000, steps]) { c.c4Steps(st, m - st.step, o); marks[m] = [0, 1].map(p => c.c4Stats(st.P[p], 36)); }
  const ms = Date.now() - t0;
  const f = (s) => `cos=${s.meanCos.toFixed(3)} sin=${s.meanSin.toFixed(3)} TVu=${s.tvUnif.toFixed(3)}`;
  let ref = '';
  if (err !== 'rot') { const L = meanCos(c.c2Marginal({ a: 1, b: 1, sigma, alpha: 0, err }).dens), T = meanCos(c.c2Marginal({ a: 1, b: 1, sigma, alpha: aTS, err }).dens); ref = ` | C2 exact: L cos=${L[0].toFixed(3)}  TS cos=${T[0].toFixed(3)}`; }
  console.log(`sigma=${sigma} err=${err} [${ms}ms] resets=${st.resets}`);
  for (const m of Object.keys(marks)) console.log(`  step ${m}: L ${f(marks[m][0])} | TS ${f(marks[m][1])}`);
  console.log('  ' + ref);
}
