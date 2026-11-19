// C2 ellipse: windowed vs full quadrature (accuracy) and timing per update at sigma = 0.01
const c = require('./core.js');
const maxdiff = (p, q) => { let m = 0; for (let j = 0; j < p.length; j++) m = Math.max(m, Math.abs(p[j] - q[j]) / Math.max(q[j], 1e-300)); return m; };
for (const [s, al, err, be] of [[0.01, 0, 'none'], [0.01, 1, 'none'], [0.01, 1.5, 'none'], [0.01, 1.9, 'none'], [0.05, 1, 'c3'], [0.2, 1.5, 'none'], [0.01, 0, 'tan', -1.9]]) {
  const o = { a: 1, b: 2, sigma: s, alpha: al, err, beta: be };
  const W = c.c2Marginal(o), F = c.c2Marginal({ ...o, fullQuad: true });
  console.log(`s=${s} alpha=${al} err=${err} max rel diff windowed vs full = ${maxdiff(W.dens, F.dens).toExponential(2)}  finite=${W.dens.every(isFinite)}`);
}
const time = (o, rep = 5) => { const t0 = process.hrtime.bigint(); for (let k = 0; k < rep; k++) c.c2Marginal(o); return Number(process.hrtime.bigint() - t0) / 1e6 / rep; };
for (const al of [0, 1, 1.5, 1.9]) {
  const o = { a: 1, b: 2, sigma: 0.01, alpha: al, err: 'none' };
  console.log(`ellipse s=0.01 alpha=${al}: windowed nt=180 ${time(o).toFixed(1)} ms | nt=90 (drag) ${time({ ...o, nt: 90 }).toFixed(1)} ms | full nt=180 ${time({ ...o, fullQuad: true }, 2).toFixed(1)} ms`);
}
const G = c.c2Grid({ a: 1, b: 2, sigma: 0.01, alpha: 1 }); c.c2FillRows(G, G.nt);
const t0 = process.hrtime.bigint(); for (let k = 0; k < 100; k++) c.c2Combine(G, { err: 'tan', beta: -1, lambda: 1 }); console.log('c2Combine only (cache hit, err/beta/lambda change):', (Number(process.hrtime.bigint() - t0) / 1e8).toFixed(2), 'ms');
