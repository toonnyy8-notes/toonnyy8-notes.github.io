const c = require('./core.js');
const row = (o) => {
  const t0 = Date.now(), r = c.c2Marginal(o), ms = Date.now() - t0;
  const pd = c.refCurve('pdata', o.a, o.b), un = c.refCurve('unif', o.a, o.b);
  let extra = '';
  if (o.err === 'c3') extra = ' TV(c3lim)=' + c.tv(r.dens, c.refCurve('c3lim', o.a, o.b)).toFixed(4);
  if (o.lambda) extra += ' TV(post)=' + c.tv(r.dens, c.refCurve('post', o.a, o.b, o.lambda)).toFixed(4) + ' TV(postdata)=' + c.tv(r.dens, c.refCurve('postdata', o.a, o.b, o.lambda)).toFixed(4);
  console.log(`${o.a === 1 && o.b === 1 ? 'circle ' : 'ellipse'} s=${o.sigma} a=${o.alpha} err=${o.err}${o.err === 'tan' ? '(b=' + o.beta + ')' : ''} lam=${o.lambda || 0} | TV(pdata)=${c.tv(r.dens, pd).toFixed(4)} TV(unif)=${c.tv(r.dens, un).toFixed(4)} max/min=${c.maxOverMin(r.dens).toFixed(3)} w=${r.w.toFixed(3)}${extra} [${ms}ms]`);
  return r;
};
console.log('-- consistency: circle closed form vs quadrature path (forceQuad)');
for (const s of [0.01, 0.05, 0.2]) for (const al of [0, 1]) {
  const A = c.c2Marginal({ a: 1, b: 1, sigma: s, alpha: al, err: 'none' }), B = c.c2Marginal({ a: 1, b: 1, sigma: s, alpha: al, err: 'none', forceQuad: true });
  let m = 0; for (let j = 0; j < A.dens.length; j++) m = Math.max(m, Math.abs(A.dens[j] - B.dens[j]));
  console.log(`s=${s} alpha=${al} max|closed-quad|=${m.toExponential(2)}`);
}
console.log('-- acceptance table');
for (const shape of [[1, 1], [1, 2]]) {
  const [a, b] = shape;
  for (const s of [0.2, 0.05, 0.01]) for (const al of [0, 0.5, 1, 1.5]) row({ a, b, sigma: s, alpha: al, err: 'none' });
  for (const s of [0.05, 0.01]) for (const al of [0, 1]) row({ a, b, sigma: s, alpha: al, err: 'c3' });
  for (const s of [0.05, 0.01]) for (const [al, be] of [[0, -1], [0.5, -1], [1, -1], [1.5, -1], [1, 0.5]]) row({ a, b, sigma: s, alpha: al, err: 'tan', beta: be });
  for (const s of [0.05, 0.01]) for (const al of [0, 1]) row({ a, b, sigma: s, alpha: al, err: 'none', lambda: 1.5 });
}
const un = c.refCurve('unif', 1, 2); console.log('ellipse uniform-intrinsic density in t: max', Math.max(...un).toFixed(4), 'min', Math.min(...un).toFixed(4), 'ratio', c.maxOverMin(un).toFixed(3));
console.log('pdata max/min', c.maxOverMin(c.refCurve('pdata',1,1)).toFixed(1), 'pdata max', Math.max(...c.refCurve('pdata',1,1)).toFixed(4), 'min', Math.min(...c.refCurve('pdata',1,1)).toFixed(5));
const c3l = c.refCurve('c3lim', 1, 1); console.log('circle c3lim max', Math.max(...c3l).toFixed(4), 'min', Math.min(...c3l).toFixed(4));
