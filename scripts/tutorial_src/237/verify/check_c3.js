// C3: closed-form readouts of the (beta, alpha) panel at sigma = 0.01
const s = 0.01, K = 4;
const verdict = (b, a) => ({ conc: b >= -2, pdata: a === 0 && b >= 0, ts: b > -2 && a > Math.max(-b, 0) && a < 2 });
for (const [b, a] of [[-1, 1], [-1, 1.5], [-1, 0], [0.5, 1], [-2.5, 1], [-1.9, 1.95], [0, 0], [0, 1]]) {
  const v = verdict(b, a);
  console.log(`beta=${b} alpha=${a} | conc(Thm4.1-1)=${v.conc} pdata(Thm4.1-3, alpha=0)=${v.pdata} TS-uniform(Thm5.1)=${v.ts} | theta=${Math.pow(s, 2 - a).toPrecision(3)} w=${Math.pow(s, 1 - a / 2).toPrecision(3)} s^(a+b)=${Math.pow(s, a + b).toPrecision(3)} e^(2k s^a)=${Math.exp(2 * K * Math.pow(s, a)).toPrecision(4)}`);
}
// ellipse (1,2) perimeter and uniform-intrinsic density extremes (high-res trapezoid)
let L = 0; const n = 200000; for (let j = 0; j < n; j++) { const t = (j + 0.5) * 2 * Math.PI / n; L += Math.hypot(Math.sin(t), 2 * Math.cos(t)); } L *= 2 * Math.PI / n;
console.log('ellipse(1,2) perimeter', L.toFixed(4), 'density max (t=0,pi)', (2 / L).toFixed(4), 'min (t=±pi/2)', (1 / L).toFixed(4));
