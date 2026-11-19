// core.js — numerical core to be copied verbatim into build/index.html.
// Setting: VE smoothing, d=2, n=1; p_data = von Mises on the parameter angle t.
'use strict';
const KAPPA = 4, THETA0 = Math.PI;              // App. C.3 values
const TWO_PI = 2 * Math.PI;

// ---- Bessel I0, I1 (A&S 9.8.1-9.8.4 for x<50; asymptotic series for x>=50)
function logI0(x) {
  x = Math.abs(x);
  if (x < 3.75) { const t = (x / 3.75) ** 2;
    return Math.log(1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))); }
  if (x < 50) { const t = 3.75 / x;
    const p = 0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
    return x + Math.log(p) - 0.5 * Math.log(x); }
  const y = 1 / x;
  return x - 0.5 * Math.log(TWO_PI * x) + Math.log1p(y * (1 / 8 + y * (9 / 128 + y * (225 / 3072 + y * 11025 / 98304))));
}
// I1(x)/I0(x) / x  (finite at x=0, value 1/2)
function ratioOverX(x) {
  if (x < 1e-8) return 0.5;
  if (x < 3.75) { const t = (x / 3.75) ** 2;
    const i1x = 0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411)))));
    const i0 = 1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
    return i1x / i0; }
  if (x < 50) { const t = 3.75 / x;
    const p0 = 0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
    const p1 = 0.39894228 + t * (-0.03988024 + t * (-0.00362018 + t * (0.00163801 + t * (-0.01031555 + t * (0.02282967 + t * (-0.02895312 + t * (0.01787654 - t * 0.00420059)))))));
    return p1 / p0 / x; }
  const y = 1 / x;
  return (1 - y * (0.5 + y * (1 / 8 + y * (1 / 8 + y * (25 / 128 + y * 13 / 32))))) / x;
}
const LOG_NORM_VM = Math.log(TWO_PI) + logI0(KAPPA);       // log(2*pi*I0(kappa))
const logPdata = t => KAPPA * Math.cos(t - THETA0) - LOG_NORM_VM;

// ---- unit circle: exact log p_sigma and score (closed form)
function circleLogP(x1, x2, s) {
  const s2 = s * s, a = x1 / s2 + KAPPA * Math.cos(THETA0), b = x2 / s2 + KAPPA * Math.sin(THETA0);
  const R = Math.hypot(a, b);
  return -Math.log(TWO_PI * s2) - (x1 * x1 + x2 * x2 + 1) / (2 * s2) + logI0(R) - logI0(KAPPA);
}
function circleScore(x1, x2, s) {                // returns [s1, s2]
  const s2 = s * s, a = x1 / s2 + KAPPA * Math.cos(THETA0), b = x2 / s2 + KAPPA * Math.sin(THETA0);
  const q = ratioOverX(Math.hypot(a, b)) / s2;
  return [-x1 / s2 + q * a, -x2 / s2 + q * b];
}
// radial / tangential components at polar (rho, phi), computed without cancellation in the tangential part
function circleScorePolar(rho, phi, s) {
  const s2 = s * s, c = Math.cos(phi - THETA0);
  const R = Math.hypot(rho / s2 + KAPPA * c, KAPPA * Math.sin(phi - THETA0));
  const q = ratioOverX(R) / s2;
  return { rad: -rho / s2 + q * (rho / s2 + KAPPA * c), tan: -q * KAPPA * Math.sin(phi - THETA0) };
}
// Eq. (6) four terms on the circle (VE): A=-d_M/s^2, B=log p_data, C=-(d-n)/2 log(2 pi s^2), D=H=-1/2 log|H^| = -1/2 log rho
function eq6Terms(rho, phi, s) {
  const x1 = rho * Math.cos(phi), x2 = rho * Math.sin(phi);
  const exact = circleLogP(x1, x2, s);
  const A = -0.5 * (rho - 1) ** 2 / (s * s), B = logPdata(phi), C = -0.5 * Math.log(TWO_PI * s * s), D = -0.5 * Math.log(rho);
  return { exact, A, B, C, D, resid: exact - (A + B + C + D) };
}

// ---- general ellipse Phi(t) = (a cos t, b sin t): log p_sigma by periodic trapezoid in t'
// Windowed: a 128-node coarse scan finds the coarse intervals that can contribute; only their fine nodes are summed.
// Interval k is kept if (d_k - Delta)^2 <= dmin^2 + 2 s^2 (40 + 2 kappa), Delta = max|Phi'| * coarse spacing
// (true distance inside the interval >= d_k - Delta; 2 kappa = range of log p_data). Dropped mass < e^-40 relative.
const NC = 128;
function makeEllipseLogP(a, b, s, full) {
  const per = Math.max(1, Math.ceil(Math.min(6000, Math.max(256, 8 * Math.PI * Math.max(a, b) / s)) / NC));
  const N = per * NC, h = TWO_PI / N, cx = new Float64Array(N), cy = new Float64Array(N), lp = new Float64Array(N), e = new Float64Array(N);
  for (let k = 0; k < N; k++) { const t = -Math.PI + (k + 0.5) * h; cx[k] = a * Math.cos(t); cy[k] = b * Math.sin(t); lp[k] = logPdata(t); }
  const inv = 1 / (2 * s * s), c0 = -Math.log(TWO_PI * s * s) + Math.log(h);
  const Delta = Math.max(a, b) * TWO_PI / NC, slack = 2 * s * s * (40 + 2 * KAPPA), dk = new Float64Array(NC);
  const keep = new Uint8Array(NC);
  return function (x1, x2) {
    let m = -Infinity, sum = 0;
    if (full) {
      for (let k = 0; k < N; k++) { const dx = x1 - cx[k], dy = x2 - cy[k]; const v = lp[k] - (dx * dx + dy * dy) * inv; e[k] = v; if (v > m) m = v; }
      for (let k = 0; k < N; k++) if (e[k] > m - 40) sum += Math.exp(e[k] - m);
      return c0 + m + Math.log(sum);
    }
    let dmin2 = Infinity;
    for (let c = 0; c < NC; c++) { const k = c * per + (per >> 1), dx = x1 - cx[k], dy = x2 - cy[k], d2 = dx * dx + dy * dy; dk[c] = Math.sqrt(d2); if (d2 < dmin2) dmin2 = d2; }
    const thr = dmin2 + slack;
    for (let c = 0; c < NC; c++) { const lo = dk[c] - Delta; keep[c] = (lo <= 0 || lo * lo <= thr) ? 1 : 0; }
    let n = 0;
    for (let c = 0; c < NC; c++) if (keep[c]) for (let k = c * per, k1 = k + per; k < k1; k++) { const dx = x1 - cx[k], dy = x2 - cy[k]; const v = lp[k] - (dx * dx + dy * dy) * inv; e[n++] = v; if (v > m) m = v; }
    for (let q = 0; q < n; q++) if (e[q] > m - 40) sum += Math.exp(e[q] - m);
    return c0 + m + Math.log(sum);
  };
}

// ---- C2: angular marginal of pi~ ∝ exp( s^alpha (log p_s + U_err) - v ), generalized polar x=(a r cos t, b r sin t)
// Split in two: c2Grid (expensive, depends on shape, sigma, alpha, nt; cacheable; can be filled in row slices)
// and c2Combine (cheap; error field, beta, lambda). opts: {a,b,sigma,alpha,err:'none'|'c3'|'tan', beta, lambda, nt}
const C2_NT = 180, C2_NS = 49;
function c2Grid(o) {
  const { a, b, sigma: s, alpha } = o, nt = o.nt || C2_NT, w = Math.pow(s, 1 - alpha / 2);
  const r0 = Math.max(0.02, 1 - 8 * w), r1 = 1 + 8 * w, dr = (r1 - r0) / (C2_NS - 1);
  const circle = (a === 1 && b === 1 && !o.forceQuad);
  return { a, b, s, alpha, nt, w, r0, dr, circle, lpE: circle ? null : makeEllipseLogP(a, b, s, o.fullQuad), LP: new Float64Array(nt * C2_NS), rowsDone: 0 };
}
function c2FillRows(G, j1) {            // fill rows G.rowsDone .. j1-1 (page calls this in ~8 ms slices)
  for (let j = G.rowsDone; j < j1; j++) {
    const t = -Math.PI + (j + 0.5) * TWO_PI / G.nt, ct = Math.cos(t), st = Math.sin(t);
    for (let i = 0; i < C2_NS; i++) { const r = G.r0 + i * G.dr, x1 = G.a * r * ct, x2 = G.b * r * st;
      G.LP[j * C2_NS + i] = G.circle ? circleLogP(x1, x2, G.s) : G.lpE(x1, x2); }
  }
  G.rowsDone = Math.max(G.rowsDone, j1);
}
function c2Combine(G, o) {
  const { a, b, s, alpha, nt } = G, sa = Math.pow(s, alpha), tb = o.err === 'tan' ? Math.pow(s, o.beta) : 0, lam = o.lambda || 0;
  const logm = new Float64Array(nt), L = new Float64Array(C2_NS);
  for (let j = 0; j < nt; j++) {
    const t = -Math.PI + (j + 0.5) * TWO_PI / nt, ct = Math.cos(t), st = Math.sin(t);
    let m = -Infinity;
    for (let i = 0; i < C2_NS; i++) {
      const r = G.r0 + i * G.dr, x1 = a * r * ct, x2 = b * r * st;
      let ell = G.LP[j * C2_NS + i];
      if (o.err === 'c3') { const q = (x1 - 1) ** 2 + x2 * x2; ell += -0.5 * q * q; }
      else if (o.err === 'tan') ell += tb * ct;
      const v = sa * ell - lam * (1 - ct) + Math.log(a * b * r) + ((i === 0 || i === C2_NS - 1) ? Math.log(0.5) : 0);
      L[i] = v; if (v > m) m = v;
    }
    let sum = 0; for (let i = 0; i < C2_NS; i++) sum += Math.exp(L[i] - m);
    logm[j] = m + Math.log(sum);
  }
  // normalization over t also in the log domain (sigma^alpha*U_err can reach ~6e3): subtract the max before exp
  let M = -Infinity, Mlo = Infinity; for (const v of logm) { if (v > M) M = v; if (v < Mlo) Mlo = v; }
  const dens = new Float64Array(nt); let Z = 0;
  for (let j = 0; j < nt; j++) { dens[j] = Math.exp(logm[j] - M); Z += dens[j]; }
  const dt = TWO_PI / nt; for (let j = 0; j < nt; j++) dens[j] /= Z * dt;
  return { dens, w: G.w, theta: Math.pow(s, 2 - alpha), log10MaxOverMin: (M - Mlo) / Math.LN10 };
}
function c2Marginal(o) { const G = c2Grid(o); c2FillRows(G, G.nt); return c2Combine(G, o); }
// reference curves on the same t grid (densities in t)
function refCurve(kind, a, b, lambda, nt) {
  nt = nt || C2_NT;
  const out = new Float64Array(nt), dt = TWO_PI / nt; let Z = 0;
  for (let j = 0; j < nt; j++) {
    const t = -Math.PI + (j + 0.5) * dt, sp = Math.hypot(a * Math.sin(t), b * Math.cos(t));
    let v;
    if (kind === 'pdata') v = Math.exp(logPdata(t));
    else if (kind === 'unif') v = sp;
    else if (kind === 'post') v = Math.exp(-(lambda || 0) * (1 - Math.cos(t))) * sp;
    else if (kind === 'postdata') v = Math.exp(-(lambda || 0) * (1 - Math.cos(t)) + logPdata(t));
    else if (kind === 'c3lim') { const q = (a * Math.cos(t) - 1) ** 2 + (b * Math.sin(t)) ** 2; v = Math.exp(logPdata(t) - 0.5 * q * q); }
    out[j] = v; Z += v * dt;
  }
  for (let j = 0; j < nt; j++) out[j] /= Z;
  return out;
}
const tv = (p, q) => { let s = 0; for (let j = 0; j < p.length; j++) s += Math.abs(p[j] - q[j]); return 0.5 * s * TWO_PI / p.length; };
const maxOverMin = p => { let lo = Infinity, hi = 0; for (const v of p) { if (v < lo) lo = v; if (v > hi) hi = v; } return hi / lo; };

// ---- C4: particles, Euler–Maruyama in the App. C.1 clock
function mulberry32(seed) { let a = seed >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function errField(err, x1, x2) {
  if (err === 'c3') { const d1 = x1 - 1, q = d1 * d1 + x2 * x2; return [-2 * q * d1, -2 * q * x2]; }
  if (err === 'rot') { const r = Math.hypot(x1, x2); if (r < 1e-9) return [0, 0]; const g = 4 * (1 - x1 / r); return [-g * x2 / r, g * x1 / r]; }
  return [0, 0];
}
// state: {X: Float64Array(2N) per panel}, both panels share the same noise
function c4Init(N, seed) {
  const rng = mulberry32(seed), X = new Float64Array(2 * N);
  for (let i = 0; i < N; i++) { const t = -Math.PI + TWO_PI * rng(); X[2 * i] = Math.cos(t); X[2 * i + 1] = Math.sin(t); }
  return { rng, P: [X, Float64Array.from(X)], step: 0, resets: [0, 0] };
}
function c4Steps(st, n, o) {          // o: {sigma, alphas:[0, aTS], err, dt}
  const s = o.sigma, s2 = s * s, dt = o.dt, N = st.P[0].length / 2;
  const amp = o.alphas.map(al => Math.sqrt(2 * Math.pow(s, 2 - al) * dt));
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < N; i++) {
      const u1 = st.rng(), u2 = st.rng(), rr = Math.sqrt(-2 * Math.log(1 - u1)), g1 = rr * Math.cos(TWO_PI * u2), g2 = rr * Math.sin(TWO_PI * u2);
      for (let p = 0; p < 2; p++) {
        const X = st.P[p], x1 = X[2 * i], x2 = X[2 * i + 1];
        const sc = circleScore(x1, x2, s), e = errField(o.err, x1, x2);
        let y1 = x1 + dt * s2 * (sc[0] + e[0]) + amp[p] * g1, y2 = x2 + dt * s2 * (sc[1] + e[1]) + amp[p] * g2;
        if (!isFinite(y1) || !isFinite(y2) || y1 * y1 + y2 * y2 > 2500) { y1 = 1; y2 = 0; st.resets[p]++; }
        X[2 * i] = y1; X[2 * i + 1] = y2;
      }
    }
    st.step++;
  }
}
function c4Stats(X, nb) {
  const N = X.length / 2, h = new Float64Array(nb); let C = 0, S = 0;
  for (let i = 0; i < N; i++) { const t = Math.atan2(X[2 * i + 1], X[2 * i]); C += Math.cos(t); S += Math.sin(t);
    let j = Math.floor((t + Math.PI) / TWO_PI * nb); if (j >= nb) j = nb - 1; if (j < 0) j = 0; h[j]++; }
  let tvU = 0; for (let j = 0; j < nb; j++) tvU += Math.abs(h[j] / N - 1 / nb); 
  return { meanCos: C / N, meanSin: S / N, tvUnif: 0.5 * tvU, hist: h };
}
if (typeof module !== 'undefined') module.exports = { KAPPA, THETA0, logI0, ratioOverX, logPdata, circleLogP, circleScore, circleScorePolar, eq6Terms, makeEllipseLogP, c2Grid, c2FillRows, c2Combine, c2Marginal, refCurve, tv, maxOverMin, C2_NT, mulberry32, c4Init, c4Steps, c4Stats, errField };
