// core.js — synthesis page numeric core; copied verbatim into build/index.html
// (after 237/verify/core.js, which supplies circleLogP, circleScorePolar, c2Marginal, tv, KAPPA, THETA0, ...).
'use strict';
// ---- schedules (ported from 406/verify/check.py; T = 1000)
const SYN_T = 1000;
function abarFull(kind) {
  const ab = []; let p = 1;
  if (kind === 'linear') { for (let i = 0; i < SYN_T; i++) { p *= 1 - (1e-4 + (0.02 - 1e-4) * i / (SYN_T - 1)); ab.push(p); } return ab; }
  const f = u => Math.cos((u / SYN_T + 0.008) / 1.008 * Math.PI / 2) ** 2;
  for (let i = 0; i < SYN_T; i++) { p *= 1 - Math.min(1 - f(i + 1) / f(i), 0.999); ab.push(p); }
  return ab;
}
function respace(kind, Tp) {          // A[0] = 1, A[t] for t = 1..Tp
  const ab = abarFull(kind), A = [1];
  for (let i = 0; i < Tp; i++) A.push(ab[Math.round(i * (SYN_T - 1) / (Tp - 1))]);
  return A;
}
function coefs(A, t) {                 // step x_t -> x_{t-1}
  const a = A[t] / A[t - 1], b = 1 - a;
  return { alpha: a, beta: b, c1: Math.sqrt(A[t - 1]) * b / (1 - A[t]), c2: Math.sqrt(a) * (1 - A[t - 1]) / (1 - A[t]), bt: (1 - A[t - 1]) * b / (1 - A[t]) };
}
// ---- W1: multipliers on the score (equivalently on eps, since s = -eps/sigma)
const mTS = (sigma, alpha) => Math.pow(sigma, alpha);                  // 237 Eq. 8 / App. C.2
const mES = lambda => 1 / lambda;                                       // 406 Sec. 4.1
const mIPman = gamma => 1 / (1 + gamma * gamma);                        // page derivation: normal part, sigma -> 0
function mIPgauss(sigma, gamma, s2) {                                   // page toy: 1-D Gaussian data, VP abar = 1 - sigma^2
  if (!(sigma > 0 && sigma < 1)) return NaN;
  const ab = 1 - sigma * sigma; return (ab * s2 + sigma * sigma) / (ab * s2 + sigma * sigma * (1 + gamma * gamma));
}
const sigmaCross = (lambda, alpha) => Math.pow(lambda, -1 / alpha);     // sigma^alpha = 1/lambda
const alphaEq = (lambda, sigma) => (sigma > 0 && sigma < 1) ? Math.log(lambda) / -Math.log(sigma) : NaN; // ES written as sigma^alpha at this sigma
function sigmaOfT(A, t) { return Math.sqrt(1 - A[t]); }
// ---- W2: on-circle limits; tempered von Mises density in t, p^c normalised
function vmPow(c, nt) {
  nt = nt || 180; const out = new Float64Array(nt), dt = 2 * Math.PI / nt; let Z = 0;
  for (let j = 0; j < nt; j++) { const t = -Math.PI + (j + 0.5) * dt; out[j] = Math.exp(c * KAPPA * Math.cos(t - THETA0)); Z += out[j] * dt; }
  for (let j = 0; j < nt; j++) out[j] /= Z; return out;
}
// ---- W3: gamma that makes IP's training input variance equal the sampling variance of the 406 Eq. 7 model
// model 'x0': e_t = val (x0-space std, 405 nu_t / 406 e_t); model 'eps': e_t = val*sqrt((1-abar_t)/abar_t) (eps-space std val)
function gammaMatch(kind, Tp, val, model) {
  // accumulate the excess variance D_t = V_t - sigma_t^2 directly (406 App. A.1: c2^2 (1-abar_t) + bt = 1-abar_{t-1}),
  // so D_{t-1} = c2^2 D_t + (c1 e_t)^2, D_{T'} = 0 -- no subtraction of nearly equal numbers
  const A = respace(kind, Tp), D = new Float64Array(Tp + 1), rows = [];
  const eOf = t => model === 'eps' ? val * Math.sqrt((1 - A[t]) / A[t]) : val;
  for (let s = Tp; s >= 1; s--) { const c = coefs(A, s); D[s - 1] = c.c2 * c.c2 * D[s] + (c.c1 * eOf(s)) ** 2; }
  for (let t = 1; t < Tp; t++) {
    const bb = 1 - A[t], sg = Math.sqrt(bb), c = coefs(A, t + 1), one = (c.c1 * eOf(t + 1)) ** 2;
    rows.push({ t, sigma: sg, c1: c.c1, g1: Math.sqrt(one) / sg, gM: Math.sqrt(D[t]) / sg, delta: (D[t] / (Math.sqrt(bb + D[t]) + sg)) ** 2 });
  }
  return rows;
}
// ---- prose check: perfect network, Gaussian data var s2, ancestral DDPM with eps/lambda -> Var(x0_hat) (exact linear recursion)
function esGaussVar(kind, Tp, s2, lambda) {
  const A = respace(kind, Tp); let v = 1;                                // x_T ~ N(0, 1) (v = Var of x_hat_t)
  for (let t = Tp; t >= 1; t--) {
    const c = coefs(A, t), sg = Math.sqrt(1 - A[t]), k = sg / (A[t] * s2 + 1 - A[t]);   // eps*(x) = k x
    const g = (1 - sg * k / lambda) / Math.sqrt(A[t]);                   // x0_hat = g x
    const m = c.c1 * g + c.c2;                                           // x_{t-1} = m x + sqrt(bt) z
    v = m * m * v + c.bt;
  }
  return v;
}
if (typeof module !== 'undefined') module.exports = { abarFull, respace, coefs, mTS, mES, mIPman, mIPgauss, sigmaCross, alphaEq, sigmaOfT, vmPow, gammaMatch, esGaussVar };
