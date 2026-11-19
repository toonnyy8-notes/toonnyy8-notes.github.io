const c = require('./core.js'), f = (x, n = 4) => x.toFixed(n);
console.log('C1: Eq.(6) terms at rho, phi (deg) for several sigma');
for (const [rho, deg] of [[1.1, 90], [1.05, 90], [1.0, 180], [0.9, 30]]) {
  const phi = deg * Math.PI / 180;
  console.log(`rho=${rho} phi=${deg}`);
  console.log('sigma | exact | A=-dM/s2 | B=log pdata | C | D=-1/2log rho | resid | s_rad | -(rho-1)/s2 | s_rad+(rho-1)/s2 | s_tan | lim s_tan | |s_rad/s_tan| | s2*logp | -dM');
  for (const s of [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]) {
    const T = c.eq6Terms(rho, phi, s), S = c.circleScorePolar(rho, phi, s);
    const lim = -c.KAPPA * Math.sin(phi - c.THETA0) / rho;
    console.log([s, f(T.exact), f(T.A), f(T.B), f(T.C), f(T.D), T.resid.toExponential(3), f(S.rad, 3), f(-(rho - 1) / s / s, 3), f(S.rad + (rho - 1) / s / s, 4), f(S.tan), f(lim), Math.abs(S.rad / S.tan).toFixed(1), f(s * s * T.exact, 5), f(-0.5 * (rho - 1) ** 2, 5)].join(' | '));
  }
}
