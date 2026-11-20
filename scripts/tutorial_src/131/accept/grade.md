# Acceptance grade: tutorial 131

Score: 12/12 (12 correct, 0 partial, 0 wrong)

## Per-LO verdict
- LO1 correct: state, action, deterministic transition, terminal-only reward and J(pi) all match.
- LO2 correct: A-hat, r, mask rule, and the sg(r)·log-pi = grad r argument match; the clipped branch's zero gradient is covered.
- LO3 correct: Q = V(s_{t+1}), A = V(s_{t+1}) - V(s_t), sum of mu*A = 0.
- LO4 correct: pi+ closed form, Z, and eta -> 0 gives pi+ -> mu.
- LO5 correct: log Z = KL(mu||pi+) using E[A] = 0; Q adds eta*V(s_t). The KL direction note was not asked for.
- LO6 correct: sum A = R - V(x), delta(pi+) = 0, delta(mu) = eta(R - V). The tree example was not asked for.
- LO7 correct: uniqueness scope (Pi_mu, reachable states, same completion distribution, tabular) and the 4-step necessity argument match.
- LO8 correct: not covered; four steps in the right order; linearization drops delta's dependence on pi and eta. It does not give the case-I numbers, which were not asked for.
- LO9 correct: (1-mu)/(1-pi) identity is exact; omega derivation; at pi = mu, omega = r = 1 and the first update equals GRPO.
- LO10 correct: r = 1.0102 and 2.0; omega = 1.0909 and 1.0093; range [1/11, 11], clipped at C = 3.
- LO11 correct: peaks, gaps, 49.4 vs 45.5, and the limits (best checkpoint, same problem set, single run, no error bars).
- LO12 correct: eps = 0.1, C = 3, no KL, seq-mean-token-mean, and the three unspecified items.

## Fix list
No LO needs a fix. The stuck points below are already addressed on the page; the fixes are optional polish.
1. Three meanings of "eps" and the delta clash (stuck point 1). page.txt L77 and L589 already warn. One-line fix: put a bold "symbol clash" callout at the first use of eps in eq. 15 and again at the DPPO delta.
2. V as vocabulary and V as value (stuck point 2). One-line fix: write the vocabulary as a different glyph (e.g. calligraphic "Vocab") in the symbol table, or add "V^pi with a superscript is always a value" to it.
3. KL direction (stuck point 3). The page says it is a derived result. One-line fix: add a single sentence next to eq. 21: "The penalty is KL(pi||mu); the residual contains KL(mu||pi+). They are different quantities, so this is not a typo."
4. Exact versus approximate, and the opposite placement of omega and r (stuck point 4). L392 already states the identity is exact. One-line fix: add a two-row table right after eq. 15 with "r = pi/mu (importance ratio)" and "omega = (1+eps-mu)/(1+eps-pi) (KL-gradient coefficient)".
5. Demo values mistaken for paper settings (stuck point 5). L461 and L595 already say so. One-line fix: tag every demo number with a "demo" badge and give std = 0 a one-line "unspecified" note at eq. 2.
