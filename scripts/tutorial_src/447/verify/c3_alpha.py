"""C3: alpha on the page slider grid (log10 alpha step 0.1): first diverging alpha, and alpha=10^1.5."""
import numpy as np, core
A = (-1.2, -0.8); c = np.array([1.0, 1.0])
for H, N in ((32, 20), (64, 20), (32, 100)):
    first = None
    for e in np.round(np.arange(-3, 2.0001, 0.1), 1):
        al = 10 ** e
        with np.errstate(all='ignore'):
            m, hist = core.run_chain(H, N, cond=[(0, A)], guide=core.make_guide('clean', al, c, H, N), noise=False)
        div = next((N - k + 1 for k, X in enumerate(hist) if not np.all(np.abs(X) <= 1e6)), None)
        if e in (1.0, 1.2, 1.3, 1.4, 1.5): print(f'H={H} N={N} log10a={e} alpha={al:.4f} div_step={div} end={m[:,-1].round(4) if div is None else None}')
        if div is not None and first is None: first = e
    print(f'H={H} N={N} first diverging log10(alpha) on grid = {first}')
