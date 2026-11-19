"""C1: Sec. 3.1 VP noise schedule, closed form abar_i = exp(-bmin i/N - 0.5(bmax-bmin) i^2/N^2)."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from core import schedule, BMIN, BMAX
for N in (1, 2, 5, 10, 20, 50):
    b, a, ab = schedule(N)
    i = np.arange(1, N + 1)
    closed = np.exp(-BMIN * i / N - 0.5 * (BMAX - BMIN) * i ** 2 / N ** 2)
    assert np.abs(closed - ab).max() < 1e-12
    print('N=%d beta=%s' % (N, np.array2string(b, precision=4, max_line_width=200) if N <= 10 else '[%.4f ... %.4f]' % (b[0], b[-1])))
    print('     abar=%s sqrt(abar_N)=%.4f sqrt(1-abar_1)=%.4f' % (np.array2string(ab, precision=4, max_line_width=200) if N <= 10 else '[%.4f ... %.6f]' % (ab[0], ab[-1]), math.sqrt(ab[-1]), math.sqrt(1 - ab[0])))
print('abar_N for all N = exp(-(bmin+bmax)/2) = %.6f' % math.exp(-(BMIN + BMAX) / 2))
