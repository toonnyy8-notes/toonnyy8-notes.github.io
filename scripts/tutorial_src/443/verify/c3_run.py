import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, c3
c3.B = int(sys.argv[1]); c3.LR = float(sys.argv[2]); steps = int(sys.argv[3])
for N in [int(x) for x in sys.argv[4].split(',')]:
    for eta in [float(x) for x in sys.argv[5].split(',')]:
        lg, rec = c3.train(N, eta, steps, log=(steps // 4, steps // 2, steps))
        near, bot, eq, _ = c3.evaluate(lg, N)
        ws = ' '.join(np.array2string(rec[t][0], precision=3) for t in sorted(k for k in rec if k != 'avg50')) + ' avg(551-600)=' + np.array2string(rec['avg50'], precision=3)
        print('N=%2d eta=%.2f %s | eval near=%.3f bottom=%.3f EQ=%.3f' % (N, eta, ws, near, bot, eq), flush=True)
