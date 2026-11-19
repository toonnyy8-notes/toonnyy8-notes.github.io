"""R2: alternative reading 'slip = jump to a uniformly random free cell' flattens V* far from the goal."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maze
from maze import NS, IDX, move
def tp(s, a):
    out = {move(s, a): 0.75}
    for x in range(NS): out[x] = out.get(x, 0) + 0.25 / NS
    return out
maze.true_P = tp
V = maze.value_iteration()
print('V* at (0,7),(6,7)=S,(3,7):', ['%.3f' % V[IDX[p]] for p in [(0, 7), (6, 7), (3, 7)]])
