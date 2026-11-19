"""Arithmetic on numbers quoted from the paper (Fig. 2 / Fig. 6 values checked against the rendered PDF pages 8 and 20, see fig/)."""
import math
fig2 = {  # Oracle, VF, FlowCritic, Value Flows, PCBF  (mean 1-Wasserstein, +- in comments)
 'cube-triple': [92, 115, 212, 295, 204], 'cube-quadruple': [57, 179, 203, 289, 184],
 'puzzle-4x4': [20, 22, 125, 134, 118], 'humanoidmaze-giant': [25, 13, 127, 462, 122]}
fig6 = {  # Oracle, IVF, Implicit C51, Implicit IQN, DIVL
 'cube-triple': [92, 116, 299, 251, 287], 'cube-quadruple': [57, 174, 497, 233, 290],
 'puzzle-4x4': [20, 18, 279, 141, 128], 'humanoidmaze-giant': [25, 13, 205, 88, 255]}
for k, v in fig2.items():
    print(f'{k}: VF vs PCBF reduction {1 - v[1] / v[4]:.3f}; VF best of 3 prior? {v[1] < min(v[2:])}; VF/Oracle {v[1] / v[0]:.2f}')
for k, v in fig6.items():
    print(f'{k}: IVF best of 3 implicit? {v[1] < min(v[2:])}; IVF/Oracle {v[1] / v[0]:.2f}')
# OGBench overall averages (Tables 2-5)
names = ['Floq', 'FlowCritic', 'ValueFlows', 'PCBF', 'C51', 'IQN', 'QC', 'VF', 'DIVL', 'ImplC51', 'ImplIQN', 'ImplQC', 'IVF']
og = {10: [13, 34, 36, 29, 39, 39, 43, 48, 14, 26, 27, 25, 43], 25: [19, 34, 47, 26, 47, 48, 44, 54, 15, 39, 46, 38, 61],
      50: [19, 32, 46, 23, 51, 50, 45, 53, 15, 58, 57, 51, 66], 100: [16, 31, 48, 23, 52, 48, 52, 57, 18, 66, 61, 62, 65]}
for d, v in og.items():
    ex = dict(zip(names[:7], v[:7])); im = dict(zip(names[8:12], v[8:12]))
    be = max(ex, key=ex.get); bi = max(im, key=im.get)
    print(f'{d}%: VF {v[7]} vs best explicit {be} {ex[be]} (diff {v[7] - ex[be]}); IVF {v[12]} vs best implicit {bi} {im[bi]} (diff {v[12] - im[bi]})')
# Fig.5 vs Table 5 (4 ablation tasks: cube-triple t2,t4; puzzle t2,t4)
print('Table5 VF on ablation tasks mean', (98 + 78 + 98 + 100) / 4, ' IVF', (90 + 58 + 85 + 75) / 4)
# unscaled return range
for lo, name in [(-3, 'cube-triple'), (-4, 'cube-quadruple'), (-16, 'puzzle-4x4'), (-1, 'humanoidmaze-giant')]:
    print(f'{name}: [rmin,rmax]/(1-gamma) at gamma=.999 -> [{lo / 0.001:.0f}, 0]')
print('OGBench 10% of 100M =', 0.1 * 100e6 / 1e6, 'M transitions')
