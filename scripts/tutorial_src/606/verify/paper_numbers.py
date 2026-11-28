"""論文數字的算術核對（頁面引用前確認一致）。"""
xl, rae = 1652.92 * 5, 1134.59 * 80
print('cumulative fwd GFLOPs ratio RAE/RevFlow-XL = %.2f (paper: about 11x)' % (rae / xl))
print('V(a|c) <= 1-0.574 =', round(1 - 0.574, 3), '; V(a|y) = 1-0.042 =', round(1 - 0.042, 3))
print('V(h23|y)/V(h23) token = 1-0.186 =', round(1 - 0.186, 3))
print('params ratio shared/separate = %.3f' % (0.690 / 1.361))
print('T2I: RAE GenEval = 77.54-3.65 =', round(77.54 - 3.65, 2), '; DPG = 81.55-4.04 =', round(81.55 - 4.04, 2), '(Table 5: 0.74 / 77.6)')
print('Fig.7a closed form J_max Dirac K=8 v=.958 t=.02:', 0.98 / 0.02**2 * ((2*1.4951/7)**2 / 4))
