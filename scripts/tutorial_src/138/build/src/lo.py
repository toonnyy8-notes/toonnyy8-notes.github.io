import os
t = open(os.path.join(os.path.dirname(__file__), '..', '..', 'accept', 'page.txt'), encoding='utf-8').read()
checks = {
 'LO1': ['涵蓋', '\\sum_a\\mu\\cdot\\pi/\\mu=\\sum_a\\pi=1', '\\mathbb E_\\mu[rA]', '\\mathbb E_\\pi[A]', '樣本乘上 r $r$'],
 'LO2': ['收益項', '懲罰項', '\\frac{4\\xi\\gamma}{(1-\\gamma)^2}D^{\\max}_{TV}(\\pi\\|\\mu)^2', '\\eta(\\pi)\\ge\\eta(\\mu)'],
 'LO3': ['D_{TV}=\\tfrac12\\mathbb E_\\mu|r-1|', '2D_{TV}', '單樣本', '|r-1|\\le\\epsilon'],
 'LO4': ['r>1+\\epsilon', 'r<1-\\epsilon', '正優勢 token 只受上界影響', '負優勢 token 只受下界影響'],
 'LO5': ['f\'(1)\\mathbb E_\\mu[r-1]=0', 'f\'\'(1)', '單一樣本', '$r=1$ 不可微'],
 'LO6': ['\\inf_{0<V\\le1}L_f(V)/V^2', 'TV 1、χ² 4、KL 2、Hellinger 1', 'Pinsker'],
 'LO7': ['C_f^{\\rm Vajda}>0', '一次方', 'Eq. 9 到 Eq. 11 沒有定理', '不保證'],
 'LO8': ['δ = f(0.8)', 'TV [0.8, 1.2]', 'Hellinger [0.8, 1.2223]', 'KL（|log r|）[0.8, 1.25]', '11 個不同演算法'],
 'LO9': ['\\sum_a\\mu^2f(r)\\ne D_f', '不是無偏估計', 'v_{Atom}=\\mu^2v_{MC}', '沒有 bias 分析'],
 'LO10': ['DAPO 為 (KL, MC)', 'DPPO 為 (TV, Binary)', '|\\pi-\\mu|\\le\\delta', '1\\pm\\delta/\\mu', 'RIPO 為 (χ², Atomic)', '1\\pm\\sqrt{\\delta/\\mu}'],
 'LO11': ['r ≤ 1/µ', '0.0969', '0.0955', '假設性疊加'],
 'LO12': ['+49.82%', '+13.8 分', '+13.93%', '+7.3 分', '1.0 與 2.5', '34.8 < 35.1', '53.8 < 54.4', 'seed'],
}
bad = 0
for k, v in checks.items():
    miss = [s for s in v if s not in t]
    bad += bool(miss)
    print(k, 'OK' if not miss else 'MISSING ' + str(miss))
print('LO self-check:', 12 - bad, '/ 12 OK')
