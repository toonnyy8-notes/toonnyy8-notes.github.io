"""LO self-check: key terms of each reference answer (design.md section 2) must appear in accept/page.txt (answer boxes stripped)."""
import pathlib
t = (pathlib.Path(__file__).resolve().parents[2] / 'accept/page.txt').read_text()
checks = {
 'LO1': ['\\alpha_t x+(1-\\alpha_t)\\pi', '\\alpha_1=0', '\\pi=m', '\\pi=\\mathbf 1/K', '遞減'],
 'LO2': ['x_\\theta(z_t,t)', '就得到 ancestral sampler', 'NFE 等於步數', '\\mathrm{Cat}(x)', '已解開的 token 不再更新', '允許 token 改成任何值'],
 'LO3': ['式 (11) 由兩項組成', '$\\kappa_t q_{s|t}(\\cdot|z_t,x)$', '$(1-\\kappa_t)q_s(\\cdot|x)$', '標準 posterior', '0.625、0.505、0.385', '邊際始終不變'],
 'LO4': ['加權後都等於', '凸組合仍是', '(i) 用真', '真 posterior', '(iii) 第二項不依賴', '(iv)', '不再保證邊際'],
 'LO5': ['式 (12) 替換了兩處', '$q_{0|t}(\\cdot|z_t,x_\\theta)$', 'q_{0|t}=x_\\theta', 'carry-over'],
 'LO6': ['已解開 token 以', '(1-\\kappa_t)(1-\\alpha_s)', '的機率變回 [MASK]', '$(1-\\kappa_t)(1-\\alpha_s)/K$'],
 'LO7': ['$\\kappa_t=1-\\sigma_t/(1-\\alpha_s)$', '\\sigma_t>1-\\alpha_s', '\\kappa_t\\lt0', '最後幾步', '$\\kappa_t\\in[0,1]$', '$[1-2\\eta,1-\\eta]$'],
 'LO8': ['去噪器錯誤率隨正確位置比例下降', '早期以高錯誤率解開的 token 不再更新', '把 token 變回 [MASK]', '錯誤率較低的去噪器重新預測', '步數越多，校正次數越多', '不對應論文的 NFE'],
 'LO9': ['34.75 (5.35)', '19.21 (5.17)', '$p=0.9$ 與 rescale', '63.70', '110.75', '131.85', '32 步時沒有增益'],
 'LO10': ['25.63 降到 15.05', '逐步數挑最佳超參數的包絡', '選錯 κ 比 ancestral 差', '105.75', '23.93 而非 24.73', '取樣 cosine、4096 步'],
 'LO11': ['另外單獨抽', '$N=K-1$', '$U^{(1)}=V^{1/N}$', 'N-i', 'U[0,U^{(i)}]', '\\tilde\\sigma_t\\Phi^{-1}', '值與索引在分佈上精確', '個 embedding（式 13）與', '是近似'],
 'LO12': ['K-k-1', '另加乾淨 token 的', 'e^{\\tilde w/\\tau}', 'case 2 有', '(K-k)\\mu', '94.3→63.4 GiB', '81.8→121.9 samples/s', 'OWT PPL 25.2→25.2', 'k=2'],
}
bad = 0
for k, v in checks.items():
    miss = [s for s in v if s not in t]
    bad += bool(miss)
    print(k, 'OK' if not miss else 'MISSING ' + str(miss))
print('LO self-check:', 12 - bad, '/ 12 OK')
