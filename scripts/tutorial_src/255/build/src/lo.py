"""LO self-check: key terms of each reference answer (design.md section 2) must appear in accept/page.txt (answer boxes stripped)."""
import pathlib
t = (pathlib.Path(__file__).resolve().parents[2] / 'accept/page.txt').read_text()
checks = {
 'LO1': ['p(x_{0:T})=\\prod_{t=0}^{T}p(x_t\\mid x_{\\lt t})', '式 1', '$\\alpha$ 次方再正規化', '\\tau=1/\\alpha', '式 3'],
 'LO2': ['p(x)^\\alpha/p(x\')^\\alpha>p(x)/p(x\')', '\\alpha>1', '\\alpha\\in[1,\\infty]', '兩邊相等'],
 'LO3': ['0.160 對 0.125', '0.160 對 0.250', '0.90', '\\ln2/\\ln1.6\\approx1.475', 'p(ba)=p(bb)'],
 'LO4': ['次方的和', '和的次方', '式 7', '式 8', 'F(x_{\\le t})=\\sum_{x_{>t}}p(x_{>t}\\mid x_{\\le t})^\\alpha', 't=T'],
 'LO5': ['\\varepsilon\'/N^{1-1/\\alpha}\\lt\\varepsilon\\lt\\varepsilon\'', '式 15', '任一條未來', '極端模型', '沒有證明'],
 'LO6': ['不可計算', '正規化常數在分子分母相消', '不可約', '非週期', 'Def. 1', 'Neal 1993', '支撐', '雙向提議機率同時為正'],
 'LO7': ['\\prod_{t=m}^{L}Z_t(x\'_{\\lt t})/Z_t(x_{\\lt t})', 'Z_t=\\sum_v p(v\\mid x_{\\lt t})^\\alpha', '比值為 1 必接受', 'top-k', '\\tau=0.5'],
 'LO8': ['$B$ 個 token，再跑', '重取到目前區塊結尾', '\\pi_{k+1}\\propto p(x_{0:(k+1)B})^\\alpha', '索引錯誤', '不凍結前綴', 'N_{\\text{MCMC}}\\to\\infty', '沒有一致的大小順序'],
 'LO9': ['8.84', '9600', '14.14', '13.48', '14.48', '13.5–14.5', '(T/B)^2/2=6.26', 'K(K+1)/2=10'],
 'LO10': ['0.748 對 0.785、0.706 對 0.740', '0.389 對 0.399、0.318 對 0.354', '62%–77%', '22%', '不是因果歸因', '8 題對低溫 − base 5 題', 'HumanEval 與 AlpacaEval 高於 GRPO'],
 'LO11': ['k>1 時 power sampling 高於 GRPO', '與 base 收斂', 'strictly better than both GRPO and the base model', '0.785 高於 0.748', '大 k 時與 base 相同'],
 'LO12': ['Base 與 Low-temperature 列的溫度', 'Table 1 的 NMCMC', 'AlpacaEval 的 α', 'HumanEval 在 Table 1 的 α', 'EOS 處理', 'GRPO 超參數細節', 'pass@k 樣本的產生方式', '掃描的基準', '提議是否截斷', '誤差棒與種子數'],
}
bad = 0
for k, v in checks.items():
    miss = [s for s in v if s not in t]
    bad += bool(miss)
    print(k, 'OK' if not miss else 'MISSING ' + str(miss))
print('LO self-check:', 12 - bad, '/ 12 OK')
