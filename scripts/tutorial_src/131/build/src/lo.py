"""LO self-check: key terms of each reference answer (design.md section 2) must appear in accept/page.txt."""
import pathlib
t = (pathlib.Path(__file__).resolve().parents[2] / 'accept/page.txt').read_text()
checks = {
 'LO1': ['s_t=(x,y_{\\lt t})', '確定性', '終端獎勵', '\\lvert R(x,y)\\rvert\\le1', 'J(\\pi)'],
 'LO2': ['\\mathrm{std}', 'r_t^i', '1+\\epsilon_{\\text{high}}', '1-\\epsilon_{\\text{low}}', '\\nabla(\\mathrm{sg}(r)\\log\\pi)=r\\nabla\\log\\pi=\\nabla r', 'M=0'],
 'LO3': ['Q^\\pi(s_t,y_t)=V^\\pi(s_{t+1})', 'A^\\mu(s_t,y_t)=V^\\mu(s_{t+1})-V^\\mu(s_t)', '\\sum_y\\mu(y\\mid s)A^\\mu(s,y)=0'],
 'LO4': ['\\exp\\big(\\eta A^\\mu(s_t,y_t)\\big)', 'Z_\\mu(s_t)', '\\eta\\to0', '\\pi^+\\to\\mu'],
 'LO5': ['\\log Z_\\mu(s_t)=D_{\\mathrm{KL}}', '式 (10)', '\\eta V^\\mu(s_t)', '方向相反'],
 'LO6': ['R(x,y)-V^\\mu(x)', '每一條', '\\delta(x,y;\\mu,\\mu)=\\eta(R(x,y)-V^\\mu(x))', '(0.75, −0.25, −0.25, −0.25)', '0.09375'],
 'LO7': ['$\\Pi_\\mu$ ​ 上', '可達', '完成分佈', '\\mathbb E_\\mu[\\delta^2]=\\sum_t\\mathbb E_\\mu[D_t^2]', 'c(s)=1', '沒有保證參數化的 LLM'],
 'LO8': ['定理 1 只涵蓋精確目標 (18)', '不涵蓋 (14)', '沒有誤差界', '組內平均', '簡化 KL', '遮罩', '(−0.125, −0.125, 0)', '論文沒有分析', '$\\eta$'],
 'LO9': ['\\frac{1-\\mu(y\\mid s)}{1-\\pi(y\\mid s)}', '精確', '\\omega=r=1', '忽略 rollout 與訓練引擎的數值差異', 'REINFORCE', '判斷量由 r $r$ 換成 ω'],
 'LO10': ['r = 1.0102', 'ω = 1.0909', 'r = 2.0、ω = 1.0093', '[0.0909,11]', 'C = 3', '低機率', '高機率'],
 'LO11': ['50.5', '39.5、43.5、47.4、46.4', '11.0、7.0、3.1、4.1', '49.4%', '45.5%', '同一組 AIME 題目', '誤差線', '24.1', '25.3、25.4、25.8', '20.5'],
 'LO12': ['ϵ = 0.1 與 C = 3.0', '沒有 KL 懲罰', 'seq-mean-token-mean', '1e−6', 'ϵ_low／ϵ_high', 'std = 0', '母體還是樣本'],
}
bad = 0
for k, v in checks.items():
    miss = [s for s in v if s not in t]
    bad += bool(miss)
    print(k, 'OK' if not miss else 'MISSING ' + str(miss))
print('LO self-check:', 12 - bad, '/ 12 OK')
