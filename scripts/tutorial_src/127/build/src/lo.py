"""LO self-check: key terms of each reference answer (design.md section 2) must appear in accept/page.txt (answer boxes stripped)."""
import pathlib
t = (pathlib.Path(__file__).resolve().parents[2] / 'accept/page.txt').read_text()
checks = {
 'LO1': ['fail@128', '128 次全錯', '溫度 1.0', 'A(z_i)=0', '更新方向是 0', '120.0', '20.0', '10.0', '只教「用 boxed 寫出答案」的格式'],
 'LO2': ['(1-p)^{128}=0.774', '1-(1-p)^{32}=0.062', '仍可能偶爾答對'],
 'LO3': ['都從同一個預訓練模型初始化', '看不到難題', '正確率進步'],
 'LO4': ['\\theta\'(\\mathcal X)=\\text{RL-update}(\\theta,\\mathcal X)', '外層用 RLOO、學生進步只當黑箱獎勵', '不需要對內層求導'],
 'LO5': ['R(\\mathcal X_d)=\\mathrm{Acc}(\\pi^S_{\\theta\'_d}(Q_R))-\\mathrm{Acc}(\\pi^S_\\theta(Q_R))', '貪婪解碼在 64 道', '減基準讓不同步的獎勵可比', 'r=4$ 個學生降低變異'],
 'LO6': ['3 步移動平均超過', '\\tau=0.01$ 時，先選獎勵最高的資料集', '選獎勵最高的資料集', '獎勵為中位數', '所有觸發晉升的資料集的聯集', '學生每次內層後回到基準'],
 'LO7': ['\\pi_0(S_{acc})>0', '\\ln\\pi(z)=\\ln\\pi_0(z)-\\ln\\pi_0(S_{acc})', 'RLOO 的優勢總和為 0', 'REINFORCE 的優勢總和不為 0'],
 'LO8': ['PS 直接用晉升學生推論', 'PQ 用 PQ＋fail@128 訓練集從頭訓練新學生', '幸運的訓練軌跡', '4 倍取樣量'],
 'LO9': ['18.9 − 9.6 = 9.3', '百分點', '(18.9 − 9.6)/(22.0 − 9.6) = 0.75', '分母兩者都用 MATH train (Full)', '只高 2.8 個百分點'],
 'LO10': ['MATH pass@32 是 8.3%', '基礎模型是 7.5%', '6.8% 對 6.4%', '大致獨立'],
 'LO11': ['\\bar s_i=0$', 'u_i=0$', 'u_i=1-\\bar s_i$', '$u_i=1-\\bar s_i$', '32 個學生解答', '10.82', '34.91', '31.99', '34.66', '三個種子中有一個在所有資料集上崩潰'],
 'LO12': ['32.8%', '64.6%', '63%', '55.5%', 'Correct-only（82 題）', 'Well-posed（144 題）', '表現提高', '只在 HARP 上做，3 個種子'],
}
bad = 0
for k, v in checks.items():
    miss = [s for s in v if s not in t]
    bad += bool(miss)
    print(k, 'OK' if not miss else 'MISSING ' + str(miss))
print('LO self-check:', 12 - bad, '/ 12 OK')
