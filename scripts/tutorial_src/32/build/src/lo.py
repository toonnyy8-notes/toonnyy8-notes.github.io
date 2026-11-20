import os
t=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../accept/page.txt')).read()
checks={'LO1':['一個基底 LLM','參數分開','Challenger','Solver','輪流訓練','不使用任何外部題目或標籤'],
'LO2':['出現最多次的答案','不必過半','相同的答案比例','一致性，不是對真實答案的正確率'],
'LO3':['1-2|\\hat p-\\frac12|','$c=5$ 得 1','$c=10$ 得 0','$c=1$ 得 0.2','至少出現 1 次'],
'LO4':['\\hat p(1-\\hat p)/(2\\beta^2)','二元獎勵的變異數','熵正則化的最佳策略','真實答案的成功率','多數決一致率','只說明最大值的位置'],
'LO5':['對同一提示取','(R_i-\\text{mean})/(\\text{std}+\\varepsilon_{\\text{norm}})','所有 token 共用','策略梯度項為零','KL 項仍在'],
'LO6':['缺 <question> 標籤的輸出直接為 0','d_{ij}=1-\\mathrm{BLEU}','平均連結分群','\\tau_{\\text{BLEU}}=0.5','r_{\\text{rep}}=\\lambda|C_k|/B','\\lambda=1','r_i=\\max(0,r_{\\text{uncertainty}}-r_{\\text{rep}})'],
'LO7':['N=8{,}000','m=10','\\delta=0.25','3–7 個','去掉太易','低一致性'],
'LO8':['某個錯誤答案比正確答案更常見','多數答案是錯的','79.0%、69.0%、63.0%'],
'LO9':['答案等於偽標籤得 1，否則得 0','每題 5 個 rollout','Challenger 訓練 5 步','Solver 訓練 15 步','45 步（3 輪）'],
'LO10':['42.58 升到 49.07','+6.49','42.57 → 49.93','未訓練 Challenger 出的題目訓練','45.01','Challenger 的 RL 是否必要'],
'LO11':['49.07／31.15','45.76／28.73','47.35／26.69'],
'LO12':['Step 15、30、45','第 1、2、3 輪','70.6%、55.2%、48.8%','沒有共同的門檻','模型崩潰','Single-R-Zero']}
bad=0
for k,v in checks.items():
    miss=[s for s in v if s not in t]; bad+=bool(miss); print(k,'OK' if not miss else 'MISSING '+str(miss))
print('LO self-check:', 12-bad, '/ 12')
