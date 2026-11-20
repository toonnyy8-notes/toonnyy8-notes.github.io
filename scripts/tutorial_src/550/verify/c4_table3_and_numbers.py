# Table 3 (paper) data embedded in C4 + derived deltas; arithmetic checks of numbers quoted in the design.
T3 = {  # benchmark: (category, [R1-Zero, Dev1, Dev2, Dev3, R1])
 'MMLU (EM)':('English',[88.8,89.1,91.2,91.0,90.8]),
 'MMLU-Redux (EM)':('English',[85.6,90.0,93.0,93.1,92.9]),
 'MMLU-Pro (EM)':('English',[68.9,74.1,83.8,83.1,84.0]),
 'DROP (3-shot F1)':('English',[89.1,89.8,91.1,88.7,92.2]),
 'IF-Eval (Prompt Strict)':('English',[46.6,71.7,72.0,78.1,83.3]),
 'GPQA Diamond (Pass@1)':('English',[75.8,66.1,70.7,71.2,71.5]),
 'SimpleQA (Correct)':('English',[30.3,17.8,28.2,24.9,30.1]),
 'FRAMES (Acc.)':('English',[82.3,78.5,81.8,81.9,82.5]),
 'AlpacaEval2.0 (LC-winrate)':('English',[24.7,50.1,55.8,62.1,87.6]),
 'ArenaHard (GPT-4-1106)':('English',[53.6,77.0,73.2,75.6,92.3]),
 'LiveCodeBench (Pass@1-COT)':('Code',[50.0,57.5,63.5,64.6,65.9]),
 'Codeforces (Percentile)':('Code',[80.4,84.5,90.5,92.1,96.3]),
 'Codeforces (Rating)':('Code',[1444,1534,1687,1746,2029]),
 'SWE Verified (Resolved)':('Code',[43.2,39.6,44.6,45.6,49.2]),
 'Aider-Polyglot (Acc.)':('Code',[12.2,6.7,25.6,44.8,53.3]),
 'AIME 2024 (Pass@1)':('Math',[77.9,59.0,74.0,78.1,79.8]),
 'MATH-500 (Pass@1)':('Math',[95.9,94.2,95.9,95.4,97.3]),
 'CNMO 2024 (Pass@1)':('Math',[88.1,58.0,73.9,77.3,78.8]),
 'CLUEWSC (EM)':('Chinese',[93.1,92.8,92.6,91.6,92.8]),
 'C-Eval (EM)':('Chinese',[92.8,85.7,91.9,86.4,91.8]),
 'C-SimpleQA (Correct)':('Chinese',[66.4,58.8,64.2,66.9,63.7]),
}
ST=['R1-Zero','Dev1','Dev2','Dev3','R1']
print('== stage-to-stage deltas (points)')
for b,(c,v) in T3.items():
    d=[round(v[i+1]-v[i],1) for i in range(4)]
    print(f'{b:30s} {c:8s} {v}  d={d}')
print('AlpacaEval Dev3->R1: %.1f pts (%.1f%% rel); ArenaHard: %.1f pts (%.1f%% rel)'%(87.6-62.1,(87.6/62.1-1)*100,92.3-75.6,(92.3/75.6-1)*100))
print('== arithmetic checks')
print('batch 32x16 =',32*16,'; 8192/16 minibatches =',8192/16,'; 8192/16 outputs per q -> questions per rollout =',8192//16)
print('R1-Zero: 10400 steps x 32 q =',10400*32,'question-visits; /1.6 epochs =',10400*32/1.6)
print('Table 4 reasoning prompts 26K+17K+22K+15K =',26+17+22+15,'K (+8K bug fixing mentioned in B.3.1)')
print('GPU hours: 64*8*198 =',64*8*198,'; 64*8*80 =',64*8*80,'; total 101K+5K+41K =',101+5+41,'K; USD x2 =',2*147,'K')
print('SFT Table 5 sum =',395285+211129+10124+10395+177812,'; reasoning (math+code+stem+logic) =',395285+211129+10124+10395,'; general =',177812)
print('LC reward example: 18 target-language words of 20 ->',18/20)
print('MATH level-1 43 of 500; 95-97% of 43 ->',[round(43*x,2) for x in (0.95,0.97)],'correct, i.e. unsolved', [round(43*(1-x),2) for x in (0.95,0.97)])
print('pass@1 = mean of k binary: e.g. 50 correct of 64 ->',50/64)
print('Table 16: Distill-32B minus Qwen2.5-32B-Zero AIME pass@1 =',round(72.6-47.0,1))
print('Jailbreak gap R1: 85.9-25.2 =',round(85.9-25.2,1))
print('== C4 round1 stage card: compare with the paper Sec. 4 comparison node; integer tenths; list only |delta| >= 1.0 point')
BASE={1:(0,'R1-Zero (not same training line)'),2:(1,'Dev1 (same line)'),3:(2,'Dev2 (data source only, not same line)'),4:(3,'Dev3 (same line)')}
for i in range(1,5):
    b0,lab=BASE[i]
    d=[(b,(round(v[i]*10)-round(v[b0]*10))) for b,(c,v) in T3.items() if 'Rating' not in b]
    up=[(b,x/10) for b,x in sorted(d,key=lambda x:-x[1]) if x>=10][:3]
    dn=[(b,x/10) for b,x in sorted(d,key=lambda x:x[1]) if x<=-10][:3]
    print(ST[i],'vs',lab,'| up',up,'| down',dn)
