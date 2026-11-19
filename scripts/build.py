import re,json,os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from anywhere; database files go to the repo root,csv,collections
P=json.load(open('papers_raw.json'))
TAGS={
'Diffusion':r'diffusion|denois|score[- ]based|score matching|\bDDPM|\bDDIM',
'Flow Matching':r'flow match|flow map|rectified flow|optimal transport|transport model|normalizing flow|\bflows?\b',
'Diffusion LM':r'diffusion language|\bd(?:v)?llm|masked diffusion|discrete diffusion|llada|dllm|flow language',
'Video Generation':r'video (?:generation|diffusion)|world rollout|long video|interactive video',
'Image Generation':r'image generation|text-to-image|pixel[- ]space|latent diffusion|autoencoder|inpainting',
'Distillation':r'distill',
'Reinforcement Learning':r'reinforcement|\bRL\b|policy optim|reward|grpo|\bppo\b|on-policy',
'Reasoning':r'reason|chain-of-thought|\bcot\b|thinking',
'Test-Time Scaling':r'test[- ]time',
'Agents':r'agent|harness',
'Self-Improvement':r'self-improv|self-evolv|recursive self|\bRSI\b|self-rewrite|evolving',
'Memory':r'memory|memoriz',
'Continual Learning':r'continual|forgetting|knowledge injection',
'Interpretability':r'interpret|mechanis|probing|circuit|sparse autoencoder|\bSAE|attention sink|steer|monosemantic|superposition|feature',
'Representation Learning':r'representation|jepa|embedding|platonic',
'Learning Theory':r'theor|provabl|grokking|scaling law|generaliz|double descent|manifold hypothesis|learning dynamics',
'Optimization':r'gradient descent|optimizer|\badam|\bmuon|learning rate|zeroth|evolution strateg|loss landscape|sharpness',
'VLM / Multimodal':r'vision[- ]language|\bvlms?\b|multimodal|multi-modal|visual|omni-modal|\bmllm',
'Efficient Inference':r'kv cache|inference|quantiz|compression|speculative|efficient|sparsity|prefill',
'Architecture':r'transformer|state space|mamba|\brnns?\b|recurrent|architecture|backbone|attention|mixture-of-experts|\bmoe\b',
'Looped Models':r'loop|recurrent depth|recursive reasoning|weight sharing|universal transformer',
'Data':r'data curation|synthetic data|data selection|dataset|data reduction|zero data|curat',
'Safety & Alignment':r'alignment|safety|jailbreak|security|attack|reward hacking|monitor|uncertainty|underspecif|ambigu',
'Retrieval':r'retriev|\brag\b',
'3D':r'\b3d\b|mesh|point cloud',
'Robotics / Embodied':r'robot|embodied|\bvla\b|action manifold',
'Energy-Based':r'energy[- ]based|energy matching|\bebms?\b|boltzmann',
'Sampling / MCMC':r'sampling|langevin|monte carlo|mcmc|sampler|metropolis',
'Benchmark':r'benchmark',
'Audio / Music':r'audio|music|speech',
'Post-Training':r'post-train|fine-tun|finetun|\blora\b|sft\b',
'Pretraining':r'pretrain|pre-train|scaling up|100b',
}
CL={
'Diffusion & Generative Models':['diffusion-model','causal-diffusion','Diffusion、Flow Matching 與生成模型','Basic Diffusion','Energy based Models','3D generation','Music generation'],
'Diffusion Language Models':['dLLM','dVLM','Diffusion Language Models（dLLM）'],
'RL, Reasoning & Self-Distillation':['RL','自蒸餾、強化學習與推理','post training','Reward Hacking','Self improving','Evolution Strategies'],
'Agents & Self-Improvement':['Text Gradient (Agent)','Agent、Context Engineering 與 LLM 應用','RAG','Research idea'],
'Memory & Continual Learning':['記憶、Test-Time Training 與持續學習','Memory System','Local Support Learning / Catastrophic Forgetting'],
'Architectures & Efficiency':['Loop LLM','new backbone','高效推論、架構與系統','Hidden state Compression'],
'Interpretability & Theory':['Interpretable AI','Interpretation: High Attention','可解釋性與學習理論','LLM streering','Learning Dynamic','Gradient decent','CV interpretation','Statistics','教材'],
'Vision-Language & Embodied':['VLM 與多模態','VLM: Tool calling','VLM Test time compute','VLM 痛點','Benchmark','Alignment','Physical Robot Learning','Few-shot learning'],
'Training, Data & Representation':['LLM 訓練專題','Data curation & Auto generation','資料篩選、合成資料與資料效率','JEPA 與表徵學習'],
'Safety & Security':['Cybersecurity AI','AI security（各種use case）'],
}
BROAD={'Efficient Inference','Architecture','Reasoning','Interpretability','Representation Learning','VLM / Multimodal','Pretraining','Post-Training','Benchmark','Data','Memory','Agents','Sampling / MCMC','Learning Theory'}
cat2cl={c:k for k,v in CL.items() for c in v}
for i,p in enumerate(P,1):
  p['id']=i
  short=' '.join([p['title'],' '.join(p['cats'])]); txt=short+' '+p['abstract'][:600]
  p['tags']=[t for t,r in TAGS.items() if re.search(r,short if t in BROAD else txt,re.I)] or ['Misc']
  p['cluster']=collections.Counter(cat2cl.get(c,'Training, Data & Representation') for c in p['cats']).most_common(1)[0][0]
  ab=re.sub(r'\\(?:widetilde|tilde|mathrm|mathcal|left|right)\s*(?!\{)','',p['abstract']); ab=re.sub(r'\\[a-zA-Z]+\{([^}]*)\}',r'\1',ab); ab=re.sub(r'\\([a-zA-Z]+)',r'\1',ab).replace('$','')
  s=re.split(r'(?<=[.!?])\s',ab,1)[0] if ab else (p['note'] or '')
  p['summary']=(s[:240]+'…') if len(s)>240 else s
  p['branch']=sorted({re.search(r'\((.*?)\)',x).group(1) for x in p['sources']})
# ponytail: tag-Jaccard + shared-category bonus, O(n^2) over ~340 papers; swap for embeddings if the list grows past a few thousand
df=collections.Counter(t for p in P for t in p['tags'])
for p in P:
  sc=[]
  for q in P:
    if q is p: continue
    a,b=set(p['tags']),set(q['tags']); inter=a&b
    s=sum(1/df[t]**.5 for t in inter)/max(1,len(a|b))**.5 if inter else 0
    if set(p['cats'])&set(q['cats']): s+=.35
    elif p['cluster']==q['cluster']: s+=.1
    if s>.25: sc.append((s,q['id']))
  p['related']=[i for s,i in sorted(sc,reverse=True)[:6]]
deg=collections.Counter(i for p in P for i in p['related'])
for p in P: p['degree']=deg[p['id']]
with open('../papers.csv','w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['Paper ID','Title','Authors','Year','URL','Source File & Branch','Source Category','Cluster','Core Topic / Summary','Tags','Related Paper IDs'])
  for p in P: w.writerow([p['id'],p['title'],', '.join(p['authors'][:6])+(' et al.' if len(p['authors'])>6 else ''),p['year'],p['url'],'; '.join(p['sources']),'; '.join(p['cats']),p['cluster'],p['summary'],'; '.join(p['tags']),', '.join(map(str,p['related']))])
keep=['id','title','url','authors','year','sources','branch','cats','cluster','tags','summary','related','degree']
data=[{k:p[k] for k in keep} for p in P]
for d in data: d['authors']=d['authors'][:8]
json.dump(data,open('../papers.json','w'),ensure_ascii=False)
print(collections.Counter(p['cluster'] for p in P).most_common()); print(df.most_common())
