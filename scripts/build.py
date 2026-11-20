import re,json,os,csv,collections,sqlite3
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # run from anywhere; database files go to the repo root
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
'Architecture':r'convnet|convolutional|length extrapolation|long-context|transformer|state space|mamba|\brnns?\b|recurrent|architecture|backbone|attention|mixture-of-experts|\bmoe\b',
'Looped Models':r'loop|recurrent depth|recursive reasoning|weight sharing|universal transformer',
'Data':r'data curation|synthetic data|data selection|dataset|data reduction|zero data|curat',
'Safety & Alignment':r'alignment|safety|jailbreak|security|attack|reward hacking|monitor|uncertainty|underspecif|ambigu',
'Retrieval':r'retriev|\brag\b',
'3D':r'\b3d\b|mesh|point cloud',
'Robotics / Embodied':r'robot|embodied|\bvla\b|action manifold',
'Energy-Based':r'energy[- ]based|energy matching|\bebms?\b|boltzmann',
'Sampling / MCMC':r'sampling|langevin|monte carlo|mcmc|sampler|metropolis',
'Benchmark':r'benchmark',
'Audio / Music':r'audio|music|speech|voice|text-to-speech|\btts\b|vocoder|singing|phase reconstruction',
'GAN':r'adversarial net|\bgans?\b|stargan|seqgan|cyclegan',
'VAE':r'variational bayes|variational auto|\bvaes?\b',
'Self-Supervised':r'self-supervised|semi-supervised|contrastive|bootstrap|noisy student|siamese|自監督|半監督',
'Meta-Learning':r'meta-learn|few-shot|learning to learn|\bmaml\b|元學習',
'Post-Training':r'post-train|fine-tun|finetun|\blora\b|sft\b',
'Pretraining':r'pretrain|pre-train|scaling up|100b',
}
CL={
'Diffusion & Generative Models':['擴散模型：基礎與設計空間', '擴散模型：引導', '擴散模型：取樣器與 ODE 求解', '擴散模型：蒸餾與少步生成', '擴散模型：曝光偏差', '擴散模型：潛空間、反演與編輯', '擴散模型：最佳傳輸與耦合', '擴散模型：逆問題與 Plug-and-Play', '擴散模型 × 強化學習', '擴散模型應用：影像、3D 與多模態', 'diffusion-model', 'causal-diffusion', 'Diffusion、Flow Matching 與生成模型', 'Basic Diffusion', '3D generation'],
'Diffusion Language Models':['離散擴散與擴散語言模型', '非自回歸文字生成', 'dLLM', 'dVLM', 'Diffusion Language Models（dLLM）'],
'RL, Reasoning & Self-Distillation':['大型語言模型：推理與代理', '大型語言模型：對齊、編輯與解碼時對齊', '離線強化學習與決策模型', 'RL', '自蒸餾、強化學習與推理', 'post training', 'Reward Hacking', 'Self improving', 'Evolution Strategies'],
'Agents & Self-Improvement':['Text Gradient (Agent)', 'Agent、Context Engineering 與 LLM 應用', 'RAG', 'Research idea'],
'Memory & Continual Learning':['測試時訓練與長期記憶', '記憶、Test-Time Training 與持續學習', 'Memory System', 'Local Support Learning / Catastrophic Forgetting'],
'Architectures & Efficiency':['大型語言模型：高效訓練、推論與長度外推', 'Loop LLM', 'new backbone', '高效推論、架構與系統', 'Hidden state Compression'],
'Interpretability & Theory':['Interpretable AI', 'Interpretation: High Attention', '可解釋性與學習理論', 'LLM streering', 'Learning Dynamic', 'Gradient decent', 'CV interpretation', 'Statistics', '教材'],
'Vision-Language & Embodied':['VLM 與多模態', 'VLM: Tool calling', 'VLM Test time compute', 'VLM 痛點', 'Benchmark', 'Alignment', 'Physical Robot Learning', 'Few-shot learning'],
'Training, Data & Representation':['自監督學習', '半監督學習', '元學習', '大型語言模型：微調', '文字風格轉換', '情感支持對話', '其他', 'LLM 訓練專題', 'Data curation & Auto generation', '資料篩選、合成資料與資料效率', 'JEPA 與表徵學習'],
'Safety & Security':['Cybersecurity AI', 'AI security（各種use case）'],
'Speech & Audio':['語音：合成、轉換與聲碼器', 'Music generation'],
'GAN, VAE, Flows & EBM':['GAN', 'VAE', 'Normalizing Flows', '能量模型', 'Neural ODE 與連續時間模型', 'Energy based Models'],
}
BROAD={'GAN','VAE','Self-Supervised','Meta-Learning','Efficient Inference','Architecture','Reasoning','Interpretability','Representation Learning','VLM / Multimodal','Pretraining','Post-Training','Benchmark','Data','Memory','Agents','Sampling / MCMC','Learning Theory'}
cat2cl={c:k for k,v in CL.items() for c in v}
# Paper IDs are permanent (tutorials and notes link to #paper-<id>): keep the id from the previous papers.jsonl, append new papers after the max.
AXID=re.compile(r'(\d{4}\.\d{4,5})')
pkey=lambda url:(AXID.search(url).group(1) if re.search(r'arxiv\.org|alphaxiv\.org|huggingface\.co/papers',url) and AXID.search(url) else url.split('#')[0].rstrip('/'))
prev={pkey(q['url']):q['id'] for q in map(json.loads,open('../papers.jsonl'))} if os.path.exists('../papers.jsonl') else {}
nxt=max(prev.values(),default=0)
for p in P:
  k=p['arxiv'] or p['key']
  if k in prev: p['id']=prev[k]
  else: nxt+=1; p['id']=nxt
assert len({p['id'] for p in P})==len(P), 'duplicate paper ids'
P.sort(key=lambda p:p['id'])
for p in P:
  short=' '.join([p['title'],' '.join(p['cats'])]); txt=short+' '+p['abstract'][:600]
  p['tags']=[t for t,r in TAGS.items() if re.search(r,short if t in BROAD else txt,re.I)] or ['Misc']
  p['cluster']=collections.Counter(cat2cl.get(c,'Training, Data & Representation') for c in p['cats']).most_common(1)[0][0]
  ab=re.sub(r'\\(?:widetilde|tilde|mathrm|mathcal|left|right)\s*(?!\{)','',p['abstract']); ab=re.sub(r'\\[a-zA-Z]+\{([^}]*)\}',r'\1',ab); ab=re.sub(r'\\([a-zA-Z]+)',r'\1',ab).replace('$','')
  s=re.split(r'(?<=[.!?])\s',ab,1)[0] if ab else (p['note'] or '')
  p['summary']=(s[:240]+'…') if len(s)>240 else s
  p['branch']=sorted({re.search(r'\((.*?)\)',x).group(1) for x in p['sources']})
# Related papers. Score = citation link (either direction) + shared keywords weighted by rarity (idf) + a little for shared tags / category.
# ponytail: O(n^2) over ~600 papers; swap for embeddings if the list grows past a few thousand.
import math
KW={int(k):v for k,v in json.load(open('keywords.json')).items()}
# Model/method abbreviations (abbrevs.json, exact casing): added to keywords (lowercase) and, after scoring, to tags.
ABBR={int(k):v for k,v in json.load(open('abbrevs.json')).items()} if os.path.exists('abbrevs.json') else {}
for i,ab in ABBR.items(): KW[i]=KW.get(i,[])+[a.lower() for a in ab if a.lower() not in KW.get(i,[])]
CITE=[tuple(e) for e in json.load(open('citations.json'))] if os.path.exists('citations.json') else []
ids={p['id'] for p in P}
refs=collections.defaultdict(set); cited=collections.defaultdict(set)
for a,b in CITE:
  if a in ids and b in ids: refs[a].add(b); cited[b].add(a)
kdf=collections.Counter(k for p in P for k in KW.get(p['id'],[]))
idf={k:math.log(len(P)/n) for k,n in kdf.items()}
tdf=collections.Counter(t for p in P for t in p['tags'])
for p in P:
  p['keywords']=KW.get(p['id'],[])
  p['refs']=sorted(refs[p['id']]); p['cited_by']=sorted(cited[p['id']])
for p in P:
  sc=[]; kp=set(p['keywords'])
  for q in P:
    if q is p: continue
    s=1.5 if (q['id'] in refs[p['id']] or q['id'] in cited[p['id']]) else 0
    s+=sum(idf[k] for k in kp&set(q['keywords']))/4
    inter=set(p['tags'])&set(q['tags'])
    s+=.3*sum(1/tdf[t]**.5 for t in inter)/max(1,len(set(p['tags'])|set(q['tags'])))**.5 if inter else 0
    if set(p['cats'])&set(q['cats']): s+=.2
    if s>.6: sc.append((s,q['id']))
  p['related']=[i for s,i in sorted(sc,reverse=True)[:6]]
deg=collections.Counter(i for p in P for i in p['related'])
for p in P: p['degree']=deg[p['id']]
for p in P: p['abbr']=ABBR.get(p['id'],[]); p['tags']+=[a for a in p['abbr'] if a not in p['tags']]
with open('../papers.csv','w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['Paper ID','Title','Authors','Year','URL','Source File & Branch','Source Category','Cluster','Core Topic / Summary','Tags','Keywords','Cites (Paper IDs)','Cited By (Paper IDs)','Related Paper IDs'])
  for p in P: w.writerow([p['id'],p['title'],', '.join(p['authors'][:6])+(' et al.' if len(p['authors'])>6 else ''),p['year'],p['url'],'; '.join(p['sources']),'; '.join(p['cats']),p['cluster'],p['summary'],'; '.join(p['tags']),'; '.join(p['keywords']),', '.join(map(str,p['refs'])),', '.join(map(str,p['cited_by'])),', '.join(map(str,p['related']))])
keep=['id','title','url','authors','year','sources','branch','cats','cluster','tags','abbr','keywords','summary','refs','cited_by','related','degree']
data=[{k:p[k] for k in keep} for p in P]
for d in data: d['authors']=d['authors'][:8]
with open('../papers.jsonl','w') as f:  # one paper per line, so git diffs show only the papers that changed
  for d in data: f.write(json.dumps(d,ensure_ascii=False)+'\n')
if os.path.exists('../papers.db'): os.remove('../papers.db')
db=sqlite3.connect('../papers.db')  # same data, normalised for SQL: many-valued fields get their own tables
db.executescript("""CREATE TABLE papers(id INTEGER PRIMARY KEY,title TEXT,url TEXT,year TEXT,cluster TEXT,summary TEXT,degree INTEGER);
CREATE TABLE authors(paper_id INTEGER REFERENCES papers,pos INTEGER,name TEXT);
CREATE TABLE tags(paper_id INTEGER REFERENCES papers,tag TEXT);
CREATE TABLE keywords(paper_id INTEGER REFERENCES papers,keyword TEXT);
CREATE TABLE citations(citing_id INTEGER REFERENCES papers,cited_id INTEGER REFERENCES papers);
CREATE TABLE sources(paper_id INTEGER REFERENCES papers,source TEXT,branch TEXT);
CREATE TABLE categories(paper_id INTEGER REFERENCES papers,category TEXT);
CREATE TABLE related(paper_id INTEGER REFERENCES papers,related_id INTEGER REFERENCES papers,rank INTEGER);""")
for p in P:
  db.execute('INSERT INTO papers VALUES(?,?,?,?,?,?,?)',(p['id'],p['title'],p['url'],p['year'],p['cluster'],p['summary'],p['degree']))
  db.executemany('INSERT INTO authors VALUES(?,?,?)',[(p['id'],i,a) for i,a in enumerate(p['authors'])])
  db.executemany('INSERT INTO tags VALUES(?,?)',[(p['id'],t) for t in p['tags']])
  db.executemany('INSERT INTO keywords VALUES(?,?)',[(p['id'],k) for k in p['keywords']])
  db.executemany('INSERT INTO citations VALUES(?,?)',[(p['id'],r) for r in p['refs']])
  db.executemany('INSERT INTO sources VALUES(?,?,?)',[(p['id'],s,re.search(r'\((.*?)\)',s).group(1)) for s in p['sources']])
  db.executemany('INSERT INTO categories VALUES(?,?)',[(p['id'],c) for c in p['cats']])
  db.executemany('INSERT INTO related VALUES(?,?,?)',[(p['id'],r,i+1) for i,r in enumerate(p['related'])])
db.commit(); db.close()
print(collections.Counter(p['cluster'] for p in P).most_common()); print(tdf.most_common()); print(len(CITE),"citations;",len(kdf),"keywords")
