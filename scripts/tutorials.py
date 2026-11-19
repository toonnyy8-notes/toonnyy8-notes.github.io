import json,re,os,html
P=json.load(open('papers.json')); byid={p['id']:p for p in P}
grad=[p['id'] for p in P if 'Gradient decent' in p['cats']]
T=[
 ('fors-illustrated','只用 Score 的高精度取樣（FORS 圖解）',[273]),
 ('fors-from-zero','從零讀懂 FORS',[273]),
 ('flow-matching-coupling','Flow Matching 配對入門',[214,215,233]),
 ('fourier-features','Fourier Features 入門',[]),
 ('discoloop','DiscoLoop 互動導讀',[79]),
 ('loopcd','LoopCD 入門',[82]),
 ('looped-lm-intro','迴圈式語言模型入門',[p['id'] for p in P if re.search(r'loop|recurrent depth',p['title'],re.I)]),
 ('zeroth-order','零階之路',sorted(set(grad+[122,123,148,200,258]))),
 ('rejection-mh','拒絕取樣與 Metropolis–Hastings 入門',[273,1]),
 ('mala','MALA 入門',[273,146,228]),
]
os.makedirs('out/tutorials',exist_ok=True)
BAR='''<nav id="pm-bar" style="position:fixed;left:16px;bottom:16px;z-index:9999;display:flex;gap:8px;font:500 13px/1 Inter,'Noto Sans TC',system-ui,sans-serif">
<a href="../index.html" style="padding:10px 14px;border-radius:999px;background:#1f1e1c;color:#faf9f5;text-decoration:none;box-shadow:0 4px 16px rgba(0,0,0,.2)">← Paper Master</a>
<a href="index.html" style="padding:10px 14px;border-radius:999px;background:#faf9f5;color:#1f1e1c;text-decoration:none;border:1px solid #ddd9cd;box-shadow:0 4px 16px rgba(0,0,0,.12)">所有教學</a></nav>'''
out=[]
for slug,title,ids in T:
  src=f'tutorials_raw/{slug}.html'
  if not os.path.exists(src): continue
  s=open(src).read()
  s=re.sub(r'</body>',BAR+'\n</body>',s,count=1,flags=re.I) if re.search(r'</body>',s,re.I) else s+BAR
  open(f'out/tutorials/{slug}.html','w').write(s)
  out.append({'file':f'{slug}.html','title':title,'papers':ids})
ax={}
for p in P:
  m=re.search(r'(\d{4}\.\d{4,5})',p['url'])
  if m: ax[m.group(1)]=p['id']
NOTES=[
 ('flow-matching-lm-2026','2026 連續 Flow / Flow Matching 語言模型研究近況',r'flow.*language|continuous.*(language|diffusion)|language.*flow'),
 ('weight-sharing-lora','層間權重共享 × 層級 LoRA 鬆綁',r'loop|recursive transformer|weight shar|lora'),
 ('latent-diffusion-rae','Latent Diffusion：Diffusability 指標與 RAE',r'latent|autoencoder|\brae\b|diffusab|tokeniz'),
 ('blockwise-smc-separation','長序列一致性與 Zero-Shot Diffusion Source Separation + SMC',r'forcing|monte carlo|\bsmc\b|separation|autoregressive video|long video'),
 ('causal-attention-diffusion','Causal Attention × Diffusion：連續序列生成',r'causal|forcing|autoregressive (video|diffusion)|streaming'),
 ('ttt-sequence-modeling','Test-Time Training 作為序列建模架構',r'test[- ]time train|learn at test time|titans|\bttt\b|memoriz.*test'),
]
NT=open('note.template.html').read()
for slug,title,rx in NOTES:
  src=f'notes_raw/{slug}.md'
  if not os.path.exists(src): continue
  md=open(src).read()
  ids=sorted({ax[i] for i in re.findall(r'(\d{4}\.\d{4,5})',md) if i in ax}|{p['id'] for p in P if re.search(rx,p['title'],re.I)})
  rel=('<section class="related"><h2>資料庫中的相關論文</h2>'+''.join(f'<a href="../index.html#paper-{i}">{html.escape(byid[i]["title"])}</a>' for i in ids)+'</section>') if ids else ''
  h1=re.search(r'^#\s+(.+)',md,re.M); pt=h1.group(1) if h1 else title
  open(f'out/tutorials/{slug}.html','w').write(NT.replace('__TITLE__',html.escape(pt)).replace('__RELATED__',rel).replace('__MD__',md.replace('</script','<\\/script')))
  out.append({'file':f'{slug}.html','title':title,'papers':ids,'kind':'note'})
json.dump(out,open('tutorials.json','w'),ensure_ascii=False)
cards=''.join(f'''<a class="card" href="{t['file']}"><h2>{html.escape(t['title'])}</h2><p>{'相關論文：'+'、'.join(html.escape(byid[i]['title']) for i in t['papers'][:3])+(f' 等 {len(t["papers"])} 篇' if len(t['papers'])>3 else '') if t['papers'] else '主題教學（未對應特定論文）'}</p><span>開啟 →</span></a>''' for t in out)
open('out/tutorials/index.html','w').write(f'''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>互動教學</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Noto+Sans+TC:wght@400;500;700&family=Noto+Serif+TC:wght@600&display=swap" rel="stylesheet">
<style>:root{{--bg:#faf9f5;--surface:#fff;--ink:#1f1e1c;--muted:#6b6862;--line:#e4e1da;--accent:#c96442}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#191917;--surface:#232320;--ink:#ecebe7;--muted:#a19e96;--line:#34332f;--accent:#e08a6b}}}}
:root[data-theme=dark]{{--bg:#191917;--surface:#232320;--ink:#ecebe7;--muted:#a19e96;--line:#34332f;--accent:#e08a6b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,"Noto Sans TC",system-ui,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:56px 16px 80px}}a.back{{color:var(--muted);text-decoration:none;font-size:14px}}a.back:hover{{color:var(--accent)}}
h1{{font-family:"Noto Serif TC",serif;font-weight:600;font-size:clamp(28px,4vw,40px);margin:16px 0 8px}}.lead{{color:var(--muted);margin:0 0 36px;max-width:40em}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}
.card{{display:flex;flex-direction:column;gap:10px;padding:22px;border-radius:20px;background:var(--surface);border:1px solid var(--line);color:inherit;text-decoration:none;transition:transform .25s cubic-bezier(.2,0,0,1),box-shadow .25s}}
.card:hover{{transform:translateY(-3px);box-shadow:0 8px 24px -12px rgba(0,0,0,.25)}}.card h2{{font-family:"Noto Serif TC",serif;font-size:19px;font-weight:600;margin:0;line-height:1.35}}
.card p{{margin:0;color:var(--muted);font-size:13px;flex:1}}.card span{{color:var(--accent);font-weight:500;font-size:14px}}</style></head>
<body><main><a class="back" href="../index.html">← Paper Master 論文關聯圖</a><h1>互動教學</h1><p class="lead">從論文清單中收集、由 Claude 製作的互動式教學頁。每頁左下角可以回到關聯圖；在關聯圖點選相關論文時，詳情面板也會列出對應的教學。</p><div class="grid">{cards}</div></main></body></html>''')
print(len(out),'tutorials')
