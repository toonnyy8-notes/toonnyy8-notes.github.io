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
import yaml,glob
NT=open('note.template.html').read()
notes={}
for f in sorted(glob.glob('notes_split/*.md')):
  raw=open(f).read()
  _,fm,md=raw.split('---',2)
  m=yaml.safe_load(fm); m['md']=md.strip(); notes[m['slug']]=m
IDX=json.load(open('notes_split/index.json'))
hubkids={g['hub']:[n['slug'] for n in g['notes'] if n['slug']!=g['hub']] for g in IDX}
order=list(dict.fromkeys(n['slug'] for g in IDX for n in g['notes']))
order=[s for s in order if s in notes]+[s for s in notes if s not in order]
for n in notes.values():
  n['papers']=sorted({ax[str(i)] for i in (n.get('arxiv') or []) if str(i) in ax})
for n in notes.values():  # hubs inherit their children's papers
  if n.get('kind')=='hub':
    n['papers']=sorted(set(n['papers']).union(*[notes[r]['papers'] for r in n.get('related') or [] if r in notes]))
for slug in order:
  n=notes[slug]
  rn=[r for r in n.get('related') or [] if r in notes and r!=slug]
  backs=[o for o in order if slug in (notes[o].get('related') or []) and o not in rn and o!=slug]
  rel=''
  if rn or backs:
    rel+='<section class="related"><h2>相關筆記</h2>'+''.join(f'<a href="{r}.html">{html.escape(notes[r]["title"])}</a>' for r in rn+backs)+'</section>'
  if n['papers']:
    rel+='<section class="related"><h2>資料庫中的相關論文</h2>'+''.join(f'<a href="../index.html#paper-{i}">{html.escape(byid[i]["title"])}</a>' for i in n['papers'])+'</section>'
  kick=('主題總覽' if n.get('kind')=='hub' else '研究筆記')+' · '+html.escape(n.get('cluster',''))
  open(f'out/tutorials/{slug}.html','w').write(NT.replace('__TITLE__',html.escape(n['title'])).replace('<div class="kicker">研究筆記</div>',f'<div class="kicker">{kick}</div>').replace('__RELATED__',rel).replace('__MD__',n['md'].replace('</script','<\\/script')))
  out.append({'file':f'{slug}.html','title':n['title'],'papers':n['papers'],'kind':n.get('kind','topic'),'cluster':n.get('cluster',''),'related':rn})
json.dump(out,open('tutorials.json','w'),ensure_ascii=False)
inter=[t for t in out if 'kind' not in t]
cards=''.join(f'''<a class="card" href="{t['file']}"><h2>{html.escape(t['title'])}</h2><p>{'相關論文：'+'、'.join(html.escape(byid[i]['title']) for i in t['papers'][:3])+(f' 等 {len(t["papers"])} 篇' if len(t['papers'])>3 else '') if t['papers'] else '主題教學（未對應特定論文）'}</p><span>開啟 →</span></a>''' for t in inter)
hubs=[t for t in out if t.get('kind')=='hub']
seen=set(t['file'][:-5] for t in hubs)
byslug={t['file'][:-5]:t for t in out if t.get('kind')}
def grp(h):
  kids=[byslug[r] for r in hubkids.get(h['file'][:-5],[]) if r in byslug]
  seen.update(k['file'][:-5] for k in kids)
  return f'''<section class="hub"><a class="hubt" href="{h['file']}"><span>{html.escape(h['cluster'])}</span><h3>{html.escape(h['title'])}</h3></a><ol>'''+''.join(f'<li><a href="{k["file"]}">{html.escape(k["title"])}</a></li>' for k in kids)+'</ol></section>'
notes_html=''.join(grp(h) for h in hubs)
rest=[t for t in out if t.get('kind')=='topic' and t['file'][:-5] not in seen]
if rest: notes_html+='<section class="hub"><div class="hubt"><h3>其他筆記</h3></div><ol>'+''.join(f'<li><a href="{k["file"]}">{html.escape(k["title"])}</a></li>' for k in rest)+'</ol></section>'
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
.card p{{margin:0;color:var(--muted);font-size:13px;flex:1}}.card span{{color:var(--accent);font-weight:500;font-size:14px}}
h2.sec{{font-family:"Noto Serif TC",serif;font-weight:600;font-size:24px;margin:56px 0 16px}}
.hubs{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}}
.hub{{padding:22px;border-radius:20px;background:var(--surface);border:1px solid var(--line)}}
.hubt{{display:block;color:inherit;text-decoration:none}}.hubt span{{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}}
.hubt h3{{font-family:"Noto Serif TC",serif;font-size:19px;font-weight:600;margin:6px 0 10px;line-height:1.35}}a.hubt:hover h3{{color:var(--accent)}}
.hub ol{{margin:0;padding-left:1.4em;font-size:14px}}.hub li{{margin:6px 0}}.hub li::marker{{color:var(--muted)}}.hub li a{{color:inherit;text-decoration:none}}.hub li a:hover{{color:var(--accent)}}</style></head>
<body><main><a class="back" href="../index.html">← Paper Master 論文關聯圖</a><h1>教學與筆記</h1><p class="lead">從論文清單中收集、由 Claude 製作的互動式教學頁與研究筆記。每頁左下角可以回到關聯圖；在關聯圖點選相關論文時，詳情面板也會列出對應的教學。</p><h2 class="sec">互動教學</h2><div class="grid">{cards}</div><h2 class="sec">研究筆記</h2><p class="lead">每篇筆記只談一個主題；同一概念只寫一次，其他筆記用連結引用。先讀各區塊標題的總覽，再依序讀子筆記。</p><div class="hubs">{notes_html}</div></main></body></html>''')
print(len(out),'tutorials')
