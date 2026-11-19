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
 ('prediction-targets-k-diff','擴散模型的預測目標：從 ε、v、x 到連續的 k',[238]),
]
os.makedirs('out/tutorials',exist_ok=True)
BAR='''<nav id="pm-bar" aria-label="Paper Master" style="position:fixed;left:16px;bottom:16px;z-index:9999;display:flex;gap:8px;font:500 13px/1 'IBM Plex Sans','Noto Sans TC',system-ui,sans-serif">
<a href="../index.html" style="padding:10px 14px;border-radius:999px;background:#1D1C19;color:#F6F4EE;min-height:20px;display:inline-flex;align-items:center;text-decoration:none;box-shadow:0 4px 16px rgba(0,0,0,.2)">← Paper Master</a>
<a href="index.html" style="padding:10px 14px;border-radius:999px;background:#F6F4EE;color:#1D1C19;text-decoration:none;border:1px solid #DCD7CB;min-height:20px;display:inline-flex;align-items:center;box-shadow:0 4px 16px rgba(0,0,0,.12)">教學與筆記</a></nav>'''
out=[]
for slug,title,ids in T:
  src=f'tutorials_raw/{slug}.html'
  if not os.path.exists(src): continue
  s=open(src).read()
  if 'id="pm-bar"' not in s:
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
nid={s:i+1 for i,s in enumerate(order)}
noteno=lambda s:f'N{nid[s]:02d}'
for slug in order:
  n=notes[slug]
  rn=[r for r in n.get('related') or [] if r in notes and r!=slug]
  backs=[o for o in order if slug in (notes[o].get('related') or []) and o not in rn and o!=slug]
  rel=''
  if rn or backs:
    rel+='<section class="related"><h2>相關筆記</h2>'+''.join(f'<a href="{r}.html"><span>{noteno(r)}</span>{html.escape(notes[r]["title"])}</a>' for r in rn+backs)+'</section>'
  if n['papers']:
    rel+='<section class="related"><h2>資料庫中的相關論文</h2>'+''.join(f'<a href="../index.html#paper-{i}"><span>{i:03d}</span>{html.escape(byid[i]["title"])}</a>' for i in n['papers'])+'</section>'
  kick=f'<b>{noteno(slug)}</b><span>'+('主題總覽' if n.get('kind')=='hub' else '研究筆記')+'</span><span>'+html.escape(n.get('cluster',''))+'</span>'
  open(f'out/tutorials/{slug}.html','w').write(NT.replace('__TITLE__',html.escape(n['title'])).replace('__KICKER__',kick).replace('__RELATED__',rel).replace('__MD__',n['md'].replace('</script','<\\/script')))
  out.append({'file':f'{slug}.html','title':n['title'],'papers':n['papers'],'kind':n.get('kind','topic'),'cluster':n.get('cluster',''),'related':rn})
json.dump(out,open('tutorials.json','w'),ensure_ascii=False)
inter=[t for t in out if 'kind' not in t]
hubs=[t for t in out if t.get('kind')=='hub']
byslug={t['file'][:-5]:t for t in out if t.get('kind')}
ntopic=sum(1 for t in out if t.get('kind')=='topic')
def paper_line(t):
  if not t['papers']: return '主題教學，未對應特定論文'
  first=html.escape(byid[t['papers'][0]]['title'])
  return f'對應 {len(t["papers"])} 篇論文，包含 {first}' if len(t['papers'])>1 else f'對應論文：{first}'
rows=''.join(f'<li><a href="{t["file"]}"><span class="no">T{i:02d}</span><span class="tt">{html.escape(t["title"])}</span><span class="meta">{paper_line(t)}</span></a></li>' for i,t in enumerate(inter,1))
def grp(h):
  slug=h['file'][:-5]
  kids=[byslug[r] for r in hubkids.get(slug,[]) if r in byslug]
  return (f'<section class="hub"><header><span class="cl">{html.escape(h["cluster"])}</span><a href="{h["file"]}"><span class="no">{noteno(slug)}</span><h3>{html.escape(h["title"])}</h3></a>'
          f'<p class="cnt">{len(kids)} 篇主題筆記 · {len(h["papers"])} 篇相關論文</p></header><ol>'
          +''.join(f'<li><a href="{k["file"]}"><span class="no">{noteno(k["file"][:-5])}</span>{html.escape(k["title"])}</a></li>' for k in kids)+'</ol></section>')
notes_html=''.join(grp(h) for h in hubs)
open('out/tutorials/index.html','w').write(f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>教學與筆記 · Paper Master</title>
<meta name="description" content="Paper Master 收錄的 {len(inter)} 份互動教學與 {len(hubs)+ntopic} 篇研究筆記。">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Noto+Serif+TC:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Sans+TC:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{--bg:#F6F4EE;--surface:#FFFFFF;--surface-2:#EEEBE3;--ink:#1D1C19;--ink-2:#57534B;--ink-3:#67635A;--line:#DCD7CB;--accent:#B4532A;
  --f-display:"Newsreader","Noto Serif TC",Georgia,serif;--f-cjk-serif:"Noto Serif TC","Songti TC",Georgia,serif;--f-body:"IBM Plex Sans","Noto Sans TC",system-ui,sans-serif;--f-mono:"JetBrains Mono",ui-monospace,monospace;--ease:cubic-bezier(.2,0,0,1);color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#161513;--surface:#1F1E1B;--surface-2:#2A2925;--ink:#EEEBE4;--ink-2:#B5B0A5;--ink-3:#8E897F;--line:#36342F;--accent:#E58C66;color-scheme:dark}}}}
:root[data-theme=dark]{{--bg:#161513;--surface:#1F1E1B;--surface-2:#2A2925;--ink:#EEEBE4;--ink-2:#B5B0A5;--ink-3:#8E897F;--line:#36342F;--accent:#E58C66;color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 var(--f-body);-webkit-font-smoothing:antialiased}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}}
.top nav{{max-width:1200px;margin:0 auto;padding:0 24px;height:56px;display:flex;align-items:center;gap:24px;font-size:14px;border-bottom:1px solid var(--line)}}
.top a{{color:var(--ink-2);text-decoration:none}}.top a:hover{{color:var(--accent)}}.top .brand{{font-family:var(--f-display);font-size:20px;font-weight:500;color:var(--ink)}}
main{{max-width:1200px;margin:0 auto;padding:0 24px 120px}}
.hero{{display:grid;grid-template-columns:repeat(12,1fr);gap:24px;padding:96px 0 72px;border-bottom:1px solid var(--ink)}}
.hero h1{{grid-column:1/8;margin:0;font-family:var(--f-cjk-serif);font-weight:700;font-size:clamp(44px,7vw,88px);line-height:1.05;letter-spacing:-.01em}}
.hero .facts{{grid-column:9/13;align-self:end;margin:0;display:grid;grid-template-columns:auto 1fr;gap:6px 16px;font-size:14px;color:var(--ink-2)}}
.hero dt{{font:500 28px/1 var(--f-display);color:var(--ink);text-align:right}}.hero dd{{margin:0;align-self:end}}
.hero p{{grid-column:1/8;margin:0;color:var(--ink-2);max-width:36em}}
.sec{{display:grid;grid-template-columns:repeat(12,1fr);gap:24px;padding-top:64px}}
.sec>h2{{grid-column:1/4;margin:0;font-family:var(--f-cjk-serif);font-weight:600;font-size:26px;line-height:1.3}}
.sec>h2 small{{display:block;margin-top:8px;font:400 14px/1.6 var(--f-body);color:var(--ink-2)}}
.sec>.body{{grid-column:4/13;min-width:0}}
.tlist{{list-style:none;margin:0;padding:0;border-top:1px solid var(--line)}}
.tlist a{{display:grid;grid-template-columns:56px minmax(0,1fr) minmax(0,1.1fr);gap:24px;align-items:baseline;padding:18px 0;border-bottom:1px solid var(--line);text-decoration:none;color:var(--ink);transition:background .16s var(--ease),padding .16s var(--ease)}}
.tlist a:hover{{background:var(--surface);padding-left:12px}}
.tlist .tt{{font-family:var(--f-cjk-serif);font-weight:600;font-size:19px;line-height:1.35}}
.tlist .meta{{font-size:13px;color:var(--ink-2);line-height:1.5}}
.no{{font:12px var(--f-mono);color:var(--ink-3);font-variant-numeric:tabular-nums}}
.hubs{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:48px 32px}}
.hub header{{border-top:1px solid var(--ink);padding-top:14px}}
.hub .cl{{font:600 11px/1 var(--f-body);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}}
.hub header a{{display:block;margin-top:12px;text-decoration:none;color:var(--ink)}}
.hub header a .no{{color:var(--accent)}}
.hub h3{{margin:4px 0 0;font-family:var(--f-cjk-serif);font-weight:600;font-size:21px;line-height:1.35;transition:color .16s var(--ease)}}
.hub header a:hover h3{{color:var(--accent)}}
.hub .cnt{{margin:6px 0 12px;font-size:13px;color:var(--ink-3)}}
.hub ol{{list-style:none;margin:0;padding:0}}
.hub li a{{display:grid;grid-template-columns:44px 1fr;gap:8px;padding:8px 0;border-top:1px solid var(--line);text-decoration:none;color:var(--ink);font-size:15px;line-height:1.45;transition:color .16s var(--ease)}}
.hub li a:hover{{color:var(--accent)}}.hub li .no{{line-height:1.9}}
@media (max-width:900px){{.hero,.sec{{grid-template-columns:1fr}}.hero>*,.sec>*{{grid-column:1/-1!important}}.hero{{padding:56px 0 40px}}.hero .facts{{justify-self:start}}
  .tlist a{{grid-template-columns:44px 1fr;gap:4px 12px}}.tlist .meta{{grid-column:2}}.hubs{{grid-template-columns:1fr}}main,.top nav{{padding-left:16px;padding-right:16px}}}}
@media (prefers-reduced-motion:reduce){{*{{transition-duration:.001ms!important}}}}
</style></head>
<body><header class="top"><nav aria-label="網站"><a class="brand" href="../index.html">Paper Master</a><a href="../index.html">論文關聯圖</a><a href="index.html" aria-current="page">教學與筆記</a></nav></header>
<main>
<section class="hero"><h1>教學與筆記</h1>
<dl class="facts"><dt>{len(inter)}</dt><dd>份互動教學</dd><dt>{len(hubs)}</dt><dd>篇主題總覽</dd><dt>{ntopic}</dt><dd>篇主題筆記</dd></dl>
<p>互動教學由 Claude 製作，從論文清單中收集。研究筆記每篇只處理一個主題，同一內容只寫在一篇，其他篇以連結引用。每篇頁尾列出對應的論文，點選後回到關聯圖並選取該論文。</p></section>
<section class="sec"><h2>互動教學<small>可操作的圖表與演算法示範。</small></h2><div class="body"><ul class="tlist">{rows}</ul></div></section>
<section class="sec"><h2>研究筆記<small>先讀總覽，再依編號順序讀主題筆記。</small></h2><div class="body"><div class="hubs">{notes_html}</div></div></section>
</main></body></html>""")
print(len(out),'tutorials')
