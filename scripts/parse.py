import re,json,urllib.request,time
SRC=[('ai-research-topic-papers.md','raymin-notes'),('diffusion-channels-papers.md','raymin-notes'),('waiting-to-read-papers.md','toonnyy8-notes')]
AX=re.compile(r'(?:arxiv\.org/(?:abs|pdf|html)/|alphaxiv\.org/(?:abs|overview)/|huggingface\.co/papers/)(\d{4}\.\d{4,5})')
papers={};order=[]
for fn,br in SRC:
  cat=None;sub=None
  for line in open(fn).read().split('## 二')[0].splitlines():
    m=re.match(r'^(\d+)\.\s+(.*)',line)
    if m: cat=re.sub(r'[\[\]]','',m.group(2)).strip(); continue
    m=re.match(r'^\s*-\s+\[(.+?)\]\((\S+?)\)(.*)',line)
    if not m: continue
    title,url,note=m.groups()
    a=AX.search(url); key=a.group(1) if a else url.split('#')[0].rstrip('/')
    if key in papers:
      p=papers[key]; s=f'{fn} ({br})'
      if s not in p['sources']: p['sources'].append(s)
      if cat not in p['cats']: p['cats'].append(cat)
      continue
    papers[key]={'key':key,'title':title.strip(),'url':url,'arxiv':a.group(1) if a else None,'sources':[f'{fn} ({br})'],'cats':[cat],'note':note.strip('（）() ')}
    order.append(key)
ids=[papers[k]['arxiv'] for k in order if papers[k]['arxiv']]
meta={}
for i in range(0,len(ids),50):
  q=','.join(ids[i:i+50])
  t=urllib.request.urlopen(f'https://export.arxiv.org/api/query?id_list={q}&max_results=60',timeout=120).read().decode()
  for e in t.split('<entry>')[1:]:
    aid=re.sub(r'v\d+$','',re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1])
    g=lambda k:' '.join(re.search(f'<{k}>(.*?)</{k}>',e,re.S).group(1).split())
    meta[aid]={'authors':re.findall(r'<name>(.*?)</name>',e),'year':g('published')[:4],'abstract':g('summary')}
  time.sleep(3)
out=[]
for k in order:
  p=papers[k]; m=meta.get(p['arxiv'] or '',{})
  p.update(authors=m.get('authors',[]),year=m.get('year',''),abstract=m.get('abstract',''))
  out.append(p)
json.dump(out,open('papers_raw.json','w'),ensure_ascii=False,indent=1)
print(len(out),'papers;',len(ids),'arxiv;',len(meta),'meta')
