import json,os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
t=open('index.template.html').read()
data='['+','.join(l.strip() for l in open('../papers.jsonl'))+']'
tut=open('tutorials.json').read() if os.path.exists('tutorials.json') else '[]'
open('../index.html','w').write(t.replace('__DATA__',data.replace('</','<\\/')).replace('__TUTORIALS__',tut.replace('</','<\\/')))
print('ok', os.path.getsize('../index.html'))
