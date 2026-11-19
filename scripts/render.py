import json,os
t=open('index.template.html').read()
data=open('papers.json').read()
tut=open('tutorials.json').read() if os.path.exists('tutorials.json') else '[]'
os.makedirs('out',exist_ok=True)
open('out/index.html','w').write(t.replace('__DATA__',data.replace('</','<\\/')).replace('__TUTORIALS__',tut.replace('</','<\\/')))
print('ok', os.path.getsize('out/index.html'))
