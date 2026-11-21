import re,json
import os; B=os.path.abspath(os.path.join(os.path.dirname(__file__),'../../../../../..'))
ref=open(B+'/wt-paper-master/tutorials/prediction-targets-k-diff.html').read()
css=ref[ref.index('<style>')+7:ref.index('</style>')]
head=ref[:ref.index('<style>')]
head=re.sub(r'<title>.*?</title>','<title>Token 縮減系譜</title>',head)
head=re.sub(r'<meta name="description" content="[^"]*">','<meta name="description" content="系列互動教學：Transformer 的 token 縮減與解析度還原，從 Hourglass、ToMe、ToMeSD 到像素擴散的 RTI 與 HiT，比較分組、預算、縮減位置與還原方式。">',head)
P=B+'/wt-paper-master/scripts/tutorial_src/series-token-compression/build/src/'
tl=open(B+'/wt-paper-master/scripts/tutorial_src/series-token-compression/notes/timeline.json').read()
tl=json.dumps(json.loads(tl),ensure_ascii=False).replace('</','<\\/')
html=head+'<style>'+css+open(P+'extra.css').read()+'</style>\n</head>\n<body>\n'+open(P+'body.html').read()+'\n<script type="application/json" id="timeline-data">'+tl+'</script>\n<script>\n'+open(P+'app.js').read()+'</script>\n</body>\n</html>\n'
open(B+'/wt-paper-master/scripts/tutorial_src/series-token-compression/build/index.html','w').write(html)
print(len(html))
