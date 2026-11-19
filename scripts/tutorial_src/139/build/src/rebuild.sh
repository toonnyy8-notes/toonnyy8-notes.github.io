#!/bin/sh
# NOTE: the scratchpad paths below (local KaTeX, hide.js) are only for the headless local preview/screenshots; point K at any local KaTeX 0.16.9 copy. The shipped index.html uses the CDN.
cd "$(dirname "$0")"
K=/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/katex/0.16.9
python3 - <<'PY'
import json
P='parts/'
tl=json.dumps(json.load(open('../notes/timeline.json')),ensure_ascii=False).replace('</','<\\/')
s=open(P+'p1_head.html').read()+open(P+'p2_css.html').read()+open(P+'p2b_body.html').read()+open(P+'p2c_body.html').read()
s+='<script type="application/json" id="timeline-data">'+tl+'</script>\n<script>\n(()=>{\n'+open(P+'p3a_helpers.js').read()+open(P+'p3b_widgets.js').read()
s=s.replace('頁面末的 A.2 筆誤：','另外，A.2 有一處筆誤：')
open('index.html','w').write(s)
h=open('/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/hide.js').read()
K='/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/katex/0.16.9'
pv='\n'.join(l for l in s.replace('https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9','file://'+K).split('\n') if 'fonts.googleapis.com/css2' not in l).replace('</body>',h+'</body>')
open('preview.html','w').write(pv)
PY
