#!/bin/sh
B=$(dirname "$0")/..
{ cat $B/parts/a_head.html $B/parts/b_css.html $B/parts/c_body.html
  printf '<script type="application/json" id="timeline-data">'; python3 -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])),ensure_ascii=False).replace('</','<\\\\/'),end='')" $B/../notes/timeline.json; printf '</script>\n'
  cat $B/parts/e_js.html; } > $B/index.html
# local preview copy for headless runs (CDN rewritten)
K=/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/katex/0.16.9
sed -e "s#https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9#file://$K#g" -e '/fonts.googleapis\|fonts.gstatic/d' $B/index.html > $B/.local.html
