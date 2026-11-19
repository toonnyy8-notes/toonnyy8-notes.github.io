#!/bin/sh
# Assemble build/index.html from parts (run: sh assemble.sh)
P=$(dirname "$0"); D=$P/../..
{ cat $P/head.html $P/base.css $P/extra.css $P/body1.html $P/body2.html $P/body3.html $P/body4.html $P/body5.html
  printf '<script type="application/json" id="timeline-data">'; python3 -c 'import json,sys;print(json.dumps(json.load(open(sys.argv[1])),ensure_ascii=False),end="")' $D/notes/timeline.json; printf '</script>\n'
  cat $P/app.js; } > $P/../index.html
