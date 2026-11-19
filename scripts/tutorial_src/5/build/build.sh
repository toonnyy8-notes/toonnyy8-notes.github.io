#!/bin/sh
# Assemble build/index.html: reference shell CSS (copied verbatim) + this page's parts.
set -e
B=$(cd "$(dirname "$0")" && pwd); REF="$B/../../../../tutorials/prediction-targets-k-diff.html"
{
  cat "$B/parts/0_head.html"
  sed -n '15,340p' "$REF"            # CSS tokens, layout, plates, callouts, algo grid, timeline, pm-bar
  cat "$B/parts/1_extra.css"
  printf '</style>\n</head>\n'
  cat "$B/parts/2_body.html"
  printf '<script type="application/json" id="timeline-data">'
  python3 -I -c 'import json,sys;print(json.dumps(json.load(open(sys.argv[1])),ensure_ascii=False),end="")' "$B/../notes/timeline.json"
  printf '</script>\n<script>\n'
  cat "$B/parts/3_app.js"
  printf '</script>\n</body>\n</html>\n'
} > "$B/index.html"
echo "built $(wc -c < "$B/index.html") bytes"
