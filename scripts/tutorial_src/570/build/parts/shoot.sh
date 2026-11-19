#!/bin/bash
# usage: shoot.sh <elementId> <W> <H> <out.png> [extra chrome flags]; shows only the plate with that id (uses build/.preview.html)
cd "$(dirname "$0")/.."
python3 - "$1" <<'PY'
import sys; i=sys.argv[1]
css=f"<style>main>section,footer,#pm-bar{{display:none!important}} main>section:has(#{i}){{display:block!important}} main>section:has(#{i})>*:not(#{i}){{display:none!important}} section.ch{{padding-top:8px!important}}</style></head>"
open('.shot.html','w').write(open('.preview.html').read().replace('</head>',css,1))
PY
timeout 60 google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --virtual-time-budget=6000 --window-size=$2,$3 $5 --screenshot="$4" "file://$PWD/.shot.html" 2>/dev/null
rm -f .shot.html
