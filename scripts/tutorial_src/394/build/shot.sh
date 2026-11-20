# usage: bash shot.sh <query> <width> <height> <out.png> [extra chrome flags]; builds .t.html (local KaTeX, no web fonts) first
K=/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/katex/0.16.9
cd "$(dirname "$0")"
sed -e "s#https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9#file://$K#g" -e 's#<link[^>]*fonts.googleapis[^>]*>##' index.html > .t.html
q=$1; w=$2; h=$3; out=$4; shift 4
timeout 60 google-chrome --headless=new --disable-gpu --no-sandbox --hide-scrollbars --virtual-time-budget=8000 --window-size=$w,$h "$@" --screenshot=$out "file://$PWD/.t.html$q" >/dev/null 2>&1
