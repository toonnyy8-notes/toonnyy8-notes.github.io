#!/bin/sh
# usage: ./localtest.sh  -> prints selfTest title (uses local KaTeX, no Google Fonts)
D=$(cd "$(dirname "$0")" && pwd); K=/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/katex/0.16.9
sed -e "s#https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9#file://$K#g" -e '/fonts.googleapis.com\/css2/d' "$D/index.html" > "$D/.t.html"
timeout 120 google-chrome --headless=new --disable-gpu --no-sandbox --virtual-time-budget=9000 --enable-logging=stderr --v=0 --dump-dom "file://$D/.t.html?test" 2>"$D/.console.txt" | grep -o '<title>[^<]*' ; grep -i "uncaught\|error" "$D/.console.txt" | grep -v "gpu\|dbus\|GCM\|fontconfig" | head
