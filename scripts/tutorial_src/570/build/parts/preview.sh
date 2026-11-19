#!/bin/bash
# assemble, then write build/.preview.html (local KaTeX, no Google Fonts) for headless checks
cd "$(dirname "$0")/.." && python3 parts/assemble.py
K=/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/katex/0.16.9
sed -e "s#https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9#file://$K#g" -e '/fonts.googleapis\|fonts.gstatic/d' index.html > .preview.html
