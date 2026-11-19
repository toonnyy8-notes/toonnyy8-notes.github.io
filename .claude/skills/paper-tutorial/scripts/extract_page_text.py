"""usage: extract_page_text.py <tutorial_src/ID dir>
Renders build/index.html headless, strips scripts/SVG and the <details class="check"> answer boxes,
and writes accept/page.txt (what a reader can learn from body text) and accept/questions.txt (from design.md).
Renders a copy without Google Fonts; if headless Chrome cannot reach cdnjs, set KATEX_DIR to a local KaTeX 0.16.9
directory (katex.min.js, katex.min.css, contrib/) and the copy loads KaTeX from there."""
import sys, os, re, html, subprocess, pathlib
d = pathlib.Path(sys.argv[1]).resolve(); (d / 'accept').mkdir(exist_ok=True)
src = re.sub(r'<link[^>]*(fonts\.googleapis|fonts\.gstatic)[^>]*>', '', (d / 'build/index.html').read_text())
if os.environ.get('KATEX_DIR'):
    src = src.replace('https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9', 'file://' + os.path.abspath(os.environ['KATEX_DIR']))
tmp = d / 'build/.extract.html'; tmp.write_text(src)
try:
    dom = subprocess.run(['google-chrome', '--headless=new', '--disable-gpu', '--no-sandbox', '--virtual-time-budget=8000',
                          '--dump-dom', f'file://{tmp}'], capture_output=True, text=True, timeout=120).stdout
finally:
    tmp.unlink()
if not dom.strip(): sys.exit('empty DOM: headless Chrome hung or failed (try KATEX_DIR=<local katex>)')
s = re.sub(r'<(script|style|svg)[\s\S]*?</\1>', '', dom)
s = re.sub(r'<details class="check[\s\S]*?</details>', '', s)
s = re.sub(r'<annotation encoding="application/x-tex">([\s\S]*?)</annotation>', lambda m: ' $' + m.group(1) + '$ ', s)
s = re.sub(r'<span class="katex-html"[\s\S]*?</span>\s*</span>\s*</span>', '', s)   # drop the rendered duplicate of each formula
s = re.sub(r'<(br|/p|/h\d|/li|/tr|/div|/section|/summary)[^>]*>', '\n', s)
s = html.unescape(re.sub(r'<[^>]+>', '', s))
s = re.sub(r'\n\s*\n+', '\n\n', re.sub(r'[ \t]+', ' ', s)).strip()
(d / 'accept/page.txt').write_text(s)
qs = [f'{c[1].strip()} | {c[3].strip()}' for c in (re.split(r'\s\|\s', ' '+l.strip()+' ') for l in (d / 'design.md').read_text().splitlines())
      if len(c) > 4 and re.match(r'\s*LO\d+\s*$', c[1])]
(d / 'accept/questions.txt').write_text('\n'.join(qs) + '\n')
print(f'page.txt {len(s)} chars, {len(qs)} questions')
