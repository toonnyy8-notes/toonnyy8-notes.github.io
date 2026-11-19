"""Assemble build/index.html from the reference shell head + parts. Run: python3 assemble.py"""
import pathlib, re
here = pathlib.Path(__file__).resolve().parent
src = here.parents[1]; repo = src.parents[2]
ref = (repo / 'tutorials/prediction-targets-k-diff.html').read_text()
head = ref[:ref.index('</style>')]
head = re.sub(r'<title>.*?</title>', '<title>修正型擴散語言模型</title>', head)
head = re.sub(r'<meta name="description" content=".*?">', '<meta name="description" content="圖解 Corrective Diffusion Language Models：遮罩擴散只在遮罩位置學習，本文另外監督被隨機替換的可見字，讓模型對錯字給低信心，再以信心門檻原地修正程式。">', head)
head = head.replace('--c-par:#3F6E8C;--c-perp:#6B7A3A;', '--c-clean:#3F6E8C;--c-err:#7A4E8C;--hatch:color-mix(in srgb,var(--ink) 35%,transparent);')
head = head.replace('--c-par:#8DB4D3;--c-perp:#A9BC7C;', '--c-clean:#8DB4D3;--c-err:#C3A1D6;--hatch:color-mix(in srgb,var(--ink) 45%,transparent);')
head = head.replace('details.check', 'details:is(.check,.more)')
# drop reference-only blocks (k axis, algorithm grid) and rules that use the removed colours
head = re.sub(r'/\* k axis.*?(?=/\* plates \*/)', '', head, flags=re.S)
head = re.sub(r'/\* algorithm grid \*/.*?(?=/\* timeline \*/)', '', head, flags=re.S)
head = '\n'.join(l for l in head.split('\n') if 'c-par' not in l and 'c-perp' not in l)
assert 'c-par' not in head and '--c-clean' in head
out = head + (here / 'extra.css').read_text() + '</style>\n</head>\n' + (here / 'body.html').read_text()
out += '\n<script type="application/json" id="timeline-data">' + (src / 'notes/timeline.json').read_text().strip() + '</script>\n'
out += (here / 'app.js').read_text()
(src / 'build/index.html').write_text(out)
print('wrote', src / 'build/index.html', len(out), 'bytes')
