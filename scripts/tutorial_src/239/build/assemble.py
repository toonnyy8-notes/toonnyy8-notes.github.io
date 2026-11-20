#!/usr/bin/env python3
"""Assemble build/index.html from build/src/ pieces and notes/timeline.json. Run: python3 assemble.py"""
import json, pathlib
b = pathlib.Path(__file__).resolve().parent
src = b / 'src'
tl = json.loads((b.parent / 'notes/timeline.json').read_text())
tl_s = json.dumps(tl, ensure_ascii=False).replace('</', '<\\/')
js = (src / 'app.js').read_text()
assert '</script' not in js
out = ''.join((src / f).read_text() for f in ('head.html', 'body1.html', 'body2.html', 'body3.html'))
out += f'\n<script type="application/json" id="timeline-data">{tl_s}</script>\n<script>\n{js}</script>\n</body>\n</html>\n'
assert 'file://' not in out
(b / 'index.html').write_text(out)
print('index.html', len(out.encode()), 'bytes')
