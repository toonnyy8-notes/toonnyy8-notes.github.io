"""Assemble build/index.html from the pieces in this folder. Run from anywhere: python3 assemble.py"""
import json, pathlib, re
H = pathlib.Path(__file__).resolve().parent
B = H.parent.parent  # scripts/tutorial_src/255
rd = lambda n: (H / n).read_text(encoding='utf-8')
body = rd('body1.html') + rd('body2.html') + rd('body3.html')
# guard: a raw "<" directly followed by a letter inside $...$ TeX would be parsed as a tag (No.131 bug)
for m in re.finditer(r'\$[^$]*\$', body):
    assert not re.search(r'<[A-Za-z]', m.group(0)), 'raw < in TeX: ' + m.group(0)[:60]
app = rd('app.js').replace('/*@TOY@*/', rd('toy.js'))
assert '/*@' not in app
tl = json.dumps(json.load(open(B / 'notes/timeline.json', encoding='utf-8')), ensure_ascii=False).replace('</', '<\\/')
out = (rd('head.html') + '<style>\n' + rd('css_base.css') + rd('css_add.css') + '</style>\n</head>\n' + body
       + f'\n<script type="application/json" id="timeline-data">{tl}</script>\n<script>\n' + app + '</script>\n</body>\n</html>\n')
(B / 'build/index.html').write_text(out, encoding='utf-8')
print(len(out.encode()), 'bytes')
