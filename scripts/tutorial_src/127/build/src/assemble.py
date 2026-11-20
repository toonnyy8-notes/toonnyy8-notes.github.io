"""Assemble build/index.html from the pieces in this folder (No. 127). Run from anywhere."""
import json, pathlib
H = pathlib.Path(__file__).resolve().parent
B = H.parent.parent  # scripts/tutorial_src/127
head = (H / 'head.html').read_text()
css = ''.join((H / f).read_text() for f in ('css_base.css', 'css_shared.css', 'css_add.css'))
body = ''.join((H / f'body{i}.html').read_text() for i in (1, 2, 3))
esc = lambda o: json.dumps(o, ensure_ascii=False).replace('</', '<\\/')
tl = esc(json.load(open(B / 'notes/timeline.json')))
res = esc(json.load(open(B / 'verify/results.json')))
out = (head + '<style>\n' + css + '</style>\n</head>\n' + body +
       f'\n<script type="application/json" id="timeline-data">{tl}</script>\n<script type="application/json" id="results-data">{res}</script>\n<script>\n' + ''.join((H / f).read_text() for f in ('app_head.js', 'app_main.js', 'app_shell.js', 'app_tail.js')) + '</script>\n</body>\n</html>\n')
(B / 'build/index.html').write_text(out)
print(len(out.encode()))
