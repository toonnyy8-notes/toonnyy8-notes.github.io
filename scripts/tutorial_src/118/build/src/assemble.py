"""Assemble build/index.html from the pieces in this folder. Run from anywhere."""
import json, pathlib
H = pathlib.Path(__file__).resolve().parent
B = H.parent.parent  # scripts/tutorial_src/118
head = (H / 'head.html').read_text()
css = (H / 'css_base.css').read_text() + (H / 'css_add.css').read_text()
body = ''.join((H / f'body{i}.html').read_text() for i in (1, 2, 3))
tl = json.dumps(json.load(open(B / 'notes/timeline.json')), ensure_ascii=False).replace('</', '<\\/')
out = (head + '<style>\n' + css + '</style>\n</head>\n' + body +
       f'\n<script type="application/json" id="timeline-data">{tl}</script>\n<script>\n' + (H / 'app.js').read_text() + '</script>\n</body>\n</html>\n')
(B / 'build/index.html').write_text(out)
print(len(out.encode()))
