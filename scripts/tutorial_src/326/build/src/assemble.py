"""Assemble build/index.html from the pieces in this folder. Run from anywhere.
Embeds notes/timeline.json and verify/data.json (written by `python3 verify/check.py design.md --json verify/data.json`)."""
import json, pathlib
H = pathlib.Path(__file__).resolve().parent
B = H.parent.parent  # scripts/tutorial_src/326
emb = lambda p: json.dumps(json.load(open(p)), ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
head = (H / 'head.html').read_text()
css = (H / 'css_base.css').read_text() + (H / 'css_add.css').read_text()
body = ''.join((H / f'body{i}.html').read_text() for i in (1, 2, 3))
out = (head + '<style>\n' + css + '</style>\n</head>\n' + body +
       f'\n<script type="application/json" id="timeline-data">{emb(B / "notes/timeline.json")}</script>'
       f'\n<script type="application/json" id="fpr-data">{emb(B / "verify/data.json")}</script>\n<script>\n' +
       (H / 'app.js').read_text() + '</script>\n</body>\n</html>\n')
(B / 'build/index.html').write_text(out)
print(len(out.encode()))
