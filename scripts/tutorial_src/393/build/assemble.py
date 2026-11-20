"""python3 assemble.py -> index.html (single self-contained file; no file:// paths)."""
import json, pathlib
d = pathlib.Path(__file__).resolve().parent; src = d / 'src'
tl = json.loads((d.parent / 'notes' / 'timeline.json').read_text())
tl_json = json.dumps(tl, ensure_ascii=False).replace('</', '<\\/')
out = (src / 'head.html').read_text() + '<style>\n' + (src / 'style.css').read_text() + '\n</style>\n</head>\n<body>\n' \
    + (src / 'body.html').read_text() \
    + f'\n<script type="application/json" id="timeline-data">{tl_json}</script>\n<script>\n' + (src / 'app.js').read_text() + '</script>\n</body>\n</html>\n'
assert 'file://' not in out
(d / 'index.html').write_text(out); print('index.html', len(out), 'bytes')
