"""Assemble build/index.html from the 238 shell (head + CSS) and this tutorial's parts."""
import json, pathlib, re
here = pathlib.Path(__file__).resolve().parent
D = here.parents[1]; REPO = D.parents[2]
shell = (REPO / 'tutorials/prediction-targets-k-diff.html').read_text()
css = shell[shell.index('<style>') + 7: shell.index('</style>')]
head = shell[:shell.index('<style>')]
head = re.sub(r'<title>.*?</title>', '<title>Flow Map Language Models</title>', head)
head = re.sub(r'<meta name="description" content="[^"]*">',
              '<meta name="description" content="圖解 Flow Map Language Models：把 token 寫成 one-hot 向量做 flow matching，以交叉熵學 denoiser，再用 two-time denoiser 與 KL 半群目標蒸餾成一步生成的 flow map。">', head)
timeline = json.dumps(json.loads((D / 'notes/timeline.json').read_text()), ensure_ascii=False)
out = (head + '<style>' + css + (here / 'extra.css').read_text() + '</style>\n</head>\n<body>\n'
       + (here / 'body.html').read_text()
       + '\n<script type="application/json" id="timeline-data">' + timeline + '</script>\n<script>\n'
       + (here / 'app.js').read_text() + '</script>\n</body>\n</html>\n')
(D / 'build/index.html').write_text(out)
print(len(out), 'bytes')
