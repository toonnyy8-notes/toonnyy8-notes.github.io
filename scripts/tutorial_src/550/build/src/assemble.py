import json, os
B = os.path.dirname(os.path.abspath(__file__)) + '/../../'
S = os.path.dirname(os.path.abspath(__file__)) + '/'
rd = lambda f: open(S + f).read()
tl = json.dumps(json.load(open(B + 'notes/timeline.json')), ensure_ascii=False).replace('</', '<\\/')
out = (rd('head.html') + '<style>\n' + rd('css_base.css') + rd('css_add.css') + '</style>\n</head>\n'
       + rd('body1.html') + rd('body2.html') + rd('body3.html')
       + f'\n<script type="application/json" id="timeline-data">{tl}</script>\n<script>\n' + rd('app.js') + '</script>\n</body>\n</html>\n')
open(B + 'build/index.html', 'w').write(out)
print(len(out.encode()), 'bytes')
