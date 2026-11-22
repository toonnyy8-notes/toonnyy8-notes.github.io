"""Assemble ../index.html from the reference shell head/CSS + extra.css + body*.html + timeline.json + app.js."""
import json, re, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
ref = open('../../../../../tutorials/prediction-targets-k-diff.html').read().split('\n')
head = '\n'.join(ref[0:13]).replace('<title>擴散模型的預測目標 k</title>', '<title>Model-Dowser 重要性探測</title>')
head = re.sub(r'<meta name="description" content="[^"]*">', '<meta name="description" content="圖解 Model-Dowser（ICML 2026）：以一階近似的輸出位移 ‖J‖·|W|·|h| 為每個權重評分，用模型自生成的探測輸入估計，不需資料；微調時凍結高分權重，減輕 MLLM 的災難性遺忘。">', head)
css = '\n'.join(ref[13:340])
body = ''.join(open(f).read() for f in ('body1.html', 'body2.html', 'body3.html'))
body = re.sub(r'〔本頁推導(?:[，：]([^〕]*))?〕', lambda m: '<span class="tag">本頁推導</span>' + (f'<span class="tagn">（{m.group(1)}）</span>' if m.group(1) else ''), body)
tl = json.dumps(json.load(open('../../notes/timeline.json')), ensure_ascii=False)
out = (head + '\n' + css + '\n' + open('extra.css').read() + '</style>\n</head>\n' + body +
       '\n<script type="application/json" id="timeline-data">' + tl + '</script>\n<script>\n' + open('app.js').read() + '</script>\n</body>\n</html>\n')
open('../index.html', 'w').write(out)
print(len(out), 'bytes')
