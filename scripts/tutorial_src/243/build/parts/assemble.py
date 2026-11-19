# Assemble build/index.html from the 238 shell head + parts. Run from repo root.
import json,pathlib
D=pathlib.Path('scripts/tutorial_src/243');P=D/'build/parts'
tpl=pathlib.Path('tutorials/prediction-targets-k-diff.html').read_text().split('\n')
head='\n'.join(tpl[:340])  # through the last CSS rule, before </style>
head=head.replace('<title>擴散模型的預測目標 k</title>','<title>Titans 測試時記憶</title>')
import re
head=re.sub(r'<meta name="description" content="[^"]*">','<meta name="description" content="圖解 Behrouz、Zhong、Mirrokni 的 Titans：以梯度（驚訝度）、動量與遺忘在測試時更新的深層神經記憶，chunk 平行化訓練，以及 MAC、MAG、MAL 三種架構。">',head)
css_start=head.index('<style>')
head=head[:css_start]+head[css_start:].replace('details.check',':is(details.check,details.more)')
tl=json.loads((D/'notes/timeline.json').read_text())
out=head+'\n'+(P/'extra.css').read_text()+'</style>\n</head>\n'+(P/'body.html').read_text()+'\n<script type="application/json" id="timeline-data">'+json.dumps(tl,ensure_ascii=False).replace('</','<\\/')+'</script>\n<script>\n'+(P/'app.js').read_text()+'</script>\n</body>\n</html>\n'
(D/'build/index.html').write_text(out);print(len(out))
