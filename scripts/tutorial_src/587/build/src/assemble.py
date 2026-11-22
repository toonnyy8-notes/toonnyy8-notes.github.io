import re,pathlib,json
b=pathlib.Path(__file__).resolve().parent
ref=pathlib.Path(b,'../../../../../tutorials/prediction-targets-k-diff.html').resolve().read_text()
head=ref[:ref.index('</style>')]
head=head.replace('<title>擴散模型的預測目標 k</title>','<title>RIPO 等距 clipping</title>')
head=re.sub(r'<meta name="description" content="[^"]*">','<meta name="description" content="圖解 Cai 等人的 ICML 2026 論文 RIPO：PPO-Clip 以固定比值範圍衡量策略差異，與 KL 散度的二次近似不一致；RIC 把範圍改成 1 ± √(δ/π_old)，使每個 token 的距離上限相同。">',head)
head=head.replace('--c-par:#3F6E8C;--c-perp:#6B7A3A;','--c-dapo:#3F6E8C;').replace('--c-par:#8DB4D3;--c-perp:#A9BC7C;','--c-dapo:#8DB4D3;')
head=head.replace('var(--c-par)','var(--c-dapo)').replace('var(--c-perp)','var(--ink-3)')
tl=pathlib.Path(b,'../../notes/timeline.json').read_text()
json.loads(tl)
out=head+pathlib.Path(b,'extra.css').read_text()+'</style>\n</head>\n<body>\n'+pathlib.Path(b,'body.html').read_text()+\
 '\n<script type="application/json" id="timeline-data">'+json.dumps(json.loads(tl),ensure_ascii=False)+'</script>\n<script>\n'+pathlib.Path(b,'app.js').read_text()+'</script>\n</body>\n</html>\n'
pathlib.Path(b,'../index.html').write_text(out)
print(len(out))
