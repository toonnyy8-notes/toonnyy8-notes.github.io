import re,json,pathlib
B=pathlib.Path(__file__).resolve().parent; D=B.parent.parent
ref=open('/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/wt-paper-master/tutorials/prediction-targets-k-diff.html').read()
head=ref[:ref.index('</style>')]
head=head.replace('<title>擴散模型的預測目標 k</title>','<title>分數學到流形幾何</title>')
head=re.sub(r'<meta name="description" content="[^"]*">','<meta name="description" content="圖解 Li、Shen、Hsieh、He 的論文：小噪音時 log p_σ 的主導項只含到資料流形的距離，生成分佈集中到流形只需 o(σ⁻²) 的分數誤差，還原資料分佈需要 o(1)；把 Langevin 的分數乘上 σ^α 可取樣流形上的均勻分佈。">',head)
head=head.replace("/* Layout: editorial reading column (≤700px) beside a sticky contents rail; each\n   interactive figure sits on a wider \"plate\". One accent colour; two figure-only\n   series colours (parallel / perpendicular mode). */","/* Layout: editorial reading column (≤700px) beside a sticky contents rail; each\n   interactive figure sits on a wider \"plate\". One accent colour; two figure-only\n   series colours (geometry/normal = --c-par, density/tangent = --c-perp). */")
css_extra=open(B/'extra.css').read()
script=ref[ref.index('<script>\n(()=>{')+len('<script>\n'):]
helpers=script[script.index("(()=>{"):script.index('/* ---------- shared math ---------- */')]
toc=script[script.index('/* ---------- toc, theme, appbar ---------- */'):script.index('/* ---------- self test ---------- */')]
core=open(D/'verify/core.js').read()
core=core.replace("'use strict';\n",'',1)
core=core[:core.index("if (typeof module !== 'undefined')")]
widgets=open(B/'widgets.js').read()
body=open(B/'body.html').read()
tl=json.dumps(json.load(open(D/'notes/timeline.json')),ensure_ascii=False)
out=f'''{head}{css_extra}</style>
</head>
<body>
<a class="sr" href="#main">跳到內容</a>
<header class="appbar" id="appbar"><div class="appbar-in">
  <span class="no">No. 237</span>
  <span class="ttl">小噪音時，分數的主導項只含流形的幾何</span>
  <button class="icon-btn" id="theme" type="button" aria-label="切換深色／淺色模式"><svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 3a9 9 0 1 0 9 9c0-.46-.04-.92-.1-1.36a5.39 5.39 0 0 1-4.4 2.26 5.4 5.4 0 0 1-3.14-9.8A9 9 0 0 0 12 3z"/></svg></button>
</div></header>
<nav class="tocbar" id="tocbar" aria-label="章節"></nav>

<div class="shell">
<nav class="toc" aria-label="章節目錄"><ol id="toc"></ol></nav>
<main id="main">

{body}
</main>
</div>

<footer>
  <p>Xiang Li, Zebang Shen, Ya-Ping Hsieh, Niao He. When Scores Learn Geometry: Rate Separations under the Manifold Hypothesis. <a href="https://arxiv.org/abs/2509.24912">arXiv:2509.24912</a> (v2, 2026).</p>
  <p>本頁的方程式與數字取自論文 v2；標示「本頁推導」的內容為本頁自行推導，並非論文結論。互動元件的數值在瀏覽器中即時計算。</p>
  <p id="selftest" class="mono"></p>
</footer>

<nav id="pm-bar" aria-label="Paper Master">
  <a href="../index.html">← Paper Master</a>
  <a href="index.html">教學與筆記</a>
</nav>

<script type="application/json" id="timeline-data">{tl}</script>
<script>
{helpers}
/* ---------- core (verify/core.js, verbatim) ---------- */
{core}
{toc}
{widgets}
}})();
</script>
</body>
</html>
'''
open(B.parent/'index.html','w').write(out)
# local preview copy for headless chrome (no CDN access)
K='/tmp/claude-1000/-media-datatank-toonnyy8-toonnyy8-notes-github-io/5b278533-609c-4705-9f64-8b225c95d4e4/scratchpad/katex/0.16.9'
loc=out.replace('https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9',K)
loc=re.sub(r'<link href="https://fonts.googleapis.com[^>]*>','',loc)
loc=re.sub(r'<link rel="preconnect"[^>]*>','',loc)
open(B.parent/'local.html','w').write(loc)
print(len(out))
