"""Assemble build/index.html from the k-diff shell (tutorials/prediction-targets-k-diff.html: CSS tokens, helpers, toc/theme)
and the 443 parts (body.html, widgets.js, extra.css) plus notes/timeline.json and verify/c4_data.json."""
import pathlib, re, json
here = pathlib.Path(__file__).resolve().parent
root = here.parents[4]
shell = (root / 'tutorials/prediction-targets-k-diff.html').read_text()
head, rest = shell.split('</style>', 1)
head = re.sub(r'<title>.*?</title>', '<title>Diffusion-QL 擴散策略</title>', head)
head = re.sub(r'<meta name="description" content=".*?">', '<meta name="description" content="圖解 Wang、Hunt 與 Zhou 的 ICLR 2023 論文：把離線強化學習的策略寫成條件擴散模型，並在去噪損失上加入 Q 值項（Diffusion-QL）。">', head)
js = rest.split('<script>\n(()=>{', 1)[1]
helpers = js.split('\n/* ---------- shared math', 1)[0]
tocth = '/* ---------- toc, theme' + js.split('/* ---------- toc, theme', 1)[1].split('/* ---------- self test', 1)[0]
timeline = (here / '../../notes/timeline.json').read_text().strip(); json.loads(timeline)
c4 = json.dumps(json.loads((here / '../../verify/c4_data.json').read_text()), ensure_ascii=False, separators=(',', ':'))
widgets = (here / 'widgets.js').read_text().replace('/*C4DATA*/null', c4)
boot = r"""
/* ---------- boot ---------- */
initTheme();initToc();initC1();initC2();initC3();initC4();renderTimeline();
renderC1();renderC2();
let lastW=innerWidth,rz=0;
addEventListener('resize',()=>{if(innerWidth===lastW)return;lastW=innerWidth;cancelAnimationFrame(rz);rz=requestAnimationFrame(()=>{renderC1();renderC2();renderC3();renderC4()})});
function renderMath(){if(typeof renderMathInElement!=='function')return;renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','svg'],throwOnError:false})}
if(document.readyState==='loading')addEventListener('DOMContentLoaded',renderMath);else renderMath();
addEventListener('load',()=>{if(!document.querySelector('.katex'))renderMath()});
if(/[?&]test\b/.test(location.search))selfTest();
})();
</script>
</body>
</html>
"""
out = (head + '\n' + (here / 'extra.css').read_text() + '</style>\n</head>\n<body>\n' + (here / 'body.html').read_text()
       + '\n<script type="application/json" id="timeline-data">' + timeline + '</script>\n<script>\n(()=>{' + helpers + '\n'
       + widgets + '\n' + tocth + boot)
(here / '../index.html').write_text(out)
print(len(out), 'bytes')
