"""Assemble build/index.html from the 406 shell (CSS + generic JS helpers) and the 570 parts."""
import pathlib, re, json
here = pathlib.Path(__file__).resolve().parent
shell = (here / '../../../406/build/index.html').resolve().read_text()
head, rest = shell.split('</style>', 1)
head = re.sub(r'<title>.*?</title>', '<title>TD3+BC 極簡離線強化學習</title>', head)
head = re.sub(r'<meta name="description" content=".*?">', '<meta name="description" content="圖解 Fujimoto 與 Gu 的 NeurIPS 2021 論文：只在 TD3 的策略更新加入一項行為複製並正規化狀態，就達到當時最佳離線強化學習方法的表現。">', head)
js = rest.split('<script>\n(()=>{', 1)[1]
def cut(a, b): return js.split(a, 1)[1].split(b, 1)[0]
helpers = js.split('\n/* ---------- shared math', 1)[0]
ui = '/* ---------- UI helpers' + cut('/* ---------- UI helpers', '/* ---------- C1 ----------')
tl = cut('/* ---------- timeline ---------- */', '/* ---------- toc, theme')
tl = re.sub(r"const TL_GROUP=\{[\s\S]*?\};\nconst TL_NAME=\{[\s\S]*?\};", lambda m: TLG, tl, count=1) if False else tl
tocth = '/* ---------- toc, theme' + cut('/* ---------- toc, theme', '/* ---------- self test')
boot = js.split('/* ---------- boot ---------- */', 1)[1]
boot = boot.replace('initC1();initC2();initC3();initC4();', 'initC1();initC2();initC4();').replace('renderC1();renderC2();renderC3();renderC4()', 'renderC1();renderC2();renderC4()')
TLG = """const TL_GROUP={'1509.02971':['ac','ddpg-td3-actor-critic'],'1802.09477':['ac','ddpg-td3-actor-critic'],
 '1709.10089':['rlbc','rl-plus-behavior-cloning-prior-work'],
 '1709.06560':['impl','rl-implementation-details-reproducibility'],'2005.12729':['impl','rl-implementation-details-reproducibility'],
 '1812.02900':['pc','batch-constrained-policy-constraint-methods'],'1906.00949':['pc','batch-constrained-policy-constraint-methods'],'1911.11361':['pc','batch-constrained-policy-constraint-methods'],
 '1910.00177':['awbc','advantage-weighted-bc-methods'],'2006.15134':['awbc','advantage-weighted-bc-methods'],
 '2004.07219':['bench','td3bc-algorithm-and-d4rl'],'2006.04779':['bench','td3bc-algorithm-and-d4rl'],'2103.08050':['bench','td3bc-algorithm-and-d4rl'],
 '2106.06860':['me','td3bc-algorithm-and-d4rl']};
const TL_NAME={ac:'Actor-critic',rlbc:'RL + BC',impl:'實作細節',pc:'策略約束',awbc:'加權 BC',bench:'基準與比較方法',me:'本文',other:'其他'};"""
tl = re.sub(r"const TL_GROUP=\{[\s\S]*?\};\nconst TL_NAME=\{[^\n]*\};", lambda m: TLG, tl, count=1)
assert 'ddpg-td3-actor-critic' in tl
timeline = (here / '../../notes/timeline.json').read_text().strip(); json.loads(timeline)
out = (head + (here / 'extra.css').read_text() + '</style>\n</head>\n<body>\n' + (here / 'body.html').read_text()
       + '\n<script type="application/json" id="timeline-data">' + timeline + '</script>\n<script>\n(()=>{' + helpers + '\n' + ui
       + (here / 'widgets.js').read_text() + '\n/* ---------- timeline ---------- */' + tl + tocth
       + '/* ---------- boot ---------- */' + boot)
(here / '../index.html').write_text(out)
print(len(out), 'bytes')
