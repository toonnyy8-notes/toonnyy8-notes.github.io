"""Assemble build/index.html: shell CSS from tut/447/build (k-diff shell) + parts. Run: python3 -I assemble.py"""
import os, re, json
here = os.path.dirname(os.path.abspath(__file__))
src447 = open(os.path.join(here, '../../../447/build/index.html')).read()
css = src447[src447.index('<style>') + 7: src447.index('/* ---- No. 447 additions ---- */')]
p = lambda n: open(os.path.join(here, n)).read()
tl = json.load(open(os.path.join(here, '../../notes/timeline.json')))
tl = [{k: e[k] for k in ('year', 'title', 'authors_short', 'arxiv')} for e in tl]   # role/source not shown
head = '''<!DOCTYPE html>
<html lang="zh-Hant-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Flow-GRPO 線上強化學習</title>
<meta name="description" content="圖解 Flow-GRPO（arXiv 2505.05470）：把流匹配模型的確定性 ODE 取樣換成邊際分布相同的 SDE，以 GRPO 的組相對優勢做線上強化學習，並以去噪步數縮減加速訓練；附四個即時計算的互動元件。">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Noto+Serif+TC:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Sans+TC:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<style>'''
out = (head + css + p('extra.css') + '</style>\n</head>\n<body>\n' + p('body.html')
       + '<script type="application/json" id="timeline-data">' + json.dumps(tl, ensure_ascii=False) + '</script>\n<script>\n'
       + p('app.js') + '</script>\n</body>\n</html>\n')
# guard: '<' directly inside inline math would be parsed as a tag
for m in re.finditer(r'\$[^$\n]{0,300}\$', p('body.html')):
    assert '<' not in m.group(0) or '&lt;' in m.group(0) or '<' not in m.group(0).replace('&lt;', ''), m.group(0)
open(os.path.join(here, '../index.html'), 'w').write(out)
print('index.html', len(out))
