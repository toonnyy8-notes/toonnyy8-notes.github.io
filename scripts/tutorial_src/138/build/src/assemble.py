"""Assemble build/index.html from the pieces in this directory. Run from anywhere: python3 assemble.py"""
import json, os, re
H = os.path.dirname(os.path.abspath(__file__))
B = os.path.normpath(os.path.join(H, '..', '..'))            # tutorial_src/138
S441 = os.path.normpath(os.path.join(B, '..', '441', 'build', 'src', 'app.js'))
rd = lambda p: open(os.path.join(H, p), encoding='utf-8').read()

# shared helpers copied verbatim from the No.441 shell
a441 = open(S441, encoding='utf-8').read().splitlines()
common = '\n'.join(a441[2:20])                                 # lines 3-20: $, el, tx, lin, lp, widthOf, f4..., announce, segInit
full441 = '\n'.join(a441)
tl = full441[full441.index('function renderTimeline'):full441.index('return data}') + len('return data}')]
toc = full441[full441.index('function initToc'):full441.index('/* ================= self test')]

# Table 1 (paper, verbatim; identical to verify/check.py T / PRINTED and design.md 5.6)
T1 = {
 'math': [('TV–MC（GRPO）', [34.6, 20.8, 33.3, 21.9], '27.7 (+0.00)'), ('TV–Binary（DPPO）', [46.7, 32.5, 42.5, 34.2], '39.0 (+40.79)'),
          ('TV–Atomic', [45.0, 28.3, 40.8, 30.9], '36.3 (+31.05)'), ('Hellinger–MC', [37.9, 30.8, 35.4, 23.1], '31.8 (+14.80)'),
          ('Hellinger–Binary', [42.1, 32.9, 37.5, 29.0], '35.4 (+27.80)'), ('Hellinger–Atomic', [42.9, 32.5, 43.3, 29.0], '36.9 (+33.21)'),
          ('χ²–MC', [34.6, 20.8, 33.3, 21.9], '27.7 (+0.00)'), ('χ²–Binary', [45.0, 29.2, 36.7, 28.1], '34.8 (+25.63)'),
          ('χ²–Atomic（RIPO）', [47.9, 32.5, 47.5, 34.2], '40.5 (+46.21)'), ('KL–MC（DAPO）', [45.0, 27.5, 37.5, 28.5], '34.6 (+24.91)'),
          ('KL–Binary', [47.1, 29.6, 42.9, 29.0], '37.2 (+34.30)'), ('KL–Atomic', [48.9, 33.9, 48.7, 34.4], '41.5 (+49.82)'),
          ('GSPO', [42.9, 28.7, 36.3, 32.5], '35.1 (+26.71)'), ('GMPO', [41.7, 30.0, 35.8, 33.0], '35.1 (+26.71)'),
          ('SAPO', [37.3, 28.7, 41.7, 32.3], '35.0 (+26.35)'), ('CISPO', [31.7, 23.8, 38.3, 28.0], '30.5 (+10.11)'),
          ('Clip-Cov', [36.3, 22.9, 36.6, 30.2], '31.5 (+13.72)'), ('Base Model', [3.8, 3.3, 15.0, 10.8], '8.2 (−70.40)')],
 'code': [('TV–MC（GRPO）', [61.9, 57.4, 36.2, 54.2], '52.4 (+0.00)'), ('TV–Binary（DPPO）', [63.5, 58.6, 37.6, 55.6], '53.8 (+2.67)'),
          ('TV–Atomic', [64.2, 60.5, 38.1, 56.7], '54.9 (+4.77)'), ('Hellinger–MC', [66.4, 58.2, 36.3, 56.8], '54.4 (+3.82)'),
          ('Hellinger–Binary', [66.3, 60.2, 39.2, 57.7], '55.9 (+6.68)'), ('Hellinger–Atomic', [68.2, 63.7, 40.3, 60.3], '58.1 (+10.88)'),
          ('χ²–MC', [61.9, 57.4, 36.2, 54.2], '52.4 (+0.00)'), ('χ²–Binary', [64.8, 59.3, 38.6, 56.3], '54.8 (+4.58)'),
          ('χ²–Atomic（RIPO）', [68.2, 62.3, 39.3, 58.9], '57.2 (+9.16)'), ('KL–MC（DAPO）', [64.0, 59.1, 37.4, 56.8], '54.3 (+3.63)'),
          ('KL–Binary', [66.4, 61.4, 38.3, 57.3], '55.9 (+6.68)'), ('KL–Atomic', [70.7, 65.1, 42.3, 60.8], '59.7 (+13.93)'),
          ('GSPO', [63.8, 57.9, 36.6, 56.3], '53.7 (+2.48)'), ('GMPO', [64.0, 59.2, 37.4, 56.8], '54.4 (+3.82)'),
          ('SAPO', [63.7, 57.0, 37.5, 55.9], '53.5 (+2.10)'), ('CISPO', [61.6, 54.6, 35.1, 53.9], '51.3 (−2.10)'),
          ('Clip-Cov', [62.9, 57.4, 36.0, 55.5], '53.0 (+1.15)'), ('Base Model', [32.3, 32.1, 16.5, 32.1], '28.3 (−45.99)')]}
def rows(dom):
    out = []
    for nm, v, avg in T1[dom]:
        hl = ' class="hl"' if nm in ('KL–Atomic',) else ''
        out.append(f'<tr{hl}><td>{nm}</td>' + ''.join(f'<td class="num">{x:.1f}</td>' for x in v) + f'<td class="num">{avg}</td></tr>')
    return ''.join(out)

body = rd('body1.html') + rd('body2.html') + rd('body3.html')
body = body.replace('0<V', '0&lt;V')                          # keep "<V" from being parsed as a tag
body = body.replace('<tbody id="t1-math-body"></tbody>', '<tbody>' + rows('math') + '</tbody>')
body = body.replace('<tbody id="t1-code-body"></tbody>', '<tbody>' + rows('code') + '</tbody>')
assert '<tbody id="t1' not in body
app = rd('app.js').replace('/*@COMMON@*/', common).replace('/*@TIMELINE@*/', tl).replace('/*@TOCTHEME@*/', toc)
assert '/*@' not in app
tlj = json.dumps(json.load(open(os.path.join(B, 'notes', 'timeline.json'), encoding='utf-8')), ensure_ascii=False).replace('</', '<\\/')
out = (rd('head.html') + '<style>\n' + rd('css_base.css') + rd('css_add.css') + '</style>\n</head>\n' + body
       + f'\n<script type="application/json" id="timeline-data">{tlj}</script>\n'
       + '<script type="text/js-worker" id="w-vajda">\n' + rd('worker.js') + '</script>\n<script>\n' + app + '</script>\n</body>\n</html>\n')
open(os.path.join(B, 'build', 'index.html'), 'w', encoding='utf-8').write(out)
print(len(out.encode()), 'bytes')
