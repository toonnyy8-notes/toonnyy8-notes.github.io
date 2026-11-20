import sys, json
sys.path.insert(0, '../wt-paper-master/scripts/tutorial_src/441/verify')
import tables
B = '../wt-paper-master/scripts/tutorial_src/441/'
rows = []
for ds, env, v, se in tables.T1 + tables.T1K:
    mx = max(x for x in v if x is not None)
    cells = ''.join(f'<td class="num">{"—" if x is None else (str(x) + (f" ±{se}" if j == 7 else ""))}{" ▲" if x == mx else ""}</td>' for j, x in enumerate(v))
    rows.append(f'<tr><td>{ds}</td><td>{env}</td>{cells}</tr>')
rows.append('<tr class="hl"><td colspan="2">運動控制平均</td>' + ''.join(f'<td class="num">{x}</td>' for x in tables.T1_AVG) + '</tr>')
rows.append('<tr class="hl"><td colspan="2">Kitchen 平均</td>' + ''.join(f'<td class="num">{"—" if x is None else x}</td>' for x in tables.T1K_AVG) + '</tr>')
head = open('head.html').read()
css = open('css_base.css').read() + open('css_add.css').read()
body = open('body1.html').read() + open('body2.html').read() + open('body3.html').read()
body = body.replace('<tbody id="t1-body"></tbody>', '<tbody>' + ''.join(rows) + '</tbody>')
body = body.replace('<caption>Table 1：離線 RL（正規化分數）</caption>', '<caption>Table 1：離線 RL（正規化分數；▲ 為該列最高）</caption>')
tl = json.dumps(json.load(open(B + 'notes/timeline.json')), ensure_ascii=False).replace('</', '<\\/')
out = head + '<style>\n' + css + '</style>\n</head>\n' + body + f'\n<script type="application/json" id="timeline-data">{tl}</script>\n<script>\n' + open('app.js').read() + '</script>\n</body>\n</html>\n'
import os; os.makedirs(B + 'build', exist_ok=True)
open(B + 'build/index.html', 'w').write(out)
print(len(out.encode()))
