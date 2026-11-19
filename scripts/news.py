"""Render news.json → ../news.html (最新消息). Entries: {time 'YYYY-MM-DD HH:MM', type, title, body, links:[{label, href}]}.
Usage: python3 news.py            render only
       python3 news.py add <type> "<title>" "<body>" [label=href ...]   append an entry stamped now, then render"""
import json, os, sys, html, datetime, collections
os.chdir(os.path.dirname(os.path.abspath(__file__)))
TYPES = {'update': '更新', 'paper': '新論文', 'tutorial': '互動教學', 'note': '研究筆記', 'doc': '文件'}
news = json.load(open('news.json'))
if len(sys.argv) > 1 and sys.argv[1] == 'add':
    t, title, body, *links = sys.argv[2:]
    assert t in TYPES, f'type must be one of {list(TYPES)}'
    news.append({'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), 'type': t, 'title': title, 'body': body,
                 'links': [dict(zip(('label', 'href'), l.split('=', 1))) for l in links]})
    json.dump(news, open('news.json', 'w'), ensure_ascii=False, indent=1)

esc = html.escape
news.sort(key=lambda n: n['time'], reverse=True)
days = collections.OrderedDict()
for n in news: days.setdefault(n['time'][:10], []).append(n)
count = collections.Counter(n['type'] for n in news)
chips = ''.join(f'<button type="button" data-t="{k}" aria-pressed="false">{v}<span>{count[k]}</span></button>' for k, v in TYPES.items() if count[k])
def item(n):
    links = ''.join(f'<a href="{esc(l["href"])}">{esc(l["label"])} →</a>' for l in n['links'])
    return (f'<li data-t="{n["type"]}"><time datetime="{n["time"].replace(" ", "T")}">{n["time"][11:]}</time>'
            f'<div><span class="ty ty-{n["type"]}">{TYPES[n["type"]]}</span><h3>{esc(n["title"])}</h3>'
            f'<p>{esc(n["body"])}</p>{f"<p class=links>{links}</p>" if links else ""}</div></li>')
body = ''.join(f'<section class="day"><h2><time datetime="{d}">{d}</time></h2><ol>{"".join(item(n) for n in ns)}</ol></section>' for d, ns in days.items())

open('../news.html', 'w').write(f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>最新消息 · Paper Master</title><meta name="description" content="Paper Master 的更新公告、新加入的論文、互動教學與文件。">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Noto+Serif+TC:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Sans+TC:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{--bg:#F6F4EE;--surface:#FFFFFF;--surface-2:#EEEBE3;--ink:#1D1C19;--ink-2:#57534B;--ink-3:#67635A;--line:#DCD7CB;--accent:#B4532A;--accent-soft:#F3E2D8;
  --f-display:"Newsreader","Noto Serif TC",Georgia,serif;--f-cjk-serif:"Noto Serif TC","Songti TC",Georgia,serif;--f-body:"IBM Plex Sans","Noto Sans TC",system-ui,sans-serif;--f-mono:"JetBrains Mono",ui-monospace,monospace;--ease:cubic-bezier(.2,0,0,1);color-scheme:light}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#161513;--surface:#1F1E1B;--surface-2:#2A2925;--ink:#EEEBE4;--ink-2:#B5B0A5;--ink-3:#9C978C;--line:#36342F;--accent:#E58C66;--accent-soft:#3B2820;color-scheme:dark}}}}
:root[data-theme=dark]{{--bg:#161513;--surface:#1F1E1B;--surface-2:#2A2925;--ink:#EEEBE4;--ink-2:#B5B0A5;--ink-3:#9C978C;--line:#36342F;--accent:#E58C66;--accent-soft:#3B2820;color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 var(--f-body);-webkit-font-smoothing:antialiased}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}}
.top nav{{max-width:1200px;margin:0 auto;padding:0 24px;height:56px;display:flex;align-items:center;gap:24px;font-size:14px;border-bottom:1px solid var(--line)}}
.top a{{color:var(--ink-2);text-decoration:none}}.top a:hover{{color:var(--accent)}}.top .brand{{font-family:var(--f-display);font-size:20px;font-weight:500;color:var(--ink)}}
.top a[aria-current]{{color:var(--ink);font-weight:600}}
main{{max-width:1200px;margin:0 auto;padding:0 24px 120px;display:grid;grid-template-columns:repeat(12,1fr);gap:24px}}
.hero{{grid-column:1/-1;padding:96px 0 40px;border-bottom:1px solid var(--ink);display:grid;grid-template-columns:repeat(12,1fr);gap:24px;align-items:end}}
.hero h1{{grid-column:1/8;margin:0;font-family:var(--f-cjk-serif);font-weight:700;font-size:clamp(44px,7vw,88px);line-height:1.05}}
.hero p{{grid-column:8/13;margin:0;color:var(--ink-2);font-size:15px}}
.filters{{grid-column:1/4;padding-top:48px}}
.filters .h{{font:600 11px/1 var(--f-body);letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);margin:0 0 12px}}
.filters div{{display:flex;flex-direction:column;gap:2px;position:sticky;top:24px}}
.filters button{{all:unset;box-sizing:border-box;cursor:pointer;display:flex;justify-content:space-between;align-items:center;min-height:40px;padding:8px 12px;border-radius:8px;font:500 14px var(--f-body);color:var(--ink);transition:background .16s var(--ease)}}
.filters button:hover{{background:var(--surface-2)}}.filters button:focus-visible{{outline:2px solid var(--accent);outline-offset:-2px}}
.filters button span{{font:12px var(--f-mono);color:var(--ink-3)}}
.filters button[aria-pressed=true]{{background:var(--accent-soft);color:var(--ink)}}.filters button[aria-pressed=true] span{{color:var(--accent)}}
.feed{{grid-column:4/13;min-width:0;padding-top:40px}}
.day h2{{margin:0;padding:8px 0;font:500 13px var(--f-mono);color:var(--ink-3);border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:1}}
.day ol{{list-style:none;margin:0 0 40px;padding:0}}
.day li{{display:grid;grid-template-columns:64px 1fr;gap:16px;padding:20px 0;border-bottom:1px solid var(--line)}}
.day li[hidden]{{display:none}}
.day time{{font:13px/1.9 var(--f-mono);color:var(--ink-3)}}
.ty{{display:inline-block;font:600 11px/1 var(--f-body);letter-spacing:.04em;padding:5px 8px;border-radius:6px;background:var(--surface-2);color:var(--ink-2)}}
.ty-tutorial{{background:var(--accent-soft);color:var(--accent)}}
.day h3{{margin:8px 0 4px;font-family:var(--f-cjk-serif);font-weight:600;font-size:20px;line-height:1.4}}
.day p{{margin:0;color:var(--ink-2);font-size:15px}}
.day .links{{margin-top:8px;display:flex;flex-wrap:wrap;gap:4px 20px}}
.day .links a{{color:var(--accent);text-decoration:none;font-weight:500;font-size:14px}}.day .links a:hover{{text-decoration:underline}}
.empty{{color:var(--ink-3);padding:24px 0}}
@media (max-width:900px){{main,.hero{{grid-template-columns:1fr}}.hero>*,main>*{{grid-column:1/-1!important}}.hero{{padding:56px 0 32px}}
  .filters{{padding-top:24px}}.filters div{{flex-direction:row;flex-wrap:wrap;position:static}}.filters button{{gap:8px;border:1px solid var(--line)}}
  .feed{{padding-top:8px}}.day li{{grid-template-columns:1fr;gap:4px}}main,.top nav{{padding-left:16px;padding-right:16px}}}}
@media (prefers-reduced-motion:reduce){{*{{transition-duration:.001ms!important}}}}
@media (max-width:600px){{.top nav{{gap:16px;font-size:13px;white-space:nowrap;overflow-x:auto;scrollbar-width:none}}.top .brand{{font-size:18px}}}}
</style></head>
<body><header class="top"><nav aria-label="網站"><a class="brand" href="index.html">Paper Master</a><a href="index.html">論文關聯圖</a><a href="tutorials/index.html">教學與筆記</a><a href="news.html" aria-current="page">最新消息</a></nav></header>
<main>
<section class="hero"><h1>最新消息</h1><p>更新公告、新加入的論文、互動教學與文件，依時間由新到舊排列。共 {len(news)} 則，最後更新 {news[0]['time']}。</p></section>
<aside class="filters" aria-label="依類型篩選"><p class="h">類型</p><div role="group">{chips}</div></aside>
<div class="feed">{body}<p class="empty" hidden>這個類型目前沒有消息。</p></div>
</main>
<script>
const on = new Set(), btns = [...document.querySelectorAll('.filters button')];
btns.forEach(b => b.onclick = () => {{
  on.has(b.dataset.t) ? on.delete(b.dataset.t) : on.add(b.dataset.t); b.setAttribute('aria-pressed', on.has(b.dataset.t));
  let shown = 0;
  document.querySelectorAll('.day li').forEach(li => {{ li.hidden = on.size > 0 && !on.has(li.dataset.t); shown += !li.hidden; }});
  document.querySelectorAll('.day').forEach(d => d.hidden = !d.querySelector('li:not([hidden])'));
  document.querySelector('.empty').hidden = shown > 0;
}});
</script></body></html>""")
print('news.html', len(news), 'entries')
