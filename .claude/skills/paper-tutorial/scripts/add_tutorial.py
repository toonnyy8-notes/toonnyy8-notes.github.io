"""usage: add_tutorial.py <paper-id> <slug> "<title>"
Publishes scripts/tutorial_src/<id>/: build/index.html → scripts/tutorials_raw/<slug>.html, notes/*.md → scripts/notes_split/
(new hub group appended to index.json), registers the page in tutorial_list.json, then rebuilds all pages."""
import sys, json, re, shutil, subprocess, pathlib, yaml
pid, slug, title = int(sys.argv[1]), sys.argv[2], sys.argv[3]
S = pathlib.Path(__file__).resolve().parents[4] / 'scripts'; src = S / 'tutorial_src' / str(pid)
shutil.copy(src / 'build/index.html', S / 'tutorials_raw' / f'{slug}.html')
meta = {}
for f in sorted((src / 'notes').glob('*.md')):
    dst = S / 'notes_split' / f.name
    if dst.exists() and dst.read_text() != f.read_text():
        sys.exit(f'note slug collision: {f.name} already exists with different content')
    shutil.copy(f, dst)
    m = yaml.safe_load(f.read_text().split('---', 2)[1]); meta[m['slug']] = {k: m.get(k) for k in ['slug', 'title', 'cluster', 'kind', 'related']}
if meta:
    hub = next(s for s, m in meta.items() if m['kind'] == 'hub')
    linked = re.findall(r'\]\(([a-z0-9-]+)\.html\)', (src / 'notes' / f'{hub}.md').read_text())
    order = [s for s in dict.fromkeys(linked) if s in meta and s != hub] + [s for s in meta if s not in linked and s != hub]
    idx = [g for g in json.loads((S / 'notes_split/index.json').read_text()) if g['hub'] != hub]
    idx.append({'hub': hub, 'notes': [meta[hub]] + [meta[s] for s in order]})
    (S / 'notes_split/index.json').write_text(json.dumps(idx, ensure_ascii=False, indent=1))
lst = json.loads((S / 'tutorial_list.json').read_text())
entry = {'slug': slug, 'title': title, 'papers': [pid]}
pos = next((i for i, t in enumerate(lst) if t['slug'] == slug), None)   # re-publishing keeps the page's position
if pos is None: lst.append(entry)
else: lst[pos] = entry
(S / 'tutorial_list.json').write_text(json.dumps(lst, ensure_ascii=False, indent=1))
for script in ['tutorials.py', 'render.py']:
    subprocess.run([sys.executable, str(S / script)], check=True)
# announce on the news page (最新消息)
subprocess.run([sys.executable, str(S / 'news.py'), 'add', 'tutorial', f'互動教學：{title}（No.{pid}）',
                f'新增互動教學與 {len(meta)} 篇研究筆記。', f'開啟教學=tutorials/{slug}.html', f'論文 No.{pid}=index.html#paper-{pid}'], check=True)
print(f'published tutorials/{slug}.html with {len(meta)} notes')
