"""usage: merge_notes.py <mapping.json> [...]   (mapping: {"merged": {"<removed-slug>": "<canonical-slug>"}, "trimmed": [...]})
Deletes removed notes, repoints every link / related entry / index.json slot to the canonical note, rebuilds pages.
Only `related:` lines that contain a removed slug are rewritten."""
import json, re, glob, os, subprocess, sys
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..')); maps = [os.path.abspath(p) for p in sys.argv[1:]]; os.chdir(R)
M = {}
for f in maps: M.update(json.load(open(f)).get('merged', {}))
for a, b in list(M.items()):          # resolve chains a→b→c
    while b in M: b = M[b]
    M[a] = b
assert all(os.path.exists(f'scripts/notes_split/{b}.md') for b in M.values()), 'canonical missing'
print('merges:', M)
if not M: sys.exit(0)
link = re.compile(r'(?<![\w-])(' + '|'.join(map(re.escape, M)) + r')\.html') if M else None
def fix_related(fm, self_slug):
    def rep(m):
        items = [M.get(x.strip(' \'"'), x.strip(' \'"')) for x in m.group(2).split(',') if x.strip(' \'"')]
        out = []; [out.append(x) for x in items if x != self_slug and x not in out]
        return f'{m.group(1)}[{", ".join(out)}]' if any(x.strip(' \'"') in M for x in m.group(2).split(',')) else m.group(0)
    return re.sub(r'^(related:\s*)\[(.*?)\]', rep, fm, flags=re.M)
for f in glob.glob('scripts/notes_split/*.md'):
    slug = os.path.basename(f)[:-3]
    if slug in M: continue
    s = open(f).read(); fm_end = s.index('\n---', 4)
    t = fix_related(s[:fm_end], slug) + link.sub(lambda m: M[m.group(1)] + '.html', s[fm_end:])
    if t != s: open(f, 'w').write(t)
for f in glob.glob('scripts/tutorials_raw/*.html') + glob.glob('scripts/tutorial_src/*/build/index.html'):
    s = open(f).read(); t = link.sub(lambda m: M[m.group(1)] + '.html', s)
    if t != s: open(f, 'w').write(t)
idx = json.load(open('scripts/notes_split/index.json'))
entry = {n['slug']: n for g in idx for n in g['notes'] if n['slug'] not in M}
for g in idx:
    out = []
    for n in g['notes']:
        n = dict(entry[M[n['slug']]]) if n['slug'] in M else n
        n['related'] = [x for x in dict.fromkeys(M.get(r, r) for r in n.get('related') or []) if x != n['slug']]
        if n['slug'] not in [o['slug'] for o in out]: out.append(n)
    g['notes'] = out
json.dump(idx, open('scripts/notes_split/index.json', 'w'), ensure_ascii=False, indent=1)
for a in M:
    subprocess.run(['git', 'rm', '-q', f'scripts/notes_split/{a}.md'], check=True)
    if os.path.exists(f'tutorials/{a}.html'): subprocess.run(['git', 'rm', '-q', f'tutorials/{a}.html'], check=True)
os.chdir('scripts'); subprocess.run([sys.executable, 'tutorials.py'], check=True); subprocess.run([sys.executable, 'render.py'], check=True)
os.chdir(R)
left = subprocess.run(['grep', '-rlE', r'(' + '|'.join(M) + r')\.html', 'tutorials', 'index.html', 'scripts/notes_split'], capture_output=True, text=True).stdout
print('dangling links:', left or 'none')
