import re, json, sys, pathlib
d = pathlib.Path(sys.argv[1])
notes = {}
for p in sorted(d.glob('*.md')):
    t = p.read_text()
    fm, body = t.split('---\n', 2)[1], t.split('---\n', 2)[2]
    meta = dict(l.split(': ', 1) for l in fm.strip().splitlines())
    assert meta['slug'] == p.stem, p
    # drop any previously generated keys (idempotent)
    fm = ''.join(l + '\n' for l in fm.strip().splitlines() if not l.startswith(('arxiv:', 'related:')))
    ids = list(dict.fromkeys(re.findall(r'(?<![\d.])(\d{4}\.\d{4,5})(?![\d])', body)))
    rel = list(dict.fromkeys(m for m in re.findall(r'\]\(([a-z0-9-]+)\.html\)', body) if m != p.stem))
    notes[p.stem] = dict(meta=meta, fm=fm, body=body, ids=ids, rel=rel)
bad = [(s, r) for s, n in notes.items() for r in n['rel'] if r not in notes]
assert not bad, bad
q = lambda xs: '[' + ', '.join(f'"{x}"' for x in xs) + ']'
for s, n in notes.items():
    (d / f'{s}.md').write_text(f"---\n{n['fm']}arxiv: {q(n['ids'])}\nrelated: {q(n['rel'])}\n---\n{n['body']}")
# reading order: each hub followed by children in the order its body links them
hubs = ['continuous-flow-lm-overview', 'weight-sharing-recursive-overview', 'latent-diffusion-rae-overview',
        'long-sequence-consistency-overview', 'causal-attention-diffusion-overview', 'ttt-sequence-modeling-overview']
seen, groups = set(), []
for h in hubs:
    sec = notes[h]['body'].split('## 建議閱讀順序')[1]
    kids = [k for k in dict.fromkeys(re.findall(r'\]\(([a-z0-9-]+)\.html\)', sec)) ]
    seen.update([h, *kids])
    groups.append(dict(hub=h, notes=[dict(slug=k, title=notes[k]['meta']['title'], cluster=notes[k]['meta']['cluster'],
                       kind=notes[k]['meta']['kind'], related=notes[k]['rel']) for k in [h, *kids]]))
orphans = set(notes) - seen
assert not orphans, orphans
(d / 'index.json').write_text(json.dumps(groups, ensure_ascii=False, indent=2))
for g in groups: print(g['hub'], [n['slug'] for n in g['notes'][1:]])
print(len(notes), 'notes')
