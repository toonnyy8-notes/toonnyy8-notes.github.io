"""Fetch each paper's reference list from Semantic Scholar and keep the citations between papers in this database.
usage: python3 scripts/fetch_refs.py   (reads ../papers.jsonl, writes citations.json = sorted [citing_id, cited_id] pairs)
Only public arXiv IDs / DOIs are sent. References are matched to database papers by arXiv ID, then by normalized title."""
import json, re, time, urllib.request, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
P = [json.loads(l) for l in open('../papers.jsonl')]
AX = re.compile(r'(?:arxiv\.org/(?:abs|pdf|html)/|alphaxiv\.org/(?:abs|overview)/|huggingface\.co/papers/)(\d{4}\.\d{4,5})')
norm = lambda t: re.sub(r'[^a-z0-9]', '', (t or '').lower())
ids, byax, bytitle = {}, {}, {}
for p in P:
    m = AX.search(p['url'])
    if m: ids[p['id']] = 'ARXIV:' + m.group(1); byax[m.group(1)] = p['id']
    elif 'doi.org/' in p['url']: ids[p['id']] = 'DOI:' + p['url'].split('doi.org/')[1]
    bytitle[norm(p['title'])] = p['id']
edges, keys = set(), list(ids.items())
for i in range(0, len(keys), 100):
    chunk = keys[i:i + 100]
    body = json.dumps({'ids': [v for _, v in chunk]}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request('https://api.semanticscholar.org/graph/v1/paper/batch?fields=references.externalIds,references.title',
                                         data=body, headers={'Content-Type': 'application/json'})
            res = json.load(urllib.request.urlopen(req, timeout=120)); break
        except Exception as e:  # 429 rate limit: back off
            print('retry', i, e); time.sleep(10 * (attempt + 1))
    else: raise SystemExit('Semantic Scholar kept failing')
    for (pid, _), r in zip(chunk, res):
        for ref in (r or {}).get('references') or []:
            q = byax.get((ref.get('externalIds') or {}).get('ArXiv', '')) or bytitle.get(norm(ref.get('title')))
            if q and q != pid: edges.add((pid, q))
    time.sleep(3)
json.dump(sorted(edges), open('citations.json', 'w'))
print(len(edges), 'citations between', len({x for e in edges for x in e}), 'papers')
