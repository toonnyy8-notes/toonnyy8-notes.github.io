"""Parse Tables 4-6 (all rows x 5 k, mean ± SD) from paper.txt -> results.json (page constant RESULTS)."""
import re, json, sys, pathlib
src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parent.parent / 'src/paper.txt').read_text()
ROW = re.compile(r'^\s+(\S.*?)\s{2,}((?:\d+\.\d\s*±\s*\d+\.\d\s*){5})$')
def table(start, n):
    i = src.index(start); out = []
    for line in src[i:].splitlines()[1:]:
        m = ROW.match(line)
        if m:
            nums = [float(x) for x in re.findall(r'\d+\.\d', m.group(2))]
            out.append({'name': m.group(1).strip(), 'vals': [[nums[2*j], nums[2*j+1]] for j in range(5)]})
            if len(out) == n: return out
    raise SystemExit(f'only {len(out)} rows for {start}')
R = {'MATH': table('Table 4 MATH pass@k', 13), 'HARP': table('Table 5 HARP pass@k', 12), 'OLYMP': table('Table 6 Olympiad pass@k', 12)}
out = pathlib.Path(__file__).parent / 'results.json'
out.write_text(json.dumps(R, ensure_ascii=False, indent=1))
for ds, rows in R.items(): print(ds, len(rows), [r['name'] for r in rows])
