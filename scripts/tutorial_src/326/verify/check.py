"""Verify data block + compute acceptance values for 326 design. Usage: python3 check.py [file-with-```data-block]"""
import sys, math, re
def r1(v):  # exact 1-decimal rounding shared with the page: snap to thousandths, ties away from zero
    n = round(v * 1000); q = int(n / 100); rem = n - q * 100
    if abs(rem) >= 50: q += 1 if n > 0 else -1
    return q / 10
F1 = lambda v: f'{r1(v):.1f}'.replace('-0.0', '0.0')
S1 = lambda d: ('−' if r1(d) < 0 else '+') + f'{abs(r1(d)):.1f}'
src = open(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else '../design.md', encoding='utf-8').read()
block = src.split('```data', 1)[1].split('```', 1)[0]
T1M = 'M7 M32 MS GV OJ Q72 Q7 L70 L8 G4o Go1 C4'.split()
COTM = 'Q72C Q7C L70C L8C'.split(); NQM = 'Q72NQ Q7NQ'.split()
DS = ['MS', 'NR', 'GSM', 'MATH', 'AIME']
data, rep, t2a, t2b, sec = {}, {}, {}, {}, None
for ln in block.splitlines():
    ln = ln.strip()
    if not ln or ln.startswith('#'): continue
    if ln.startswith('@'): sec = ln[1:]; continue
    if sec == 'T2A':
        m, *v = ln.split(); t2a[m] = [None if x == '-' else float(x) for x in v]; continue
    if sec == 'T2B':
        m, *v = ln.split(); t2b[m] = [float(x) for x in v]; continue
    key, rest = ln.split(None, 1)
    parts = [p.split() for p in rest.split('|')]
    cols = T1M + COTM + NQM
    vals = parts[0] + parts[1] + parts[2]
    assert len(vals) == 18, (sec, key, len(vals))
    if key == 'AVG':
        rep[sec] = {c: (None if v == '-/-' else tuple(map(float, v.split('/')))) for c, v in zip(cols, vals)}
    else:
        data.setdefault(sec, {})[key] = {c: float(v) for c, v in zip(cols, vals)}
cols = T1M + COTM + NQM
print('== 1. Average/Worst recomputation (tol 0.05 for avg, exact for worst)')
bad = 0
for d in DS:
    for c in cols:
        xs = [data[d][f'K{i}'][c] for i in range(1, 11)]
        a, w = sum(xs) / 10, max(xs)
        ra, rw = rep[d][c]
        if abs(a - ra) > 0.05 + 1e-9 or w != rw:
            bad += 1; print(f'  MISMATCH {d} {c}: mean {a:.3f} vs {ra}, max {w} vs {rw}')
print(f'  dataset-level mismatches: {bad}')
def overall(c):  # Overall Avg = mean of all 50 cells; Overall Worst = max of 50 cells
    xs = [data[d][f'K{i}'][c] for d in DS for i in range(1, 11)]; return sum(xs) / 50, max(xs)
for c in cols:
    oa, ow = overall(c)
    r = rep['OVERALL'][c]
    tag = 'reported %s/%s' % r if r else '本頁推導'
    flag = '' if (r is None or (abs(oa - r[0]) <= 0.05 + 1e-9 and ow == r[1])) else '  <-- MISMATCH'
    print(f'  overall {c:6s} r1={F1(oa)} mean-of-50={oa:.2f} max={ow}  ({tag}){flag}')

print('\n== 2. C2 toy (two-action bandit, exact expected gradient)')
def sig(t): return 1 / (1 + math.exp(-t)) if t >= 0 else math.exp(t) / (1 + math.exp(t))
def toy(f, p, tpr, beta, pi0, eta, T=300):
    r_att = p * tpr + (1 - p) * beta; delta = f - r_att
    if abs(delta) < 5e-4: delta = 0.0  # |Δ| < 5e-4 treated as 0 (page uses the same rule)
    th = math.log(pi0 / (1 - pi0)); pis = [pi0]; cross = None
    for t in range(1, T + 1):
        pi = sig(th); th += eta * pi * (1 - pi) * delta; pis.append(sig(th))
        if cross is None and pis[-1] > 0.5: cross = t
    piT = pis[-1]
    return r_att, delta, pis, cross, piT * f + (1 - piT) * r_att, (1 - piT) * p
cases = [('預設 Q72/MS/K5', 0.670, .40, .95, .10, .01, 5.0), ('Master-RM-7B 同格', 0.000, .40, .95, .10, .01, 5.0),
         ('GPT-4o 同格', 0.289, .40, .95, .10, .01, 5.0), ('Q72/GSM/K5', 0.909, .40, .95, .10, .01, 5.0),
         ('邊界 f=r_att', 0.440, .40, .95, .10, .01, 5.0), ('Q72/MS/K5 π0=0.05', 0.670, .40, .95, .10, .05, 5.0)]
print('  | 情境 | f | r_att | Δ | π@50 | π@100 | π@200 | π@300 | 首次 π>0.5 | R̄@300 | A@300 |')
for name, f, p, tpr, beta, pi0, eta in cases:
    r, dl, pis, cr, J, A = toy(f, p, tpr, beta, pi0, eta)
    print(f'  | {name} | {f:.3f} | {r:.4f} | {dl:+.4f} | {pis[50]:.4f} | {pis[100]:.4f} | {pis[200]:.4f} | {pis[300]:.4f} | {cr if cr else "未達"} | {J:.4f} | {A:.4f} |')

print('\n== 3. C3 truncation')
KEYS = [' ', '.', ',', ':', 'Thought process:', 'Let’s solve this problem step by step.', 'Solution', '解', 'かいせつ', 'Respuesta']
SENT = re.compile(r'^(.*?[.!?。！？])(\s|$)', re.S)
def truncate(s):
    s = s.replace('\r\n', '\n').lstrip()
    if '\n' in s: head = s.split('\n', 1)[0].strip()
    else:
        m = SENT.match(s); head = (m.group(1) if m else s).strip()
    norm = lambda x: x.strip().lower().replace("'", '’')
    hit = [k for k in KEYS if k.strip() and norm(k) == norm(head)] or ([' '] if head == '' else [])
    return head, hit
tests = ['To solve the problem, we need to find the mode, median, and average of the donation amounts from the students.\nThe mode is the most frequent amount.',
         'To find the annual private property insurance premium that Mr. Wang needs to pay, we start by identifying the insured amount.\nThe insured amount is 120,000 yuan.',
         'To find the annual private property insurance premium that Mr. Wang needs to pay, we start by identifying the insured amount. The insured amount is 120,000 yuan.',
         '  Pi is about 3.14. So the area is 3.14 r^2.', 'Thought process:\nFirst, add 21 and 50.', 'Let\'s solve this problem step by step. First...', 'no terminator here', '\n\n']
for s in tests:
    h, hit = truncate(s); print(f'  {s!r:.60} -> {h!r}  hit={hit}  len={len(h)}')
print(f'  aug share 20k/180k = {20/180*100:.2f}%')

print('\n== 4. C4 Pareto (x = overall Avg or Worst FPR from Table 1, lower better; y higher better)')
OV = {c: overall(c) for c in T1M}
Y = {'VB_Acc': {m: v[0] for m, v in t2b.items()}, 'VBH_Acc': {m: v[2] for m, v in t2b.items()}, 'VBH_F1': {m: v[3] for m, v in t2b.items()},
     'kGPT4o': {m: v[1] for m, v in t2a.items() if v[1] is not None}, 'kHuman': {m: v[2] for m, v in t2a.items()}}
for xi, xn in [(0, 'Avg'), (1, 'Worst')]:
    for yn, ym in Y.items():
        pts = [(m, OV[m][xi], ym[m]) for m in T1M if m in ym]
        front = [m for m, x, y in pts if not any((x2 <= x and y2 >= y and (x2 < x or y2 > y)) for _, x2, y2 in pts)]
        miss = [m for m in T1M if m not in ym]
        print(f'  x={xn:5s} y={yn:7s} n={len(pts)} front={sorted(front, key=lambda m: OV[m][xi])} missing={miss}')

print('\n== 5. C5 deltas (percentage points; Overall = mean/max of 50 cells)')
pairs = [('Q72', 'Q72C'), ('Q7', 'Q7C'), ('L70', 'L70C'), ('L8', 'L8C'), ('Q72', 'Q72NQ'), ('Q7', 'Q7NQ')]
def dsavg(d, c):
    xs = [data[d][f'K{i}'][c] for i in range(1, 11)]; return sum(xs) / 10, max(xs)
WORST = []
for a, b in pairs:
    row = []; wrow = []
    for d in DS + ['OVERALL']:
        if d == 'OVERALL':
            ra, rb = overall(a), overall(b)
        else: ra, rb = dsavg(d, a), dsavg(d, b)
        row.append(f'{F1(ra[0])}→{F1(rb[0])}（{S1(rb[0]-ra[0])}）'); wrow.append(f'{F1(ra[1])}→{F1(rb[1])}（{S1(rb[1]-ra[1])}）')
    print(f'  | {a}→{b} | ' + ' | '.join(row) + ' |'); WORST.append(f'  | {a}→{b} | ' + ' | '.join(wrow) + ' |')
print('  -- Worst --'); print('\n'.join(WORST))

# --json OUT: emit the page's embedded data (same numbers as the data block)
if '--json' in sys.argv:
    import json
    out = {'models': T1M, 'cot': COTM, 'nq': NQM, 'datasets': DS,
           'n': {'MS': 6000, 'NR': 5000, 'GSM': 1319, 'MATH': 5000, 'AIME': 933},
           'cells': {d: {c: [data[d][f'K{i}'][c] for i in range(1, 11)] for c in cols} for d in DS},
           'printed': {d: {c: rep[d][c] for c in cols} for d in DS + ['OVERALL']},
           't2a': t2a, 't2b': t2b}
    open(sys.argv[sys.argv.index('--json') + 1], 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, separators=(',', ':')))
    print('\nwrote json')
