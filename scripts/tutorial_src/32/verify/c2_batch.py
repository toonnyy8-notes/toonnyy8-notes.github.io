"""C2: Challenger batch reward pipeline. Pure-python port of the page algorithm
(BLEU = nltk sentence_bleu + method1, average-linkage cut at tau, r_rep=|C|/B,
r=max(0,r_unc-r_rep), GRPO z-score with sample std). Run with the venv python
to cross-check against nltk/sklearn; plain python3 skips the cross-check."""
import math, re, sys
from collections import Counter

RAW = [
 "<question>Find the number of positive integers n less than 100 such that n^2 + 1 is divisible by 5.</question>",
 "<question>Find the number of positive integers n less than 200 such that n^2 + 1 is divisible by 5.</question>",
 "<question>Find the number of positive integers k less than 500 for which k^3 - k is divisible by 7.</question>",
 "<question>A circle of radius 3 is inscribed in a right triangle with hypotenuse 15. Find the area of the triangle.</question>",
 "<question>A circle of radius 4 is inscribed in a square. Find the area of the region inside the square but outside the circle.</question>",
 "<question>Let f(x) = x^2 - 6x + 10. Find the minimum value of f(f(x)) over all real x.</question>",
 "<question>How many ways can 6 distinct books be arranged on a shelf so that two specific books are not adjacent?</question>",
 "<question>Compute the sum of all primes below 50.",
]
COUNTS = [5, 6, 4, 7, 3, 9, 5, 5]   # majority counts c_i out of m=10 (demo values)
M, B = 10, len(RAW)
EPS_NORM = 1e-6

def extract(s):
    mt = re.search(r"<question>(.*?)</question>", s, re.S)
    return mt.group(1).strip() if mt else None

def ngrams(t, n): return Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1))

def bleu(hyp, ref):  # nltk sentence_bleu([ref], hyp, smoothing=method1), weights .25x4
    h, r = hyp.split(), ref.split()
    nums, dens = [], []
    for n in range(1, 5):
        hc, rc = ngrams(h, n), ngrams(r, n)
        nums.append(sum(min(v, rc[g]) for g, v in hc.items()))
        dens.append(max(1, sum(hc.values())))
    if nums[0] == 0: return 0.0
    c, rl = len(h), len(r)
    bp = 1.0 if c > rl else (0.0 if c == 0 else math.exp(1 - rl / c))
    ps = [(0.1 / d) if nn == 0 else nn / d for nn, d in zip(nums, dens)]
    return bp * math.exp(math.fsum(0.25 * math.log(p) for p in ps))

def dist_matrix(qs):  # d[i][j] for i<j uses hyp=qs[i], ref=qs[j]; mirrored (upper triangle rule)
    n = len(qs); d = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            d[i][j] = d[j][i] = 1 - bleu(qs[i], qs[j])
    return d

def avg_linkage(d, tau):  # merge closest pair while average-linkage distance < tau
    cl = [[i] for i in range(len(d))]
    while len(cl) > 1:
        best = None
        for a in range(len(cl)):
            for b in range(a+1, len(cl)):
                v = sum(d[i][j] for i in cl[a] for j in cl[b]) / (len(cl[a])*len(cl[b]))
                if best is None or v < best[0] - 1e-12: best = (v, a, b)
        if best[0] >= tau: break
        _, a, b = best; cl[a] = cl[a] + cl[b]; del cl[b]
    return cl

def pipeline(raw, counts, tau, use_rep=True):
    qs = [extract(s) for s in raw]
    valid = [i for i, q in enumerate(qs) if q is not None]
    d = dist_matrix([qs[i] for i in valid])
    clusters = [[valid[k] for k in c] for c in avg_linkage(d, tau)]
    size = {i: len(c) for c in clusters for i in c}
    rows = []
    for i in range(len(raw)):
        if qs[i] is None: rows.append((i, None, None, None, 0.0)); continue
        ru = 1 - 2*abs(counts[i]/M - 0.5)
        rr = size[i]/B if use_rep else 0.0
        rows.append((i, size[i], ru, rr, max(0.0, ru - rr)))
    rs = [r[-1] for r in rows]
    mu = sum(rs)/len(rs); sd = math.sqrt(sum((x-mu)**2 for x in rs)/(len(rs)-1))
    adv = [(x-mu)/(sd+EPS_NORM) for x in rs]
    return qs, valid, d, clusters, rows, mu, sd, adv

if __name__ == "__main__":
    for tau, rep in [(0.5, True), (0.5, False), (0.65, True), (0.7, True), (0.95, True)]:
        qs, valid, d, clusters, rows, mu, sd, adv = pipeline(RAW, COUNTS, tau, rep)
        print(f"\n== tau={tau} rep={rep} clusters={[[i+1 for i in c] for c in clusters]}")
        for (i, sz, ru, rr, r), a in zip(rows, adv):
            print(f"q{i+1} size={sz} r_unc={ru} r_rep={rr} r={r:.4f} A={a:+.4f}")
        print(f"mean={mu:.4f} std={sd:.4f}")
    qs, valid, d, *_ = pipeline(RAW, COUNTS, 0.5)
    print("\nBLEU distance matrix (valid q1..q7):")
    for row in d: print(" ".join(f"{x:.3f}" for x in row))
    print("asymmetry example BLEU(q1,q2) vs BLEU(q2,q1):", round(bleu(qs[0], qs[1]), 4), round(bleu(qs[1], qs[0]), 4),
          "| q1,q3:", round(bleu(qs[0], qs[2]), 4), round(bleu(qs[2], qs[0]), 4),
          "| q4,q6:", round(bleu(qs[3], qs[5]), 4), round(bleu(qs[5], qs[3]), 4))
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        print("nltk/sklearn not available: cross-check skipped"); sys.exit()
    sm = SmoothingFunction().method1
    V = [qs[i] for i in valid]; mx = 0
    for a in V:
        for b in V:
            if a is b: continue
            mx = max(mx, abs(bleu(a, b) - sentence_bleu([b.split()], a.split(), smoothing_function=sm)))
    print("max |BLEU port - nltk| =", mx); assert mx < 1e-12
    full = np.array([[0 if i == j else 1 - sentence_bleu([V[j].split()], V[i].split(), smoothing_function=sm)
                      for j in range(len(V))] for i in range(len(V))])
    for tau in [0.3, 0.5, 0.65, 0.7, 0.95]:
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=tau, metric="precomputed",
                                      linkage="average").fit(full).labels_
        sk = sorted(sorted(valid[k] for k in range(len(V)) if lab[k] == L) for L in set(lab))
        mine = sorted(sorted(c) for c in avg_linkage(dist_matrix(V), tau))
        mine = sorted(sorted(valid[k] for k in c) for c in mine)
        print(f"tau={tau}: sklearn(full asym matrix)={sk} port(upper triangle)={mine}"); assert sk == mine
    # sklearn on an asymmetric precomputed matrix reads only the upper triangle (i<j)
    A = np.array([[0, .4, .9], [.6, 0, .9], [.9, .9, 0]])
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=.5, metric="precomputed", linkage="average").fit(A).labels_
    print("asym test labels (upper .4 / lower .6, tau .5):", lab); assert lab[0] == lab[1]
    print("cross-check OK")
