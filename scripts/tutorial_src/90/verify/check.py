"""Acceptance numbers for tutorial 90 (Engram). Same algorithms as the page (design.md Sec. 4).
Run: python3 -I check.py   -> prints all tables; asserts the cross-checks against paper numbers."""
import math, unicodedata, re

# ---------- shared: mulberry32 (identical to the page's rng) ----------
def mulberry32(seed):
    a = seed & 0xFFFFFFFF
    def r():
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return r

def is_prime(n):
    if n < 2: return False
    if n % 2 == 0: return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0: return False
        i += 2
    return True

def head_primes(M, K, orders=2):
    """demo find_next_prime: smallest unused primes >= M, K per order, orders consecutive."""
    seen, out, c = set(), [], M
    for _ in range(orders):
        row = []
        while len(row) < K:
            if is_prime(c) and c not in seen:
                seen.add(c); row.append(c)
            c += 1
        out.append(row)
    return out

def multipliers(seed=0, n=3):
    r = mulberry32(seed)
    return [2 * math.floor(r() * 2**30) + 1 for _ in range(n)]

# ---------- C1: tokenizer compression ----------
TOK_RE = re.compile(r"\s*[^\W_]+|\s*[^\s\w]|\s*_", re.UNICODE)
def normalize(tok):
    s = unicodedata.normalize('NFKC', tok)
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch)[0] != 'M')
    s = s.lower()
    s = re.sub(r"[ \t\r\n]+", " ", s)
    if s == " ": return " "
    s = s.strip()
    return s if s else tok

def compress(text):
    toks = TOK_RE.findall(text)
    raw, canon, rid, cid = {}, {}, [], []
    for t in toks:
        raw.setdefault(t, len(raw) + 1)
        k = normalize(t)
        canon.setdefault(k, len(canon) + 1)   # canonical ids start at 1; 0 = pad
        rid.append(raw[t]); cid.append(canon[k])
    return toks, rid, cid, len(raw), len(canon)

def hash_heads(cid, t, n, mults, primes_n):
    toks = [cid[t - i] if t - i >= 0 else 0 for i in range(n)]  # tokens[0]=current
    mix = toks[0] * mults[0]
    for i in range(1, n):
        mix ^= toks[i] * mults[i]
    return mix, [mix % p for p in primes_n]

print("== C1 pipeline ==")
mults = multipliers(0)
print("multipliers(seed 0):", mults)
P = head_primes(10000, 8)
print("primes M=10000 K=8:", P)
for text in ["Only Alexander the Great could tame the horse Bucephalus.",
             "The cat saw the CAT. Thé café, the Café."]:
    toks, rid, cid, V, Vp = compress(text)
    print(repr(text)); print(" tokens", toks); print(" raw", rid); print(" canon", cid)
    print(f" |V|={V} |V'|={Vp} reduction={1-Vp/V:.4f}")
    t = 3 if len(toks) > 3 else len(toks) - 1
    for n in (2, 3):
        mix, idx = hash_heads(cid, t, n, mults, P[n - 2])
        print(f"  t={t} n={n} mix={mix} idx={idx}")

# ---------- C1: collision analytics ----------
def collide_stats(U, M, K):
    p1 = -math.expm1((U - 1) * math.log1p(-1.0 / M))
    pall = p1 ** K
    log10_full = math.log10(U - 1) - K * math.log10(M) if U > 1 else -math.inf
    return p1, pall, log10_full
# All K heads are residues of one mix (CRT): a full K-head collision needs equal mix once prod(M) > mix range R.
# Demo scale: token id < 2^17, multiplier < 2^31 -> mix < 2^48. Floor model: E_full ~ (U-1)/min(M^K, R).
LOG10_R = 48 * math.log10(2)
def full_floor(U, M, K):
    return math.log10(U - 1) - min(K * math.log10(M), LOG10_R) if U > 1 else -math.inf
print("\n== C1 analytic ==")
print("U, M, K -> p1, p1^K, log10 E_full (independent), log10 E_full (mix<2^48 floor)")
for U, M, K in [(20000, 10007, 1), (20000, 10007, 2), (20000, 10007, 8), (1e9, 2262400, 8),
                (1e6, 2262400, 1), (1e6, 2262400, 8), (1e10, 2262400, 8)]:
    p1, pa, lf = collide_stats(U, M, K)
    print(f"{U:g} {M:g} {K}: p1={p1:.6f} p1^K={pa:.6g} log10E={lf:.3f} floor={full_floor(U, M, K):.3f}")
assert abs(full_floor(1e9, 2262400, 8) - (9 - LOG10_R)) < 1e-6   # ~3.55e-6 per n-gram

def empirical(U, M, K, V=1000):  # page: Map keyed by index string, chunks of 2000 (same counts)
    pr = head_primes(M, K, orders=1)[0]
    occ = [dict() for _ in range(K)]; tup = {}
    keys = []
    for i in range(U):
        a, b = divmod(i, V)           # x_{t-1}=a+1, x_t=b+1 (ids start at 1)
        mix = (b + 1) * mults[0] ^ (a + 1) * mults[1]
        idx = tuple(mix % p for p in pr)
        keys.append(idx)
        for j in range(K): occ[j][idx[j]] = occ[j].get(idx[j], 0) + 1
        tup[idx] = tup.get(idx, 0) + 1
    c1 = sum(1 for k in keys if occ[0][k[0]] > 1) / U
    call = sum(1 for k in keys if all(occ[j][k[j]] > 1 for j in range(K))) / U
    full = sum(1 for k in keys if tup[k] > 1) / U
    return c1, call, full
print("\n== C1 empirical (pairs enumerated, V'=1000) ==")
for U, M, K in [(20000, 10000, 1), (20000, 10000, 2), (20000, 10000, 8), (5000, 1000, 4), (20000, 50000, 8)]:
    c1, ca, fu = empirical(U, M, K)
    pr0 = head_primes(M, K, 1)[0][0]
    p1, pa, _ = collide_stats(U, pr0, K)
    print(f"U={U} M={M}(p0={pr0}) K={K}: emp head1={c1:.4f} (ana {p1:.4f}); emp all={ca:.4f} (ana {pa:.4f}); full={fu:.5f}")
    print(f"   counts (denominator U={U}): head1={round(c1*U)} all={round(ca*U)} full={round(fu*U)}")

# ---------- C2: gate ----------
def sig(s): return 1 / (1 + math.exp(-s)) if s >= 0 else math.exp(s) / (1 + math.exp(s))
def gate(d, c, g, sq):
    s = g * math.sqrt(d) * c
    if sq: s = math.copysign(math.sqrt(abs(s)), s) if s != 0 else 0.0
    return s, sig(s), sig(-s)
print("\n== C2 gate ==  d cos gamma sqrt -> s alpha 1-alpha")
for d, c, g, sq in [(16, 0.0625, 1, False), (16, 0.0625, 1, True), (2560, 0.3, 1, False), (2560, 0.3, 1, True), (2560, 0.05, 1, False), (2560, -0.3, 1, False),
                    (2560, 0, 1, False), (16, 0.3, 1, False), (2560, 0.3, 0.1, False), (4096, 1, 2, False), (4096, -1, 2, False)]:
    s, a, oma = gate(d, c, g, sq)
    print(f"{d} {c} {g} {sq}: s={s:.4f} alpha={a:.8f} 1-alpha={oma:.3e}")
print("|cos| giving |s|<5 at d=2560:", 5 / math.sqrt(2560))

# ---------- C3: receptive field ----------
def field(t, N, w, delta, zero_init=False):
    taps = [0] if zero_init else range(w)
    S = set()
    for j in taps:
        c = t - j * delta
        for i in range(N): S.add(c - i)
    return sorted(S)
print("\n== C3 receptive field ==")
for t, N, w, dl, z in [(9, 3, 4, 3, False), (11, 3, 4, 3, False), (11, 3, 4, 1, False), (11, 2, 4, 2, False), (11, 3, 4, 3, True),
                       (4, 3, 4, 3, False), (13, 4, 4, 4, False), (13, 3, 6, 5, False)]:
    f = field(t, N, w, dl, z)
    inseq = [p for p in f if p >= 0]
    print(f"t={t} N={N} w={w} d={dl} zero={z}: positions={f} -> in-seq {len(inseq)}, span {f[-1]-f[0]+1}, pad {len(f)-len(inseq)}")

# ---------- C4: allocation ----------
print("\n== C4 allocation ==")
k = 6
def experts(rho, E0): return k + rho * (E0 - k)
for name, E0 in [("2e20", 106), ("6e20", 99), ("27B", 72)]:
    print(name, {r: round(experts(r, E0), 2) for r in (1.0, 0.8, 0.75, 0.4)})
assert round(experts(0.4, 106)) == 46 and round(experts(0.4, 99)) == 43
rho27 = (55 - k) / (72 - k); print("rho(55 experts)=", rho27)
emb = 2 * 129280 * 2560 / 1e9
Ptot27 = 26.7 - emb; Ps27 = Ptot27 - 3.8
print(f"vocab emb+head={emb:.4f}B, Ptot(excl)={Ptot27:.4f}, Psparse={Ps27:.4f}, Engram=(1-rho)Psparse={(1-rho27)*Ps27:.4f}B, rho from 5.7B={1-5.7/Ps27:.4f}")
for name, Pt, Pa in [("2e20", 5.7, 0.568), ("6e20", 9.9, 0.993)]:
    Ps = Pt - Pa
    print(name, "Psparse", round(Ps, 3), {r: round((1 - r) * Ps, 3) for r in (0.8, 0.75, 0.4)}, "Ptot/Pact", round(Pt / Pa, 2))
Ps27b = 26.7 - 3.8   # alternative reading: 26.7B already excludes embed/head
print(f"27B slider grid rho=0.745: E={experts(0.745, 72):.2f}, Engram={(1-0.745)*Ps27:.4f}B")
print(f"alt reading Psparse={Ps27b:.2f}, Engram=(1-rho)Psparse={(1-rho27)*Ps27b:.3f}B")
# units (推定): Table 5 'Engram Vocab Size' = rows per head; row = d_mem/((N-1)K) = 1280/16 = 80 dims
print("table5 engram params (2 layers x 2 orders x 8 heads x V_E x 80) 27B:", 2*2*8*2262400*80/1e9, " 40B:", 2*2*8*7239680*80/1e9)
print("V/128k-vocab:", 2262400 / 129280, 7239680 / 129280)
print("slots x1280:", 2.58e5 * 1280 / 1e9, 1e7 * 1280 / 1e9)
print("loss gain 6e20:", round(1.7248 - 1.7109, 4))

# ---------- C5: prefetch ----------
print("\n== C5 prefetch ==")
print("Table4 drop 4B:", (9031.62 - 8858.28) / 9031.62, " 8B:", (6315.52 - 6140.02) / 6315.52)
def prefetch(tokens, dmem, bytes_, bw_gbs, layer_ms, ell, mode):
    vol = tokens * dmem * bytes_                     # bytes per step per Engram layer
    tx = vol / (bw_gbs * 1e9) * 1e3                  # ms
    window = (ell - 1) * layer_ms if mode == "engram" else 0.0
    stall = max(0.0, tx - window)
    return vol, tx, window, stall
for args in [(512, 1280, 2, 25, 0.5, 2, "engram"), (512, 1280, 2, 25, 0.5, 2, "moe"), (512, 1280, 2, 25, 0.01, 2, "engram"),
             (65536, 1280, 2, 25, 0.5, 2, "engram"), (65536, 1280, 2, 25, 0.5, 15, "engram"), (512, 1280, 2, 25, 0.5, 1, "engram")]:
    v, tx, w, s = prefetch(*args)
    print(args, f"vol={v/1e6:.4f}MB tx={tx:.5f}ms window={w:.3f}ms stall={s:.5f}ms")
print("\nall asserts passed")
