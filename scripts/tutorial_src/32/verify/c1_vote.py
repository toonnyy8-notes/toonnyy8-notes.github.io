"""C1: uncertainty reward and filter band for m=10 (same rule as page)."""
m, delta = 10, 0.25
print("c  p_hat  r_unc  4p(1-p)  pass")
for c in range(1, m + 1):
    p = c / m
    r = 1 - 2 * abs(p - 0.5)
    ok = abs(p - 0.5) <= delta + 1e-12
    print(f"{c:2d} {p:5.2f} {r:6.3f} {4*p*(1-p):7.3f} {ok}")
# default slots of the page
slots = list("AAAAAABBBC")
from collections import Counter
cnt = Counter(slots); c = max(cnt.values())
tied = sorted(a for a, v in cnt.items() if v == c)
print("default:", dict(cnt), "c=", c, "tied=", tied, "r=", 1 - 2 * abs(c / m - 0.5))
assert abs((1 - 2 * abs(0.1 - 0.5)) - 0.2) < 1e-12
