"""C4 + prose numbers: Avg/H (Eq. 9) recomputed from paper tables; random-vs-ours gaps (Tables 8/9);
Hamming scale; BWT check (Table 13). The TABLE dict is pasted verbatim into the page as C4 data."""
RHO = [0.1, 0.25, 0.5, 0.75, 0.9]
# (A_up, A_down, Avg, H) ; random entries also carry the std of H
TABLE = {
    "NVILA|ImageNet-R": {"full": (13.8, 92.3, 53.1, 24.0),
        "rand": [(54.5, 90.7, 72.6, 68.0, 0.8), (49.0, 91.5, 70.2, 63.8, 0.5), (39.2, 92.2, 65.7, 55.0, 0.8), (25.6, 92.6, 59.1, 40.1, 5.5), (20.4, 92.6, 56.5, 33.4, 1.9)],
        "ours": [(56.6, 90.5, 73.6, 69.7), (53.0, 91.7, 72.4, 67.2), (47.5, 92.2, 69.8, 62.7), (35.5, 92.5, 64.0, 51.3), (24.9, 92.6, 58.7, 39.2)]},
    "LLaVA|ImageNet-R": {"full": (9.4, 89.8, 49.6, 17.1),
        "rand": [(54.6, 88.5, 71.6, 67.6, 0.1), (43.3, 89.3, 66.3, 58.3, 0.2), (20.4, 90.0, 55.2, 33.2, 0.5), (12.0, 89.8, 50.9, 21.2, 0.3), (9.8, 89.7, 49.8, 17.6, 0.2)],
        "ours": [(55.8, 88.6, 72.2, 68.5), (48.2, 89.5, 68.9, 62.7), (27.9, 89.7, 58.8, 42.6), (13.6, 89.7, 51.6, 23.6), (10.2, 89.9, 50.1, 18.3)]},
    "NVILA|COCO": {"full": (12.9, 98.2, 55.6, 22.9),
        "rand": [(62.3, 134.8, 98.6, 85.2, 0.1), (59.9, 124.3, 92.1, 80.8, 0.3), (44.8, 105.2, 75.0, 62.9, 1.4), (27.1, 98.6, 62.8, 42.5, 3.4), (19.7, 97.8, 58.7, 32.7, 0.9)],
        "ours": [(62.7, 135.1, 98.9, 85.7), (61.4, 123.8, 92.6, 82.1), (55.2, 106.4, 80.8, 72.7), (36.7, 98.5, 67.6, 53.5), (22.9, 97.7, 60.3, 37.1)]},
    "LLaVA|COCO": {"full": (4.5, 101.5, 53.0, 8.5),
        "rand": [(58.8, 120.8, 89.8, 79.1, 0.2), (55.9, 105.2, 80.6, 73.0, 0.3), (33.1, 100.5, 66.8, 49.8, 3.5), (11.0, 100.5, 55.8, 19.9, 4.2), (6.3, 100.5, 53.4, 11.9, 2.7)],
        "ours": [(59.0, 120.8, 89.9, 79.3), (57.3, 104.5, 80.9, 74.0), (39.2, 100.8, 70.0, 56.5), (13.4, 100.9, 57.2, 23.7), (6.7, 101.3, 54.0, 12.5)]},
}

# std of random A_up (Tables 8/9), same order as RHO
AUP_SD = {"NVILA|ImageNet-R": [1.0, 0.7, 0.9, 4.4, 1.4], "LLaVA|ImageNet-R": [0.1, 0.2, 0.3, 0.2, 0.2],
          "NVILA|COCO": [0.1, 0.4, 1.4, 2.7, 0.7], "LLaVA|COCO": [0.3, 0.2, 3.0, 2.5, 1.5]}


def avg(u, d):
    return (u + d) / 2


def hs(u, d):
    return 2 * u * d / (u + d) if u + d > 0 else 0.0


def rounding_ok(u, d, a_pr, h_pr, step=0.001):
    """True if some unrounded (U, D) with |U-u|<=0.05, |D-d|<=0.05 gives Avg and H that round
    (to 0.1) to the printed values. Distinguishes rounding from genuine inconsistency."""
    n = int(0.05 / step)
    for p in range(-n, n + 1):
        U = u + p * step
        for q in range(-n, n + 1):
            D = d + q * step
            if abs(avg(U, D) - a_pr) <= 0.05 + 1e-9 and abs(hs(U, D) - h_pr) <= 0.05 + 1e-9:
                return True
    return False


if __name__ == "__main__":
    print("## Tables 8/9 (Ours rows): recomputed Avg/H vs paper, gap H(ours)-H(rand), gap/sd")
    for key, t in TABLE.items():
        for rho, o, r in zip(RHO, t["ours"], t["rand"]):
            a, h = avg(o[0], o[1]), hs(o[0], o[1])
            ok = rounding_ok(o[0], o[1], o[2], o[3])
            gap = o[3] - r[3]
            print(f"| {key} | ρ={rho} | Avg {a:.2f}/{o[2]} | H {h:.2f}/{o[3]} | {'ok' if ok else 'MISMATCH'} | gap {gap:+.1f} | gap/sd {gap/r[4]:.1f} |")
        f = t["full"]
        print(f"| {key} | Full FT | Avg {avg(f[0], f[1]):.2f}/{f[2]} | H {hs(f[0], f[1]):.2f}/{f[3]} | {'ok' if rounding_ok(*f) else 'MISMATCH'} |")

    print("\n## Table 1 NVILA COCO Dowser")
    up = [69.3, 48.6, 68.8, 77.7, 50.8, 60.7]
    A = sum(up) / 6
    print(f"A_up={A:.4f} Avg={avg(A, 135.5):.2f} H={hs(A, 135.5):.2f}")
    print("## Table 1 NVILA ImageNet-R Tailor (paper Avg 66.9, H 47.2)  rounding_ok:", rounding_ok(33.38, 80.3, 66.9, 47.2))
    up = [28.8, 18.7, 23.9, 60.2, 42.5, 26.2]
    A = sum(up) / 6
    print(f"A_up={A:.4f} Avg={avg(A, 80.3):.2f} H={hs(A, 80.3):.2f}")
    print("## IconQA NVILA zero-shot: A_up from 6 = ", round(sum([70.8, 51.2, 69.4, 78.1, 42.2, 62.3]) / 6, 4),
          "Avg", round(avg(sum([70.8, 51.2, 69.4, 78.1, 42.2, 62.3]) / 6, 0.1), 2), "H", round(hs(sum([70.8, 51.2, 69.4, 78.1, 42.2, 62.3]) / 6, 0.1), 2))

    print("\n## Hamming scale (two masks of density p): distance = 2p(1-overlap)")
    for p, hd in ((0.1, 0.050), (0.1, 0.081), (0.1, 0.226)):
        ov = 1 - hd / (2 * p)
        print(f"p={p} Hamming={hd} -> overlap={ov:.3f}  (max distance {2*p}, independent-random expectation {2*p*(1-p):.3f})")

    print("\n## Table 13 BWT: T/(T-1)*(MFN-MFT) vs table")
    for name, mft, mfn, bwt in (("Full-FT", 70.2, 54.7, -15.6), ("Tailor", 69.1, 55.9, -13.2), ("MoELoRA", 67.9, 60.0, -7.8), ("Dowser", 69.1, 66.6, -2.5)):
        print(f"{name}: MFN-MFT={mfn-mft:+.1f}, 1.25*(MFN-MFT)={1.25*(mfn-mft):+.2f}, table {bwt}")
