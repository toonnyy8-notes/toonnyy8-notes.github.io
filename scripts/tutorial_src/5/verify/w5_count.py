"""W5 acceptance values: counting behind the masked (Thm 8.1, Eq. 100) and uniform (Thm A.1) lower bounds.
All unspecified constants set to 1 (self-chosen, labelled on the page): C in Eq. (96), C_Lambda and A in Eq. (107),
candidate count for uniform taken as ceil(sqrt(d/Lambda)) (same order as 1/w). kappa- = 0.1, kappa+ = 0.5.
Run: python3 -I w5_count.py
"""
import math
LN2 = math.log(2)
KM, KP = 0.1, 0.5

def counts(d, eta, rho):
    K = math.floor(KP * d / LN2) - math.ceil(KM * d / LN2) + 1
    ell = math.ceil(math.log2(d / (eta * rho ** 2)))
    nM = math.ceil(K / (2 * ell + 1))
    Lam = math.log(d / (eta * rho ** 2))
    nU = math.ceil(math.sqrt(d / Lam))
    return K, ell, nM, Lam, nU

def eq100(d, eta, rho, Q):
    K, ell, *_ = counts(d, eta, rho)
    return (2 * ell + 1) * Q / K, 2 ** (1 - ell) / eta, Q * 2 ** (-ell)

if __name__ == "__main__":
    for eta, rho in [(1e-3, 0.2), (1e-6, 0.2), (0.09, 0.2)]:
        print(f"eta={eta} rho={rho}")
        for p in range(6, 21, 2):
            d = 2 ** p
            K, ell, nM, Lam, nU = counts(d, eta, rho)
            qmax = rho ** 2 * K / (4 * (2 * ell + 1))
            print(f"  d={d:6d} |K|={K:6d} ell={ell:2d} nM={nM:5d} Lambda={Lam:7.3f} nU={nU:4d} ratio={nM/nU:6.2f} Q_T1<=rho^2/4: {qmax:.3f}")
    d, eta, rho = 1024, 1e-3, 0.2
    for Q in (1, 6, 12):
        print(f"Eq.100 d={d} eta={eta} rho={rho} Q={Q}:", ["%.4g" % x for x in eq100(d, eta, rho, Q)])
