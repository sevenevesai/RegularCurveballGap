"""Row-trade Curveball on binary matrices with row sums r and column sums c: when is gap(H) = m/2?
(Section 8.)

Omega is the set of m x n binary matrices with every row sum r and column sums c (uniform law);
the paper writes k for the number n of columns here, and rows and columns are numbered from 0.
For rows a = {x, w}, E_a averages over the matrices that agree with A off rows x, w and have the
same column sums on rows x, w (the 2q singleton columns are split q/q uniformly);
h_a = I - E_a and H = sum_a h_a over the binom(m,2) row pairs. Proposition 8.1 gives
gap(H) = min(m/2, mu_inv), with mu_inv (the paper's mu_sym) the least eigenvalue of H on
centered functions of the multiset of rows. This script computes mu_inv on the row-multiset
quotient for every margin in a grid and decides mu_inv versus m/2 exactly. It does not assume
the answer.

Usage, from the scripts directory:
    python bipartite.py           # m<=6, n<=7
    python bipartite.py --large   # adds larger quotients
Exits 1 if an internal check fails; the last line starts with PASS otherwise.

EXACT: the quotient is the integer matrix LA = L * A with L the lcm of the fiber sizes;
Proposition 8.1(1), the row exchange of Proposition 8.1(2) and the quotient itself are checked
against the labeled space in integers; verdicts
on mu_inv vs m/2 come from a Fraction LDL of B = M W (2 LA - m L I) + 2 m L w w^T (sign of the
centered spectrum shifted by m/2) or, for larger quotients, from an integer Rayleigh quotient
of a rounded eigenvector (certifies mu_inv < m/2) or an exactly verified congruence
X^T B X that is diagonally dominant (certifies mu_inv > m/2). FLOAT: printed eigenvalues, the
co-occurrence Ritz values, and the dense cross-checks of the quotient against the full H.
Verdicts printed with a leading '?' are float only.

Only 2r <= n is enumerated: complementing every entry maps margin (r, c) to (n-r, m-c) and
a-fibers onto a-fibers, so the quotients agree; `complement_ok` checks that exactly.
"""
import argparse
import itertools
import math
import sys
import time
from fractions import Fraction

import numpy as np
import sympy
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse.linalg import LinearOperator, eigsh

LDL_MAX = 200          # exact Fraction LDL of B up to this many classes
LDL_CROSS = 60         # also run LDL on witnessed failures up to this size, as a cross-check
EXACT_VALUE_MAX = 30   # exact minimal polynomial of mu_inv (biregular and non-failing margins)
DENSE_Q = 3500         # dense eigensolver on the quotient up to this size, else Lanczos
FULL_EXACT = 20000     # exact labeled-space checks of (i), (ii) and the quotient up to this |Omega|
FULL_DENSE = 1500      # dense eigensolve of the full H up to this |Omega|
FULL_SPARSE = 30000    # (--large) Lanczos on the full H up to this |Omega|
TOL = 1e-9


class TooMany(Exception):
    pass


def margins(m, n, r):
    """Weakly decreasing column sums c with 0 < c_y < m and sum m*r."""
    out = []

    def rec(prefix, left):
        k = len(prefix)
        if k == n:
            if left == 0:
                out.append(tuple(prefix))
            return
        rem = n - k
        for v in range(min(prefix[-1] if prefix else m - 1, left - (rem - 1)), 0, -1):
            if v * rem < left:
                break
            rec(prefix + [v], left - v)

    rec([], m * r)
    return out


def row_masks(n, r):
    masks = sorted(sum(1 << i for i in s) for s in itertools.combinations(range(n), r))
    return masks, [[y for y in range(n) if R >> y & 1] for R in masks]


def classes(m, n, r, c, cap=None):
    """Row multisets with column sums c, each as its sorted tuple of row bitmasks."""
    masks, bits = row_masks(n, r)
    out, chosen, resid = [], [], list(c)

    def rec(start):
        t = m - len(chosen)
        if t == 0:
            out.append(tuple(chosen))
            if cap is not None and len(out) > cap:
                raise TooMany
            return
        if max(resid) > t or sum(v == t for v in resid) > r:
            return
        for i in range(start, len(masks)):
            if all(resid[y] for y in bits[i]):
                for y in bits[i]:
                    resid[y] -= 1
                chosen.append(masks[i])
                rec(i)
                chosen.pop()
                for y in bits[i]:
                    resid[y] += 1

    rec(0)
    return out


def labeled_states(m, n, r, c):
    """All of Omega, enumerated row by row independently of `classes`; an M x m array."""
    masks, bits = row_masks(n, r)
    out, chosen, resid = [], [], list(c)

    def rec():
        t = m - len(chosen)
        if t == 0:
            out.append(list(chosen))
            return
        if max(resid) > t:
            return
        for R, b in zip(masks, bits):
            if all(resid[y] for y in b):
                for y in b:
                    resid[y] -= 1
                chosen.append(R)
                rec()
                chosen.pop()
                for y in b:
                    resid[y] += 1

    rec()
    return np.array(out, dtype=np.int64).reshape(-1, m)


def orbit_size(C):
    return math.factorial(len(C)) // math.prod(math.factorial(C.count(R)) for R in set(C))


def fiber_lcm(n, r):
    return math.lcm(*(math.comb(2 * q, q) for q in range(min(r, n - r) + 1)))


def quotient(m, n, r, cls):
    """LA = L * (H on row-multiset classes): (H 1_D)(A) = LA[class(A), D] / L, and weights w."""
    index = {C: i for i, C in enumerate(cls)}
    L = fiber_lcm(n, r)
    pairs = list(itertools.combinations(range(m), 2))
    splits = {}
    I, J, V = [], [], []
    for i, C in enumerate(cls):
        counts = {}
        for x, y in pairs:
            S = C[x] ^ C[y]
            if not S:
                counts[i] = counts.get(i, 0) + L
                continue
            if S not in splits:
                sb = [1 << b for b in range(n) if S >> b & 1]
                q = len(sb) // 2
                splits[S] = ([sum(X) for X in itertools.combinations(sb, q)],
                             L // math.comb(2 * q, q))
            subsets, wt = splits[S]
            com = C[x] & C[y]
            rest = [C[z] for z in range(m) if z != x and z != y]
            for X in subsets:
                j = index[tuple(sorted(rest + [com | X, com | (S ^ X)]))]
                counts[j] = counts.get(j, 0) + wt
        for j, v in counts.items():
            I.append(i)
            J.append(j)
            V.append(v)
    k = len(cls)
    P = csr_matrix((np.array(V, dtype=np.int64), (I, J)), shape=(k, k))
    LA = (L * math.comb(m, 2) * identity(k, dtype=np.int64, format="csr") - P).tocsr()
    LA.eliminate_zeros()
    w = np.array([orbit_size(C) for C in cls], dtype=np.int64)
    return LA, w, L


def quotient_ok(LA, w):
    """Exact: LA kills constants and W LA is symmetric (reversibility w.r.t. class sizes)."""
    WLA = (diags(w) @ LA).tocsr()
    return not np.any(np.asarray(LA.sum(axis=1))) and (WLA - WLA.T).count_nonzero() == 0


def lowest(LA, w, L, m):
    """FLOAT: mu_inv and its eigenfunction on classes (None for a single class)."""
    k = len(w)
    if k == 1:
        return None, None
    s = np.sqrt(w.astype(float))
    S = diags(s) @ LA.astype(float) @ diags(1 / s) / L
    S = ((S + S.T) / 2).tocsr()
    u = s / np.linalg.norm(s)
    K = math.comb(m, 2) + 1  # H <= binom(m,2) I, so the shifted constant is never the minimum
    if k <= DENSE_Q:
        val, vec = eigh(S.toarray() + K * np.outer(u, u), subset_by_index=[0, 0])
    else:
        op = LinearOperator((k, k), matvec=lambda x: S @ x + K * u * (u @ x), dtype=float)
        val, vec = eigsh(op, k=1, which="SA", tol=1e-12, maxiter=20000)
    return float(val[0]), vec[:, 0] / s


def witness(LA, w, L, f, bits=24):
    """EXACT Rayleigh quotient of H at the centered part of a rounded copy of f."""
    fi = np.rint(f / np.abs(f).max() * 2 ** bits).astype(np.int64)
    g = (LA @ fi).astype(object)
    fo, wo = fi.astype(object), w.astype(object)
    num = int(np.dot(fo * wo, g))            # = L <f, H f>_W, constants drop out
    M = int(w.sum())
    den = M * int(np.dot(wo, fo * fo)) - int(np.dot(wo, fo)) ** 2   # = M |f - mean|_W^2
    return Fraction(num * M, L * den)


def ldl_psd(B):
    """EXACT: (is B PSD, is B singular) for a symmetric integer matrix, by LDL in Fractions."""
    k = len(B)
    U = [[Fraction(B[i][j]) for j in range(k)] for i in range(k)]
    singular = False
    for i in range(k):
        d = U[i][i]
        nz = [j for j in range(i + 1, k) if U[i][j] != 0]
        if d < 0 or (d == 0 and nz):
            return False, None
        if d == 0:
            singular = True
            continue
        row = U[i]
        for j in nz:
            f = row[j] / d
            Uj = U[j]
            for l in nz:
                if l >= j:
                    Uj[l] -= f * row[l]
    return True, singular


def compare_exact(LA, w, L, p, q):
    """EXACT sign of mu_inv - p/q (+1, 0, -1). B = M W (q LA - p L I) + (p+q) L w w^T is PSD iff
    mu_inv >= p/q, and singular iff equality: the constants and the centered functions are
    B-orthogonal, and B is (q L M^2) > 0 on the constants."""
    k, M = len(w), int(w.sum())
    D = LA.toarray()
    wl = [int(x) for x in w]
    B = [[M * q * wl[i] * int(D[i, j]) + (p + q) * L * wl[i] * wl[j] for j in range(k)]
         for i in range(k)]
    for i in range(k):
        B[i][i] -= M * p * L * wl[i]
    psd, singular = ldl_psd(B)
    return -1 if not psd else (0 if singular else 1)


def crt_primes(k, bits):
    """Primes p < 2^20 with k (p-1)^2 < 2^53, so float64 matrix products of residues are exact,
    whose product exceeds 2^bits."""
    out, P, p = [], 1, 1 << 20
    while P.bit_length() <= bits:
        p = int(sympy.prevprime(p))
        assert k * (p - 1) ** 2 < 2 ** 53
        out.append(p)
        P *= p
    return out


def pd_certificate(LA, w, L, p, q, xbits=36, block=2048):
    """EXACT certificate that mu_inv > p/q, for quotients too large for ldl_psd. With B as in
    compare_exact, find (in floats) an upper-triangular integer X with nonzero diagonal and
    X^T B X close to a positive diagonal; then compute C = X^T B X exactly (residues modulo
    primes via exact float64 products, reconstructed by CRT) and check that C is strictly
    diagonally dominant. Then C > 0, X is invertible, so B > 0. Returns None if it fails."""
    k, M = len(w), int(w.sum())
    wmax, lamax = int(w.max()), int(abs(LA).max())
    if M * q * wmax * lamax + M * p * L * wmax + (p + q) * L * wmax ** 2 >= 2 ** 53:
        return None
    wl = w.astype(np.int64)
    B = ((M * q) * (diags(wl) @ LA)).toarray() + (p + q) * L * np.outer(wl, wl)
    B[np.diag_indices(k)] -= M * p * L * wl
    if np.any(np.diag(B) <= 0):
        return None
    d = np.sqrt(np.diag(B).astype(float))
    try:
        R = cholesky(B / np.outer(d, d), lower=False)
    except np.linalg.LinAlgError:
        return None
    X = solve_triangular(R, np.eye(k), lower=False) / d[:, None]
    del R
    X = np.rint(X * 2.0 ** (xbits - math.ceil(math.log2(np.abs(X).max())))).astype(np.int64)
    if np.any(np.diag(X) == 0):
        return None
    bound = k * k * int(np.abs(X).max()) ** 2 * int(np.abs(B).max())
    primes = crt_primes(k, bound.bit_length() + 1)
    P = math.prod(primes)
    # Garner: C = sum_i g_i prod_{j<i} p_j with digits 0 <= g_i < p_i; inv[i][j] = p_j^-1 mod p_i
    inv = [[pow(primes[j], -1, primes[i]) for j in range(i)] for i in range(len(primes))]
    # |C_ij| < p0 p1 p2 / 2 < 2^59 is read off the first three Garner digits in int64: the
    # remaining digits are all 0 (C_ij >= 0) or all p_t - 1 (C_ij < 0). Other entries (the
    # diagonal, and any large off-diagonal) are rebuilt as Python ints.
    w3 = primes[0] * primes[1] * primes[2]
    diag = np.zeros(k, dtype=object)
    off = [0] * k
    for j0 in range(0, k, block):
        J = slice(j0, min(k, j0 + block))
        res = []
        for pr in primes:
            Bp = np.mod(B, pr).astype(float)
            Xp = np.mod(X, pr).astype(float)
            Y = np.mod(Bp @ Xp[:, J], pr)
            res.append(np.mod(Xp.T @ Y, pr).astype(np.int64))
        g = []
        for i, pr in enumerate(primes):
            t = res[i]
            for j in range(i):
                t = np.mod((t - g[j]) * inv[i][j], pr)
            g.append(t)
        hi_zero = np.all([g[i] == 0 for i in range(3, len(primes))], axis=0)
        hi_full = np.all([g[i] == primes[i] - 1 for i in range(3, len(primes))], axis=0)
        low = g[0] + primes[0] * (g[1] + primes[1] * g[2])
        absval = np.abs(np.where(hi_zero, low, low - w3))
        cols = np.arange(J.start, J.stop)
        smallv = (hi_zero | hi_full) & (np.arange(k)[:, None] != cols[None, :])
        absval = np.where(smallv, absval, 0)
        lo30 = (absval & ((1 << 30) - 1)).sum(axis=1)   # split so the int64 row sums cannot overflow
        hi30 = (absval >> 30).sum(axis=1)
        for i in range(k):
            off[i] += (int(hi30[i]) << 30) + int(lo30[i])
        for i, jj in zip(*np.nonzero(~smallv)):
            v = 0
            for t in range(len(primes) - 1, -1, -1):
                v = v * primes[t] + int(g[t][i, jj])
            v = v - P if v > P // 2 else v
            if cols[jj] == i:
                diag[i] = v
            else:
                off[i] += abs(v)
    return all(diag[i] > off[i] for i in range(k))


def cooccurrence_ritz(LA, w, L, cls, n):
    """FLOAT: least Rayleigh quotient of H over centered combinations of the column
    co-occurrence counts N_yz(A) = #{rows of A containing columns y and z}."""
    pidx = {pr: i for i, pr in enumerate(itertools.combinations(range(n), 2))}
    F = np.zeros((len(cls), len(pidx)))
    for i, C in enumerate(cls):
        for R in C:
            for pr in itertools.combinations([y for y in range(n) if R >> y & 1], 2):
                F[i, pidx[pr]] += 1
    wf = w.astype(float)
    F -= (wf @ F) / wf.sum()
    G = F.T @ (wf[:, None] * F)
    Hm = F.T @ (wf[:, None] * (LA @ F) / L)
    ev, U = np.linalg.eigh(G)
    keep = ev > 1e-9 * max(ev.max(), 1e-300)
    if not np.any(keep):
        return None
    T = U[:, keep] / np.sqrt(ev[keep])
    return float(np.linalg.eigvalsh(T.T @ ((Hm + Hm.T) / 2) @ T)[0])


def exact_value(LA, L, mu):
    """EXACT: the irreducible factor over Q of the characteristic polynomial of A = LA / L whose
    real root is mu (rational value if linear)."""
    t, s = sympy.symbols("t s")
    p = sympy.Matrix(LA.toarray().tolist()).charpoly(t).as_expr()
    for fac, _ in sympy.factor_list(p)[1]:
        if any(abs(complex(z) - L * mu) < 1e-6 for z in sympy.Poly(fac, t).nroots()):
            g = sympy.Poly(sympy.expand(fac.subs(t, L * s)), s).monic()
            if g.degree() == 1:
                return str(-g.all_coeffs()[1])
            return f"root of {g.as_expr()}"
    return "?"


def complement_ok(m, n, r, c, cls, LA, w):
    """EXACT: complementing entries (and reversing columns) maps the quotient of (r, c) onto the
    quotient of (n-r, m-c reversed)."""
    full = (1 << n) - 1
    flip = {R: int(format(R ^ full, f"0{n}b")[::-1], 2) for C in cls for R in C}
    cc = tuple(m - v for v in reversed(c))
    cls2 = classes(m, n, n - r, cc)
    LA2, w2, _ = quotient(m, n, n - r, cls2)
    index2 = {C: i for i, C in enumerate(cls2)}
    if len(cls2) != len(cls):
        return False
    perm = np.array([index2[tuple(sorted(flip[R] for R in C))] for C in cls])
    return (np.array_equal(w2[perm], w)
            and (LA2[perm][:, perm] - LA).count_nonzero() == 0)


def full_checks(m, n, r, c, cls, LA, w, L, mu, large):
    """Checks on the labeled space Omega. EXACT: |Omega|, fiber sizes binom(2q,q), the row
    exchange of Proposition 8.1(2) (the row swap stays in Omega and in the a-fiber),
    Proposition 8.1(1) (2 L H 1[x,y] = L (m 1[x,y] - c_y) pointwise), and the quotient
    (L H 1_D)(A) = LA[class(A), D] for every A and D.
    FLOAT (small Omega): dense H gives the invariant-sector least eigenvalue, H >= m/2 off the
    invariant sector, H - L_T/2 >= 0, and gap(H) = min(m/2, mu_inv)."""
    out = {}
    R = labeled_states(m, n, r, c)
    M = len(R)
    out["M_ok"] = M == int(w.sum())
    if M > FULL_EXACT and not (large and M <= FULL_SPARSE):
        return out
    pos = {tuple(row): i for i, row in enumerate(R.tolist())}
    index = {C: i for i, C in enumerate(cls)}
    cl = np.array([index[tuple(sorted(row))] for row in R.tolist()])
    k = len(cls)
    pairs = list(itertools.combinations(range(m), 2))
    ent = {(x, y): (R[:, x] >> y) & 1 for x in range(m) for y in range(n)}
    Hent = {key: np.zeros(M, dtype=np.int64) for key in ent}   # L * H 1[x,y]
    do_quot = M * k <= 6_000_000
    PA = np.zeros((M, k), dtype=np.int64) if do_quot else None
    sizes_ok = swap_ok = True
    labs, swaps = [], []
    for x, y in pairs:
        key = R.copy()
        key[:, x] = R[:, x] & R[:, y]
        key[:, y] = R[:, x] | R[:, y]
        _, lab = np.unique(key, axis=0, return_inverse=True)
        lab = lab.ravel()
        size = np.bincount(lab)
        q2 = np.bitwise_count(R[:, x] ^ R[:, y]).astype(np.int64)
        expect = np.array([math.comb(int(t), int(t) // 2) for t in q2])
        sizes_ok &= bool(np.array_equal(size[lab], expect)) and not np.any(L % size)
        sw = R.copy()
        sw[:, [x, y]] = R[:, [y, x]]
        r_sw = np.array([pos.get(tuple(row), -1) for row in sw.tolist()])
        swap_ok &= bool(np.all(r_sw >= 0)) and bool(np.array_equal(lab[r_sw], lab))
        scale = (L // size)[lab]
        for e_key, e in ent.items():
            # fiber sums are integers no larger than the fiber size, exact in float64
            sums = np.rint(np.bincount(lab, weights=e, minlength=len(size))).astype(np.int64)
            Hent[e_key] += L * e - scale * sums[lab]
        if do_quot:
            N = coo_matrix((np.ones(M, dtype=np.int64), (lab, cl)), shape=(len(size), k)).toarray()
            PA += scale[:, None] * N[lab]
        labs.append((lab, size))
        swaps.append(r_sw)
    out["fiber_sizes"] = sizes_ok
    out["fact_ii"] = swap_ok
    out["fact_i"] = all(np.array_equal(2 * Hent[(x, y)], L * (m * ent[(x, y)] - c[y]))
                        for x in range(m) for y in range(n))
    if do_quot:
        LAd = LA.toarray()
        Hq = L * math.comb(m, 2) * np.eye(k, dtype=np.int64)[cl] - PA
        out["quotient_exact"] = bool(np.array_equal(Hq, LAd[cl]))
    if M <= FULL_DENSE:
        H = math.comb(m, 2) * np.eye(M)
        LT = np.zeros((M, M))
        for (lab, size), r_sw in zip(labs, swaps):
            Z = np.zeros((M, len(size)))
            Z[np.arange(M), lab] = 1
            H -= (Z / size) @ Z.T
            T = np.zeros((M, M))
            T[r_sw, np.arange(M)] = 1
            LT += np.eye(M) - T
        ev = np.linalg.eigvalsh(H)
        Q = np.zeros((M, k))
        Q[np.arange(M), cl] = 1 / np.sqrt(w[cl])
        Hq = Q.T @ H @ Q
        u = np.sqrt(w / w.sum())
        big = math.comb(m, 2) + 1
        mu_full = np.linalg.eigvalsh(Hq + big * np.outer(u, u))[0] if k > 1 else None
        Pp = np.eye(M) - Q @ Q.T
        off_inv = np.linalg.eigvalsh(Pp @ H @ Pp + big * Q @ Q.T)[0]
        out["dense_mu"] = mu_full is None and mu is None or (
            mu_full is not None and abs(mu_full - mu) < TOL)
        out["dense_off_inv"] = bool(off_inv >= m / 2 - TOL)
        out["dense_H_minus_LT"] = bool(np.linalg.eigvalsh(H - LT / 2)[0] >= -TOL)
        out["dense_gap"] = bool(abs(ev[1] - (m / 2 if mu is None else min(m / 2, mu))) < TOL)
        out["dense_ran"] = True
    elif large:
        Hs = math.comb(m, 2) * identity(M, format="csr")
        for lab, size in labs:
            Z = csr_matrix((np.ones(M), (np.arange(M), lab)), shape=(M, len(size)))
            Hs = Hs - (Z @ diags(1.0 / size) @ Z.T)
        vals = np.sort(eigsh(Hs.tocsr(), k=2, which="SA", tol=1e-12, maxiter=20000)[0])
        out["sparse_gap"] = bool(abs(vals[1] - (m / 2 if mu is None else min(m / 2, mu))) < 1e-7)
    return out


def analyse(m, n, r, c, large, cap, cert_max):
    t0 = time.time()
    cls = classes(m, n, r, c, cap)
    if not cls:
        return None
    LA, w, L = quotient(m, n, r, cls)
    rec = dict(m=m, n=n, r=r, c=c, M=int(w.sum()), k=len(cls), qok=quotient_ok(LA, w),
               bireg=len(set(c)) == 1)
    mu, f = lowest(LA, w, L, m)
    rec["mu"] = mu
    rec["verdict"], rec["method"], rec["rq"], rec["agree"] = None, "", None, None
    if mu is None:
        rec["verdict"], rec["method"] = "vacuous", "1 class"
    else:
        if mu < m / 2 - 1e-6:
            rq = witness(LA, w, L, f)
            rec["rq"] = rq
            if rq < Fraction(m, 2):
                rec["verdict"], rec["method"] = "<", "witness"
        if rec["k"] <= LDL_MAX and (rec["verdict"] is None or rec["k"] <= LDL_CROSS):
            s = compare_exact(LA, w, L, m, 2)
            v = {-1: "<", 0: "=", 1: ">"}[s]
            if rec["verdict"] is None:
                rec["verdict"], rec["method"] = v, "LDL"
            else:
                rec["agree"] = v == rec["verdict"]
                rec["method"] = "witness+LDL"
            # the certificate must succeed exactly when LDL finds B > 0
            rec["cert_agree"] = bool(pd_certificate(LA, w, L, m, 2)) == (v == ">")
        if rec["verdict"] is None and m == 2:
            # one row pair whose fiber is all of Omega: A = I - 1 w^T / M
            D = LA.toarray() * rec["M"]
            if np.array_equal(D, L * rec["M"] * np.eye(rec["k"], dtype=np.int64)
                              - L * np.outer(np.ones(rec["k"], dtype=np.int64), w)):
                rec["verdict"], rec["method"] = "=", "H = I - Pi"
        if rec["verdict"] is None and mu > m / 2 + 1e-6 and rec["k"] <= cert_max:
            if pd_certificate(LA, w, L, m, 2):
                rec["verdict"], rec["method"] = ">", "PD cert"
        if rec["verdict"] is None:
            rec["verdict"], rec["method"] = ("?>=" if mu >= m / 2 - 1e-6 else "?<"), "float only"
        if rec["k"] <= 100:
            rec["vs1"] = compare_exact(LA, w, L, 1, 1)
        if rec["k"] <= EXACT_VALUE_MAX and (rec["bireg"] or rec["verdict"] != "<"):
            rec["exact_mu"] = exact_value(LA, L, mu)
        rec["cooc"] = cooccurrence_ritz(LA, w, L, cls, n)
        if set(c) == {1}:
            rec["ones"] = abs(mu - (n - 1) / (2 * r - 1)) < TOL
            if rec["k"] <= LDL_MAX:
                rec["ones_exact"] = compare_exact(LA, w, L, n - 1, 2 * r - 1) == 0
    if rec["k"] <= 300 and 2 * r <= n:
        rec["compl"] = complement_ok(m, n, r, c, cls, LA, w)
    if rec["M"] <= (FULL_SPARSE if large else FULL_EXACT) or rec["M"] <= FULL_DENSE:
        rec["full"] = full_checks(m, n, r, c, cls, LA, w, L, mu, large)
    rec["t"] = time.time() - t0
    return rec


def fmt_row(rec):
    mu = rec["mu"]
    cs = "".join(str(v) for v in rec["c"])
    mus = "-" if mu is None else f"{mu:.6f}"
    dif = "-" if mu is None else f"{mu - rec['m'] / 2:+.6f}"
    rq = "" if rec["rq"] is None else f"{float(rec['rq']):.6f}"
    return (f"{rec['m']} {rec['n']} {rec['r']} {cs:<9} {rec['M']:>9} {rec['k']:>6}  {mus:>9} {dif:>10}"
            f"  {rec['verdict']:<7} {rec['method']:<12} {rq:>9} "
            + ("        -" if rec.get("cooc") is None else f"{rec['cooc']:9.6f}")
            + f" {rec['t']:6.2f}s"
            + (f"  mu_inv = {rec['exact_mu']}" if "exact_mu" in rec else ""))


HEADER = ("m n r c         |Omega|  #cls   mu_inv(f) mu-m/2(f)  exact   method       "
          "witnessRQ  coocRQ(f)   time")


def sweep(cases, large, cap, cert_max):
    recs = []
    print(HEADER)
    for m, n, r, c in cases:
        try:
            rec = analyse(m, n, r, c, large, cap, cert_max)
        except TooMany:
            print(f"{m} {n} {r} {''.join(map(str, c)):<9} skipped: more than {cap} classes", flush=True)
            continue
        if rec is None:
            continue
        print(fmt_row(rec), flush=True)
        recs.append(rec)
    return recs


def grid(mmax, nmax, mmin=2, nmin=2, bireg_only=False):
    out = []
    for m in range(mmin, mmax + 1):
        for n in range(nmin, nmax + 1):
            for r in range(1, n // 2 + 1):
                for c in margins(m, n, r):
                    if not bireg_only or len(set(c)) == 1:
                        out.append((m, n, r, c))
    return out


def report(recs):
    ok = True
    bir = [x for x in recs if x["bireg"]]
    non = [x for x in recs if not x["bireg"]]
    print("\n(a) biregular margins (all c_y equal); mu_inv FLOAT, verdict EXACT")
    print(HEADER)
    for x in bir:
        print(fmt_row(x))
    bad = [x for x in bir if x["verdict"] in ("<", "?<")]
    print(f"biregular: {len(bir)} margins, {sum(x['verdict'] == 'vacuous' for x in bir)} vacuous "
          f"(one class), {len(bad)} with mu_inv < m/2"
          + ("" if not bad else " -> the biregular analogue FAILS"))
    print("\n(b) non-column-regular margins")
    real = [x for x in non if x["verdict"] != "vacuous"]
    fails = [x for x in real if x["verdict"] in ("<", "?<")]
    eqs = [x for x in real if x["verdict"] == "="]
    gts = [x for x in real if x["verdict"] in (">", "?>=")]
    print(f"tested {len(non)} ({len(non) - len(real)} vacuous); mu_inv < m/2: {len(fails)}; "
          f"= m/2: {len(eqs)}; > m/2: {len(gts)}")
    print("smallest failures by |Omega|:")
    for x in sorted(fails, key=lambda x: (x["M"], x["m"]))[:6]:
        print("  " + fmt_row(x))
    no_edge = [x for x in fails if not any(v in (1, x["m"] - 1) for v in x["c"])]
    print(f"failures with no column sum in {{1, m-1}}: {len(no_edge)}"
          + ("" if not no_edge else "; smallest: " + fmt_row(min(no_edge, key=lambda x: x["M"]))))
    print("non-failing non-vacuous margins (all, both kinds):")
    for x in [x for x in recs if x["verdict"] in ("=", ">", "?>=")]:
        print("  " + fmt_row(x))
    print("mu_inv by m over non-vacuous margins (FLOAT): m  #margins  #fail  min mu  max mu")
    for m in sorted({x["m"] for x in recs}):
        xs = [x for x in recs if x["m"] == m and x["mu"] is not None]
        if xs:
            print(f"  {m}  {len(xs):>4}  {sum(x['verdict'] in ('<', '?<') for x in xs):>4}  "
                  f"{min(x['mu'] for x in xs):.6f}  {max(x['mu'] for x in xs):.6f}")
    v1 = [x for x in recs if "vs1" in x]
    print(f"EXACT mu_inv vs 1 on {len(v1)} margins with <= 100 classes: "
          f"{sum(x['vs1'] < 0 for x in v1)} below, {sum(x['vs1'] == 0 for x in v1)} equal, "
          f"{sum(x['vs1'] > 0 for x in v1)} above")
    eq1 = [x for x in v1 if x["vs1"] == 0]
    for x in eq1[:12]:
        print("  mu_inv = 1 exactly: " + fmt_row(x))
    allf = [x for x in recs if x["verdict"] in ("<", "?<")]
    co = [x for x in allf if x.get("cooc") is not None and x["cooc"] < x["m"] / 2 - TOL]
    print(f"failures (both kinds) where centered column co-occurrence counts N_yz already have "
          f"Rayleigh quotient < m/2 (FLOAT Ritz value, column coocRQ): {len(co)} of {len(allf)}")
    nf = [x for x in recs if x["verdict"] in ("=", ">", "?>=") and x.get("cooc") is not None]
    print(f"non-failing margins with co-occurrence Ritz value >= m/2: "
          f"{sum(x['cooc'] >= x['m'] / 2 - TOL for x in nf)} of {len(nf)}")
    ones = [x for x in recs if "ones" in x]
    print(f"c = 1^n: mu_inv = (n-1)/(2r-1) on {sum(x['ones'] for x in ones)} of {len(ones)} margins "
          f"(FLOAT); exactly (LDL) on {sum(x.get('ones_exact', False) for x in ones)} of "
          f"{sum('ones_exact' in x for x in ones)} with <= {LDL_MAX} classes")
    print("\n(c) min of mu_inv - m/2 over biregular non-vacuous margins (FLOAT)")
    bnv = [x for x in bir if x["mu"] is not None]
    if bnv:
        x = min(bnv, key=lambda x: x["mu"] - x["m"] / 2)
        print(f"  {x['mu'] - x['m'] / 2:+.6f} at " + fmt_row(x))
        x = max(bnv, key=lambda x: x["mu"] - x["m"] / 2)
        print(f"  max {x['mu'] - x['m'] / 2:+.6f} at " + fmt_row(x))

    print("\nInternal checks")
    agg = {}
    for x in recs:
        agg.setdefault("quotient reversible, LA 1 = 0 (EXACT)", []).append(x["qok"])
        if x["agree"] is not None:
            agg.setdefault("witness and LDL agree (EXACT)", []).append(x["agree"])
        if "cert_agree" in x:
            agg.setdefault("PD certificate succeeds iff LDL says > (EXACT)", []).append(x["cert_agree"])
        if "compl" in x:
            agg.setdefault("complement (r,c)->(n-r,m-c) maps quotient (EXACT)", []).append(x["compl"])
        for key, val in x.get("full", {}).items():
            if key != "dense_ran":
                agg.setdefault(key, []).append(val)
    names = {"M_ok": "|Omega| by labeled enumeration = sum of class sizes (EXACT)",
             "fiber_sizes": "fiber sizes = binom(2q,q) (EXACT)",
             "fact_ii": "Proposition 8.1(2): row swap stays in the a-fiber (EXACT)",
             "fact_i": "Proposition 8.1(1): H(1[x,y]-c_y/m) = (m/2)(1[x,y]-c_y/m) (EXACT)",
             "quotient_exact": "quotient = L H 1_D on every labeled matrix (EXACT)",
             "dense_mu": "dense full H: invariant-sector min = quotient mu_inv (FLOAT)",
             "dense_off_inv": "dense full H: H >= m/2 off the invariant sector (FLOAT)",
             "dense_H_minus_LT": "dense full H: H - L_T/2 >= 0 (FLOAT)",
             "dense_gap": "dense full H: gap(H) = min(m/2, mu_inv) (FLOAT)",
             "sparse_gap": "Lanczos full H: gap(H) = min(m/2, mu_inv) (FLOAT)"}
    for key, vals in agg.items():
        good = all(vals)
        ok &= good
        print(f"  {names.get(key, key)}: {sum(vals)}/{len(vals)} {'ok' if good else 'FAILED'}")
    dense = [x for x in recs if x.get("full", {}).get("dense_ran") and x["mu"] is not None]
    quot = [x for x in recs if "quotient_exact" in x.get("full", {}) and x["mu"] is not None]
    print(f"  of these, non-vacuous margins: dense full-H cross-check {len(dense)} "
          f"(largest |Omega| {max((x['M'] for x in dense), default=0)}), exact labeled quotient "
          f"cross-check {len(quot)} (largest |Omega| {max((x['M'] for x in quot), default=0)})")
    return ok, bir, non


def known_example():
    """EXACT: the 3x4 example of Section 8, r=2 and c=(1,1,2,2) (columns reordered), has
    mu_inv = 1 exactly."""
    cls = classes(3, 4, 2, (2, 2, 1, 1))
    LA, w, L = quotient(3, 4, 2, cls)
    return compare_exact(LA, w, L, 1, 1) == 0 and compare_exact(LA, w, L, 3, 2) == -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--large", action="store_true",
                    help="add all margins with m = 7, n <= 7; r = 2 with m, n in {7, 8}; "
                         "biregular m, n <= 8; c = 1^n with n = m r <= 12 "
                         "(quotients up to --cap classes)")
    ap.add_argument("--cap", type=int, default=40000, help="class cap for --large")
    args = ap.parse_args()
    t0 = time.time()
    print("Default grid: 2 <= m <= 6, 2 <= n <= 7, 1 <= r <= n/2, c weakly decreasing, 0 < c_y < m")
    recs = sweep(grid(6, 7), False, None, 2500)
    if args.large:
        extra = grid(7, 7, mmin=7)
        extra += [x for x in grid(8, 8, mmin=7, nmin=7) if x[2] == 2]
        extra += [x for x in grid(8, 8, bireg_only=True)]
        extra += [(m, m * r, r, (1,) * (m * r)) for m in range(2, 7) for r in range(2, 7)
                  if 7 < m * r <= 12]
        seen = {(x["m"], x["n"], x["r"], x["c"]) for x in recs}
        extra = [x for x in dict.fromkeys(extra) if x not in seen]
        print(f"\n--large: {len(extra)} more margins (class cap {args.cap})")
        recs += sweep(extra, True, args.cap, 7000)
    ok, _, _ = report(recs)
    ex = known_example()
    print(f"  3x4 example of Section 8 has mu_inv = 1 exactly and < 3/2 (EXACT): {'ok' if ex else 'FAILED'}")
    ok &= ex
    print(f"\ntotal {time.time() - t0:.1f}s")
    if not ok:
        print("FAIL: an internal check did not hold")
        sys.exit(1)
    print("PASS: Proposition 8.1(1), (2), the quotient, and its cross-checks hold on every margin checked")


if __name__ == "__main__":
    main()
