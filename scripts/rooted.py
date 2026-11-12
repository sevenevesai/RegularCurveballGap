"""Exact certificate of gap(H) = n-1 from rooted isomorphism classes, without listing Omega
(Theorem 1.2 and Section 6).

Usage, from the scripts directory (exact matrix products need python-flint==0.7.1):
    python rooted.py N D [N D ...] [--workers W] [--exact-max M] [--threshold-shift S]
Exits 0 with a last line starting PASS when every case is certified, and 1 otherwise.

By Theorem 1.1(3), gap(H) = n-1 iff H >= n-1 on centered rooted-class functions. Vertices
are numbered from 0 and graphs are rooted at vertex 0; the paper roots them at vertex n,
and by symmetry the choice of root does not matter. Breadth-first search over Curveball
moves (the chain is irreducible)
finds every rooted class c with a representative G_c, the exact quotient
A[c, c'] = sum_a (1[c'=c] - |fiber_a(G_c) ∩ c'| / |fiber_a(G_c)|), and the class sizes
w_c = (n-1)! / |Aut_0(G_c)| from the canonical-form search.

Checked exactly in every case: detailed balance w_c A[c,c'] = w_c' A[c',c]; each pair's fiber
probabilities sum to 1; M = sum_c w_c equals |Omega| from an independent degree recursion and,
where tabulated, OEIS; the number of unlabeled classes equals OEIS where tabulated.

Decided exactly for at most --exact-max rooted classes, with mu = n-1+S and K = mu+1: is
B = W(A - mu I) + K w w^T / M positive definite? It is iff H > mu strictly on centered
rooted-class functions, since B agrees with the form of H - mu on w-perp and B 1 = (K-mu) w;
for S = 0 this is the matrix B of Lemma 6.1. A positive --threshold-shift asks for more than
the theorem and is a negative control: (8,3) with S = 2 must fail with an exact witness.
True needs an integer Q with Q^T (scaled B) Q strictly diagonally dominant; then Q is
nonsingular and B is congruent to a positive definite matrix. False needs an integer f with
f^T (scaled B) f < 0. Both are checked in integer arithmetic; floating eigenvectors only
propose Q and f, and anything else leaves the case undecided and failed. For at most
BAREISS_MAX rooted classes fraction-free elimination decides again and must agree.

Evidence only: the floating least eigenvalue of H on centered rooted-class functions.
"""
import argparse
import functools
import itertools
import math
import os
import sys
import time
from fractions import Fraction
from multiprocessing import Pool

import flint
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

BAREISS_MAX = 64
DENSE_FLOAT_MAX = 20000
BLOCK = 2048

# Labeled d-regular graphs on n nodes, OEIS A001205 (d=2), A002829 (d=3), A005815 (d=4),
# A338978 (d=5), read 2026-11-12; every entry also equals count_labeled(n, d).
LABELED = {
    **{(n, 2): c for n, c in zip(range(3, 13), (1, 3, 12, 70, 465, 3507, 30016, 286884, 3026655, 34944085))},
    **{(n, 3): c for n, c in zip(range(4, 13, 2), (1, 70, 19355, 11180820, 11555272575))},
    **{(n, 4): c for n, c in zip(range(5, 13), (1, 15, 465, 19355, 1024380, 66462606, 5188453830, 480413921130))},
    **{(n, 5): c for n, c in zip(range(6, 13, 2), (1, 3507, 66462606, 2977635137862))},
}
# Unlabeled d-regular graphs on n nodes, OEIS A008483 (d=2: partitions of n into parts >= 3),
# A005638 (d=3), A033301 (d=4), A165626 (d=5), read 2026-11-12.
UNLABELED = {
    **{(n, 2): c for n, c in zip(range(3, 13), (1, 1, 1, 2, 2, 3, 4, 5, 6, 9))},
    **{(n, 3): c for n, c in zip(range(4, 13, 2), (1, 2, 6, 21, 94))},
    **{(n, 4): c for n, c in zip(range(5, 13), (1, 1, 2, 6, 16, 60, 266, 1547))},
    **{(n, 5): c for n, c in zip(range(6, 13, 2), (1, 3, 60, 7849))},
}


def known(table, n, d):
    """Table entry for (n, d) or, by complementation, for (n, n-1-d)."""
    return table.get((n, d), table.get((n, n - 1 - d)))


def count_labeled(n, d):
    """Labeled d-regular graphs on n nodes: give the vertex of largest residual degree its
    remaining neighbours, grouped by residual degree; the count depends only on that multiset."""

    @functools.cache
    def count(state):  # state[j-1] = vertices still needing j edges
        top = max((j for j in range(1, len(state) + 1) if state[j - 1]), default=0)
        if not top:
            return 1
        rest = list(state)
        rest[top - 1] -= 1
        total = 0

        def choose(j, need, cur, ways):
            nonlocal total
            if need == 0:
                total += ways * count(tuple(cur))
                return
            if j == 0:
                return
            for k in range(min(need, rest[j - 1]) + 1):
                nxt = cur[:]
                nxt[j - 1] -= k
                if j > 1:
                    nxt[j - 2] += k
                choose(j - 1, need - k, nxt, ways * math.comb(rest[j - 1], k))

        choose(len(rest), top, rest, 1)
        return total

    return count(tuple(n if j == d else 0 for j in range(1, d + 1)))


# ---------- canonical form of a graph rooted at vertex 0 ----------


def _refine(nbrs, colors, shift):
    k = len(set(colors))
    while True:
        sig = [(c, sum(1 << shift * colors[u] for u in nb)) for c, nb in zip(colors, nbrs)]
        rank = {s: i for i, s in enumerate(sorted(set(sig)))}
        colors = [rank[s] for s in sig]
        if len(rank) == k:
            return colors
        k = len(rank)


def canonical(masks, auts=False):
    """(certificate, |Aut_0|[, automorphisms fixing 0]) for neighbour bitmasks `masks`, root 0.

    Individualization-refinement without pruning from the invariant colouring (root, triangle
    count): the certificate is the least relabelled adjacency over all leaves, and the leaves
    attaining it are the images of one of them under Aut_0, one per automorphism."""
    n = len(masks)
    nbrs = [[u for u in range(n) if m >> u & 1] for m in masks]
    shift = n.bit_length()
    tri = [sum((masks[v] & masks[u]).bit_count() for u in nbrs[v]) for v in range(n)]
    inv = [(v != 0, tri[v]) for v in range(n)]
    rank = {s: i for i, s in enumerate(sorted(set(inv)))}
    best, leaves = [None, 0], []

    def search(colors):
        colors = _refine(nbrs, colors, shift)
        if max(colors) == n - 1:
            cert = 0
            for v in range(n):
                row = 0
                for u in nbrs[v]:
                    row |= 1 << colors[u]
                cert |= row << n * colors[v]
            if best[0] is None or cert < best[0]:
                best[0], best[1] = cert, 0
                leaves.clear()
            if cert == best[0]:
                best[1] += 1
                if auts:
                    leaves.append(colors)
            return
        size = [0] * n
        for c in colors:
            size[c] += 1
        target = next(c for c in range(n) if size[c] > 1)
        for v in range(n):
            if colors[v] == target:
                c2 = [2 * c + 1 for c in colors]
                c2[v] -= 1
                search(c2)

    search([rank[s] for s in inv])
    if not auts:
        return best[0], best[1]
    at = {p: v for v, p in enumerate(leaves[0])}
    return best[0], best[1], [[at[p] for p in leaf] for leaf in leaves]


def unrooted(masks):
    """Unlabeled class key: least rooted certificate over all choices of root."""
    n = len(masks)
    best = None
    for r in range(n):
        perm = [r] + [v for v in range(n) if v != r]
        pos = {v: i for i, v in enumerate(perm)}
        relabeled = tuple(sum(1 << pos[u] for u in range(n) if masks[v] >> u & 1) for v in perm)
        cert, _ = canonical(relabeled)
        best = cert if best is None or cert < best else best
    return best


# ---------- Curveball pair fibers of one representative ----------


def process(job):
    """Row of Dn * sum_a P_a(G -> class) for G = masks, summed over pair orbits of Aut_0(G).

    Pairs in one Aut_0-orbit have equal rows. For a pair avoiding 0, X and S \\ X give graphs
    related by the transposition (x w), which fixes the root, so only X containing min S is
    canonicalised. Returns (certificate, |Aut_0|, unlabeled key, row, example per new class)."""
    masks, Dn = job
    n = len(masks)
    cert, aut, perms = canonical(masks, auts=True)
    row, examples, seen = {}, {}, set()
    for x, w in itertools.combinations(range(n), 2):
        if (x, w) in seen:
            continue
        orbit = {(min(g[x], g[w]), max(g[x], g[w])) for g in perms}
        seen |= orbit
        sx, sw = masks[x] & ~(1 << w), masks[w] & ~(1 << x)
        single = sx ^ sw
        S = [y for y in range(n) if single >> y & 1]
        own = sx & single
        q = own.bit_count()
        assert 2 * q == len(S), "fiber is not balanced: degrees not regular"
        size = math.comb(2 * q, q)
        assert Dn % size == 0
        half = q > 0 and 0 not in (x, w)
        unit = len(orbit) * (2 if half else 1) * (Dn // size)
        choices = itertools.combinations(S[1:], q - 1) if half else itertools.combinations(S, q)
        for chosen in choices:
            X = sum(1 << y for y in chosen) | (1 << S[0] if half else 0)
            if X == own or (half and X == single ^ own):
                c = cert
            else:
                new = list(masks)
                new[x] = masks[x] & ~single | X
                new[w] = masks[w] & ~single | single ^ X
                for y in S:
                    new[y] = new[y] & ~(1 << x | 1 << w) | (1 << x if X >> y & 1 else 1 << w)
                c, _ = canonical(new)
                if c not in examples:
                    examples[c] = tuple(new)
            row[c] = row.get(c, 0) + unit
    return cert, aut, unrooted(masks), row, examples


def circulant(n, d):
    steps = list(range(1, d // 2 + 1))
    masks = [0] * n
    for v in range(n):
        for s in steps:
            for u in ((v + s) % n, (v - s) % n):
                masks[v] |= 1 << u
        if d % 2:
            masks[v] |= 1 << ((v + n // 2) % n)
    assert all(bin(m).count("1") == d for m in masks)
    return tuple(masks)


def explore(n, d, workers):
    """Rooted classes by BFS: (reps, |Aut_0| list, unlabeled keys, rows {class: numerator}, Dn)."""
    Dn = math.lcm(*(math.comb(2 * q, q) for q in range(min(d, (n - 2) // 2) + 1)))
    start = circulant(n, d)
    ids, reps, info = {canonical(start)[0]: 0}, [start], [None]
    todo = [0]
    # Pool processes import numpy but only search; one BLAS thread each keeps many processes
    # inside the Windows commit limit (24 processes with default BLAS threading exhausted the
    # paging file).
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    with Pool(workers) as pool:
        while todo:
            jobs = [(reps[c], Dn) for c in todo]
            chunk = max(1, len(jobs) // (8 * workers))
            for c, (cert, aut, key, row, ex) in zip(todo, pool.imap(process, jobs, chunksize=chunk)):
                assert ids[cert] == c, "representative certificate mismatch"
                r = {}
                for cc, v in row.items():
                    if cc not in ids:
                        ids[cc] = len(reps)
                        reps.append(ex[cc])
                        info.append(None)
                    r[ids[cc]] = v
                info[c] = (aut, key, r)
            todo = [c for c in range(len(reps)) if info[c] is None]
    auts, keys, rows = zip(*info)
    return reps, list(auts), list(keys), list(rows), Dn


# ---------- floating evidence ----------


def float_min(n, w, rows, Dn):
    """Floating least eigenvalue of H on centered rooted-class functions (evidence only)."""
    m = len(w)
    if m == 1:
        return float("inf")
    pairs = n * (n - 1) // 2
    s = np.sqrt(np.array(w, float))
    I, J, V = [], [], []
    for c, r in enumerate(rows):
        I.append(c), J.append(c), V.append(float(pairs))
        for c2, v in r.items():
            I.append(c), J.append(c2), V.append(-v / Dn * s[c] / s[c2])
    S = coo_matrix((V, (I, J)), shape=(m, m)).tocsr()
    S = (S + S.T) / 2
    if m <= DENSE_FLOAT_MAX:
        ev = eigh(S.toarray(), eigvals_only=True, subset_by_index=[0, 1])
        assert abs(ev[0]) < 1e-8, ev
        return float(ev[1])
    u = s / np.linalg.norm(s)
    lift = 4.0 * pairs
    op = LinearOperator((m, m), matvec=lambda x: S @ x + lift * u * (u @ x), dtype=float)
    ev = eigsh(op, k=1, which="SA", tol=1e-10, ncv=96, maxiter=100000, return_eigenvectors=False)
    return float(ev[0])


# ---------- exact linear algebra ----------


def positive_definite(B):
    """Bareiss fraction-free elimination on an integer matrix: all leading minors > 0."""
    m = len(B)
    B = [row[:] for row in B]
    prev = 1
    for k in range(m):
        if B[k][k] <= 0:
            return False
        for i in range(k + 1, m):
            for j in range(k + 1, m):
                B[i][j] = (B[i][j] * B[k][k] - B[i][k] * B[k][j]) // prev
        prev = B[k][k]
    return True


def scaled_form(n, w, rows, Dn, mu):
    """Integers (Bz, c0) with M mu_d Dn B = M Bz + c0 w w^T and Bz = mu_d Dn W(A - mu I)."""
    pairs = n * (n - 1) // 2
    md, mn = mu.denominator, mu.numerator
    Bz = []
    for c, r in enumerate(rows):
        b = {c2: -w[c] * md * v for c2, v in r.items() if c2 != c}
        b[c] = w[c] * (md * (Dn * pairs - r.get(c, 0)) - Dn * mn)
        Bz.append(b)
    return Bz, Dn * (mn + md)


def exact_verdict(w, Bz, c0, M):
    """(True | False | None, how): congruence certificate, negative witness, or undecided."""
    m = len(w)
    wmax = max(w)
    D = [max(1, round(256 * math.sqrt(wmax / x))) for x in w]
    z = [d * x for d, x in zip(D, w)]
    Y = np.zeros((m, m))
    scale = float(M) * 65536.0 * wmax
    for c, b in enumerate(Bz):
        for c2, v in b.items():
            Y[c, c2] = float(D[c] * v * D[c2] * M) / scale
    zf = np.array(z, float) / math.sqrt(scale)
    Y += np.outer(zf, zf) * float(c0)
    lam, U = eigh(Y, overwrite_a=True, driver="evd")
    del Y
    if lam[0] <= 0:
        u = np.rint(U[:, 0] * 2.0 ** 40).astype(np.int64)
        f = [D[i] * int(u[i]) for i in range(m)]
        val = M * sum(f[c] * v * f[c2] for c, b in enumerate(Bz) for c2, v in b.items())
        val += c0 * sum(x * fi for x, fi in zip(w, f)) ** 2
        return (False, "witness f^T B f < 0") if val < 0 else (None, "witness failed")
    flat = [0] * (m * m)
    for c, b in enumerate(Bz):
        for c2, v in b.items():
            flat[c * m + c2] = D[c] * v * D[c2]
    DBD = flint.fmpz_mat(m, m, flat)
    del flat
    Z = flint.fmpz_mat(m, 1, z)
    for k in (40, 50):
        Q = np.rint(U * 2.0 ** k).astype(np.int64)
        blocks = [(j, flint.fmpz_mat(m, min(BLOCK, m - j), Q[:, j:j + BLOCK].ravel().tolist()))
                  for j in range(0, m, BLOCK)]
        tblocks = [(j, Qj.transpose()) for j, Qj in blocks]
        v = [int(x) for _, Qt in tblocks for x in (Qt * Z).entries()]
        diag, rowabs = [0] * m, [0] * m
        for j, Qj in blocks:
            bj = Qj.ncols()
            T = DBD * Qj
            vj = flint.fmpz_mat(1, bj, v[j:j + bj])
            for i, Qt in tblocks:
                bi = Qt.nrows()
                C = (Qt * T) * M + (flint.fmpz_mat(bi, 1, v[i:i + bi]) * vj) * c0
                e = C.entries()
                for r in range(bi):
                    rowabs[i + r] += sum(map(abs, e[r * bj:(r + 1) * bj]))
                    if i == j:
                        diag[i + r] = int(e[r * bj + r])
        if all(dg > 0 and 2 * dg > ra for dg, ra in zip(diag, rowabs)):
            return True, f"diagonally dominant congruence, 2^{k} scaling"
    return None, "certificate failed"


def run(n, d, workers, exact_max, shift):
    t0 = time.time()
    reps, auts, keys, rows, Dn = explore(n, d, workers)
    m = len(reps)
    fact = math.factorial(n - 1)
    pairs = n * (n - 1) // 2
    problems = []
    if any(fact % a for a in auts):
        problems.append("|Aut_0| does not divide (n-1)!")
    w = [fact // a for a in auts]
    if not all(sum(r.values()) == Dn * pairs for r in rows):
        problems.append("fiber probabilities do not sum to 1")
    if not all(w[c] * v == w[c2] * rows[c2].get(c, 0) for c, r in enumerate(rows) for c2, v in r.items()):
        problems.append("detailed balance fails")
    M = sum(w)
    classes = len(set(keys))
    count, oeis, oeis_u = count_labeled(n, d), known(LABELED, n, d), known(UNLABELED, n, d)
    if M != count or (oeis is not None and M != oeis):
        problems.append(f"sum w = {M}, recursion {count}, OEIS {oeis}")
    if oeis_u is not None and classes != oeis_u:
        problems.append(f"unlabeled classes {classes}, OEIS {oeis_u}")
    low = float_min(n, w, rows, Dn)
    mu = n - 1 + shift
    t1 = time.time()
    verdict, how = None, f"skipped: {m} rooted classes > --exact-max {exact_max}"
    if m <= exact_max:
        Bz, c0 = scaled_form(n, w, rows, Dn, mu)
        verdict, how = exact_verdict(w, Bz, c0, M)
        if m <= BAREISS_MAX:
            bareiss = positive_definite([[M * Bz[i].get(j, 0) + c0 * w[i] * w[j] for j in range(m)]
                                         for i in range(m)])
            if verdict is None:
                verdict, how = bareiss, f"{how}; Bareiss elimination decides"
            elif verdict == bareiss:
                how += "; Bareiss agrees"
            else:
                problems.append(f"certificate says {verdict}, Bareiss says {bareiss}")
                verdict = None
    if verdict is False:
        problems.append(f"H > {mu} on centered rooted-class functions is false (exact)")
    elif verdict is None:
        problems.append(f"H > {mu} on centered rooted-class functions undecided")
    print(f"({n},{d}) |Omega|={M} [recursion {count}, OEIS {oeis}] classes={classes} [OEIS {oeis_u}] "
          f"rooted={m} detailed-balance={'detailed balance fails' not in problems}\n"
          f"  floating min H on centered rooted-class functions = {low:.6f} (n-1+S = {float(mu):g})\n"
          f"  exact H > {mu} on centered rooted-class functions: {verdict} ({how}, {time.time() - t1:.1f}s)\n"
          f"  total {time.time() - t0:.1f}s" + "".join(f"\n  FAIL: {p}" for p in problems), flush=True)
    return not problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("nd", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--exact-max", type=int, default=DENSE_FLOAT_MAX)
    ap.add_argument("--threshold-shift", type=Fraction, default=Fraction(0))
    args = ap.parse_args()
    if len(args.nd) % 2:
        ap.error("give N D pairs")
    cases = list(zip(args.nd[::2], args.nd[1::2]))
    for n, d in cases:
        if not (1 <= d <= n - 2 and n * d % 2 == 0):
            ap.error(f"need 1 <= d <= n-2 and n*d even: ({n},{d})")
    flint.ctx.threads = args.workers
    passed = [(n, d) for n, d in cases if run(n, d, args.workers, args.exact_max, args.threshold_shift)]
    failed = [c for c in cases if c not in passed]
    S = args.threshold_shift
    claim = "gap(H) = n-1" if S == 0 else f"H > n-1+({S}) on centered rooted-class functions"
    if failed:
        print(f"FAIL: {claim} not certified for (n,d) in {failed}; certified for {passed}")
        sys.exit(1)
    print(f"PASS: {claim} certified exactly for (n,d) in {passed}")
