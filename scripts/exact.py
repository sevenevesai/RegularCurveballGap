"""Exact (integer) checks of Lemma 3.1, Proposition 3.2 and Lemma 4.1 on twelve regular
fibers: ten with 5 <= n <= 8, plus (6,4) and (8,6) to exercise d = n-2.

Usage, from the scripts directory:  python exact.py

1. Lemma 3.1 (balanced trades): E_{xw} 1[xy] = (1[xy] + 1[wy]) / 2 for every graph, pair
   {x,w}, and y outside it.
2. Proposition 3.2 (edge eigenfunctions): (H 1[xy])(G) = (n-1) 1[xy](G) - d for every graph
   and pair {x,y}.
3. Lemma 4.1 (the exchange is a trade): relabeling G by (x w) stays in the {x,w}-fiber of G.
4. d in {1, n-2} (Remark 5.3 and, by complementation, Lemma 3.3): every fiber is {G, (x w)G},
   so E_a = (I + T_a)/2 and H = L_T / 2.
Fiber averages are scaled by L = lcm of all fiber sizes, so every comparison is between
integers; no floating point enters a verdict. Vertices are numbered from 0. Exits 1 if any
check fails; the last line starts with PASS otherwise.
"""
import math
import sys
import time

import numpy as np

from fibers import Fiber, regular_cases


def check(n, d):
    F = Fiber([d] * n)
    edge = {p: ((F.states >> F.bit[p]) & 1).astype(np.int64) for p in F.pairs}
    sizes = [np.bincount(lab) for lab in F.labels]
    L = math.lcm(*{int(s) for sz in sizes for s in sz})
    marg_ok = swap_ok = small_ok = True
    H_edge = {p: np.zeros(F.M, dtype=np.int64) for p in F.pairs}  # L * (H 1[p])
    for idx, (x, w) in enumerate(F.pairs):
        lab, sz = F.labels[idx], sizes[idx]
        scale = (L // sz)[lab]
        perm = list(range(n))
        perm[x], perm[w] = w, x
        r = F.relabel(perm)
        swap_ok &= bool(np.array_equal(lab[r], lab))
        if d in (1, n - 2):
            small_ok &= bool(np.all((sz[lab] == 2) & (r != np.arange(F.M)) | (sz[lab] == 1) & (r == np.arange(F.M))))
        for p in F.pairs:
            sums = np.zeros(len(sz), dtype=np.int64)
            np.add.at(sums, lab, edge[p])
            mean_L = scale * sums[lab]  # L * fiber mean of 1[p], an exact integer
            H_edge[p] += L * edge[p] - mean_L
            u, v = p
            if (u in (x, w)) != (v in (x, w)):
                a, y = (u, v) if u in (x, w) else (v, u)
                other = w if a == x else x
                q = (min(other, y), max(other, y))
                marg_ok &= bool(np.array_equal(2 * mean_L, L * (edge[p] + edge[q])))
    eig_ok = all(np.array_equal(H_edge[p], L * ((n - 1) * edge[p] - d)) for p in F.pairs)
    return F.M, marg_ok, eig_ok, swap_ok, (small_ok if d in (1, n - 2) else None)


if __name__ == "__main__":
    print("(n,d)  M      marginal  H-edge  swap-in-fiber  fiber={G,tG}")
    cases = regular_cases() + [(6, 4), (8, 6)]
    failed = []
    for n, d in cases:
        t0 = time.time()
        M, a, b, c, s = check(n, d)
        print(f"({n},{d})  {M:<6} {a!s:<9} {b!s:<7} {c!s:<14} {s!s:<12} {time.time() - t0:.1f}s", flush=True)
        if not (a and b and c and s is not False):
            failed.append((n, d))
    if failed:
        print(f"FAIL: a check is False on (n,d) in {failed}")
        sys.exit(1)
    print(f"PASS: Lemma 3.1, Proposition 3.2, Lemma 4.1 and, for d in {{1, n-2}}, fibers {{G, (x w)G}} "
          f"hold exactly on all {len(cases)} fibers")
