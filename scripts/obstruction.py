"""Exact computations of Proposition 7.2: the triple term S and the disjoint-pair term D of
the expansion H^2 = H + S + D (Section 7) cannot be bounded separately on regular fibers.

Usage, from the scripts directory:  python obstruction.py

Vertices are numbered from 0; the paper's vertex v is vertex v-1 here.
A. Part (1). In Omega_{7,2}, G = the 7-cycle 0-3-1-4-5-2-6-0 and T = {0,1,2}: the triple
   cell of G (closure under the three pair fibers inside T) has 15 graphs and H_T has the
   exact eigenvalue 1 there, with eigenvector the pairing sign. So H_T^2 >= (3/2) H_T fails.
B. Part (2). In Omega_{8,3}, the centered triangle count t satisfies <t,S t> < 3 <t,H t>
   exactly, while every centered edge indicator has D = 3H (Proposition 7.1). So no pair of
   inequalities S >= alpha H, D >= beta H with alpha + beta = n - 2 = 6 holds on Omega_{8,3}.
All arithmetic is in Fractions. Exits 1 if any check fails; the last line starts with PASS
otherwise.
"""
import itertools
import sys
import time
from fractions import Fraction

import numpy as np

from fibers import Fiber


def groups(lab, members):
    out = {}
    for k in members:
        out.setdefault(int(lab[k]), []).append(k)
    return out


def part_a():
    F = Fiber([2] * 7)
    edges = [(0, 3), (3, 1), (1, 4), (4, 5), (5, 2), (2, 6), (6, 0)]
    mask = sum(1 << F.bit[e] for e in edges)
    start = int(np.flatnonzero(F.states == mask)[0])
    T = (0, 1, 2)
    labs = [F.labels[F.pairs.index(p)] for p in itertools.combinations(T, 2)]
    cell, frontier = {start}, [start]
    while frontier:  # close under the three pair fibers inside T
        k = frontier.pop()
        for lab in labs:
            for j in np.flatnonzero(lab == lab[k]):
                if int(j) not in cell:
                    cell.add(int(j))
                    frontier.append(int(j))
    cell = sorted(cell)
    pos = {k: i for i, k in enumerate(cell)}
    m = len(cell)
    HT = [[Fraction(3 if i == j else 0) for j in range(m)] for i in range(m)]
    for lab in labs:
        for g in groups(lab, cell).values():
            for i in g:
                for j in g:
                    HT[pos[i]][pos[j]] -= Fraction(1, len(g))

    def owner(k, y):  # the unique vertex of T adjacent to y (y of T-degree 1)
        return [x for x in T if (F.states[k] >> F.bit[(min(x, y), max(x, y))]) & 1][0]

    def missing(k, y):  # the unique vertex of T not adjacent to y (y of T-degree 2)
        return [x for x in T if not (F.states[k] >> F.bit[(min(x, y), max(x, y))]) & 1][0]

    # outside vertices: 4 and 5 have one neighbour in T, 3 and 6 have two
    f = []
    for k in cell:
        a, b, c, e = owner(k, 4), owner(k, 5), missing(k, 3), missing(k, 6)
        f.append(Fraction((a == e and b == c and a != b) - (a == c and b == e and a != b)))
    Hf = [sum(HT[i][j] * f[j] for j in range(m)) for i in range(m)]
    print(f"A: cell size {m}; f nonzero: {any(f)}; H_T f == 1 * f exactly: {Hf == f}")
    return any(f) and Hf == f, m


def part_b():
    F = Fiber([3] * 8)
    n = 8
    E = {p: ((F.states >> F.bit[p]) & 1).astype(np.int64) for p in F.pairs}
    t = sum(E[(a, b)] * E[(a, c)] * E[(b, c)] for a, b, c in itertools.combinations(range(n), 3))
    t = [int(v) for v in t]
    fibs = [list(groups(lab, range(F.M)).values()) for lab in F.labels]

    def h(idx, f):  # exact h_a f
        out = [Fraction(0)] * F.M
        for g in fibs[idx]:
            mean = sum((f[k] for k in g), Fraction(0)) / len(g)
            for k in g:
                out[k] = f[k] - mean
        return out

    hs = [h(i, t) for i in range(len(F.pairs))]
    form_H = sum(sum(Fraction(t[k]) * v[k] for k in range(F.M)) for v in hs)
    form_S3 = Fraction(0)
    for T in itertools.combinations(range(n), 3):
        idx = [F.pairs.index(p) for p in itertools.combinations(T, 2)]
        HTt = [sum(hs[i][k] for i in idx) for k in range(F.M)]
        form_S3 += sum(v * v for v in HTt) - sum(t[k] * HTt[k] for k in range(F.M))
    ratio = form_S3 / form_H
    print(f"B: <t,S t>/<t,H t> = {ratio} = {float(ratio):.6f}; below (n-2)/2 = 3: {ratio < 3}")
    return ratio < 3, ratio


if __name__ == "__main__":
    t0 = time.time()
    ok_a, cell = part_a()
    ok_b, ratio = part_b()
    print(f"time {time.time() - t0:.1f}s")
    if not (ok_a and ok_b):
        print(f"FAIL: Proposition 7.2 part (1) {ok_a}, part (2) {ok_b}")
        sys.exit(1)
    print(f"PASS: Proposition 7.2: H_T has eigenvalue 1 on a {cell}-graph cell of Omega_{{7,2}}; "
          f"<t,S t>/<t,H t> = {ratio} < 3 on Omega_{{8,3}}")
