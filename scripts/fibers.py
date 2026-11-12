"""Labeled graphs with a fixed degree vector, their Curveball pair fibers, and the operators.

Shared by exact.py and obstruction.py. `realizations` lists every labeled graph with the given
degrees and `fiber_labels` groups the graphs into the pair fibers of Section 2. States are
bitmasks over the pairs of {0, ..., n-1}, and all operators are sparse matrices in the counting
basis, which is orthogonal for the uniform law.
"""
import itertools

import numpy as np
from scipy.sparse import csr_matrix, identity


def realizations(d):
    n = len(d)
    out = []

    def rec(v, resid, mask):
        if v == n:
            out.append(mask)
            return
        need = resid[v]
        later = [w for w in range(v + 1, n) if resid[w] > 0]
        if need > len(later):
            return
        for nb in itertools.combinations(later, need):
            r = list(resid)
            mk = mask
            for w in nb:
                r[w] -= 1
                mk |= 1 << bit[(v, w)]
            r[v] = 0
            rec(v + 1, r, mk)

    bit = {}
    for i in range(n):
        for j in range(i + 1, n):
            bit[(i, j)] = bit[(j, i)] = len(bit) // 2
    rec(0, list(d), 0)
    return out, bit


def fiber_labels(states, bit, n):
    """One label array per vertex pair: states with equal labels share a Curveball fiber."""
    all_labels = []
    for u in range(n):
        for v in range(u + 1, n):
            keys = {}
            labels = np.empty(len(states), dtype=np.int64)
            for k, s in enumerate(states):
                clear = 0
                for w in range(n):
                    if w in (u, v):
                        continue
                    bu, bv = 1 << bit[(u, w)], 1 << bit[(v, w)]
                    if bool(s & bu) != bool(s & bv):
                        clear |= bu | bv
                labels[k] = keys.setdefault(s & ~clear, len(keys))
            all_labels.append(labels)
    return all_labels


def pairs(n):
    """Vertex pairs in the order used by fiber_labels."""
    return [(u, v) for u in range(n) for v in range(u + 1, n)]


class Fiber:
    """Omega_d with E_a, h_a = I - E_a, H = sum_a h_a, and the relabeling action of S_n."""

    def __init__(self, d):
        self.d = tuple(d)
        self.n = n = len(d)
        states, self.bit = realizations(d)
        self.states = np.array(states, dtype=np.int64)
        self.M = M = len(states)
        self.pairs = pairs(n)
        self.labels = fiber_labels(states, self.bit, n)
        self.eye = identity(M, format="csr")
        self.h = [self.eye - self._average(lab) for lab in self.labels]
        self.H = sum(self.h[1:], self.h[0])
        self._order = np.argsort(self.states)

    def _average(self, lab):
        sizes = np.bincount(lab)
        rows = np.arange(self.M)
        incidence = csr_matrix((np.ones(self.M), (rows, lab)), shape=(self.M, len(sizes)))
        return (incidence.multiply(1.0 / sizes[lab][:, None]) @ incidence.T).tocsr()

    def h_of(self, u, v):
        return self.h[self.pairs.index((min(u, v), max(u, v)))]

    def relabel(self, perm):
        """Index array r with states[r[k]] = perm applied to states[k]; perm maps old->new labels."""
        new = np.zeros(self.M, dtype=np.int64)
        for (i, j) in self.pairs:
            b = self.bit[(i, j)]
            new |= ((self.states >> b) & 1) << self.bit[(perm[i], perm[j])]
        pos = np.searchsorted(self.states[self._order], new)
        r = self._order[pos]
        assert np.array_equal(self.states[r], new), "relabeling left the fiber (degrees not regular?)"
        return r

    def permutation_matrix(self, perm):
        r = self.relabel(perm)
        return csr_matrix((np.ones(self.M), (r, np.arange(self.M))), shape=(self.M, self.M))

    def transposition(self, i, j):
        perm = list(range(self.n))
        perm[i], perm[j] = j, i
        return self.permutation_matrix(perm)

    def transposition_laplacian(self):
        """L_T = sum over transpositions t of (I - T_t); acts on the lambda-isotypic part
        as binom(n,2) - content(lambda)."""
        return sum((self.eye - self.transposition(i, j) for i, j in self.pairs),
                   0 * self.eye).tocsr()

    def disjoint_sum(self):
        """D = sum over ordered disjoint pairs (a, b) of h_a h_b."""
        D = 0 * self.eye
        for x, a in enumerate(self.pairs):
            for y, b in enumerate(self.pairs):
                if x < y and not set(a) & set(b):
                    D = D + self.h[x] @ self.h[y] + self.h[y] @ self.h[x]
        return D.tocsr()

    def triple_slack(self):
        """S3 = sum over vertex triples T of (H_T^2 - H_T), with H_T = sum_{a in T} h_a."""
        S3 = 0 * self.eye
        for T in itertools.combinations(range(self.n), 3):
            HT = sum((self.h_of(u, v) for u, v in itertools.combinations(T, 2)), 0 * self.eye)
            S3 = S3 + HT @ HT - HT
        return S3.tocsr()

    def orbits(self, fixed_blocks):
        """Orbit labels of the subgroup of S_n preserving each block in fixed_blocks (a list of
        vertex lists partitioning [n]); generated by adjacent transpositions inside blocks."""
        parent = np.arange(self.M)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for block in fixed_blocks:
            for i, j in zip(block, block[1:]):
                perm = list(range(self.n))
                perm[i], perm[j] = j, i
                r = self.relabel(perm)
                for k in range(self.M):
                    a, b = find(k), find(r[k])
                    if a != b:
                        parent[a] = b
        roots = np.array([find(k) for k in range(self.M)])
        _, lab = np.unique(roots, return_inverse=True)
        return lab

    def orbit_basis(self, lab):
        """Orthonormal basis (columns) of functions constant on the given orbits."""
        sizes = np.bincount(lab)
        return csr_matrix((1.0 / np.sqrt(sizes[lab]), (np.arange(self.M), lab)),
                          shape=(self.M, len(sizes)))


def regular_cases():
    """Ten regular fibers (n, d) with 5 <= n <= 8, checked by exact.py."""
    return [(5, 2), (6, 1), (6, 2), (6, 3), (7, 2), (7, 4), (8, 1), (8, 2), (8, 3), (8, 4)]
