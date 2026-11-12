# The Curveball chain on regular graphs: spectral gap at most 2/n and a comparison with random transpositions

Mark T. Short, Independent researcher, [seveneves.ai](https://seveneves.ai).

[Read the 15-page paper](paper/main.pdf) or [its LaTeX source](paper/main.tex).
Version 1.0.0, DOI [10.5281/zenodo.24180369](https://doi.org/10.5281/zenodo.24180369).

The Curveball chain `K` performs a uniform trade at a uniformly chosen vertex pair of a
`d`-regular graph on `n` vertices, `n >= 4` and `1 <= d <= n-2`. Write `H = binom(n,2)(I-K)`,
`L_T` for the random-transposition Laplacian on vertex labels, and `mu` for the least
eigenvalue of `H` on centered functions of the isomorphism class of the graph rooted at one
vertex. Proved by hand (Theorem 1.1): `n-1` is an eigenvalue of `H` at every centered edge
indicator, so `1-2/n` is an eigenvalue of `K` and `gap(K) <= 2/n`; `H >= L_T/2`, which gives
`H >= n-1` on every isotypic component other than `(n)` and `(n-1,1)`; `gap(H) = min{n-1, mu}`;
and `gap(K) = 2/n` for `d in {1, n-2}` (for `d = 1` this follows from Diaconis and Holmes).
Computer-assisted, by exact computation in integers and rationals: `gap(K) = 2/n` for every
`n <= 11` and for `n = 12` with `d not in {5,6}` (Theorem 1.2); the triple and disjoint-pair
terms of the expansion of `H^2` cannot be bounded separately (Proposition 7.2); and for binary
matrices with equal row sums under row trades the corresponding equality fails, for example
`mu_sym = 4/3 < 2` for `4 x 4` matrices with all sums 2 (Section 8). Floating-point evidence
only: the `mu` column of Table 1, and `mu` about 14.62 > 11 for `n = 12`, `d in {5,6}`. Open:
whether `gap(K) = 2/n` for every regular degree sequence (Conjecture 9.1).

Status: the hand proofs are complete in the paper. The computer-assisted statements rest on the
exact programs below; CI runs them on every push, except the `n = 12`, `d = 4` row of Theorem
1.2 (27 minutes, 14 GB) and the larger binary margins. The only review is one independent AI
review pass; there is no human review and no Lean formalization. The manuscript, the programs
and the verification were produced with AI assistance.

## Reproduce

Python 3.13 and `pip install -r requirements.txt` (NumPy 2.1.3, SciPy 1.15.1, python-flint
0.7.1, SymPy 1.14.0). Run from `scripts/`; `fibers.py` is the module shared by `exact.py` and
`obstruction.py`. The programs number vertices, rows and columns from 0; the paper numbers them
from 1. CI runs the first five rows; the negative control passes there only if it fails.
Times are for a 24-thread desktop.

| Command | Supports | Arithmetic | Success output | Time |
|---|---|---|---|---|
| `python exact.py` | Lemma 3.1, Proposition 3.2, Lemma 4.1 and the two-element fibers for `d in {1, n-2}` (Remark 5.3), on 12 fibers with `n <= 8` | exact integers | exit 0; last line `PASS: Lemma 3.1, Proposition 3.2, Lemma 4.1 and, ... hold exactly on all 12 fibers` | 5 s |
| `python obstruction.py` | Proposition 7.2 | exact rationals | exit 0; last line `PASS: Proposition 7.2: ... <t,S t>/<t,H t> = 2314/835 < 3 on Omega_{8,3}` | 10 s |
| `python rooted.py 6 2 7 2 8 2 8 3 9 2 9 4 10 2 10 3 10 4 11 2 11 4 12 2 12 3 --workers 4` | Theorem 1.2: every row of Table 1 except `n = 12`, `d = 4` | exact integers (python-flint); `mu` printed in floating point | exit 0; last line `PASS: gap(H) = n-1 certified exactly for (n,d) in [...]` | 45 s |
| `python rooted.py 8 3 --threshold-shift 2` | negative control: asks for `mu > 9`, while `mu` is about 8.22 | exact integers | expected to FAIL: exit 1, `exact H > 9 on centered rooted-class functions: False (witness f^T B f < 0; Bareiss agrees ...)`, last line `FAIL: ...` | 1 s |
| `python bipartite.py` | `mu_sym` against `m/2` on all 149 margins with `m <= 6` rows and `k <= 7` columns, including the Section 8 failures (`m = k` in `{4,5,6}` with all sums 2, `m = k = 6` with all sums 3) and the `3 x 4` example; Proposition 8.1(1), (2) on the labeled space of 109 of them | exact verdicts; eigenvalues printed in floating point | exit 0; last line `PASS: Proposition 8.1(1), (2), the quotient, and its cross-checks hold on every margin checked` | 40 s |
| `python rooted.py 12 4` | Theorem 1.2, `n = 12`, `d = 4` | exact integers | exit 0; last line `PASS: ...` | 27 min, 14 GB; not in CI |
| `python rooted.py 12 5` | evidence for `n = 12`, `d in {5,6}`: 83,858 rooted classes, beyond the exact step | floating point | `floating min H ... = 14.623656`; exits 1 by design (`undecided`) | 6 min; not in CI |
| `python bipartite.py --large` | Section 8 for `m in {7,8}`: `mu_sym > m/2` for square matrices with all sums 2 | exact verdicts | exit 0; last line `PASS: ...` | 12 min; not in CI |

`rooted.py` also checks exactly that the rooted-class sizes sum to the number of labeled
graphs from an independent recursion and, where tabulated, from OEIS, that the quotient satisfies
detailed balance, and, for at most 64 rooted classes, that fraction-free elimination agrees with
the certificate. Floating point only proposes the congruence matrix of Section 6; its
verification uses integers. The pairs of Theorem 1.2 outside Table 1 follow from Theorem 1.1(4),
from complementation (Lemma 3.3), or have a single rooted class.

Build with `tectonic --keep-logs paper/main.tex`. The release build has 15 pages and
zero LaTeX, package, reference or box warnings.

## Versions

- 1.0.0, 2026-11-13: initial release.

## Cite and reuse

See [CITATION.cff](CITATION.cff); the archived record is https://doi.org/10.5281/zenodo.24180369.
The paper is CC BY 4.0; code is MIT licensed. A bounded literature search found no prior source
for the eigenvalue `n-1`, the comparison with random transpositions or the conjecture; no
priority claim is made. The arXiv identifier will be added after submission.

```bibtex
@misc{Short2026RegularCurveball,
  author = {Short, Mark T.},
  title = {The Curveball chain on regular graphs: spectral gap at most 2/n and a comparison with random transpositions},
  year = {2026},
  month = nov,
  version = {1.0.0},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.24180369},
  url = {https://doi.org/10.5281/zenodo.24180369},
  note = {Preprint}
}
```
