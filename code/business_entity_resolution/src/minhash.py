"""MinHash + LSH banding over the cached character 3-grams.

Why this stage exists
---------------------
`findings.md` records that **23.74% of true pairs are unreachable by any exact
key at any cap or budget**, and that exact-key blocking is capped at 76.26%
reachable recall. The failures are character-level by construction — typos
(`fnesnius`/`fresenius`), concatenation (`dentcoffee`/`coffee dent`),
transliteration residue (`kanstraksans`/`constructions`), web debris
(`com jgsreal`/`jgs real`). They share most of their 3-grams while sharing no
token exactly, so a similarity search over 3-gram sets reaches them and nothing
built out of equality does.

Why NOT an inverted index over the trigrams
-------------------------------------------
This is the obvious design and it does not survive the data. The measured
3-gram vocabulary over this corpus is **15,285 distinct grams** against roughly
258M target postings, so mean document frequency is about 17,000. There are no
rare trigrams to invert on: any document-frequency threshold low enough to
bound the fan-out also discards essentially the whole vocabulary. An inverted
index needs a heavy tail of rare terms and a 3-gram alphabet does not have one.

Why NOT an ANN index
--------------------
The previous attempt (TF-IDF + truncated SVD + FAISS, on a branch that is no
longer to hand) measured its SVD-128 basis at 34.3% of variance, so retrieval
ran against a lossy projection of the very similarity being measured. It also
needed two dependencies, has no GPU wheels on Windows, and the journal records
three separate memory and tuning failures in it.

What this does instead
----------------------
MinHash reduces each entity's 3-gram SET to K integers such that, for any one
permutation, `P[minhash(A) == minhash(B)] = Jaccard(A, B)` exactly. Grouping
those K values into `bands` groups of `rows` and requiring a whole group to
agree turns a *similarity* question into an *equality* question:

    P[at least one band agrees] = 1 - (1 - J**rows)**bands

So each band signature is an ordinary exact blocking key, and `src.blocking`
handles it with no new machinery at all — the same key-outer target build, the
same bucket cap, the same inverse-frequency weighting, the same per-entity
budget. No new dependency, no approximation of the similarity beyond the
MinHash estimator itself, and the memory shape that a 16GB machine already
survives.

A property worth naming, because it falls out free: a genuinely similar pair
agrees on MANY bands, and `blocking._merge` sums the weight of every key that
found a pair. So `block_score` becomes a monotone estimator of trigram Jaccard
rather than a flat count, which is exactly what the per-entity budget needs to
rank on. The candidate that agrees on 11 of 16 bands outranks the one that
agrees on 1 without anyone writing a similarity function.

Detection rate at the defaults (bands=16, rows=3):

    trigram Jaccard  0.30   0.40   0.50   0.60   0.70   0.80
    P[some band]     0.35   0.65   0.88   0.98   0.998  1.000

Chosen for recall rather than precision, on the evidence in `findings.md` that
the model reaches AP 0.9901 on held-out entities and can reject what blocking
over-collects — "extra candidates are now cheap" — and because transliteration
residue sits at the low end of that table. `--bands`/`--rows` move it.

Arithmetic constraints, both load-bearing
-----------------------------------------
`P_MOD` is 2**31-1 and NOT a larger Mersenne prime. polars UInt64 arithmetic
**wraps silently** on overflow — no exception, no warning — so a permutation
`(a*x + b) % P` with `a` near 2**61 and `x` a UInt32 gram computes `a*x` at
2**93, wraps, and yields a hash family that is neither the intended one nor
provably anything. Verified directly: at P=2**61-1 polars returns
2305843000623759375 where Python returns 2305843004918726663. At 2**31-1 the
worst product is 2**62 and polars agrees with Python exactly.

Band packing truncates each MinHash to `min(64 // rows, 31)` bits. That is a
hash of a hash, so it preserves "equal signature iff equal MinHash" up to a
per-position false-agreement rate of 2**-keep_bits, and a whole band agrees by
chance at 2**-(keep_bits * rows): 2**-63 at the default rows=3, 2**-64 at
rows=4, 2**-62 at rows=2 and 2**-31 at rows=1. Only rows=1 is anywhere near
loose, and rows=1 is degenerate as an LSH configuration anyway.

One honest imprecision, stated rather than glossed. The MinHash identity
`P[min_h(A) == min_h(B)] == J(A, B)` holds exactly for a permutation of the
universe, and `(a*x + b) mod P_MOD` is a genuine bijection on `[0, P_MOD)` for
any `a != 0`. But the grams arrive as full UInt32 and are folded into that
range by `% P_MOD` first, which is 2-to-1. Two DISTINCT trigrams sharing a
residue become indistinguishable and inflate similarity slightly. The bound is
small and computable rather than hand-waved: the measured vocabulary is 15,285
grams, so the chance that any collision exists anywhere in it is about
15285**2 / 2 / 2**31 ~= 5%, and if one does it merges one gram pair out of
15,285. It is a rounding error on the estimator, not a correctness problem, and
it is the price of keeping every product inside UInt64.
"""

from __future__ import annotations

import random

import polars as pl

# Mersenne prime. See the module docstring: this value is an overflow
# constraint, not a tuning choice, and raising it silently corrupts every
# signature.
P_MOD = (1 << 31) - 1

# Fixed, so the permutations are identical across the three sources, across
# runs, and across machines. The same reasoning as `TRIGRAM_HASH_SEED`: a
# MinHash is only ever compared against another MinHash from the same family,
# and mixing families across sources would make every cross-source band
# comparison silently return "no agreement" — no crash, no warning, and a
# blocking key that quietly becomes a constant.
MINHASH_SEED = 0x51E9A7C3

# Defaults. `rows` is the sensitivity knob: lower catches weaker similarity and
# emits more candidates, higher is stricter. `bands` trades runtime for recall
# roughly linearly, since each band is one more key-outer join in `blocking`.
DEFAULT_BANDS = 16
DEFAULT_ROWS = 3

BAND_PREFIX = "band_"


def _coefficients(k: int) -> list[tuple[int, int]]:
    """`k` (a, b) pairs for the family `h(x) = (a*x + b) mod P_MOD`.

    Generated from a fixed seed rather than stored as a literal table so that
    `--bands`/`--rows` can change K without a code edit — but note the
    consequence, which is the reason this function takes k and not a range:
    `random.Random` is a stream, so the first K coefficients are the same for
    every larger K. A run at bands=8 therefore uses exactly the first 8 bands
    of a run at bands=16, and the two candidate sets are nested rather than
    unrelated. That makes a bands sweep interpretable as a curve on ONE
    normalization pass instead of a re-normalize per point.

    That nesting holds ONLY while `rows` is unchanged. Band `b` consumes
    `coef[b*rows : (b+1)*rows]`, so changing `rows` re-partitions the whole
    stream and every band signature changes. Changing `rows` invalidates the
    normalized parquet; changing `bands` alone does not.
    """
    rng = random.Random(MINHASH_SEED)
    # `a` must be non-zero mod P, otherwise the permutation is the constant b
    # and that band degenerates into a single bucket holding the entire corpus.
    return [(rng.randrange(1, P_MOD), rng.randrange(0, P_MOD)) for _ in range(k)]


def band_columns(bands: int = DEFAULT_BANDS) -> tuple[str, ...]:
    """The column names this module produces, zero-padded so they sort."""
    return tuple(f"{BAND_PREFIX}{i:02d}" for i in range(bands))


def band_expressions(
    col: str = "name_trigrams",
    bands: int = DEFAULT_BANDS,
    rows: int = DEFAULT_ROWS,
) -> list[pl.Expr]:
    """One UInt64 LSH signature per band, from a hashed-3-gram list column.

    Takes a COLUMN NAME, not an expression, for the reason
    `text_norm.trigram_expression` does: building these on top of the
    expression that produces the trigrams would make polars recompute the
    entire `name_core` chain and all 48 trigram slices once per permutation.
    Requiring a name makes a caller materialize its input first, which turns
    the session-4 mistake from "remember not to" into "cannot be written".

    A row whose trigram list is EMPTY — a name that normalized to nothing —
    yields `list.min() == null` for every permutation, which propagates through
    the packing to a null band key, which `blocking._usable` drops. That is the
    wanted behaviour and it is structural rather than guarded: without it every
    blank name in the corpus would share one signature and form a single
    enormous spurious bucket. This is the `_jaccard` empty-vs-empty bug family
    (journal, sessions 3 and 4) arriving at a stage where the arithmetic
    refuses it instead of a flag having to catch it.
    """
    if bands < 1 or rows < 1:
        raise ValueError(f"bands and rows must be >= 1, got {bands} and {rows}")
    bits = 64 // rows
    if bits < 1:
        raise ValueError(
            f"rows={rows} leaves {bits} bits per MinHash; at most 64 rows fit "
            "in a UInt64 band signature."
        )
    # Each MinHash is < P_MOD < 2**31, so truncating to more than 31 bits is
    # not truncation at all and the packing is exact. Asserted rather than
    # assumed because it is what bounds the collision rate quoted in the
    # docstring.
    keep_bits = min(bits, 31)
    coef = _coefficients(bands * rows)

    # The packed value must fit UInt64 or it wraps SILENTLY — polars raises
    # nothing on UInt64 overflow (measured: at P=2**61-1 it returns
    # 2305843000623759375 where Python returns 2305843004918726663). This is a
    # static property of (bands, rows), so it is checked once here rather than
    # once per band, and it is checked at the widest possible value: every slot
    # full.
    top = sum(((1 << keep_bits) - 1) << (keep_bits * i) for i in range(rows))
    if top >= (1 << 64):
        raise ValueError(
            f"rows={rows} packs to {top.bit_length()} bits, which overflows "
            "the UInt64 band signature silently."
        )

    out: list[pl.Expr] = []
    for b in range(bands):
        packed: pl.Expr | None = None
        for r in range(rows):
            a, c = coef[b * rows + r]
            # `% P_MOD` on the gram BEFORE multiplying: a gram is a full
            # UInt32 and can exceed P_MOD, and reducing first is what keeps
            # the family the textbook universal one rather than something
            # close to it.
            h = (
                pl.col(col)
                .list.eval(
                    ((pl.element().cast(pl.UInt64) % P_MOD) * a + c) % P_MOD
                )
                .list.min()
            )
            # Low `keep_bits` bits, then shifted into position. Multiplication
            # rather than a shift operator, and the top slot is filled last so
            # the widest partial product is checked below.
            term = (h % (1 << keep_bits)) * (1 << (keep_bits * (rows - 1 - r)))
            packed = term if packed is None else packed + term
        assert packed is not None  # rows >= 1 is enforced above
        out.append(packed.cast(pl.UInt64).alias(f"{BAND_PREFIX}{b:02d}"))
    return out


def detection_rate(jaccard: float, bands: int, rows: int) -> float:
    """`1 - (1 - J**rows)**bands` — the probability some band agrees.

    Exposed so the diagnostics and the tests state the same number the design
    was chosen against, rather than each carrying its own copy of the formula.
    """
    return 1.0 - (1.0 - jaccard ** rows) ** bands
