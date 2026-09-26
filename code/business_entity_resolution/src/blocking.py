"""Stage A candidate generation: the union of several exact blocking keys.

Why a union
-----------
Exact-match agreement tops out around 53% on the best single name key, so no
one key is enough on its own. But these keys are nearly free — a hash join, not
a similarity search — and they *disagree* with each other, so their union
recovers substantially more than the best of them alone.

The counterintuitive measurement that shaped this file: agreement and
usefulness are close to uncorrelated. `addr_sorted` has the weakest agreement
of any key (23.48%) and the highest unique contribution (11.20%) — it finds
more true pairs that nothing else finds than every name key combined.
`name_core` agreed on 49.40% of true pairs and contributed 0.00% unique; it is
carried here only so the diagnostics can confirm or refute that under the
current normalization, and it should be dropped if the result holds.

Country is folded into every key as a hard partition. It agrees on 100% of true
training pairs, so it costs no recall and removes all cross-country
comparisons. It is used as an opaque label and never enumerated, because the
test split contains France and training does not.

Memory shape
------------
Five resident target key tables would be roughly 3-5GB at full scale, which a
16GB machine feels. So the loop is key-outer instead: build one target table,
join all of source 1 against it, spill the hits to parquet, free it, move to
the next key. Peak memory stays at about one key table — the same order the
single-key pipeline already occupied — and the normalized parquet makes the
repeated reads columnar and cheap.

The merge is then the peak-memory step, and a naive one is how a long run dies
at the very end: concatenating every partial and taking a global sort over
~100M rows wants 8-12GB. The candidate budget is per entity, so entities are
completely independent and no step needs a global view. The merge hashes the
entity ID into buckets and processes them one at a time. Bucket count does not
affect the output, only the peak.

Usage:

    python -m src.blocking --split train
    python -m src.blocking --split train --budget 20 --merge-buckets 16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from . import minhash, paths

# How many LSH band keys exist. FIXED at module level, deliberately, even
# though `--bands` selects how many actually block.
#
# `KEY_BITS` assigns one `key_mask` bit per key by position, and
# `features.FEATURE_COLUMNS` derives one `hit_<key>` column per key from
# `KEY_COLUMNS` — which `model.py` persists alongside the booster and refuses a
# mismatch against. If the band count came from the CLI, the feature schema and
# the meaning of every mask bit would change between two runs that differ only
# in `--bands`, and a saved model would silently be read against the wrong
# columns.
#
# So normalization always writes all `N_BANDS` signatures and `--bands N`
# selects the first N. `minhash._coefficients` is a seeded stream, so the first
# N bands of a 16-band run ARE the whole of an N-band run — the configurations
# are nested, which is what makes a bands curve readable off one normalization
# pass instead of one per point.
N_BANDS = minhash.DEFAULT_BANDS
BAND_KEYS: tuple[str, ...] = minhash.band_columns(N_BANDS)

# Weight and cap for every band key. One shared pair rather than sixteen,
# because the bands are exchangeable by construction — band 3 is not a more
# informative permutation than band 11, so a per-band weight would be fitting
# noise.
#
# The weight is the one genuinely open parameter in this change, and it is set
# by the job `block_score` has to do rather than by tuning. Its ONLY use is
# ranking an entity's candidates for the per-entity budget cut; the model gets
# `key_mask` and `n_keys` and learns the real weighting itself. Two constraints
# fix it to about this value:
#
#   * a true fuzzy-only match must outrank a same-building false positive. That
#     false positive arrives through `addr_sorted` at 0.85/log2(1+bucket), so
#     roughly 0.15 in a bucket of 50. A true match agreeing on 8 of 16 bands in
#     small buckets scores about 8*0.10/1.5 = 0.53, and wins comfortably.
#   * an exact name match should not be displaced by a merely similar one.
#     Total band score saturates near 1.6 when all 16 agree, against
#     `name_sorted`'s 1.0 — so only near-identical trigrams outrank identical
#     sorted tokens, which is the intended ordering.
#
# The cap is tighter than the name keys' 2000. A band bucket holding 1000
# records is 1000 records with near-identical 3-gram sets, which on this data
# means a generic name rather than a cluster of true matches.
BAND_WEIGHT = 0.10
BAND_CAP = 1000

# (column, weight, bucket cap).
#
# Weights and caps were derived from per-key diagnostics measured under the
# SUPERSEDED normalization, against key definitions that have since changed —
# `name_core` now strips legal-form tokens and expands abbreviations, so
# agreement, bucket sizes and unique contributions have all moved. Run
# `python -m src.block_eval` before trusting any number here.
#
#   key            agrees  usable@cap  unique
#   name_sorted    52.92%    52.92%     2.68%
#   name_core      49.40%    49.40%     0.00%   <- redundant with name_sorted
#   name_nospace   52.79%    52.79%     2.66%
#   addr_sorted    23.48%    23.45%    11.20%   <- weakest agreement, most unique
#   name_acronym   63.21%     5.76%     0.98%   <- huge agreement, mostly useless
#
# `name_acronym` agrees more often than any other key and almost always
# duplicates one, while having a 17,606-record bucket in source 1 alone. A
# loose cap on it floods entities with thousands of tied low-score candidates
# that then evict true matches when the budget binds. It gets a tight cap and a
# low weight rather than being removed, because its 0.98% unique contribution
# is small but real.
KEY_SPECS: tuple[tuple[str, float, int], ...] = (
    ("name_sorted", 1.00, 2000),
    ("name_core", 0.70, 2000),
    ("name_nospace", 0.80, 2000),
    ("addr_sorted", 0.85, 2000),
    ("name_acronym", 0.25, 50),
    # LSH bands. Appended AFTER the exact keys so that every existing
    # `key_mask` bit keeps the value it had before this change — an old
    # candidate or feature parquet stays readable, and the diagnostics in
    # `findings.md` that quote per-key contributions stay comparable.
    *((k, BAND_WEIGHT, BAND_CAP) for k in BAND_KEYS),
)

KEY_COLUMNS: tuple[str, ...] = tuple(k for k, _, _ in KEY_SPECS)

# The exact keys, as distinct from the fuzzy ones. `block_eval` reports the two
# groups separately, since "what the exact keys reach" is the 76.26% ceiling
# every number in `findings.md` is measured against and the point of this
# change is to leave it behind.
EXACT_KEYS: tuple[str, ...] = tuple(k for k in KEY_COLUMNS if k not in BAND_KEYS)

# Band signatures are UInt64, every other key is Utf8. `blocking_key` and
# `_usable` are the two places that care, and both dispatch on this rather than
# on a schema lookup, matching how `main._NAME_KEYS`/`_ADDR_KEYS` already split
# keys by prefix.
def is_band(key: str) -> bool:
    return key.startswith(minhash.BAND_PREFIX)

# Which key found a pair, as one bit per key in a `key_mask` column.
#
# The candidate file used to carry `n_keys` — how many keys agreed — but not
# which ones, and that turned out to be the difference between a question that
# can be measured and one that has to be argued. When multi-key blocking came
# back with precision at 0.6702, the obvious question was what share of the
# false positives arrived through `addr_sorted` alone, since those are exactly
# the ones an address scorer cannot reject. It was unanswerable from the
# artefact, and the failure mode had to be demonstrated on a hand-built example
# instead of measured on the real 23.6M-pair candidate set.
#
# It is also a strictly stronger model feature than the count: "found by
# name_sorted AND addr_sorted" is independent corroboration from two different
# fields, while "found by two keys" could equally be name_sorted and
# name_nospace, which are near-duplicates of each other.
#
# Bits are indexed against KEY_COLUMNS, not against the selected subset, so a
# mask means the same thing across runs that pass different `--keys`.
#
# The mask WAS a UInt16, which held the five exact keys with room to spare.
# Sixteen band keys take the total to 21 and overflow it — bit 16 and up would
# be truncated on the cast, so `hit_band_11` and everything after it would read
# as a constant zero in `features.py`: no crash, no warning, and eleven model
# features silently dead. Widened to UInt32 rather than sized exactly, so the
# next few keys do not repeat this.
KEY_MASK_DTYPE = pl.UInt32
_MASK_BITS = 32
assert len(KEY_SPECS) <= _MASK_BITS, (
    f"key_mask is a {KEY_MASK_DTYPE} holding {_MASK_BITS} bits and there are "
    f"{len(KEY_SPECS)} keys — widen the dtype in _emit_key and _merge before "
    "adding another."
)
assert len(set(KEY_COLUMNS)) == len(KEY_COLUMNS), (
    "duplicate blocking key name; two keys would share one key_mask bit"
)
KEY_BITS: dict[str, int] = {k: 1 << i for i, k in enumerate(KEY_COLUMNS)}

# `name_core` is measured redundant and is excluded from blocking by default.
# It contributed 0.00% unique true pairs under the superseded normalization AND
# 0.00% again under the current one — two independent measurements, so this is
# a property of the key and not of one rule set. It is fully subsumed by
# `name_sorted`, which is the same token set in a canonical order. Dropping it
# removed 91.8M raw pairs, about a fifth of the blocking work, at no cost in
# recall.
#
# It stays in KEY_SPECS rather than being deleted so `src.block_eval` can keep
# measuring it — the claim above is worth re-checking whenever the
# normalization rules change, and that is impossible once the key is gone.
# `--keys` puts it back.
DEFAULT_KEYS: tuple[str, ...] = tuple(k for k in KEY_COLUMNS if k != "name_core")


def default_keys(bands: int | None = None) -> tuple[str, ...]:
    """The default key set with only the first `bands` band keys in it.

    `bands=None` means all of them; `bands=0` turns the fuzzy stage off
    entirely, which is what reproduces the pre-LSH candidate set and is how a
    regression gets attributed to this change rather than bisected.
    """
    if bands is None:
        return DEFAULT_KEYS
    if not 0 <= bands <= N_BANDS:
        raise ValueError(
            f"--bands must be between 0 and {N_BANDS} (normalization writes "
            f"{N_BANDS} signatures); got {bands}"
        )
    keep = set(BAND_KEYS[:bands])
    return tuple(k for k in DEFAULT_KEYS if not is_band(k) or k in keep)


# --------------------------------------------------------------------------
# key construction
# --------------------------------------------------------------------------

def blocking_key(key: str) -> pl.Expr:
    """`<country>|<key value>`, the value actually joined on.

    `concat_str` propagates nulls, so a null key value yields a null `bk` and is
    dropped by `_usable`. That is the intended behaviour, not an accident worth
    working around — and it is load-bearing for the band keys, where a row
    whose name normalized to nothing has an empty trigram list, hence a null
    MinHash, hence a null signature. Without that propagation every blank name
    in the corpus would share one signature and form a single enormous
    spurious bucket.
    """
    value = pl.col(key)
    if is_band(key):
        # Band signatures are UInt64 and `concat_str` will not take an integer
        # column. The cast is transient — it exists only inside this join and
        # is never stored, which is why the signatures live on disk as 8-byte
        # integers rather than 20-character strings across 12.5M rows.
        value = value.cast(pl.Utf8)
    return pl.concat_str(
        [pl.col("country_norm").fill_null(""), pl.lit("|"), value]
    ).alias("bk")


def _usable(key: str, min_len: int = 0) -> pl.Expr:
    """Rows whose key value is present, non-empty and long enough to mean much.

    A row whose name normalizes to nothing would otherwise land in a single
    `<country>|` bucket shared with every other such row — a spurious block,
    not a match.

    `min_len` exists for `name_acronym`, where key length and bucket size are
    directly related: a two-letter acronym has a few hundred possible values
    across millions of records and its buckets are enormous, while a six-letter
    one is nearly unique. A flat bucket cap handles that badly — it throws away
    the whole key's contribution for 54.77% of the true pairs it agrees on,
    because the short values crowd out the long ones. Dropping the short values
    outright and then capping loosely targets the mechanism instead of the
    symptom.
    """
    if is_band(key):
        # A UInt64 signature has no length and `str.len_chars()` on it raises.
        # Nullness is the whole test here and it is sufficient: an empty
        # trigram list is the only way to reach a null signature, and there is
        # no analogue of a "too short to mean much" band — every signature
        # carries the same number of bits. `min_len` is meaningless for a band
        # and passing one is a caller bug rather than something to ignore.
        if min_len > 1:
            raise ValueError(f"min_len does not apply to the band key {key!r}")
        return pl.col(key).is_not_null()
    ok = pl.col(key).is_not_null() & (pl.col(key).str.len_chars() > 0)
    if min_len > 1:
        ok = ok & (pl.col(key).str.len_chars() >= min_len)
    return ok


# --------------------------------------------------------------------------
# one key at a time
# --------------------------------------------------------------------------

def _target_table(
    split: str, work: Path, key: str, cap: int, min_len: int = 0
) -> pl.DataFrame:
    """Source-2/3 side of one key, with over-large buckets removed.

    `bucket_size` is carried through rather than discarded: it becomes the
    inverse-frequency term in the score, so a match in a bucket of 2 outranks
    one in a bucket of 2000.
    """
    sources = pl.concat(
        [
            pl.scan_parquet(paths.normalized_parquet(split, s, work))
            .select("entity_id", "country_norm", key)
            .filter(_usable(key, min_len))
            .select(
                pl.col("entity_id").alias("candidate_entity_id"), blocking_key(key)
            )
            for s in (2, 3)
        ],
        how="vertical",
    )

    # Bucket census first, on its own: it collects to one row per distinct key
    # value, which is small, and gives the "dropped" figure without holding the
    # full table and a join output at the same time.
    sizes = sources.group_by("bk").len().collect()
    n_buckets = sizes.height
    n_dropped = int(sizes.filter(pl.col("len") > cap).height)
    del sizes

    kept = (
        sources.with_columns(pl.len().over("bk").cast(pl.UInt32).alias("bucket_size"))
        .filter(pl.col("bucket_size") <= cap)
        .collect()
    )
    note = f" (min len {min_len})" if min_len > 1 else ""
    print(
        f"    {key:<14} targets {kept.height:>12,} rows{note}   "
        f"{n_dropped:,}/{n_buckets:,} buckets over cap {cap:,} dropped"
    )
    return kept


def _emit_key(
    split: str,
    work: Path,
    key: str,
    weight: float,
    cap: int,
    chunk: int,
    min_len: int = 0,
) -> Path | None:
    """Join source 1 against one key's target table, spill the hits to parquet.

    Returns the partial's path, or None if the key produced no candidates at
    all — which is normal for a key whose buckets all exceed the cap on a small
    input, and must not crash the merge.
    """
    import pyarrow.parquet as pq

    targets = _target_table(split, work, key, cap, min_len)

    # Source 1, narrow and materialized once. Slicing a DataFrame is a
    # zero-copy view, so the chunk loop below is a real cursor. Repeatedly
    # calling `.slice(offset, chunk).collect()` on a LazyFrame is not — each
    # collect re-reads from the start and discards everything before the
    # offset, which at 250k chunks over a 10M-row source is 42 full passes.
    s1 = (
        pl.scan_parquet(paths.normalized_parquet(split, 1, work))
        .select("entity_id", "country_norm", key)
        .filter(_usable(key, min_len))
        .select(pl.col("entity_id").alias("source1_entity_id"), blocking_key(key))
        .collect()
    )

    dst = paths.partial_parquet(split, key, work)
    writer = None
    emitted = 0
    try:
        for offset in range(0, max(s1.height, 1), chunk):
            block = s1.slice(offset, chunk)
            if block.is_empty():
                continue
            hits = block.join(targets, on="bk", how="inner").select(
                pl.col("source1_entity_id"),
                pl.col("candidate_entity_id"),
                # Inverse-frequency weighting. Sharing a name with one other
                # record is strong evidence; sharing it with 2000 is nearly
                # none. Without this the budget cap cannot tell tied candidates
                # apart and evicts true matches effectively alphabetically —
                # worth 2.4 points of macro recall at an IDENTICAL candidate
                # count, which is pure re-ranking.
                #
                # bucket_size is at least 1, so the log argument is at least 2
                # and the divisor is at least 1. A singleton bucket scores the
                # full weight; nothing can divide by zero.
                (
                    pl.lit(weight, dtype=pl.Float32)
                    / (pl.col("bucket_size").cast(pl.Float32) + 1.0).log(2.0)
                ).alias("w"),
                # Provenance. Constant within this key's partial, and OR-ed
                # together across keys by the merge.
                pl.lit(KEY_BITS[key], dtype=KEY_MASK_DTYPE).alias("key_bit"),
            )
            if hits.is_empty():
                continue
            table = hits.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
            writer.write_table(table)
            emitted += hits.height
    finally:
        if writer is not None:
            writer.close()

    del targets, s1
    if writer is None:
        print(f"    {key:<14} no candidates emitted")
        return None
    print(f"    {key:<14} {emitted:>12,} raw pairs -> {dst.name}")
    return dst


# --------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------

def _merge(
    partials: list[Path], dst: Path, budget: int, n_buckets: int
) -> tuple[int, int]:
    """Combine every key's hits, score, and keep the best `budget` per entity.

    Entities are independent — the budget is per entity and no candidate's fate
    depends on another entity's candidates — so the work partitions cleanly by
    a hash of the source-1 ID. Every bucket count produces the same output set;
    only the peak memory changes.
    """
    import pyarrow.parquet as pq

    writer = None
    kept = 0
    raw = 0
    for b in range(n_buckets):
        scans = [
            pl.scan_parquet(p).filter(
                (pl.col("source1_entity_id").hash() % n_buckets) == b
            )
            for p in partials
        ]
        grouped = (
            pl.concat(scans, how="vertical")
            .group_by("source1_entity_id", "candidate_entity_id")
            .agg(
                pl.col("w").sum().alias("block_score"),
                # `n_unique` rather than `pl.len()`. The two are equal whenever
                # each key emits a pair at most once, which holds as long as
                # entity IDs are unique within sources 2 and 3 — but that is an
                # assumption about the data, not about this code, and a
                # duplicated target ID would silently inflate `n_keys` while
                # leaving `key_mask` correct. Deriving both from the same bits
                # makes them consistent by construction.
                pl.col("key_bit").n_unique().cast(pl.UInt8).alias("n_keys"),
                # Bitwise OR, expressed as a sum over the DISTINCT bits
                # present. Each bit is a distinct power of two, so summing them
                # once each is exactly an OR; `unique` is what makes that hold
                # under a repeated emission rather than only in the happy case.
                pl.col("key_bit").unique().sum().cast(KEY_MASK_DTYPE)
                .alias("key_mask"),
            )
            .collect()
        )
        raw += grouped.height
        if grouped.is_empty():
            continue
        # Eager, deliberately. The sort must happen before the positional
        # filter, and a lazy optimizer is free to reorder them. Ties break on
        # evidence (more keys agreeing) and then on candidate id, so the
        # survivors are chosen deterministically rather than by whatever row
        # order the join happened to produce.
        capped = grouped.sort(
            ["block_score", "n_keys", "candidate_entity_id"],
            descending=[True, True, False],
        ).filter(pl.int_range(pl.len()).over("source1_entity_id") < budget)
        table = capped.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
        writer.write_table(table)
        kept += capped.height
        print(f"    merge bucket {b + 1}/{n_buckets}   kept {kept:,}", end="\r")
    if writer is not None:
        writer.close()
    print()
    return raw, kept


def _emit_ngram(
    split: str,
    work: Path,
    n: int = 5,
    max_bucket: int = 200,
    chunk: int = 250_000,
) -> Path | None:
    """Character n-gram blocking partial table."""
    import pyarrow.parquet as pq

    tgt_scan = pl.concat([
        pl.scan_parquet(paths.normalized_parquet(split, s, work)).select(
            pl.col("entity_id").alias("candidate_entity_id"),
            pl.col("country_norm").fill_null(""),
            pl.col("name_nospace").fill_null(""),
        )
        for s in (2, 3)
    ])

    def add_ngrams(col: pl.Expr) -> pl.Expr:
        return (
            col.str.to_lowercase()
            .str.replace_all(r"[^a-z0-9]+", "")
            .map_elements(
                lambda x: [x[i:i + n] for i in range(max(0, len(x) - n + 1))],
                return_dtype=pl.List(pl.Utf8),
            )
            .list.unique()
        )

    targets = (
        tgt_scan
        .with_columns(add_ngrams(pl.col("name_nospace")).alias("gram"))
        .explode("gram")
        .filter(pl.col("gram") != "")
        .with_columns(
            pl.concat_str([pl.col("country_norm"), pl.lit("|"), pl.col("gram")]).alias("bk")
        )
        .select("candidate_entity_id", "bk")
        .collect()
    )

    sizes = targets.group_by("bk").len().rename({"len": "bucket_size"})
    keep = sizes.filter(pl.col("bucket_size") <= max_bucket)
    targets = targets.join(keep, on="bk", how="inner")

    s1 = (
        pl.scan_parquet(paths.normalized_parquet(split, 1, work))
        .select(
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("country_norm").fill_null(""),
            pl.col("name_nospace").fill_null(""),
        )
        .with_columns(add_ngrams(pl.col("name_nospace")).alias("gram"))
        .explode("gram")
        .filter(pl.col("gram") != "")
        .with_columns(
            pl.concat_str([pl.col("country_norm"), pl.lit("|"), pl.col("gram")]).alias("bk")
        )
        .select("source1_entity_id", "bk")
        .collect()
    )

    dst = paths.partial_parquet(split, f"ngram_n{n}", work)
    writer = None
    emitted = 0
    try:
        for offset in range(0, max(s1.height, 1), chunk):
            block = s1.slice(offset, chunk)
            if block.is_empty():
                continue
            hits = block.join(targets, on="bk", how="inner").select(
                pl.col("source1_entity_id"),
                pl.col("candidate_entity_id"),
                (pl.lit(0.75, dtype=pl.Float32) / (pl.col("bucket_size").cast(pl.Float32) + 1.0).log(2.0)).alias("w"),
            )
            if hits.is_empty():
                continue
            table = hits.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
            writer.write_table(table)
            emitted += hits.height
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        print(f"    ngram_n{n:<8} no candidates emitted")
        return None
    print(f"    ngram_n{n:<8} {emitted:>12,} raw pairs -> {dst.name}")
    return dst


def _emit_prefix(
    split: str,
    work: Path,
    prefix_len: int = 3,
    max_bucket: int = 500,
    chunk: int = 250_000,
) -> Path | None:
    """Prefix blocking partial table."""
    import pyarrow.parquet as pq

    tgt_scan = pl.concat([
        pl.scan_parquet(paths.normalized_parquet(split, s, work)).select(
            pl.col("entity_id").alias("candidate_entity_id"),
            pl.col("country_norm").fill_null(""),
            pl.col("name_nospace").fill_null(""),
        )
        for s in (2, 3)
    ])

    targets = (
        tgt_scan
        .with_columns(
            pl.col("name_nospace")
            .str.to_lowercase()
            .str.replace_all(r"[^a-z0-9]+", "")
            .str.slice(0, prefix_len)
            .alias("pfx")
        )
        .filter(pl.col("pfx") != "")
        .with_columns(
            pl.concat_str([pl.col("country_norm"), pl.lit("|"), pl.col("pfx")]).alias("bk")
        )
        .select("candidate_entity_id", "bk")
        .collect()
    )

    sizes = targets.group_by("bk").len().rename({"len": "bucket_size"})
    keep = sizes.filter(pl.col("bucket_size") <= max_bucket)
    targets = targets.join(keep, on="bk", how="inner")

    s1 = (
        pl.scan_parquet(paths.normalized_parquet(split, 1, work))
        .select(
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("country_norm").fill_null(""),
            pl.col("name_nospace").fill_null(""),
        )
        .with_columns(
            pl.col("name_nospace")
            .str.to_lowercase()
            .str.replace_all(r"[^a-z0-9]+", "")
            .str.slice(0, prefix_len)
            .alias("pfx")
        )
        .filter(pl.col("pfx") != "")
        .with_columns(
            pl.concat_str([pl.col("country_norm"), pl.lit("|"), pl.col("pfx")]).alias("bk")
        )
        .select("source1_entity_id", "bk")
        .collect()
    )

    dst = paths.partial_parquet(split, f"prefix_len{prefix_len}", work)
    writer = None
    emitted = 0
    try:
        for offset in range(0, max(s1.height, 1), chunk):
            block = s1.slice(offset, chunk)
            if block.is_empty():
                continue
            hits = block.join(targets, on="bk", how="inner").select(
                pl.col("source1_entity_id"),
                pl.col("candidate_entity_id"),
                (pl.lit(0.70, dtype=pl.Float32) / (pl.col("bucket_size").cast(pl.Float32) + 1.0).log(2.0)).alias("w"),
            )
            if hits.is_empty():
                continue
            table = hits.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
            writer.write_table(table)
            emitted += hits.height
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        print(f"    prefix_len{prefix_len:<6} no candidates emitted")
        return None
    print(f"    prefix_len{prefix_len:<6} {emitted:>12,} raw pairs -> {dst.name}")
    return dst


def _emit_tfidf(
    split: str,
    work: Path,
    top_k: int = 20,
    batch_size: int = 500,
) -> Path | None:
    """Batched sparse TF-IDF partial table."""
    import numpy as np
    import pyarrow.parquet as pq
    from sklearn.feature_extraction.text import TfidfVectorizer

    s1_df = pl.scan_parquet(paths.normalized_parquet(split, 1, work)).select(
        pl.col("entity_id"),
        pl.col("name_sorted").fill_null(""),
    ).collect()

    tgt_df = pl.concat([
        pl.scan_parquet(paths.normalized_parquet(split, s, work)).select(
            pl.col("entity_id"),
            pl.col("name_sorted").fill_null(""),
        )
        for s in (2, 3)
    ]).collect()

    s1_names = s1_df["name_sorted"].to_list()
    tgt_names = tgt_df["name_sorted"].to_list()
    s1_ids = s1_df["entity_id"].to_list()
    tgt_ids = tgt_df["entity_id"].to_list()

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True, min_df=2)
    all_vecs = vectorizer.fit_transform(s1_names + tgt_names)
    s1_mat = all_vecs[:len(s1_names)]
    tgt_mat = all_vecs[len(s1_names):]

    dst = paths.partial_parquet(split, f"tfidf_top{top_k}", work)
    writer = None
    emitted = 0

    num_s1 = len(s1_names)
    for b in range(0, num_s1, batch_size):
        b_end = min(b + batch_size, num_s1)
        s1_sub = s1_mat[b:b_end]
        sims = (s1_sub * tgt_mat.T).toarray()

        rows = []
        for local_idx, global_idx in enumerate(range(b, b_end)):
            row = sims[local_idx]
            if len(row) > top_k:
                top_indices = np.argpartition(row, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(-row[top_indices])]
            else:
                top_indices = np.argsort(-row)

            for idx in top_indices:
                score = float(row[idx])
                if score > 0.0:
                    rows.append((s1_ids[global_idx], tgt_ids[idx], score * 0.85))

        if rows:
            table = pl.DataFrame(
                rows,
                schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8, "w": pl.Float32},
                orient="row",
            ).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
            writer.write_table(table)
            emitted += len(rows)

    if writer is not None:
        writer.close()
    if writer is None:
        print(f"    tfidf_top{top_k:<6} no candidates emitted")
        return None
    print(f"    tfidf_top{top_k:<6} {emitted:>12,} raw pairs -> {dst.name}")
    return dst


def block_split(
    split: str,
    work: Path,
    budget: int,
    max_bucket: int | None,
    chunk: int,
    merge_buckets: int,
    keys: tuple[str, ...] | None = None,
    acronym_cap: int | None = None,
    acronym_min_len: int = 0,
    bands: int | None = None,
    char_ngram: bool = False,
    ngram_n: int = 5,
    ngram_max_bucket: int = 200,
    prefix: bool = False,
    prefix_len: int = 3,
    prefix_max_bucket: int = 500,
    tfidf: bool = False,
    tfidf_top_k: int = 20,
) -> Path:
    """Run stage A end to end and return the candidate parquet's path.

    `bands` trims the default key set to the first N LSH bands. An explicit
    `keys` wins over it, because `--keys` is the attribution tool and it should
    mean exactly what it says.
    """
    if keys is not None and bands is not None:
        raise ValueError(
            "pass --keys or --bands, not both: --keys already names the exact "
            "band subset to block on."
        )
    selected = keys if keys is not None else default_keys(bands)
    specs = [s for s in KEY_SPECS if s[0] in selected]
    if not specs and not (char_ngram or prefix or tfidf):
        raise ValueError(f"no keys selected; available: {list(KEY_COLUMNS)}")
    unknown = set(selected) - set(KEY_COLUMNS)
    if unknown:
        raise ValueError(
            f"unknown keys {sorted(unknown)}; available: {list(KEY_COLUMNS)}"
        )

    missing = [
        paths.normalized_parquet(split, s, work)
        for s in (1, 2, 3)
        if not paths.normalized_parquet(split, s, work).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{missing[0]} missing — run `python -m src.main --split {split} "
            "--normalize-only` first."
        )

    # Check every selected key is actually IN the normalized parquet before
    # doing any work. A cache written before the band columns existed is the
    # expected way to get here, and without this the run dies inside the first
    # `_emit_key` join with a bare ColumnNotFound after building a target
    # table — several minutes in, naming a column rather than the cause.
    #
    # `main._rule_mtime` should already have invalidated such a cache, since
    # both `main.py` and `blocking.py` changed. This is the belt to that
    # braces: the journal records a normalization cache that was reused
    # silently across a rule change, and the failure there was invisible rather
    # than loud.
    have = set(pl.scan_parquet(paths.normalized_parquet(split, 1, work))
               .collect_schema().names())
    absent = [k for k in selected if k not in have]
    if absent:
        raise RuntimeError(
            f"the normalized parquet has no {absent[0]!r} column (and "
            f"{len(absent) - 1} other selected key(s)). It predates these "
            f"keys — re-run `python -m src.main --split {split} "
            "--normalize-only --force-normalize`."
        )

    started = time.perf_counter()
    partials: list[Path] = []
    for key, weight, cap in specs:
        if key == "name_acronym":
            cap = acronym_cap or cap
            min_len = acronym_min_len
        else:
            min_len = 0
        partials.append(
            _emit_key(split, work, key, weight, max_bucket or cap, chunk, min_len)
        )

    if char_ngram:
        partials.append(_emit_ngram(split, work, n=ngram_n, max_bucket=ngram_max_bucket, chunk=chunk))
    if prefix:
        partials.append(_emit_prefix(split, work, prefix_len=prefix_len, max_bucket=prefix_max_bucket, chunk=chunk))
    if tfidf:
        partials.append(_emit_tfidf(split, work, top_k=tfidf_top_k))

    partials = [p for p in partials if p is not None]
    if not partials:
        raise RuntimeError(
            "every key produced zero candidates — check the normalized parquet."
        )

    dst = paths.candidates_parquet(split, work)
    print("  merging:")
    raw, kept = _merge(partials, dst, budget, merge_buckets)
    elapsed = time.perf_counter() - started
    print(
        f"  {raw:,} distinct pairs -> {kept:,} after budget {budget} "
        f"-> {dst.name} ({elapsed:.1f}s)"
    )
    return dst



def main() -> None:
    p = argparse.ArgumentParser(description="Stage A blocking: exact-key union.")
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--work-root", default=None)
    p.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Max candidates kept per Source-1 entity (default: %(default)s). "
        "The maximum true match count on the training set is 11 and the 99th "
        "percentile is 8, so this is roughly 2x the worst case.",
    )
    p.add_argument(
        "--max-bucket",
        type=int,
        default=None,
        help="Override every key's own cap with one global value. Leave unset, "
        "which is what you want: the name keys need a loose cap and "
        "name_acronym needs a tight one.",
    )
    p.add_argument(
        "--keys",
        default=None,
        help="Comma-separated subset of keys, for attributing a change to one "
        f"key. Default: {','.join(DEFAULT_KEYS)}. Also available but off by "
        "default because it measured 0.00%% unique contribution twice: "
        f"{','.join(k for k in KEY_COLUMNS if k not in DEFAULT_KEYS)}",
    )
    p.add_argument(
        "--bands", type=int, default=None,
        help=f"How many of the {N_BANDS} LSH band keys to block on (default: "
        "all of them). 0 turns fuzzy retrieval off and reproduces the "
        "exact-key candidate set, which is how a regression gets attributed "
        "to this stage rather than bisected. The bands are NESTED — band k is "
        "the same permutation group at every setting — so a curve over this "
        "reads off one normalization pass. Mutually exclusive with --keys.",
    )
    p.add_argument(
        "--acronym-cap", type=int, default=None,
        help="Bucket cap for name_acronym alone (default: its KEY_SPECS value, "
        "50). At that cap the key agrees on 60.58%% of true pairs but only "
        "5.81%% survive, so the cap is discarding 54.77%% of what it finds.",
    )
    p.add_argument(
        "--acronym-min-len", type=int, default=0,
        help="Drop acronyms shorter than this before blocking on them. Bucket "
        "size and acronym length are directly related, so this removes the "
        "huge buckets at the source and lets --acronym-cap be loosened "
        "without flooding the candidate set. Try `--acronym-min-len 4 "
        "--acronym-cap 500` as a single designed experiment.",
    )
    p.add_argument("--chunk", type=int, default=250_000)
    p.add_argument(
        "--merge-buckets",
        type=int,
        default=16,
        help="Hash partitions for the merge. Affects peak memory only, not the "
        "output. Raise it if the merge is the step that runs out of memory.",
    )
    a = p.parse_args()

    work = paths.work_root(a.work_root)
    if a.keys and a.bands is not None:
        p.error("pass --keys or --bands, not both.")
    keys = tuple(k.strip() for k in a.keys.split(",")) if a.keys else None
    selected = keys if keys is not None else default_keys(a.bands)
    # Twenty-one key names on one line is noise. The bands are exchangeable and
    # share one weight and one cap, so they are summarized as a count and the
    # exact keys are listed in full — which is what a reader is actually
    # checking when they scan this header.
    exact_sel = [k for k in selected if not is_band(k)]
    n_band_sel = sum(1 for k in selected if is_band(k))
    caps = (
        f"global {a.max_bucket}"
        if a.max_bucket
        else ", ".join(
            f"{k}={a.acronym_cap or c if k == 'name_acronym' else c}"
            for k, _, c in KEY_SPECS
            if k in exact_sel
        ) + (f", {n_band_sel} bands={BAND_CAP}" if n_band_sel else "")
    )
    print(f"blocking [{a.split}]  work={work}  budget={a.budget}")
    print(f"  exact keys: {', '.join(exact_sel) or '(none)'}")
    print(f"  fuzzy: {n_band_sel}/{N_BANDS} LSH bands"
          + (f" (rows={minhash.DEFAULT_ROWS}, weight {BAND_WEIGHT}, detects "
             f"J=0.5 at {minhash.detection_rate(0.5, n_band_sel, minhash.DEFAULT_ROWS):.0%})"
             if n_band_sel else " — fuzzy retrieval OFF"))
    print(f"  bucket caps: {caps}"
          + (f"   acronym min len: {a.acronym_min_len}"
             if a.acronym_min_len > 1 else ""))
    block_split(
        a.split, work, a.budget, a.max_bucket, a.chunk, a.merge_buckets, keys,
        a.acronym_cap, a.acronym_min_len, a.bands if keys is None else None,
    )


# --------------------------------------------------------------------------
# standalone / experimental blocking functions
# --------------------------------------------------------------------------

def character_ngram_blocking(
    source1: pl.DataFrame,
    targets: pl.DataFrame,
    n: int = 3,
    max_bucket: int = 100,
) -> pl.DataFrame:
    """
    Generate candidate pairs using character n-gram blocking.

    Expected columns:
        source1:
            entity_id
            business_name
            country

        targets:
            candidate_entity_id
            business_name
            country

    Returns:
        source1_entity_id
        candidate_entity_id
    """

    def add_ngrams(col: pl.Expr) -> pl.Expr:
        return (
            col.fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[^a-z0-9]+", "")
            .map_elements(
                lambda x: [x[i:i + n] for i in range(max(0, len(x) - n + 1))],
                return_dtype=pl.List(pl.Utf8),
            )
            .list.unique()
        )

    s1 = source1.with_columns(
        add_ngrams(pl.col("business_name")).alias("name_ngrams")
    )

    tgt = targets.with_columns(
        add_ngrams(pl.col("business_name")).alias("name_ngrams")
    )

    s1 = s1.explode("name_ngrams")
    tgt = tgt.explode("name_ngrams")

    # Remove empty n-grams.
    s1 = s1.filter(pl.col("name_ngrams") != "")
    tgt = tgt.filter(pl.col("name_ngrams") != "")

    # Remove extremely common n-grams.
    counts = tgt.group_by("name_ngrams").len()
    valid = counts.filter(
        pl.col("len") <= max_bucket
    ).select("name_ngrams")

    s1 = s1.join(valid, on="name_ngrams", how="inner")
    tgt = tgt.join(valid, on="name_ngrams", how="inner")

    candidates = (
        s1.join(
            tgt,
            on="name_ngrams",
            how="inner",
            suffix="_target",
        )
        .select(
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("candidate_entity_id"),
        )
        .unique()
    )

    return candidates


def prefix_blocking(
    source1: pl.DataFrame,
    targets: pl.DataFrame,
    prefix_length: int = 4,
    max_bucket: int = 100,
) -> pl.DataFrame:
    """
    Generate candidate pairs using a normalized business-name prefix.

    Example:
        "ABC Technologies Pvt Ltd"
                    ↓
        prefix = "abct"

    Returns:
        source1_entity_id
        candidate_entity_id
    """

    def prefix_key(col: pl.Expr) -> pl.Expr:
        return (
            col.fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[^a-z0-9]+", "")
            .str.slice(0, prefix_length)
        )

    s1 = source1.with_columns(
        prefix_key(pl.col("business_name")).alias("prefix_key")
    )

    tgt = targets.with_columns(
        prefix_key(pl.col("business_name")).alias("prefix_key")
    )

    # Ignore empty prefixes.
    s1 = s1.filter(pl.col("prefix_key") != "")
    tgt = tgt.filter(pl.col("prefix_key") != "")

    # Remove prefixes shared by too many target records.
    counts = tgt.group_by("prefix_key").len()

    valid = counts.filter(
        pl.col("len") <= max_bucket
    ).select("prefix_key")

    s1 = s1.join(valid, on="prefix_key", how="inner")
    tgt = tgt.join(valid, on="prefix_key", how="inner")

    candidates = (
        s1.join(
            tgt,
            on="prefix_key",
            how="inner",
            suffix="_target",
        )
        .select(
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("candidate_entity_id"),
        )
        .unique()
    )

    return candidates


def tfidf_retrieval_blocking(
    source1: pl.DataFrame,
    targets: pl.DataFrame,
    top_k: int = 20,
) -> pl.DataFrame:
    """
    Generate candidates using TF-IDF cosine similarity.

    Each Source-1 business retrieves its top_k most similar
    target business names.

    Returns:
        source1_entity_id
        candidate_entity_id
        tfidf_similarity
    """

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    s1_names = (
        source1["business_name"]
        .fill_null("")
        .cast(pl.Utf8)
        .to_list()
    )

    target_names = (
        targets["business_name"]
        .fill_null("")
        .cast(pl.Utf8)
        .to_list()
    )

    # Character TF-IDF is useful for spelling variations.
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        lowercase=True,
        min_df=1,
    )

    all_names = s1_names + target_names

    matrix = vectorizer.fit_transform(all_names)

    s1_matrix = matrix[:len(s1_names)]
    target_matrix = matrix[len(s1_names):]

    similarities = cosine_similarity(
        s1_matrix,
        target_matrix,
    )

    source1_ids = source1["entity_id"].to_list()
    target_ids = targets["candidate_entity_id"].to_list()

    candidate_rows = []

    for i, source1_id in enumerate(source1_ids):

        scores = similarities[i]

        # Get only the top-k target records.
        if len(scores) > top_k:
            top_indices = scores.argsort()[-top_k:][::-1]
        else:
            top_indices = scores.argsort()[::-1]

        for idx in top_indices:

            score = float(scores[idx])

            candidate_rows.append(
                (
                    source1_id,
                    target_ids[idx],
                    score,
                )
            )

    return pl.DataFrame(
        candidate_rows,
        schema=[
            "source1_entity_id",
            "candidate_entity_id",
            "tfidf_similarity",
        ],
        orient="row",
    )


if __name__ == "__main__":
    main()

