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

from . import paths

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
)

KEY_COLUMNS: tuple[str, ...] = tuple(k for k, _, _ in KEY_SPECS)

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


# --------------------------------------------------------------------------
# key construction
# --------------------------------------------------------------------------

def blocking_key(key: str) -> pl.Expr:
    """`<country>|<key value>`, the value actually joined on.

    `concat_str` propagates nulls, so a null key value yields a null `bk` and is
    dropped by `_usable`. That is the intended behaviour, not an accident worth
    working around.
    """
    return pl.concat_str(
        [pl.col("country_norm").fill_null(""), pl.lit("|"), pl.col(key)]
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
                pl.len().cast(pl.UInt8).alias("n_keys"),
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
) -> Path:
    """Run stage A end to end and return the candidate parquet's path."""
    selected = keys if keys is not None else DEFAULT_KEYS
    specs = [s for s in KEY_SPECS if s[0] in selected]
    if not specs:
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
        "default because it measured 0.00% unique contribution twice: "
        f"{','.join(k for k in KEY_COLUMNS if k not in DEFAULT_KEYS)}",
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
    keys = tuple(k.strip() for k in a.keys.split(",")) if a.keys else None
    selected = keys if keys is not None else DEFAULT_KEYS
    caps = (
        f"global {a.max_bucket}"
        if a.max_bucket
        else ", ".join(
            f"{k}={a.acronym_cap or c if k == 'name_acronym' else c}"
            for k, _, c in KEY_SPECS
            if k in selected
        )
    )
    print(f"blocking [{a.split}]  work={work}  budget={a.budget}")
    print(f"  keys: {', '.join(selected)}")
    print(f"  bucket caps: {caps}"
          + (f"   acronym min len: {a.acronym_min_len}"
             if a.acronym_min_len > 1 else ""))
    block_split(
        a.split, work, a.budget, a.max_bucket, a.chunk, a.merge_buckets, keys,
        a.acronym_cap, a.acronym_min_len,
    )


if __name__ == "__main__":
    main()
