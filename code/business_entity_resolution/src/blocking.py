"""Step 2 — candidate generation (blocking), stage A: exact-key union.

Strategy
--------
EDA said exact-match agreement tops out around 53% on the best single name key,
so no one key is enough. But those keys are nearly free — a hash join, not a
similarity search — and they disagree with each other, so their *union* recovers
substantially more than the best of them alone. This stage takes that union; the
fuzzy n-gram stage (B) then tops up only the entities still short of budget,
which is where the expensive compute gets spent.

Country is used as a hard partition on every key. It agrees on 100% of true
pairs in training, so it costs no recall and removes all cross-country
comparisons.

Bucket capping
--------------
A blocking key is only useful when its buckets are small. `name_acronym` has a
bucket of 17,606 records in source 1 alone; joining that against its source-2/3
counterpart would emit hundreds of millions of pairs for no benefit, since a
bucket that large carries almost no evidence. Key values whose target-side
bucket exceeds `--max-bucket` are dropped before the join.

Scoring
-------
Each key carries a weight reflecting how precise it proved in EDA. A candidate's
score is the sum of the weights of the keys it matched on, so a record found by
three independent keys outranks one found only by an acronym collision. When an
entity exceeds the candidate budget, the lowest-scoring candidates are dropped.

Usage:

    python -m src.blocking --split train
    python -m src.blocking --split train --budget 30 --max-bucket 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from . import config

# (column, weight, bucket cap). All three were set from measured per-key
# diagnostics on the full training set, not from intuition:
#
#   key            agrees  usable@100  unique
#   name_sorted    52.92%    47.64%    2.48%
#   name_core      49.40%    44.54%    0.07%   <- redundant with name_sorted
#   name_nospace   52.79%    47.37%    2.41%
#   addr_sorted    23.48%    23.19%   12.33%   <- weakest agreement, most unique
#   name_acronym   63.21%     7.91%    1.34%   <- huge agreement, mostly useless
#
# addr_sorted finds more pairs that nothing else finds than every name key
# combined, so it is weighted up. name_acronym agrees constantly but almost
# always duplicates another key, and its buckets are enormous; a loose cap on it
# floods entities with thousands of tied 0.25-score candidates, which then evict
# true matches at random when the budget cap applies. It gets a tight cap.
KEY_SPECS: tuple[tuple[str, float, int], ...] = (
    ("name_sorted", 1.00, 2000),
    ("name_core", 0.70, 2000),
    ("name_nospace", 0.80, 2000),
    ("addr_sorted", 0.85, 2000),
    ("name_acronym", 0.25, 50),
)


def candidates_parquet(split: str, work: Path) -> Path:
    out = work / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{split}_candidates.parquet"


def _key_frame(lf: pl.LazyFrame, key: str) -> pl.LazyFrame:
    """(entity_id, blocking_key) with country folded in and empties dropped."""
    return (
        lf.select(
            pl.col("entity_id"),
            pl.concat_str([pl.col("country_norm"), pl.lit("|"), pl.col(key)]).alias("bk"),
            pl.col(key).alias("_raw"),
        )
        .filter(pl.col("_raw").is_not_null() & (pl.col("_raw").str.len_chars() > 0))
        .drop("_raw")
    )


def _target_keys(split: str, work: Path, key: str, max_bucket: int) -> pl.DataFrame:
    """Source-2/3 side of one key, with over-large buckets removed."""
    frames = [
        _key_frame(pl.scan_parquet(config.normalized_parquet(split, s, work)), key)
        for s in (2, 3)
    ]
    combined = pl.concat(frames, how="vertical").collect()
    sizes = combined.group_by("bk").len()
    keep = sizes.filter(pl.col("len") <= max_bucket).select("bk")
    dropped = sizes.height - keep.height
    kept = combined.join(keep, on="bk", how="inner")
    print(f"    {key:<14} target rows {combined.height:>12,} -> {kept.height:>12,} "
          f"({dropped:,} buckets over cap dropped)")
    return kept.rename({"entity_id": "candidate_entity_id"})


def block_split(
    split: str, work: Path, budget: int, max_bucket: int | None, chunk: int
) -> None:
    s1_path = config.normalized_parquet(split, 1, work)
    if not s1_path.is_file():
        raise FileNotFoundError(
            f"{s1_path} missing — run `python -m src.normalize --split {split}` first."
        )

    print("  building target key tables (source 2 + 3):")
    targets = {
        key: _target_keys(split, work, key, max_bucket or cap)
        for key, _, cap in KEY_SPECS
    }

    s1 = pl.scan_parquet(s1_path)
    n_s1 = s1.select(pl.len()).collect().item()
    dst = candidates_parquet(split, work)

    import pyarrow.parquet as pq

    writer = None
    emitted = 0
    started = time.perf_counter()
    try:
        for offset in range(0, n_s1, chunk):
            block = s1.slice(offset, chunk)
            per_key = []
            for key, weight, _ in KEY_SPECS:
                left = _key_frame(block, key).collect()
                if left.is_empty():
                    continue
                hit = left.join(targets[key], on="bk", how="inner").select(
                    pl.col("entity_id").alias("source1_entity_id"),
                    pl.col("candidate_entity_id"),
                    pl.lit(weight, dtype=pl.Float32).alias("w"),
                )
                if hit.height:
                    per_key.append(hit)
            if not per_key:
                continue

            merged = (
                pl.concat(per_key, how="vertical")
                .group_by("source1_entity_id", "candidate_entity_id")
                .agg(
                    pl.col("w").sum().alias("block_score"),
                    pl.len().cast(pl.UInt8).alias("n_keys"),
                )
                # Keep the strongest `budget` candidates per entity.
                # Sort before ranking so ties break on evidence (more keys
                # first, then candidate id) rather than arbitrary row order.
                .sort(["block_score", "n_keys", "candidate_entity_id"],
                      descending=[True, True, False])
                .filter(
                    pl.int_range(pl.len()).over("source1_entity_id") < budget
                )
            )
            table = merged.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
            writer.write_table(table)
            emitted += merged.height
            done = min(offset + chunk, n_s1)
            print(f"  {done:>10,}/{n_s1:,} entities   candidates so far: {emitted:,}")
    finally:
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - started
    print(f"  wrote {emitted:,} candidate pairs -> {dst.name} ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A blocking: exact-key union.")
    parser.add_argument("--split", choices=config.SPLITS, required=True)
    parser.add_argument("--work-root", default=None)
    parser.add_argument(
        "--budget", type=int, default=30,
        help="Max candidates kept per Source-1 entity (default: %(default)s). "
             "EDA put the 99th percentile of true matches at 8.",
    )
    parser.add_argument(
        "--max-bucket", type=int, default=None,
        help="Override the per-key bucket caps in KEY_SPECS with one global value. "
             "Leave unset to use each key's own cap, which is what you want: the "
             "name keys need a loose cap and name_acronym needs a tight one.",
    )
    parser.add_argument("--chunk", type=int, default=250_000,
                        help="Source-1 entities processed per pass.")
    args = parser.parse_args()

    work = config.work_root(args.work_root)
    caps = (f"global {args.max_bucket}" if args.max_bucket
            else ", ".join(f"{k}={c}" for k, _, c in KEY_SPECS))
    print(f"blocking [{args.split}]  work={work}  budget={args.budget}")
    print(f"  bucket caps: {caps}")
    block_split(args.split, work, args.budget, args.max_bucket, args.chunk)


if __name__ == "__main__":
    main()
