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

# (column, weight). Weights come from the matched-pair agreement EDA reports:
# name_sorted was the strongest single key, name_acronym the weakest and by far
# the most collision-prone, so it only ever acts as a tie-breaker.
KEY_SPECS: tuple[tuple[str, float], ...] = (
    ("name_sorted", 1.00),
    ("name_core", 0.90),
    ("name_nospace", 0.80),
    ("addr_sorted", 0.60),
    ("name_acronym", 0.25),
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
    split: str, work: Path, budget: int, max_bucket: int, chunk: int
) -> None:
    s1_path = config.normalized_parquet(split, 1, work)
    if not s1_path.is_file():
        raise FileNotFoundError(
            f"{s1_path} missing — run `python -m src.normalize --split {split}` first."
        )

    print("  building target key tables (source 2 + 3):")
    targets = {
        key: _target_keys(split, work, key, max_bucket) for key, _ in KEY_SPECS
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
            for key, weight in KEY_SPECS:
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
                .filter(
                    pl.col("block_score")
                    .rank("ordinal", descending=True)
                    .over("source1_entity_id")
                    <= budget
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
        "--max-bucket", type=int, default=100,
        help="Drop key values with more than this many source-2/3 records.",
    )
    parser.add_argument("--chunk", type=int, default=250_000,
                        help="Source-1 entities processed per pass.")
    args = parser.parse_args()

    work = config.work_root(args.work_root)
    print(f"blocking [{args.split}]  work={work}  budget={args.budget}  "
          f"max_bucket={args.max_bucket}")
    block_split(args.split, work, args.budget, args.max_bucket, args.chunk)


if __name__ == "__main__":
    main()
