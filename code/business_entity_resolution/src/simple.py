"""End-to-end entity resolution pipeline.

One file covering the whole path: read the data, block, score, predict, write
both submission files, and self-score against the training ground truth.
Deliberately lean on purpose — a working end-to-end pipeline before refining any
stage.

    python -m src.simple --split train --sample 200000   # ~1 min smoke test
    python -m src.simple --split train                   # full, self-scored
    python -m src.simple --split test                    # writes submission

Stages:

* normalize   lowercase, strip punctuation, collapse whitespace
* block       exact match on (country, sorted name tokens)
* score       Jaccard overlap of address tokens
* predict     keep pairs scoring >= --threshold, best --topk per entity
* output      output/matching_results.tsv and output/candidate_pairs.tsv
* evaluate    on train, macro F_0.5 against the ground truth

Memory: Source 1 is streamed in chunks; the resident structure is the source-2/3
key table, roughly 600MB-1GB at full scale. `--sample` runs on a slice for
smaller machines.

Each stage is a single clear step, so richer normalization, additional blocking
keys, fuzzy candidate generation and a learned scorer each attach to one stage
without changing the shape of the pipeline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
# A key shared by more records than this carries no evidence and would emit a
# huge number of useless pairs, so it is skipped.
DEFAULT_MAX_BUCKET = 100


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_path(split: str, source: int, data: Path) -> Path:
    return data / "dataset" / split / f"{split}_source{source}.tsv"


def ground_truth_path(data: Path) -> Path:
    return data / "dataset" / "train" / "train_ground_truth.tsv"


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

def _clean(col: pl.Expr) -> pl.Expr:
    return (
        col.fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _sorted_tokens(col: pl.Expr) -> pl.Expr:
    """Order-invariant signature, so 'Acme Foods' == 'Foods Acme'."""
    return (
        col.str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        .list.unique()
        .list.sort()
        .list.join(" ")
    )


def load_source(split: str, source: int, data: Path, sample: int | None) -> pl.LazyFrame:
    """Read one source TSV and attach the normalized columns.

    `quote_char=None` matters: addresses contain stray quote characters and
    letting polars interpret them silently merges rows. Everything is read as
    text so entity IDs and postal codes keep their exact form.
    """
    lf = pl.scan_csv(
        source_path(split, source, data),
        separator="\t",
        quote_char=None,
        has_header=True,
        schema={c: pl.Utf8 for c in SOURCE_COLUMNS},
    )
    if sample:
        lf = lf.head(sample)
    name = _clean(pl.col("business_name"))
    addr = _clean(pl.col("business_address"))
    return lf.select(
        pl.col("entity_id"),
        # Country is folded into the key rather than compared separately.
        pl.concat_str([_clean(pl.col("country")), pl.lit("|"), _sorted_tokens(name)])
        .alias("block_key"),
        addr.alias("addr"),
    )


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def build_targets(split: str, data: Path, sample: int | None, max_bucket: int) -> pl.DataFrame:
    """Source 2 + 3, keyed for joining, with over-common keys removed."""
    frames = [load_source(split, s, data, sample) for s in (2, 3)]
    targets = pl.concat(frames, how="vertical").collect()
    sizes = targets.group_by("block_key").len()
    keep = sizes.filter(pl.col("len") <= max_bucket).select("block_key")
    before = targets.height
    targets = targets.join(keep, on="block_key", how="inner")
    print(f"  targets: {before:,} rows -> {targets.height:,} after dropping "
          f"{sizes.height - keep.height:,} over-common keys")
    return targets.rename({"entity_id": "candidate_entity_id", "addr": "cand_addr"})


def _jaccard(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Token-set Jaccard. Both addresses empty counts as no evidence, not a match."""
    ta = a.str.split(" ").list.unique()
    tb = b.str.split(" ").list.unique()
    inter = ta.list.set_intersection(tb).list.len()
    union = ta.list.set_union(tb).list.len()
    return (
        pl.when(union == 0).then(0.0).otherwise(inter / union).cast(pl.Float32)
    )


def generate(
    split: str, data: Path, sample: int | None, max_bucket: int,
    topk: int, threshold: float, chunk: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (candidates, matches), one row per (source1, candidate) pair."""
    targets = build_targets(split, data, sample, max_bucket)
    s1 = load_source(split, 1, data, sample)
    n_s1 = s1.select(pl.len()).collect().item()
    print(f"  source1: {n_s1:,} entities")

    cand_parts, match_parts = [], []
    for offset in range(0, n_s1, chunk):
        block = s1.slice(offset, chunk).collect()
        hits = block.join(targets, on="block_key", how="inner")
        if hits.is_empty():
            continue
        hits = hits.with_columns(
            _jaccard(pl.col("addr"), pl.col("cand_addr")).alias("sim")
        ).select("entity_id", "candidate_entity_id", "sim")

        cand_parts.append(hits.select("entity_id", "candidate_entity_id"))
        match_parts.append(
            hits.filter(pl.col("sim") >= threshold)
            .sort(["sim", "candidate_entity_id"], descending=[True, False])
            .filter(pl.int_range(pl.len()).over("entity_id") < topk)
            .select("entity_id", "candidate_entity_id")
        )
        print(f"  {min(offset + chunk, n_s1):,}/{n_s1:,} entities", end="\r")

    empty = pl.DataFrame(schema={"entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8})
    cands = pl.concat(cand_parts, how="vertical") if cand_parts else empty
    matches = pl.concat(match_parts, how="vertical") if match_parts else empty
    print(f"\n  candidates: {cands.height:,}   predicted matches: {matches.height:,}")
    return cands, matches


def write_submission(
    pairs: pl.DataFrame, split: str, data: Path, sample: int | None,
    out_path: Path, column: str,
) -> pl.DataFrame:
    """Write a submission TSV with EVERY source-1 entity present, exactly once.

    Entities with no matches still need a row with an empty field — a missing
    entity is rejected outright by the scorer.
    """
    all_s1 = load_source(split, 1, data, sample).select("entity_id").collect()
    grouped = (
        pairs.group_by("entity_id")
        .agg(pl.col("candidate_entity_id").unique().sort().alias("ids"))
    )
    full = (
        all_s1.join(grouped, on="entity_id", how="left")
        .with_columns(pl.col("ids").fill_null([]))
        .select(
            pl.col("entity_id").alias("source1_entity_id"),
            pl.col("ids").list.join(",").alias(column),
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.write_csv(out_path, separator="\t", include_header=True, quote_style="never")
    print(f"  wrote {full.height:,} rows -> {out_path}")
    return full


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

def macro_f05(predictions: pl.DataFrame, data: Path, column: str) -> float:
    """Macro-averaged F_0.5 against the training ground truth.

    Averaged over ALL source-1 entities, singletons included: an entity with no
    true matches scores 1.0 when the prediction is empty and 0.0 otherwise.
    """
    # The truth column is renamed BEFORE the join. Both frames call it
    # `matched_entity_ids`, so joining as-is suffixes the right-hand one and
    # `pl.col("matched_entity_ids")` silently resolves to the predictions —
    # scoring every entity against itself for a perfect 1.0.
    truth = pl.read_csv(
        ground_truth_path(data), separator="\t", quote_char=None,
        schema={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
    ).rename({"matched_entity_ids": "_truth_ids"})
    if column == "_truth_ids":
        raise ValueError("prediction column name collides with the truth column")
    df = (
        predictions.join(truth, on="source1_entity_id", how="inner")
        .with_columns(
            pl.col(column).fill_null("").str.split(",")
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
            .alias("pred"),
            pl.col("_truth_ids").fill_null("").str.split(",")
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
            .alias("true"),
        )
        .with_columns(
            pl.col("pred").list.set_intersection(pl.col("true")).list.len().alias("tp"),
            pl.col("pred").list.len().alias("np"),
            pl.col("true").list.len().alias("nt"),
        )
        .with_columns(
            pl.when((pl.col("np") == 0) & (pl.col("nt") == 0))
            .then(1.0)  # correctly predicted a singleton
            .when((pl.col("np") == 0) | (pl.col("nt") == 0) | (pl.col("tp") == 0))
            .then(0.0)
            .otherwise(
                (1.25 * (pl.col("tp") / pl.col("np")) * (pl.col("tp") / pl.col("nt")))
                / (0.25 * (pl.col("tp") / pl.col("np")) + (pl.col("tp") / pl.col("nt")))
            )
            .alias("f")
        )
    )
    if df.height == 0:
        return 0.0
    singles = (df["nt"] == 0).sum()
    print(f"  scored {df.height:,} entities ({singles:,} singletons)")
    return float(df["f"].mean())


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Minimal end-to-end ER baseline.")
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--data-root", default=None, help="Folder containing dataset/")
    p.add_argument("--out-dir", default=None, help="Where the two TSVs go")
    p.add_argument("--sample", type=int, default=None,
                   help="Use only the first N rows per source (fast smoke test).")
    p.add_argument("--max-bucket", type=int, default=DEFAULT_MAX_BUCKET)
    p.add_argument("--topk", type=int, default=5,
                   help="Max predicted matches per entity (true mean is ~3.5).")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Minimum address Jaccard to predict a match.")
    p.add_argument("--chunk", type=int, default=250_000)
    a = p.parse_args()

    data = Path(a.data_root).expanduser() if a.data_root else repo_root() / "student_resource"
    out = Path(a.out_dir).expanduser() if a.out_dir else repo_root() / "output"

    started = time.perf_counter()
    print(f"simple baseline [{a.split}]  data={data}")
    cands, matches = generate(
        a.split, data, a.sample, a.max_bucket, a.topk, a.threshold, a.chunk
    )
    write_submission(cands, a.split, data, a.sample,
                     out / "candidate_pairs.tsv", "candidate_entity_ids")
    preds = write_submission(matches, a.split, data, a.sample,
                             out / "matching_results.tsv", "matched_entity_ids")

    if a.split == "train":
        score = macro_f05(preds, data, "matched_entity_ids")
        print(f"\n  MACRO F_0.5 = {score:.4f}")
    print(f"  done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
