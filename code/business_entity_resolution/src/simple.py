"""End-to-end entity resolution pipeline.

One file covering the whole path: read the data, block, score, predict, write
both submission files, and self-score against the training ground truth.
Deliberately lean on purpose — a working end-to-end pipeline before refining any
stage.

    python -m src.simple --split train --sample 200000   # ~1 min smoke test
    python -m src.simple --split train                   # full, self-scored
    python -m src.simple --split test                    # writes submission

Stages:

* normalize   romanize Indic script, fold accents, glue initialisms, expand
              abbreviations, strip legal-form tokens (see `text_norm`)
* block       exact match on (country, sorted `name_core` tokens)
* score       Jaccard overlap of `addr_core` tokens
* predict     keep pairs scoring >= --threshold, best --topk per entity
* output      output/matching_results.tsv and output/candidate_pairs.tsv
* evaluate    on train, macro F_0.5 against the ground truth

Two flags exist so the normalization change is attributable rather than bundled:
`--legacy-norm` restores the original lowercase-and-strip-punctuation behaviour
that scored 0.4670, and `--no-translit` keeps the new rules but skips
romanization. Three runs therefore separate "better rules" from "cross-script
recovery" instead of reporting one combined delta.

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

from . import text_norm, translit

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

def _clean_legacy(col: pl.Expr) -> pl.Expr:
    """The original rule: lowercase, drop anything non-alphanumeric, collapse.

    Kept verbatim so `--legacy-norm` reproduces the 0.4670 run exactly. Note it
    folds to `[^a-z0-9]`, which deletes every non-Latin character outright — an
    Indic-script name normalizes to the empty string under this rule.
    """
    return (
        col.fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _sorted_tokens_legacy(col: pl.Expr) -> pl.Expr:
    """Order-invariant signature, so 'Acme Foods' == 'Foods Acme'."""
    return (
        col.str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        .list.unique()
        .list.sort()
        .list.join(" ")
    )


def _pick(exprs: list[pl.Expr], *names: str) -> list[pl.Expr]:
    """Select aliased expressions from `text_norm` by their output name.

    Only the two columns actually used are built. `name_expressions()` and
    `address_expressions()` define six each, and materializing all twelve over
    10.3M target rows costs several GB that is then immediately discarded —
    which a 16GB machine feels. Raising rather than silently skipping a missing
    name means a rename in `text_norm` fails loudly here instead of quietly
    changing the blocking key.
    """
    by: dict[str, pl.Expr] = {e.meta.output_name(): e for e in exprs}
    missing = [n for n in names if n not in by]
    if missing:
        raise KeyError(f"text_norm no longer emits {missing}; available: {sorted(by)}")
    return [by[n] for n in names]


def scan_source(split: str, source: int, data: Path, sample: int | None) -> pl.LazyFrame:
    """Read one source TSV as raw text, with no normalization attached.

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
    return lf.head(sample) if sample else lf


def prepare(df: pl.DataFrame, legacy: bool, use_translit: bool) -> pl.DataFrame:
    """Raw rows -> (entity_id, block_key, addr).

    Eager rather than lazy: romanization is a per-string Python call that needs
    a materialized Series, so callers collect a chunk and hand it here.

    Country is folded into the block key rather than compared separately. It
    agrees on 100% of true training pairs, so it is a free hard partition that
    removes all cross-country comparisons at no cost in recall. It is used as an
    opaque label — never enumerated — because the test set contains France,
    which appears nowhere in training.
    """
    if legacy:
        name = _clean_legacy(pl.col("business_name"))
        return df.select(
            pl.col("entity_id"),
            pl.concat_str([
                _clean_legacy(pl.col("country")),
                pl.lit("|"),
                _sorted_tokens_legacy(name),
            ]).alias("block_key"),
            _clean_legacy(pl.col("business_address")).alias("addr"),
        )

    if use_translit:
        # Romanize before any cleaning: `text_norm.ACCENT_MAP` folds the IAST
        # diacritics this emits, so the order matters.
        df = df.with_columns([
            translit.romanize_series(df.get_column(c))
            for c in ("business_name", "business_address")
        ])

    (name_sorted,) = _pick(text_norm.name_expressions(), "name_sorted")
    (addr_core,) = _pick(text_norm.address_expressions(), "addr_core")
    return df.select(
        pl.col("entity_id"),
        pl.concat_str([
            text_norm.country_expression(),
            pl.lit("|"),
            name_sorted,
        ]).alias("block_key"),
        addr_core.alias("addr"),
    ).filter(
        # A row whose name normalizes to nothing would otherwise land in a
        # single "<country>|" bucket shared by every such row — a spurious
        # block, not a match. This filter is deliberately NOT applied on the
        # legacy path so `--legacy-norm` stays bit-identical to the 0.4670 run.
        ~pl.col("block_key").str.ends_with("|")
    )


def load_prepared(
    split: str, source: int, data: Path, sample: int | None,
    legacy: bool, use_translit: bool, chunk: int,
) -> pl.DataFrame:
    """Read and normalize one source in batches, keeping only the narrow result.

    Batching is about peak memory: romanization is a Python call that needs a
    materialized Series, so normalization cannot stay lazy, and normalizing
    10.3M rows at once holds the raw columns and the expression output at the
    same time. Each batch is reduced to three columns before the next is read.

    The raw source is read once, then sliced in memory. Slicing a DataFrame is
    a zero-copy view, so this is a real cursor. Two alternatives were rejected:
    `scan_csv(...).slice(offset, chunk).collect()` in a loop re-reads the TSV
    from the start every time — 42 passes over a 10M-row file — and
    `read_csv_batched` is deprecated in current polars, while its replacement
    `LazyFrame.collect_batches` does not exist in 1.18, is documented as
    unstable, and warns that it is slower than the alternatives.

    Peak memory is one raw source (~700MB for 5M rows of four short strings)
    plus the narrow result, which is the same order the target key table
    already occupied.
    """
    raw = scan_source(split, source, data, sample).collect()
    parts = [
        prepare(raw.slice(offset, chunk), legacy, use_translit)
        for offset in range(0, max(raw.height, 1), chunk)
    ]
    del raw
    return pl.concat(parts, how="vertical")


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def build_targets(
    split: str, data: Path, sample: int | None, max_bucket: int,
    legacy: bool, use_translit: bool, chunk: int,
) -> pl.DataFrame:
    """Source 2 + 3, keyed for joining, with over-common keys removed.

    `sample` here is `--sample-targets`, NOT `--sample`. Truncating the target
    sources makes any score meaningless: a source-1 entity's true partners are
    scattered anywhere across 10M target rows, so keeping only the first N
    removes most of them from reach and the run measures the truncation rather
    than the pipeline. Shrink the entity side instead.
    """
    targets = pl.concat(
        [load_prepared(split, s, data, sample, legacy, use_translit, chunk)
         for s in (2, 3)],
        how="vertical",
    )
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
    split: str, data: Path, sample: int | None, sample_targets: int | None,
    max_bucket: int, topk: int, threshold: float, chunk: int,
    legacy: bool, use_translit: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (candidates, matches), one row per (source1, candidate) pair."""
    targets = build_targets(
        split, data, sample_targets, max_bucket, legacy, use_translit, chunk
    )
    # Source 1 is normalized once, in one file pass, and held narrow — 2.2M rows
    # of three short strings. The join below still runs in chunks, but they are
    # slices of an in-memory frame rather than repeated reads of the TSV.
    s1 = load_prepared(split, 1, data, sample, legacy, use_translit, chunk)
    n_s1 = s1.height
    print(f"  source1: {n_s1:,} entities with a usable block key")

    cand_parts, match_parts = [], []
    for offset in range(0, n_s1, chunk):
        block = s1.slice(offset, chunk)
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
    # Raw scan, not the normalized path: this only needs the ID list, and every
    # source-1 entity must appear here whether or not its name normalized to
    # anything blockable.
    all_s1 = scan_source(split, 1, data, sample).select("entity_id").collect()
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
                   help="Match only the first N Source-1 entities against the FULL "
                        "target set. Fast smoke test with a meaningful score.")
    p.add_argument("--sample-targets", type=int, default=None,
                   help="Also truncate Sources 2 and 3 to N rows each. Reduces "
                        "memory, but removes most true partners from reach, so "
                        "the reported F_0.5 understates the pipeline.")
    p.add_argument("--max-bucket", type=int, default=DEFAULT_MAX_BUCKET)
    p.add_argument("--topk", type=int, default=8,
                   help="Max predicted matches per entity. Flat above 5 on the "
                        "training set; blocking rarely yields more.")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="Minimum address Jaccard to predict a match. Tuned on the "
                        "full training split; see README for the sweep.")
    p.add_argument("--chunk", type=int, default=250_000)
    p.add_argument("--legacy-norm", action="store_true",
                   help="Restore the original lowercase-and-strip-punctuation "
                        "normalization that scored 0.4670. Use it to re-measure "
                        "the baseline in the same checkout as the new rules.")
    p.add_argument("--no-translit", dest="translit", action="store_false",
                   help="Keep the new normalization rules but skip Indic "
                        "romanization, separating 'better rules' from "
                        "'cross-script recovery'. Ignored with --legacy-norm, "
                        "which never romanized.")
    a = p.parse_args()

    data = Path(a.data_root).expanduser() if a.data_root else repo_root() / "student_resource"
    out = Path(a.out_dir).expanduser() if a.out_dir else repo_root() / "output"

    started = time.perf_counter()
    if a.legacy_norm:
        mode = "legacy (lowercase + strip punctuation)"
    else:
        mode = "text_norm" + (" + romanization" if a.translit else " (no romanization)")
    print(f"simple baseline [{a.split}]  data={data}")
    print(f"  normalization: {mode}")
    if a.sample_targets:
        print("  WARNING: --sample-targets truncates the match universe, so the "
              "F_0.5 below understates the pipeline. Use it for memory, not "
              "for measurement.")
    cands, matches = generate(
        a.split, data, a.sample, a.sample_targets, a.max_bucket,
        a.topk, a.threshold, a.chunk,
        a.legacy_norm, a.translit and not a.legacy_norm,
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
