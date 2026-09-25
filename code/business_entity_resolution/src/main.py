"""The entity resolution pipeline: the single entry point for every run.

One file covering the whole path — read the data, normalize, block, score,
predict, write both submission files, and self-score against the training
ground truth. Every submission this project has made came from here, and every
improvement lands here.

    python -m src.main --split train --sample 200000   # ~1 min smoke test
    python -m src.main --split train                   # full, self-scored
    python -m src.main --split test                    # writes the submission

Architecture
------------
The pipeline is five stages, each a single function with an explicit input and
output, so a stage can be replaced without touching the ones around it:

* normalize   `prepare` — fold accents, glue initialisms, expand abbreviations,
              strip legal-form tokens, optionally romanize Indic script. Rule
              tables live in `text_norm`; edit data there, not code here.
* block       `build_targets` — exact match on (country, sorted `name_core`
              tokens), dropping keys whose bucket exceeds `--max-bucket`.
* score       `_jaccard` — token-set overlap of `addr_core`.
* predict     `generate` — keep pairs at or above `--threshold`, best `--topk`
              per entity.
* evaluate    `macro_f05` — macro F_0.5 plus the precision, recall and
              empty-prediction diagnostics that say where to work next.

That shape is what makes the next changes cheap. A second blocking key extends
`build_targets`; a fuzzy candidate stage appends to the candidate set before
`generate` filters it; a learned scorer replaces `_jaccard` and the threshold
comparison with a model over several features. None of those requires
restructuring the file.

Measurement discipline
----------------------
Every change is measured against the run it replaces, one change at a time, and
old behaviour is kept reachable behind a flag rather than deleted, so a
regression can be isolated instead of bisected. `--legacy-norm` still reproduces
the original normalization exactly. Measured on the full training split against
a do-nothing floor of 0.0558:

    legacy                  0.4670    27s
    text_norm               0.6070   176s
    text_norm + translit    0.6075   302s

Romanization is off by default on that evidence — see `--translit`.

Scale
-----
Roughly 12.5M rows per split. Sources are read once and normalized in slices;
the resident structure is the source-2/3 key table, about 600MB-1GB at full
scale, which fits comfortably in 16GB alongside one slice. `--sample` limits the
entity side only, so a smoke test still matches against the full target set and
its score stays meaningful.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from . import blocking, paths, text_norm, translit

SOURCE_COLUMNS = paths.SOURCE_COLUMNS
# A key shared by more records than this carries almost no evidence and would
# emit a very large number of useless pairs, so its bucket is dropped.
#
# Set under the superseded normalization and not revisited since. `name_core`
# strips legal-form tokens, which makes keys shorter and buckets larger: the
# number of buckets dropped at this cap rose from 1,107 to 2,755 when the new
# rules landed. Whether 100 is still the right cap is an open measurement.
DEFAULT_MAX_BUCKET = 100


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def repo_root() -> Path:
    return paths.REPO_ROOT


source_path = paths.source_tsv
ground_truth_path = paths.ground_truth_tsv


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------

def _clean_legacy(col: pl.Expr) -> pl.Expr:
    """The superseded rule: lowercase, drop anything non-alphanumeric, collapse.

    Retained as a measurement reference, not as a fallback. Keeping the previous
    behaviour reachable is what lets `--legacy-norm` reproduce its 0.4670 run
    bit-for-bit in the current checkout, so any future regression can be
    attributed rather than bisected. It is not on any default path.

    Its flaw is instructive and worth keeping visible: folding to `[^a-z0-9]`
    deletes every non-Latin character outright, so an Indic-script name
    normalizes to the empty string and roughly 1.5M target records were
    unreachable by any key.
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
# normalized parquet cache (multi-key blocking reads this)
# --------------------------------------------------------------------------

# Everything the multi-key blocker and the address scorer need, and nothing
# else. `text_norm` defines twelve columns between names and addresses;
# materializing all of them over 10.3M rows costs several GB that is
# immediately discarded.
#
# Derived from KEY_SPECS rather than written out, so adding a key there is the
# only edit needed. The prefix decides which expression builder supplies it;
# an unrecognised prefix raises instead of being silently dropped, which would
# otherwise surface as a missing-column error deep inside blocking.
_NAME_KEYS = tuple(k for k in blocking.KEY_COLUMNS if k.startswith("name_"))
# `addr_core` is both a candidate blocking key and the scorer's input. It is
# deduped rather than selected twice, which polars rejects as a duplicate
# column, and it is appended so it is present even when no address key blocks.
_ADDR_KEYS = tuple(
    dict.fromkeys(
        [k for k in blocking.KEY_COLUMNS if k.startswith("addr_")] + ["addr_core"]
    )
)
_UNKNOWN = set(blocking.KEY_COLUMNS) - set(_NAME_KEYS) - set(_ADDR_KEYS)
if _UNKNOWN:
    raise ImportError(
        f"blocking keys {sorted(_UNKNOWN)} match neither the name_ nor addr_ "
        "prefix, so no expression builder supplies them."
    )

NORMALIZED_COLUMNS: tuple[str, ...] = (
    "entity_id", "country_norm", *_NAME_KEYS, *_ADDR_KEYS,
)


def prepare_full(df: pl.DataFrame, use_translit: bool) -> pl.DataFrame:
    """Raw rows -> every blocking key plus the address used for scoring.

    Unlike `prepare`, no row is dropped here. A record whose name normalizes to
    nothing is still reachable through `addr_sorted`, and dropping it at this
    stage would silently remove it from the target universe for every key. Each
    key filters its own empties when its target table is built.
    """
    if use_translit:
        # Romanize before any cleaning: `text_norm.ACCENT_MAP` folds the IAST
        # diacritics this emits, so the order matters.
        df = df.with_columns([
            translit.romanize_series(df.get_column(c))
            for c in ("business_name", "business_address")
        ])

    out = df.select([
        pl.col("entity_id"),
        text_norm.country_expression(),
        *_pick(text_norm.name_expressions(), *_NAME_KEYS),
        *_pick(text_norm.address_expressions(), *_ADDR_KEYS),
    ])
    # Column order is load-bearing: every chunk is appended to the same parquet
    # writer, and a schema that differs between chunks fails the write. Assert
    # it here, where the message names the drift, rather than at the writer.
    assert out.columns == list(NORMALIZED_COLUMNS), (
        f"normalized columns drifted: {out.columns} != {list(NORMALIZED_COLUMNS)}"
    )
    return out


def _rule_mtime() -> float:
    """Newest mtime among the modules that define the normalization rules.

    Comparing the cache against the source TSV alone is not enough, and the gap
    is the dangerous kind: editing `text_norm` changes every key value while
    leaving the TSV untouched, so a stale cache would be reused silently and
    the next blocking measurement would describe the OLD rules while appearing
    to describe the new ones. Nothing would crash and no number would look
    obviously wrong.
    """
    here = Path(__file__).resolve().parent
    return max(
        (p.stat().st_mtime for p in (here / "text_norm.py", here / "translit.py",
                                     here / "blocking.py", here / "main.py")
         if p.is_file()),
        default=0.0,
    )


def _is_fresh(parquet: Path, source: Path, expected_rows: int | None) -> bool:
    """Whether a cached parquet can be reused.

    Three guards, all earned. The cache must be newer than the source data AND
    newer than the rules that produced it (see `_rule_mtime`), and it must hold
    the number of rows this run would write — which catches both an interrupted
    write and a cache built under a different `--sample`.

    Note what is NOT covered: toggling `--translit` changes the output without
    touching any file, so it does not invalidate the cache. Pass
    `--force-normalize` when changing it.
    """
    if not parquet.is_file():
        return False
    cached = parquet.stat().st_mtime
    if cached < source.stat().st_mtime or cached < _rule_mtime():
        return False
    try:
        import pyarrow.parquet as pq

        rows = pq.ParquetFile(parquet).metadata.num_rows
    except Exception:
        return False
    return expected_rows is None or rows == expected_rows


def ensure_normalized(
    split: str,
    data: Path,
    work: Path,
    use_translit: bool,
    chunk: int,
    force: bool,
    sample: int | None = None,
    sample_targets: int | None = None,
) -> None:
    """Normalize all three sources to parquet, reusing an up-to-date cache.

    Normalization is the expensive deterministic step — roughly 15 seconds per
    million rows, so about three minutes per full run. Caching it means
    iterating on blocking does not pay for it again, which is the whole reason
    this stage writes to disk rather than staying in memory.

    `sample` truncates source 1 only. Truncating the target sources instead
    makes any score meaningless — a source-1 entity's true partners are
    scattered anywhere across 10M target rows, so keeping the first N removes
    most of them from reach and the run measures the truncation rather than the
    pipeline. `sample_targets` exists for memory pressure and warns.

    The row count written is compared against the row count that *would* be
    written, so a cache built under one sample size is correctly rebuilt under
    another — including a sampled cache being rebuilt for a full run, which is
    the case that would otherwise silently report a truncated score.
    """
    import pyarrow.parquet as pq

    for source in (1, 2, 3):
        src = source_path(split, source, data)
        dst = paths.normalized_parquet(split, source, work)
        limit = sample if source == 1 else sample_targets
        available = _row_count(src)
        expected = (
            None if available is None
            else (min(limit, available) if limit else available)
        )
        if not force and _is_fresh(dst, src, expected):
            shape = f"{expected:,} rows" if expected is not None else "row count unverified"
            print(f"  source{source}: reusing {dst.name} ({shape})")
            continue

        raw = scan_source(split, source, data, limit).collect()
        writer = None
        try:
            for offset in range(0, max(raw.height, 1), chunk):
                block = raw.slice(offset, chunk)
                if block.is_empty():
                    continue
                table = prepare_full(block, use_translit).to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        print(f"  source{source}: normalized {raw.height:,} rows -> {dst.name}")
        del raw


def _row_count(src: Path) -> int | None:
    """Row count of a source TSV, for the cache freshness check.

    Counting is a full scan of one column, which is far cheaper than
    normalizing, but not free — so a failure here (a missing file, say) returns
    None and simply disables that half of the guard rather than crashing a run
    that would otherwise work.
    """
    try:
        return int(
            pl.scan_csv(
                src, separator="\t", quote_char=None, has_header=True,
                schema={c: pl.Utf8 for c in SOURCE_COLUMNS},
            )
            .select(pl.len())
            .collect()
            .item()
        )
    except Exception:
        return None


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


def _tokens(col: pl.Expr) -> pl.Expr:
    """Unique non-empty tokens.

    Filtering empties is not cosmetic. `"".str.split(" ")` yields `[""]` in
    polars — a list of length ONE, not zero — so without this filter two empty
    addresses have an intersection of 1 and a union of 1 and score a Jaccard of
    **1.0**, the maximum. The `union == 0` guard below never fired, and any
    candidate pair with two blank addresses was predicted as a match at top
    rank.

    Latent rather than active on the training split, which is why it survived:
    source 1 has no empty normalized addresses, so the empty-empty case needs
    both sides blank and never arises there. It is one address-only record away
    from firing on the test split, where nothing has been measured.
    """
    return (
        col.fill_null("")
        .str.split(" ")
        .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        .list.unique()
    )


def _jaccard(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """Token-set Jaccard. Both addresses empty counts as no evidence, not a match."""
    ta = _tokens(a)
    tb = _tokens(b)
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


def generate_multi(
    split: str, data: Path, work: Path, topk: int, threshold: float,
    budget: int, max_bucket: int | None, chunk: int, merge_buckets: int,
    keys: tuple[str, ...] | None, reuse_candidates: bool = False,
    acronym_cap: int | None = None, acronym_min_len: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Multi-key blocking, then the same address score and top-k as before.

    A caution that the numbers will need to settle. Until now every candidate
    was an exact name match, so address Jaccard was *independent* confirming
    evidence and a low threshold was safe. That is no longer true: candidates
    reached through `addr_sorted` agree on their address tokens by
    construction, so scoring them by address overlap is close to circular and
    they will pass almost any threshold. Expect precision to fall and the
    right threshold to move. Read the precision/recall split that `macro_f05`
    prints rather than assuming 0.15 still holds.

    The honest fix is a scorer that uses the name evidence too — `block_score`
    and `n_keys` are carried through the candidate file precisely so a model
    can use them — but that is a separate change and is deliberately not
    bundled into this one, so the blocking gain stays attributable.
    """
    cand_path = paths.candidates_parquet(split, work)
    if reuse_candidates and cand_path.is_file():
        # Blocking is the expensive step (roughly 3.5 minutes at full scale, an
        # order more than normalization), and it is deterministic given the
        # normalized parquet. Reusing it makes threshold and top-k work cheap.
        # Unlike the normalization cache there is no freshness guard here,
        # because the candidate file carries no record of the budget, caps or
        # key set it was built with — so this is opt-in and the caller owns it.
        print(f"  reusing {cand_path.name} (--reuse-candidates; "
              "not checked against --budget/--keys/--max-bucket)")
    else:
        cand_path = blocking.block_split(
            split, work, budget, max_bucket, chunk, merge_buckets, keys,
            acronym_cap, acronym_min_len,
        )

    # Address lookups, resident. Two narrow frames — IDs and one short string —
    # rather than the full normalized set.
    def _addr(sources: tuple[int, ...], id_alias: str, addr_alias: str) -> pl.DataFrame:
        return pl.concat(
            [
                pl.scan_parquet(paths.normalized_parquet(split, s, work))
                .select(
                    pl.col("entity_id").alias(id_alias),
                    pl.col("addr_core").fill_null("").alias(addr_alias),
                )
                for s in sources
            ],
            how="vertical",
        ).collect()

    s1_addr = _addr((1,), "source1_entity_id", "addr")
    tgt_addr = _addr((2, 3), "candidate_entity_id", "cand_addr")
    print(f"  address lookups: {s1_addr.height:,} source-1, {tgt_addr.height:,} target")

    cand_parts, match_parts = [], []
    # Partitioned by the same entity hash the merge used: the top-k cut is per
    # entity, so entities are independent and a bucket can be scored in
    # isolation. This keeps the join output bounded instead of materializing
    # ~44M scored pairs at once.
    for b in range(merge_buckets):
        chunk_df = (
            pl.scan_parquet(cand_path)
            .filter((pl.col("source1_entity_id").hash() % merge_buckets) == b)
            .collect()
        )
        if chunk_df.is_empty():
            continue
        scored = (
            chunk_df.join(s1_addr, on="source1_entity_id", how="inner")
            .join(tgt_addr, on="candidate_entity_id", how="inner")
            .with_columns(_jaccard(pl.col("addr"), pl.col("cand_addr")).alias("sim"))
        )
        cand_parts.append(scored.select("source1_entity_id", "candidate_entity_id"))
        match_parts.append(
            scored.filter(pl.col("sim") >= threshold)
            # Sort on the address score first, then fall back to the blocking
            # evidence, then the id. Jaccard ties are common on short
            # addresses, and an unbroken tie here picks survivors by row order.
            .sort(
                ["sim", "block_score", "n_keys", "candidate_entity_id"],
                descending=[True, True, True, False],
            )
            .filter(pl.int_range(pl.len()).over("source1_entity_id") < topk)
            .select("source1_entity_id", "candidate_entity_id")
        )
        print(f"  scoring bucket {b + 1}/{merge_buckets}", end="\r")

    empty = pl.DataFrame(
        schema={"source1_entity_id": pl.Utf8, "candidate_entity_id": pl.Utf8}
    )
    cands = pl.concat(cand_parts, how="vertical") if cand_parts else empty
    matches = pl.concat(match_parts, how="vertical") if match_parts else empty
    print(f"\n  candidates: {cands.height:,}   predicted matches: {matches.height:,}")
    return cands.rename({"source1_entity_id": "entity_id"}), matches.rename(
        {"source1_entity_id": "entity_id"}
    )


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

    # Diagnostics. F_0.5 alone cannot say which way a threshold should move:
    # the same score can come from predicting too much or too little. Reporting
    # macro precision and macro recall separately answers that in one run
    # instead of a sweep. Precision is averaged over entities that predicted
    # something and recall over entities that have something to find, since
    # neither is defined on the other's empty case.
    n = df.height
    n_true_single = int((df["nt"] == 0).sum())
    n_pred_empty = int((df["np"] == 0).sum())
    # The expensive mistake: a real entity given an empty prediction scores a
    # flat zero and no threshold change can rescue it without candidates.
    n_missed = int(((df["np"] == 0) & (df["nt"] > 0)).sum())
    scored_p = df.filter(pl.col("np") > 0)
    scored_r = df.filter(pl.col("nt") > 0)
    prec = float((scored_p["tp"] / scored_p["np"]).mean()) if scored_p.height else 0.0
    rec = float((scored_r["tp"] / scored_r["nt"]).mean()) if scored_r.height else 0.0

    print(f"  scored {n:,} entities ({n_true_single:,} true singletons, "
          f"{n_true_single / n:.2%})")
    print(f"  macro precision {prec:.4f} (over {scored_p.height:,} entities "
          f"that predicted something)")
    print(f"  macro recall    {rec:.4f} (over {scored_r.height:,} entities "
          f"with a true match)")
    print(f"  predicted empty {n_pred_empty:,} ({n_pred_empty / n:.2%}) — of "
          f"which {n_missed:,} ({n_missed / n:.2%} of all entities) have real "
          f"matches and score a flat zero")
    print(f"  mean predicted {df['np'].mean():.2f} per entity vs "
          f"{df['nt'].mean():.2f} true")
    return float(df["f"].mean())


# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Business entity resolution: normalize, block, score, "
                    "predict, write both submission files, and self-score on "
                    "the training split."
    )
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
    p.add_argument("--max-bucket", type=int, default=None,
                   help="Drop keys shared by more than this many target "
                        f"records. Default under --blocking single is "
                        f"{DEFAULT_MAX_BUCKET}; under multi it is each key's "
                        "own cap from KEY_SPECS, which is what you want — the "
                        "name keys need a loose cap and name_acronym a tight "
                        "one. Setting it explicitly overrides both.")
    p.add_argument("--topk", type=int, default=8,
                   help="Max predicted matches per entity. Flat above 5 on the "
                        "training set; blocking rarely yields more.")
    p.add_argument("--threshold", type=float, default=0.15,
                   help="Minimum address Jaccard to predict a match. This value "
                        "was tuned against the SUPERSEDED normalization; "
                        "`addr_core` expands abbreviations and drops stopwords, "
                        "so the Jaccard distribution has shifted and the "
                        "optimum has moved by an unmeasured amount. The "
                        "precision/recall figures printed after each training "
                        "run indicate which direction.")
    p.add_argument("--chunk", type=int, default=250_000)
    p.add_argument("--work-root", default=None,
                   help="Where the normalized parquet and candidate files go.")
    p.add_argument("--blocking", choices=("multi", "single"), default="multi",
                   help="multi: the union of several weighted exact keys, "
                        "inverse-frequency scored, capped per entity by "
                        "--budget. single (the 0.6070 configuration): one "
                        "exact key on (country, sorted name_core tokens). "
                        "Kept reachable so a regression can be attributed to "
                        "this stage rather than bisected.")
    p.add_argument("--budget", type=int, default=20,
                   help="Candidates kept per entity by multi-key blocking. The "
                        "maximum true match count is 11 and the 99th "
                        "percentile is 8, so this is roughly 2x the worst "
                        "case. Ignored by --blocking single.")
    p.add_argument("--keys", default=None,
                   help="Comma-separated subset of blocking keys, for "
                        f"attributing a change to one key. Default: all of "
                        f"{','.join(blocking.KEY_COLUMNS)}")
    p.add_argument("--merge-buckets", type=int, default=16,
                   help="Hash partitions for the candidate merge and the "
                        "scoring pass. Affects peak memory only, not the "
                        "output.")
    p.add_argument("--reuse-candidates", action="store_true",
                   help="Skip blocking and score the existing candidate "
                        "parquet. Blocking is the expensive step (~3.5 min at "
                        "full scale) and is deterministic given the normalized "
                        "parquet, so this makes --threshold and --topk work "
                        "cheap. NOT freshness-checked: the candidate file does "
                        "not record the budget, caps or keys it was built "
                        "with, so re-run blocking after changing any of them.")
    p.add_argument("--acronym-cap", type=int, default=None,
                   help="Bucket cap for name_acronym alone. See src.blocking.")
    p.add_argument("--acronym-min-len", type=int, default=0,
                   help="Drop acronyms shorter than this. See src.blocking.")
    p.add_argument("--force-normalize", action="store_true",
                   help="Rebuild the normalized parquet even if it looks "
                        "current. The cache is invalidated automatically when "
                        "a source file is newer or the row count differs, but "
                        "not when only --translit changes.")
    p.add_argument("--normalize-only", action="store_true",
                   help="Write the normalized parquet and stop, so "
                        "`python -m src.blocking` and `python -m src.block_eval` "
                        "can be iterated without redoing it.")
    p.add_argument("--legacy-norm", action="store_true",
                   help="Reproduce the superseded lowercase-and-strip-punctuation "
                        "normalization (0.4670) bit-for-bit. A measurement "
                        "reference for attributing a regression to this stage, "
                        "not a supported configuration.")
    p.add_argument("--translit", action="store_true",
                   help="Romanize Indic script before normalizing. OFF by "
                        "default: measured at +0.0005 macro F_0.5 for +126s on "
                        "the full training split, because exact-key blocking "
                        "needs every token to agree and a partly-romanized name "
                        "still differs on its Latin tokens. Its measured value "
                        "is character-trigram similarity (0.000 -> 0.326 on "
                        "cross-script pairs), which only pays once a fuzzy "
                        "stage exists. Ignored with --legacy-norm.")
    a = p.parse_args()

    data = paths.data_root(a.data_root)
    out = paths.output_root(a.out_dir)
    work = paths.work_root(a.work_root)

    # Multi-key blocking reads the normalized parquet, which the legacy rules
    # do not produce — they emit one fused block key and nothing else. Rather
    # than half-supporting the combination, refuse it: `--legacy-norm` exists
    # to reproduce one specific measured run, and that run was single-key.
    if a.legacy_norm and a.blocking == "multi":
        p.error("--legacy-norm reproduces the single-key 0.4670 run; pass "
                "--blocking single with it.")

    started = time.perf_counter()
    if a.legacy_norm:
        mode = "legacy (lowercase + strip punctuation)"
    else:
        mode = "text_norm" + (" + romanization" if a.translit else " (no romanization)")
    print(f"entity resolution [{a.split}]  data={data}")
    print(f"  normalization: {mode}")
    print(f"  blocking: {a.blocking}"
          + (f"  budget={a.budget}" if a.blocking == "multi" else ""))
    if a.sample_targets:
        print("  WARNING: --sample-targets truncates the match universe, so the "
              "F_0.5 below understates the pipeline. Use it for memory, not "
              "for measurement.")

    if a.blocking == "multi":
        print("  normalizing:")
        ensure_normalized(
            a.split, data, work, a.translit and not a.legacy_norm, a.chunk,
            a.force_normalize, a.sample, a.sample_targets,
        )
        if a.normalize_only:
            print(f"  done in {time.perf_counter() - started:.1f}s "
                  f"(--normalize-only; run `python -m src.blocking "
                  f"--split {a.split}` next)")
            return
        keys = tuple(k.strip() for k in a.keys.split(",")) if a.keys else None
        cands, matches = generate_multi(
            a.split, data, work, a.topk, a.threshold, a.budget,
            a.max_bucket, a.chunk, a.merge_buckets, keys,
            a.reuse_candidates, a.acronym_cap, a.acronym_min_len,
        )
    else:
        if a.normalize_only:
            p.error("--normalize-only applies to --blocking multi only.")
        cands, matches = generate(
            a.split, data, a.sample, a.sample_targets,
            a.max_bucket if a.max_bucket is not None else DEFAULT_MAX_BUCKET,
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
