"""Reading the provided TSVs and writing intermediates.

The source files are unquoted, tab-separated, and large. Two things matter:

* `separator="\\t"` and `quote_char=None` — addresses contain commas and stray
  quote characters; letting polars treat `"` as a quote char silently mangles
  rows.
* every column is read as a string. Entity IDs are `S1-00001`, postal codes have
  leading zeros, and type inference on a 10M-row file is both slow and wrong.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def scan_source(path: Path) -> pl.LazyFrame:
    """Lazily scan one `*_source<N>.tsv`, everything as Utf8."""
    return pl.scan_csv(
        path,
        separator="\t",
        quote_char=None,
        has_header=True,
        schema={c: pl.Utf8 for c in SOURCE_COLUMNS},
        skip_rows_after_header=0,
    )


def batched_source_reader(path: Path, batch_size: int):
    """Chunked reader over one source TSV, for bounded-memory passes."""
    return pl.read_csv_batched(
        path,
        separator="\t",
        quote_char=None,
        has_header=True,
        schema_overrides={c: pl.Utf8 for c in SOURCE_COLUMNS},
        infer_schema_length=0,
        batch_size=batch_size,
    )


def scan_ground_truth(path: Path) -> pl.LazyFrame:
    """Lazily scan `train_ground_truth.tsv` (two columns, list is comma-joined)."""
    return pl.scan_csv(
        path,
        separator="\t",
        quote_char=None,
        has_header=True,
        schema={c: pl.Utf8 for c in GT_COLUMNS},
    )


def ground_truth_pairs(path: Path) -> pl.LazyFrame:
    """Explode ground truth into one row per (source1_entity_id, matched_id).

    Entities with no matches are dropped here; keep the unexploded frame around
    separately when you need the singleton set.
    """
    return (
        scan_ground_truth(path)
        .with_columns(
            pl.col("matched_entity_ids").fill_null("").str.split(",").alias("matched")
        )
        .explode("matched")
        .with_columns(pl.col("matched").str.strip_chars())
        .filter(pl.col("matched").is_not_null() & (pl.col("matched") != ""))
        .select(
            pl.col("source1_entity_id"),
            pl.col("matched").alias("matched_entity_id"),
        )
    )


def write_id_list_tsv(df: pl.DataFrame, path: Path, id_column: str) -> None:
    """Write a submission-shaped TSV: `source1_entity_id` + a comma-joined list.

    `df` must have `source1_entity_id` and a List[Utf8] column named `id_column`.
    Empty lists become an empty field, which is what the scorer expects for
    singletons.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.select(
        pl.col("source1_entity_id"),
        pl.col(id_column).list.unique().list.sort().list.join(",").alias(id_column),
    )
    out.write_csv(path, separator="\t", include_header=True, quote_style="never")
