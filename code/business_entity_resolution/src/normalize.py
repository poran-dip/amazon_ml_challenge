"""Step 1 — normalization.

Reads the raw source TSVs and writes one normalized parquet per split/source.
Streaming end to end, so peak memory stays low on a laptop even at 10M rows.

Usage (from code/business_entity_resolution/):

    python -m src.normalize --split train
    python -m src.normalize --split test
    python -m src.normalize --split train --sample 50000   # quick iteration

Re-run this whenever you change a rule in `text_norm.py`. Nothing downstream
should ever touch the raw TSVs again.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from . import config, io_utils, text_norm

# Raw text is kept alongside the normalized forms: some features (casing,
# original punctuation, untouched digits) are only available pre-normalization.
OUTPUT_COLUMNS = [
    "entity_id", "source", "country", "country_norm",
    "business_name", "name_norm", "name_core", "name_sorted",
    "name_nospace", "name_acronym", "name_token_count",
    "business_address", "addr_norm", "addr_core", "addr_sorted",
    "addr_numbers", "addr_postal", "addr_token_count",
]


def normalize_lazy(lf: pl.LazyFrame, source: int) -> pl.LazyFrame:
    return (
        lf.with_columns(
            pl.lit(source, dtype=pl.UInt8).alias("source"),
            text_norm.country_expression(),
            *text_norm.name_expressions(),
            *text_norm.address_expressions(),
        )
        .select(OUTPUT_COLUMNS)
    )


def normalize_source(
    src: Path, dst: Path, source: int, batch_size: int, sample: int | None
) -> int:
    """Normalize one source file chunk by chunk, appending to a parquet file.

    Chunking rather than `sink_parquet` for two reasons: several of the string
    expressions are not supported by the streaming engine in polars 1.18, and an
    explicit batch size keeps peak RSS predictable on a laptop.
    """
    import pyarrow.parquet as pq

    reader = io_utils.batched_source_reader(src, batch_size)
    writer = None
    rows = 0
    try:
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            for batch in batches:
                if sample is not None:
                    remaining = sample - rows
                    if remaining <= 0:
                        return rows
                    batch = batch.head(remaining)
                out = batch.lazy().pipe(normalize_lazy, source).collect()
                table = out.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
                writer.write_table(table)
                rows += out.height
    finally:
        if writer is not None:
            writer.close()
    return rows


def run(
    split: str, data: Path, work: Path, sample: int | None, batch_size: int
) -> None:
    for source in config.SOURCES:
        src = config.source_tsv(split, source, data)
        if not src.is_file():
            raise FileNotFoundError(
                f"{src} not found. Unzip the provided material into "
                f"{data}, or pass --data-root / set AMLC_DATA."
            )
        dst = config.normalized_parquet(split, source, work)

        started = time.perf_counter()
        rows = normalize_source(src, dst, source, batch_size, sample)
        elapsed = time.perf_counter() - started
        print(f"  source{source}: {rows:,} rows -> {dst.name} ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw source TSVs.")
    parser.add_argument("--split", choices=config.SPLITS, required=True)
    parser.add_argument("--data-root", default=None, help="Folder containing dataset/")
    parser.add_argument("--work-root", default=None, help="Where intermediates go")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only normalize the first N rows per source (fast rule iteration).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500_000,
        help="Rows per chunk. Lower it if memory is tight (default: %(default)s).",
    )
    args = parser.parse_args()

    data = config.data_root(args.data_root)
    work = config.work_root(args.work_root)
    print(f"normalize [{args.split}]  data={data}  work={work}")
    run(args.split, data, work, args.sample, args.batch_size)


if __name__ == "__main__":
    main()
