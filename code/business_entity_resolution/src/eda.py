"""Step 0 — look at the data before tuning any more rules.

Everything in `text_norm.py` past the obvious cases is a guess until we have
seen the real noise. This script reports what the rule tables should be built
from, and — most importantly — how far normalization alone gets us on true
pairs from the ground truth.

Run after `src.normalize --split train`:

    python -m src.eda --sample 200000

Read the `matched-pair agreement` block first: for each normalized field it
shows what fraction of TRUE pairs share an identical value. Those numbers are
the ceiling for any exact-match blocking key built on that field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from . import config, io_utils

FIELDS = ["name_norm", "name_core", "name_sorted", "name_nospace",
          "name_acronym", "addr_core", "addr_sorted"]


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def basic_profile(frames: dict[int, pl.DataFrame]) -> None:
    _section("per-source profile")
    for source, df in frames.items():
        empty_name = (df["name_core"].str.len_chars() == 0).sum()
        empty_addr = (df["addr_core"].str.len_chars() == 0).sum()
        no_postal = (df["addr_postal"].list.len() == 0).sum()
        n = df.height
        print(f"source{source}: {n:,} rows")
        print(f"  empty name_core : {empty_name:,} ({empty_name / n:.2%})")
        print(f"  empty addr_core : {empty_addr:,} ({empty_addr / n:.2%})")
        print(f"  no postal code  : {no_postal:,} ({no_postal / n:.2%})")
        print(f"  countries       : {df['country_norm'].value_counts(sort=True).head(10).to_dicts()}")
        print(f"  median name toks: {df['name_token_count'].median()}   "
              f"addr toks: {df['addr_token_count'].median()}")


def match_count_distribution(frames: dict[int, pl.DataFrame], gt_path: Path) -> None:
    """How many matches each Source-1 entity has, and the do-nothing baseline.

    Under F_0.5 an entity with no true matches scores a full 1.0 when you predict
    an empty list. So the singleton rate IS the score of a submission that
    predicts nothing for everyone — the floor any pipeline must beat.
    """
    _section("matches per Source-1 entity")
    gt = (
        io_utils.scan_ground_truth(gt_path)
        .with_columns(
            pl.when(pl.col("matched_entity_ids").fill_null("").str.strip_chars() == "")
            .then(pl.lit(0, dtype=pl.UInt32))
            .otherwise(
                pl.col("matched_entity_ids").str.count_matches(",").cast(pl.UInt32) + 1
            )
            .alias("n_matches")
        )
        .select("source1_entity_id", "n_matches")
        .collect()
    )
    total = gt.height
    counts = gt["n_matches"]
    singletons = (counts == 0).sum()

    print(f"entities in ground truth : {total:,}")
    print(f"total true pairs         : {counts.sum():,}")
    print(f"SINGLETONS (0 matches)   : {singletons:,} ({singletons / total:.2%})")
    print(f"  -> predicting nothing for everyone scores F_0.5 = {singletons / total:.4f}")
    print(f"     (that is the floor; any pipeline must beat it)")
    print(f"mean matches (all)       : {counts.mean():.2f}")
    non_zero = counts.filter(counts > 0)
    if non_zero.len():
        print(f"mean matches (non-empty) : {non_zero.mean():.2f}")
        for q in (0.5, 0.9, 0.99, 1.0):
            print(f"  q{q:<5} : {non_zero.quantile(q):.0f}")

    hist = (
        gt.group_by("n_matches").len().sort("n_matches").head(12)
    )
    print("  distribution:")
    for row in hist.iter_rows(named=True):
        print(f"    {row['n_matches']:>3} matches: {row['len']:>9,} "
              f"({row['len'] / total:6.2%})")

    # Coverage: is every Source-1 record represented in the ground truth?
    s1_ids = set(frames[1]["entity_id"].to_list())
    gt_ids = set(gt["source1_entity_id"].to_list())
    missing = len(s1_ids - gt_ids)
    print(f"source1 rows absent from ground truth: {missing:,}")


def script_census(frames: dict[int, pl.DataFrame]) -> None:
    """How much of each source arrived in a non-Latin script."""
    _section("script mix (post-romanization flag)")
    for source, df in frames.items():
        if "is_indic" not in df.columns:
            print(f"source{source}: no is_indic column — re-run normalize.")
            continue
        n = df.height
        indic = df["is_indic"].sum()
        print(f"source{source}: {indic:,} of {n:,} rows ({indic / n:.2%}) "
              "contained Indic script")
        by_country = (
            df.group_by("country_norm")
            .agg(pl.col("is_indic").mean().alias("indic_rate"), pl.len())
            .sort("len", descending=True)
        )
        for row in by_country.iter_rows(named=True):
            print(f"    {row['country_norm']:<10} {row['indic_rate']:6.2%} "
                  f"of {row['len']:,}")


def key_selectivity(frames: dict[int, pl.DataFrame]) -> None:
    """How many records share each candidate blocking key.

    A key whose largest bucket holds tens of thousands of records will blow the
    20-100 candidate budget on its own.
    """
    _section("blocking-key selectivity (source 1)")
    df = frames[1]
    for field in FIELDS:
        counts = df.group_by(field).len()
        counts = counts.filter(pl.col(field).str.len_chars() > 0)
        if counts.is_empty():
            continue
        print(f"{field:14s} distinct={counts.height:,}  "
              f"max_bucket={counts['len'].max():,}  "
              f"p99_bucket={counts['len'].quantile(0.99):.0f}  "
              f"singleton_keys={(counts['len'] == 1).sum() / counts.height:.1%}")


def matched_pair_agreement(
    frames: dict[int, pl.DataFrame], gt_path: Path, sample: int | None
) -> None:
    """For true pairs, how often does each normalized field agree exactly?

    This is the number that tells you whether a rule change helped. An exact-match
    blocking key on a field can never recall more true pairs than its score here.
    """
    _section("matched-pair agreement (recall ceiling per exact-match key)")
    pairs = io_utils.ground_truth_pairs(gt_path).collect()
    if sample:
        pairs = pairs.head(sample)
    others = pl.concat([frames[2], frames[3]], how="vertical_relaxed")

    joined = (
        pairs.join(frames[1], left_on="source1_entity_id", right_on="entity_id", how="inner")
        .join(others, left_on="matched_entity_id", right_on="entity_id",
              how="inner", suffix="_r")
    )
    if joined.is_empty():
        print("no overlap between ground truth and the normalized sample — "
              "re-run normalize without --sample, or raise --sample here.")
        return
    print(f"true pairs resolved in this sample: {joined.height:,}")
    for field in FIELDS:
        agree = (joined[field] == joined[f"{field}_r"]).mean()
        print(f"  {field:14s} exact agreement: {agree:.2%}")

    same_country = (joined["country_norm"] == joined["country_norm_r"]).mean()
    print(f"  {'country_norm':14s} exact agreement: {same_country:.2%}  "
          "(if this is ~100%, country is a safe blocking partition)")

    postal_overlap = joined.select(
        pl.col("addr_postal").list.set_intersection(pl.col("addr_postal_r"))
        .list.len().gt(0).mean()
    ).item()
    print(f"  {'addr_postal':14s} any overlap    : {postal_overlap:.2%}")


def disagreement_examples(
    frames: dict[int, pl.DataFrame], gt_path: Path, field: str, n: int
) -> None:
    """True pairs that normalization failed to collapse — the rule backlog."""
    _section(f"true pairs where {field} still differs (rule backlog)")
    pairs = io_utils.ground_truth_pairs(gt_path).collect().head(200_000)
    others = pl.concat([frames[2], frames[3]], how="vertical_relaxed")
    joined = (
        pairs.join(frames[1], left_on="source1_entity_id", right_on="entity_id", how="inner")
        .join(others, left_on="matched_entity_id", right_on="entity_id",
              how="inner", suffix="_r")
        .filter(pl.col(field) != pl.col(f"{field}_r"))
        .head(n)
    )
    pl.Config.set_fmt_str_lengths(70)
    for row in joined.iter_rows(named=True):
        print(f"  {row[field]!r}\n  {row[f'{field}_r']!r}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the normalized training data.")
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--sample", type=int, default=None,
                        help="Rows per source to load (default: all).")
    parser.add_argument("--show-diffs", default="name_sorted",
                        help="Field whose unresolved true pairs to print.")
    parser.add_argument("--n-diffs", type=int, default=15)
    args = parser.parse_args()

    work = config.work_root(args.work_root)
    data = config.data_root(args.data_root)

    frames: dict[int, pl.DataFrame] = {}
    for source in config.SOURCES:
        path = config.normalized_parquet("train", source, work)
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing — run `python -m src.normalize --split train` first.")
        lf = pl.scan_parquet(path)
        if args.sample:
            lf = lf.head(args.sample)
        frames[source] = lf.collect()

    basic_profile(frames)
    match_count_distribution(frames, config.ground_truth_tsv(data))
    script_census(frames)
    key_selectivity(frames)
    matched_pair_agreement(frames, config.ground_truth_tsv(data), None)
    disagreement_examples(frames, config.ground_truth_tsv(data),
                          args.show_diffs, args.n_diffs)


if __name__ == "__main__":
    main()
