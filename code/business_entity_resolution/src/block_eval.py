"""Scoring the blocking stage against ground truth.

Blocking sets a hard ceiling on everything downstream: a true pair that never
becomes a candidate cannot be recovered by any model. The numbers that matter:

* **pair recall** — fraction of all true pairs present in the candidate set.
  This is the absolute maximum recall the matching model can reach.
* **macro recall** — pair recall computed per entity, then averaged. F_0.5 is
  scored as a macro-average, so this is the number that actually tracks the
  leaderboard; it weights an entity with 1 match the same as one with 8.
* **entities fully covered** — share of entities whose true matches are *all*
  present. These are the only entities that can still score 1.0.
* **entities under budget** — how many came out short of the candidate budget.
  This is the workload for the fuzzy top-up stage.
* **reduction ratio** — how much of the full cross product was eliminated.

Usage:

    python -m src.block_eval --split train
"""

from __future__ import annotations

import argparse

import polars as pl

from pathlib import Path

from . import blocking, config, io_utils


def key_diagnostics(
    split: str, work: Path, data: Path, max_bucket: int | None, sample_pairs: int | None
) -> None:
    """Per-key contribution to recall, and what the bucket cap costs.

    Three numbers per key, computed directly against the true pairs rather than
    against the emitted candidate set (far cheaper — 7.6M true pairs instead of
    26M candidates):

    * **agrees** — true pairs whose key values match. The key's recall if buckets
      were unlimited. This is what `src.eda` reports as matched-pair agreement.
    * **usable** — of those, the ones whose bucket also survives `--max-bucket`.
      The gap between agrees and usable is exactly what the cap costs you.
    * **unique** — true pairs this key finds that NO other key finds. A key with
      high agreement but near-zero unique contribution is redundant; a key with
      low agreement but real unique contribution earns its place.
    """
    _section = lambda t: print(f"\n{'=' * 66}\n{t}\n{'=' * 66}")
    caps = {k: c for k, _, c in blocking.KEY_SPECS}
    if max_bucket:
        caps = {k: max_bucket for k in caps}
    keys = list(caps)

    _section("per-key diagnostics")
    print("  caps: " + ", ".join(f"{k}={v}" for k, v in caps.items()))
    cols = ["entity_id", "country_norm", *keys]
    s1 = pl.scan_parquet(config.normalized_parquet(split, 1, work)).select(cols)
    others = pl.concat(
        [
            pl.scan_parquet(config.normalized_parquet(split, s, work)).select(cols)
            for s in (2, 3)
        ],
        how="vertical",
    )

    truth = io_utils.ground_truth_pairs(config.ground_truth_tsv(data))
    if sample_pairs:
        truth = truth.head(sample_pairs)

    base = (
        truth.join(s1, left_on="source1_entity_id", right_on="entity_id", how="inner")
        .join(others, left_on="matched_entity_id", right_on="entity_id",
              how="inner", suffix="_r")
        .collect()
    )
    total = base.height
    if total == 0:
        print("  no true pairs resolved — check the normalized parquet.")
        return
    print(f"true pairs examined: {total:,}\n")

    # Target-side bucket sizes, so we can tell whether a key that agrees would
    # actually have survived the cap.
    for key in keys:
        sizes = (
            others.select(
                pl.concat_str([pl.col("country_norm"), pl.lit("|"), pl.col(key)])
                .alias("bk")
            )
            .group_by("bk")
            .len()
            .collect()
        )
        base = base.with_columns(
            pl.concat_str([pl.col("country_norm_r"), pl.lit("|"), pl.col(f"{key}_r")])
            .alias("bk")
        ).join(sizes, on="bk", how="left").rename({"len": f"_size_{key}"}).drop("bk")

    base = base.with_columns([
        (
            (pl.col(key) == pl.col(f"{key}_r"))
            & (pl.col(key).str.len_chars() > 0)
        ).alias(f"_agree_{key}")
        for key in keys
    ])
    base = base.with_columns([
        (
            pl.col(f"_agree_{key}")
            & pl.col(f"_size_{key}").fill_null(10**9).le(caps[key])
        ).alias(f"_usable_{key}")
        for key in keys
    ])

    usable_any = pl.any_horizontal([pl.col(f"_usable_{k}") for k in keys])
    print(f"{'key':<14} {'agrees':>9} {'usable':>9} {'cap cost':>9} {'unique':>9}")
    for key in keys:
        agree = base[f"_agree_{key}"].sum()
        usable = base[f"_usable_{key}"].sum()
        others_usable = pl.any_horizontal(
            [pl.col(f"_usable_{k}") for k in keys if k != key]
        )
        unique = base.select(
            (pl.col(f"_usable_{key}") & ~others_usable).sum()
        ).item()
        print(f"{key:<14} {agree / total:>8.2%} {usable / total:>8.2%} "
              f"{(agree - usable) / total:>8.2%} {unique / total:>8.2%}")

    union = base.select(usable_any.sum()).item()
    union_uncapped = base.select(
        pl.any_horizontal([pl.col(f"_agree_{k}") for k in keys]).sum()
    ).item()
    print(f"\nUNION (what stage A can reach) : {union / total:.2%}")
    print(f"UNION if buckets were unlimited: {union_uncapped / total:.2%}")
    print(f"  -> the bucket cap is costing  {(union_uncapped - union) / total:.2%} "
          "of all true pairs")
    print(f"UNREACHABLE by any exact key   : {1 - union_uncapped / total:.2%} "
          "<- needs fuzzy matching, no rule or cap change will help")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate blocking recall.")
    parser.add_argument("--split", default="train", choices=config.SPLITS)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--budget", type=int, default=30,
                        help="Budget blocking was run with, for the shortfall count.")
    parser.add_argument("--max-bucket", type=int, default=None,
                        help="Global bucket-cap override for the diagnostics. Leave "
                             "unset to use each key's own cap from KEY_SPECS.")
    parser.add_argument("--sample-pairs", type=int, default=None,
                        help="Cap true pairs used for per-key diagnostics (memory).")
    parser.add_argument("--no-key-diagnostics", action="store_true")
    parser.add_argument("--candidates", default=None,
                        help="Candidate parquet to score. Defaults to stage A output; "
                             "pass the stage B file to score the merged set.")
    args = parser.parse_args()

    work = config.work_root(args.work_root)
    data = config.data_root(args.data_root)

    cand_path = (
        Path(args.candidates) if args.candidates
        else blocking.candidates_parquet(args.split, work)
    )
    if not cand_path.is_file():
        raise FileNotFoundError(
            f"{cand_path} missing — run `python -m src.blocking --split {args.split}` first."
        )
    cand = pl.scan_parquet(cand_path)

    truth = io_utils.ground_truth_pairs(config.ground_truth_tsv(data)).rename(
        {"matched_entity_id": "candidate_entity_id"}
    )

    # Per-entity: how many true matches exist, and how many blocking found.
    per_entity = (
        truth.join(
            cand.select("source1_entity_id", "candidate_entity_id").with_columns(
                pl.lit(True).alias("found")
            ),
            on=["source1_entity_id", "candidate_entity_id"],
            how="left",
        )
        .group_by("source1_entity_id")
        .agg(
            pl.len().alias("n_true"),
            pl.col("found").fill_null(False).sum().alias("n_found"),
        )
        .collect()
    )

    n_true = per_entity["n_true"].sum()
    n_found = per_entity["n_found"].sum()
    per_entity = per_entity.with_columns(
        (pl.col("n_found") / pl.col("n_true")).alias("recall")
    )
    full = (per_entity["n_found"] == per_entity["n_true"]).sum()
    none = (per_entity["n_found"] == 0).sum()

    sizes = (
        cand.group_by("source1_entity_id").agg(pl.len().alias("n_cand")).collect()
    )
    total_cand = sizes["n_cand"].sum()
    s1_total = (
        pl.scan_parquet(config.normalized_parquet(args.split, 1, work))
        .select(pl.len()).collect().item()
    )
    targets = sum(
        pl.scan_parquet(config.normalized_parquet(args.split, s, work))
        .select(pl.len()).collect().item()
        for s in (2, 3)
    )
    # Entities blocking produced nothing for do not appear in `sizes` at all.
    no_candidates = s1_total - sizes.height
    under = (sizes["n_cand"] < args.budget).sum() + no_candidates

    print(f"\n{'=' * 66}\nblocking evaluation [{args.split}]\n{'=' * 66}")
    print(f"true pairs                  : {n_true:,}")
    print(f"true pairs in candidate set : {n_found:,}")
    print(f"PAIR RECALL (ceiling)       : {n_found / n_true:.2%}")
    print(f"MACRO RECALL (tracks F_0.5) : {per_entity['recall'].mean():.2%}")
    print()
    print(f"entities with all matches found : {full:,} "
          f"({full / per_entity.height:.2%})  <- only these can score 1.0")
    print(f"entities with none found        : {none:,} ({none / per_entity.height:.2%})")
    print()
    print(f"candidate pairs emitted     : {total_cand:,}")
    print(f"mean candidates per entity  : {total_cand / max(s1_total, 1):.2f}")
    print(f"entities with no candidates : {no_candidates:,}")
    print(f"entities under budget ({args.budget})  : {under:,} "
          f"({under / max(s1_total, 1):.2%})  <- workload for fuzzy top-up")
    full_cross = s1_total * targets
    print(f"reduction ratio             : {1 - total_cand / full_cross:.8f} "
          f"(of {full_cross:,} possible pairs)")

    if not args.no_key_diagnostics:
        key_diagnostics(args.split, work, data, args.max_bucket, args.sample_pairs)


if __name__ == "__main__":
    main()
