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

from . import blocking, config, io_utils


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate blocking recall.")
    parser.add_argument("--split", default="train", choices=config.SPLITS)
    parser.add_argument("--work-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--budget", type=int, default=30,
                        help="Budget blocking was run with, for the shortfall count.")
    args = parser.parse_args()

    work = config.work_root(args.work_root)
    data = config.data_root(args.data_root)

    cand_path = blocking.candidates_parquet(args.split, work)
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


if __name__ == "__main__":
    main()
