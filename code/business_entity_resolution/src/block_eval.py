"""Scoring the blocking stage against ground truth.

Blocking sets a hard ceiling on everything downstream: a true pair that never
becomes a candidate cannot be recovered by any model, threshold or feature. So
this is the number to move before any of those are worth touching, and the
per-key diagnostics are what make a blocking change judgeable rather than a
guess.

Two sections, and they answer different questions.

**Candidate-set recall** scores the file blocking actually wrote. It is the
real ceiling, after bucket caps and after the per-entity budget.

  * *pair recall* — share of all true pairs present. The absolute maximum
    recall any downstream model can reach.
  * *macro recall* — pair recall per entity, then averaged. F_0.5 is scored as
    a macro average, so this is the number that tracks the leaderboard; it
    weights an entity with 1 true match the same as one with 8. **Track this
    one.** Pair recall and macro recall move apart, and only one of them is
    what the competition measures.
  * *fully covered* — entities whose true matches are all present. Only these
    can still score 1.0.

**Per-key diagnostics** attribute that recall to individual keys, computed
against the true pairs directly rather than the emitted candidates — 7.6M true
pairs instead of ~44M candidates, so it is much cheaper.

  * *agrees* — true pairs whose key values match. The key's recall if buckets
    were unlimited.
  * *usable* — of those, the ones whose bucket also survives the cap. The gap
    between agrees and usable is exactly what the cap costs.
  * *unique* — true pairs this key finds that NO other key finds. This is the
    column that decides whether a key earns its place, and it is close to
    uncorrelated with agreement: under the superseded normalization
    `addr_sorted` had the weakest agreement of any key and the largest unique
    contribution, while `name_core` agreed on half of all true pairs and
    contributed nothing at all.

Usage:

    python -m src.block_eval --split train
    python -m src.block_eval --split train --no-key-diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from . import blocking, paths


def ground_truth_pairs(path: Path) -> pl.LazyFrame:
    """One row per (source1_entity_id, matched_entity_id).

    The file stores the match list comma-joined in a single column; exploding
    it is what makes it joinable against a candidate set.
    """
    return (
        pl.scan_csv(
            path,
            separator="\t",
            quote_char=None,
            has_header=True,
            schema={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
        )
        .with_columns(
            pl.col("matched_entity_ids").fill_null("").str.split(",")
            .list.eval(pl.element().filter(pl.element().str.len_chars() > 0))
        )
        .explode("matched_entity_ids")
        .filter(pl.col("matched_entity_ids").is_not_null())
        .rename({"matched_entity_ids": "matched_entity_id"})
    )


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --------------------------------------------------------------------------
# what the emitted candidate set actually delivers
# --------------------------------------------------------------------------

def candidate_recall(
    split: str, work: Path, data: Path, cand_path: Path, budget: int
) -> None:
    cand = pl.scan_parquet(cand_path)
    truth = ground_truth_pairs(paths.ground_truth_tsv(data)).rename(
        {"matched_entity_id": "candidate_entity_id"}
    )

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
    if per_entity.is_empty():
        print("  no true pairs resolved — check the ground truth path.")
        return

    n_true = int(per_entity["n_true"].sum())
    n_found = int(per_entity["n_found"].sum())
    per_entity = per_entity.with_columns(
        (pl.col("n_found") / pl.col("n_true")).alias("recall")
    )
    full = int((per_entity["n_found"] == per_entity["n_true"]).sum())
    none = int((per_entity["n_found"] == 0).sum())

    sizes = cand.group_by("source1_entity_id").agg(pl.len().alias("n")).collect()
    total_cand = int(sizes["n"].sum())
    s1_total = (
        pl.scan_parquet(paths.normalized_parquet(split, 1, work))
        .select(pl.len()).collect().item()
    )
    targets = sum(
        pl.scan_parquet(paths.normalized_parquet(split, s, work))
        .select(pl.len()).collect().item()
        for s in (2, 3)
    )
    # Entities blocking produced nothing for do not appear in `sizes` at all,
    # so they have to be counted by difference rather than filtered for.
    no_candidates = s1_total - sizes.height
    under = int((sizes["n"] < budget).sum()) + no_candidates

    _section(f"candidate set [{split}] — {cand_path.name}")
    print(f"true pairs                  : {n_true:,}")
    print(f"true pairs in candidate set : {n_found:,}")
    print(f"PAIR RECALL                 : {n_found / n_true:.2%}")
    print(f"MACRO RECALL (tracks F_0.5) : {per_entity['recall'].mean():.2%}  <- track this")
    print()
    print(f"entities with all matches found : {full:,} "
          f"({full / per_entity.height:.2%})  <- only these can score 1.0")
    print(f"entities with none found        : {none:,} "
          f"({none / per_entity.height:.2%})  <- these score a flat zero")
    print()
    print(f"candidate pairs emitted     : {total_cand:,}")
    print(f"mean candidates per entity  : {total_cand / max(s1_total, 1):.2f}")
    print(f"entities with no candidates : {no_candidates:,}")
    print(f"entities under budget ({budget})  : {under:,} "
          f"({under / max(s1_total, 1):.2%})  <- headroom a fuzzy stage could fill")
    full_cross = s1_total * targets
    if full_cross:
        print(f"reduction ratio             : {1 - total_cand / full_cross:.8f} "
              f"(of {full_cross:,} possible pairs)")

    # Converts a blocking change into F_0.5 terms before any model exists.
    # An upper bound and a loose one: it assumes a perfect classifier over the
    # candidates, and F is concave in recall so averaging recall flatters it.
    #
    # Singletons must be counted from the ground truth file, NOT from
    # `per_entity`. An entity with no true matches explodes to zero rows and is
    # absent from the join entirely, so `n_true == 0` never fires and counting
    # it there silently returns zero — which would understate the ceiling by
    # the full 5.58% singleton share.
    singles = int(
        pl.scan_csv(
            paths.ground_truth_tsv(data), separator="\t", quote_char=None,
            has_header=True,
            schema={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
        )
        .filter(
            pl.col("matched_entity_ids").is_null()
            | (pl.col("matched_entity_ids").str.strip_chars().str.len_chars() == 0)
        )
        .select(pl.len()).collect().item()
    )
    ceiling = (singles + full) / max(s1_total, 1)
    print(f"\ntrue singletons             : {singles:,} "
          f"({singles / max(s1_total, 1):.2%})  <- score 1.0 for an empty prediction")
    print(f"F_0.5 ceiling with a PERFECT classifier over these candidates: "
          f"{ceiling:.4f}")
    print("  (upper bound — assumes every covered entity scores exactly 1.0)")
    print("  note: the coverage percentages above are over NON-singleton "
          f"entities ({per_entity.height:,}), this ceiling is over all "
          f"{s1_total:,}")


# --------------------------------------------------------------------------
# where that recall comes from, key by key
# --------------------------------------------------------------------------

def budget_curve(
    split: str, work: Path, data: Path, cand_path: Path, cuts: tuple[int, ...]
) -> None:
    """Macro recall at several per-entity cuts of the SAME candidate file.

    This exists to replace a budget sweep, not to be one. Blocking already
    ranked each entity's candidates by `block_score`, so a smaller budget is
    just a `head(k)` on that ranking — every cut below the budget the file was
    written at can be read off one run, with no re-blocking. Only a cut ABOVE
    that budget needs another run, because those candidates were never written.

    What it answers: whether the budget is the binding constraint. If the
    reachable union from the per-key diagnostics is far above the delivered
    macro recall and this curve is still climbing at the budget, the cap is
    evicting true matches and the budget is too tight. If the curve has
    flattened, the budget is not what is costing recall and raising it only
    adds false positives for the scorer to reject.
    """
    truth = ground_truth_pairs(paths.ground_truth_tsv(data)).rename(
        {"matched_entity_id": "candidate_entity_id"}
    )
    # Re-sort rather than trusting the file's row order. It happens to be
    # written in per-entity score order, but relying on that silently couples
    # this diagnostic to an implementation detail of the merge.
    ranked = (
        pl.read_parquet(cand_path)
        .sort(
            ["block_score", "n_keys", "candidate_entity_id"],
            descending=[True, True, False],
        )
        .with_columns(
            pl.int_range(pl.len()).over("source1_entity_id").alias("rank")
        )
    )
    n_entities = int(ranked["source1_entity_id"].n_unique())
    _section("macro recall vs per-entity budget (same candidate file)")
    print(f"{'budget':>8} {'macro recall':>14} {'pair recall':>13} "
          f"{'candidates':>13} {'mean/entity':>12}")
    for k in cuts:
        cut = ranked.filter(pl.col("rank") < k)
        per_entity = (
            truth.join(
                cut.lazy().select("source1_entity_id", "candidate_entity_id")
                .with_columns(pl.lit(True).alias("found")),
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
        if per_entity.is_empty():
            continue
        macro = float(
            (per_entity["n_found"] / per_entity["n_true"]).mean()
        )
        pair = int(per_entity["n_found"].sum()) / int(per_entity["n_true"].sum())
        print(f"{k:>8} {macro:>13.2%} {pair:>12.2%} {cut.height:>13,} "
              f"{cut.height / max(n_entities, 1):>11.2f}")
    print("\n  still climbing at the largest cut -> the budget is evicting true")
    print("  matches and is worth raising. Flat -> the budget is not the")
    print("  constraint, and raising it only hands the scorer more negatives.")


def key_diagnostics(
    split: str, work: Path, data: Path, max_bucket: int | None,
    sample_pairs: int | None,
) -> None:
    caps = {k: (max_bucket or c) for k, _, c in blocking.KEY_SPECS}
    keys = list(caps)

    _section("per-key diagnostics")
    print("  caps: " + ", ".join(f"{k}={v:,}" for k, v in caps.items()))

    cols = ["entity_id", "country_norm", *keys]
    s1 = pl.scan_parquet(paths.normalized_parquet(split, 1, work)).select(cols)
    others = pl.concat(
        [
            pl.scan_parquet(paths.normalized_parquet(split, s, work)).select(cols)
            for s in (2, 3)
        ],
        how="vertical",
    )

    truth = ground_truth_pairs(paths.ground_truth_tsv(data))
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

    # Target-side bucket sizes, so a key that agrees can be told apart from one
    # that agrees AND would have survived the cap.
    for key in keys:
        sizes = (
            others.select(blocking.blocking_key(key))
            .group_by("bk").len().collect()
        )
        base = (
            base.with_columns(
                pl.concat_str([
                    pl.col("country_norm_r").fill_null(""),
                    pl.lit("|"),
                    pl.col(f"{key}_r"),
                ]).alias("bk")
            )
            .join(sizes, on="bk", how="left")
            .rename({"len": f"_size_{key}"})
            .drop("bk")
        )
        del sizes

    base = base.with_columns([
        (
            (pl.col(key) == pl.col(f"{key}_r"))
            & pl.col(key).is_not_null()
            & (pl.col(key).str.len_chars() > 0)
        ).alias(f"_agree_{key}")
        for key in keys
    ])
    base = base.with_columns([
        (
            pl.col(f"_agree_{key}")
            # A null size means the value was not in the target table at all,
            # which cannot happen for a pair whose right side came from it —
            # but fill defensively rather than letting a null silently count
            # as usable.
            & pl.col(f"_size_{key}").fill_null(10**9).le(caps[key])
        ).alias(f"_usable_{key}")
        for key in keys
    ])

    print(f"{'key':<14} {'agrees':>9} {'usable':>9} {'cap cost':>9} {'unique':>9}")
    for key in keys:
        agree = int(base[f"_agree_{key}"].sum())
        usable = int(base[f"_usable_{key}"].sum())
        others_usable = pl.any_horizontal(
            [pl.col(f"_usable_{k}") for k in keys if k != key]
        )
        unique = int(
            base.select((pl.col(f"_usable_{key}") & ~others_usable).sum()).item()
        )
        print(f"{key:<14} {agree / total:>8.2%} {usable / total:>8.2%} "
              f"{(agree - usable) / total:>8.2%} {unique / total:>8.2%}")

    union = int(base.select(
        pl.any_horizontal([pl.col(f"_usable_{k}") for k in keys]).sum()
    ).item())
    uncapped = int(base.select(
        pl.any_horizontal([pl.col(f"_agree_{k}") for k in keys]).sum()
    ).item())
    print(f"\nUNION (what stage A can reach) : {union / total:.2%}")
    print(f"UNION if buckets were unlimited: {uncapped / total:.2%}")
    print(f"  -> the bucket cap costs        {(uncapped - union) / total:.2%} "
          "of all true pairs")
    print(f"UNREACHABLE by any exact key   : {1 - uncapped / total:.2%} "
          "<- needs fuzzy matching; no rule, weight or cap change touches it")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate blocking recall.")
    p.add_argument("--split", default="train", choices=paths.SPLITS)
    p.add_argument("--work-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--budget", type=int, default=20,
                   help="Budget blocking was run with, for the shortfall count.")
    p.add_argument("--max-bucket", type=int, default=None,
                   help="Global cap override for the diagnostics. Leave unset "
                        "to use each key's own cap from KEY_SPECS.")
    p.add_argument("--sample-pairs", type=int, default=None,
                   help="Cap the true pairs used for per-key diagnostics. For "
                        "memory only — it biases nothing, but it does make the "
                        "percentages noisier.")
    p.add_argument("--no-key-diagnostics", action="store_true")
    p.add_argument("--budget-curve", default="1,2,3,5,8,12,16,20,30",
                   help="Per-entity cuts to report macro recall at, read off "
                        "the SAME candidate file. Cuts above the budget "
                        "blocking ran with are silently unreachable. Pass an "
                        "empty string to skip.")
    p.add_argument("--no-candidate-recall", action="store_true",
                   help="Skip the emitted-candidate section, so the per-key "
                        "numbers can be read without running blocking first.")
    p.add_argument("--candidates", default=None,
                   help="Candidate parquet to score. Defaults to the stage A "
                        "output for this split.")
    a = p.parse_args()

    work = paths.work_root(a.work_root)
    data = paths.data_root(a.data_root)

    if a.split != "train":
        p.error("blocking can only be evaluated on train — there is no test "
                "ground truth.")

    for s in (1, 2, 3):
        if not paths.normalized_parquet(a.split, s, work).is_file():
            p.error(f"{paths.normalized_parquet(a.split, s, work)} missing — run "
                    f"`python -m src.main --split {a.split} --normalize-only` first.")

    if not a.no_candidate_recall:
        cand_path = (
            Path(a.candidates).expanduser() if a.candidates
            else paths.candidates_parquet(a.split, work)
        )
        if not cand_path.is_file():
            p.error(f"{cand_path} missing — run "
                    f"`python -m src.blocking --split {a.split}` first, or pass "
                    "--no-candidate-recall for the per-key numbers alone.")
        candidate_recall(a.split, work, data, cand_path, a.budget)
        cuts = tuple(
            sorted({int(c) for c in a.budget_curve.split(",") if c.strip()})
        )
        if cuts:
            budget_curve(a.split, work, data, cand_path, cuts)

    if not a.no_key_diagnostics:
        key_diagnostics(a.split, work, data, a.max_bucket, a.sample_pairs)


if __name__ == "__main__":
    main()
