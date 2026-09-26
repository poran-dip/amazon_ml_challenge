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
        # Filter on emptiness, not just nullness. A singleton's match list is
        # an empty list, and `explode` currently turns that into one NULL row —
        # but polars warns that in 2.0 `empty_as_null` defaults to False, at
        # which point an `is_not_null()` filter alone would let a phantom
        # empty-string "match" through for every singleton. That would inflate
        # the true-pair count by the full 123,247 singletons and make every
        # recall number here quietly wrong. Checking length covers both.
        .filter(
            pl.col("matched_entity_ids").is_not_null()
            & (pl.col("matched_entity_ids").str.len_chars() > 0)
        )
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

    # The cap is READ OFF THE FILE, not taken from the flag.
    #
    # `--budget` here defaulted to 20 and was used as the threshold whatever
    # the file was actually built at, so evaluating a budget-60 file without
    # repeating the flag printed "entities under budget (20): 66.39%" and
    # labelled it "headroom a fuzzy stage could fill". Both halves are wrong:
    # the threshold is not the file's cap, and being under it is not headroom.
    # It invites "the budget is not binding", which is the session-4 failure
    # verbatim, from a different direction.
    #
    # No entity can hold more candidates than the budget blocking ran at, so
    # the largest per-entity count IS that budget (on any split with at least
    # one saturated entity, which is every real run so far).
    observed = int(sizes["n"].max()) if sizes.height else 0
    at_cap = int((sizes["n"] == observed).sum()) if observed else 0
    # What those entities are missing is the upper bound on everything a
    # larger budget can recover — the number session 4 said should be read and
    # that this function never actually computed.
    capped_ids = sizes.filter(pl.col("n") == observed).select("source1_entity_id")
    missed_at_cap = int(
        per_entity.join(capped_ids, on="source1_entity_id", how="inner")
        .select((pl.col("n_true") - pl.col("n_found")).sum())
        .item() or 0
    ) if observed else 0
    under = int((sizes["n"] < observed).sum()) + no_candidates

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
    if budget != observed:
        print(f"  NOTE: --budget said {budget}, but this file's largest entity "
              f"holds {observed}. Reporting against {observed}, which is what "
              "blocking actually ran at.")
    print(f"entities AT the cap ({observed})     : {at_cap:,} "
          f"({at_cap / max(s1_total, 1):.2%})  <- only these can lose a true "
          "match to the budget")
    print(f"  true matches they are missing : {missed_at_cap:,} "
          f"({missed_at_cap / max(n_true, 1):.2%} of all true pairs)  <- the "
          "MOST a larger budget can recover")
    print(f"entities below the cap      : {under:,} "
          f"({under / max(s1_total, 1):.2%})  <- a larger budget cannot help "
          "these; only more keys can")
    full_cross = s1_total * targets
    if full_cross:
        print(f"reduction ratio             : {1 - total_cand / full_cross:.8f} "
              f"(of {full_cross:,} possible pairs)")

    # Converts a blocking change into F_0.5 terms before any model exists.
    #
    # Computed PER ENTITY, which matters. The first version of this counted
    # only fully covered entities and singletons, giving every partially
    # covered entity a flat zero — and reported 0.3044 as the "ceiling" for a
    # candidate set whose pipeline already scored 0.6070. A ceiling below the
    # achieved score is a bug report, not a number. An entity with 3 of its 5
    # true matches found scores F_0.5 = 0.68 with a perfect classifier, not 0.
    #
    # With perfect precision, F_0.5 = 1.25r / (0.25 + r). Evaluating that per
    # entity and then averaging is also what avoids the Jensen error in the
    # other direction: F is concave in r, so plugging the MEAN recall into it
    # overstates the mean F (0.8871 against the true 0.7943 on the first real
    # run — a 9-point difference, larger than most changes being considered).
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
    r = pl.col("n_found") / pl.col("n_true")
    sum_f = float(
        per_entity.select(
            pl.when(pl.col("n_found") == 0)
            .then(0.0)
            .otherwise(1.25 * r / (0.25 + r))
            .sum()
        ).item()
    )
    ceiling = (singles * 1.0 + sum_f) / max(s1_total, 1)
    print(f"\ntrue singletons             : {singles:,} "
          f"({singles / max(s1_total, 1):.2%})  <- score 1.0 for an empty prediction")
    print(f"F_0.5 CEILING over these candidates : {ceiling:.4f}")
    print("  assumes a classifier that keeps exactly the true matches and")
    print("  nothing else, so it bounds what any scorer can reach WITHOUT")
    print("  better blocking. Compare it to the end-to-end F_0.5 to see")
    print("  whether the next gain is in blocking or in scoring.")
    print("  note: the coverage percentages above are over NON-singleton "
          f"entities ({per_entity.height:,}); this ceiling is over all "
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
    observed: list[tuple[int, float]] = []
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
        observed.append((k, macro))
        print(f"{k:>8} {macro:>13.2%} {pair:>12.2%} {cut.height:>13,} "
              f"{cut.height / max(n_entities, 1):>11.2f}")

    # Read the curve rather than printing advice for both cases and leaving the
    # reader to pick. The first version printed "still climbing -> raise the
    # budget" unconditionally, directly above a table that had gone flat, which
    # is documentation contradicting its own measurement.
    if len(observed) < 2:
        return
    (k0, m0), (k1, m1) = observed[-2], observed[-1]
    gain = m1 - m0
    if gain < 0.002:
        print(f"\n  FLAT between {k0} and {k1} (+{gain:.2%}). The budget is NOT the")
        print("  constraint — raising it adds nothing but negatives for the")
        print("  scorer to reject. Recall is limited by what the keys reach.")
    else:
        print(f"\n  STILL CLIMBING between {k0} and {k1} (+{gain:.2%}). The budget is")
        print("  evicting true matches; re-run blocking with a larger --budget")
        print("  (cuts above the budget cannot be read off this file).")


def key_diagnostics(
    split: str, work: Path, data: Path, max_bucket: int | None,
    sample_pairs: int | None, acronym_cap: int | None = None,
    band_cap: int | None = None,
) -> None:
    # The caps must be the ones BLOCKING ran with, not the KEY_SPECS defaults.
    # This function used to read the defaults only, so `--acronym-cap 500` (and
    # later `--band-cap`) changed the candidate file while `usable`, `cap cost`
    # and both UNION lines went on describing a run that never happened.
    # Measured: a 40x sweep of --band-cap moved macro recall from 81.48% to
    # 77.29% while `the bucket cap costs` sat at 1.25% for every point -- a
    # number that cannot move is not a measurement, which is the session-4
    # budget-curve lesson in a third place.
    def _cap(key: str, default: int) -> int:
        if max_bucket:
            return max_bucket
        if key == "name_acronym" and acronym_cap:
            return acronym_cap
        if blocking.is_band(key) and band_cap:
            return band_cap
        return default

    caps = {k: _cap(k, c) for k, _, c in blocking.KEY_SPECS}
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

    def _agrees(key: str) -> pl.Expr:
        """Both sides present and equal.

        The emptiness test is not cosmetic and is the same one `_usable`
        makes: two records that both normalize to an empty name would
        otherwise "agree" on every name key at once and be counted as reach
        this stage does not have. For a band key the analogue is nullness — an
        empty trigram list gives a null signature — and `str.len_chars()` on a
        UInt64 raises rather than returning something wrong, which is how this
        was found.
        """
        eq = (pl.col(key) == pl.col(f"{key}_r")) & pl.col(key).is_not_null()
        if blocking.is_band(key):
            return eq & pl.col(f"{key}_r").is_not_null()
        return eq & (pl.col(key).str.len_chars() > 0)

    # Target-side bucket sizes, so a key that agrees can be told apart from one
    # that agrees AND would have survived the cap.
    #
    # Each key is reduced to TWO booleans and its inputs dropped before the
    # next one is joined. That matters now in a way it did not at five keys:
    # at twenty-one, holding every key's value for both sides plus every
    # bucket size over 7.6M true pairs is several GB, and it would be spent at
    # the end of a run that has already paid for blocking. The journal's
    # "merge would have failed at the end of the longest job" entry is the same
    # shape — peak memory in a step that runs last. Dropping as we go keeps
    # this at 2 booleans per key regardless of how many keys there are.
    for key in keys:
        sizes = (
            others.select(blocking.blocking_key(key))
            .group_by("bk").len().collect()
        )
        # Band signatures are UInt64 and `concat_str` will not take an integer
        # column, so the right-hand value needs the same cast
        # `blocking.blocking_key` applies on the left. Without it this joins
        # nothing and every band reads as 0.00% usable — a diagnostic that
        # fails silently in the direction of "the new keys do nothing", which
        # is precisely the conclusion it would be used to draw.
        value_r = pl.col(f"{key}_r")
        if blocking.is_band(key):
            value_r = value_r.cast(pl.Utf8)
        base = (
            base.with_columns(
                pl.concat_str([
                    pl.col("country_norm_r").fill_null(""),
                    pl.lit("|"),
                    value_r,
                ]).alias("bk")
            )
            .join(sizes, on="bk", how="left")
            .rename({"len": f"_size_{key}"})
            .drop("bk")
            .with_columns(_agrees(key).alias(f"_agree_{key}"))
            .with_columns(
                (
                    pl.col(f"_agree_{key}")
                    # A null size means the value was not in the target table
                    # at all, which cannot happen for a pair whose right side
                    # came from it — but fill defensively rather than letting a
                    # null silently count as usable.
                    & pl.col(f"_size_{key}").fill_null(10**9).le(caps[key])
                ).alias(f"_usable_{key}")
            )
            # The two booleans are all that is read from here on, so the key's
            # own columns and its bucket size go now rather than at the end.
            .drop(f"_size_{key}", key, f"{key}_r")
        )
        del sizes

    def _any(cols: list[str]) -> int:
        if not cols:
            return 0
        return int(base.select(pl.any_horizontal([pl.col(c) for c in cols]).sum()).item())

    exact = [k for k in keys if not blocking.is_band(k)]
    bands = [k for k in keys if blocking.is_band(k)]

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
        # A per-band "unique" is nearly always ~0 and says nothing: the bands
        # are deliberately redundant with EACH OTHER, since a similar pair is
        # meant to agree on several. The number that matters is what the bands
        # add as a GROUP, printed below. Without this note the sixteen zeroes
        # read as "the bands contribute nothing", which is the same misreading
        # the budget curve's guaranteed-flat rows produced in session 4.

    union = _any([f"_usable_{k}" for k in keys])
    uncapped = _any([f"_agree_{k}" for k in keys])
    ex_union = _any([f"_usable_{k}" for k in exact])
    bd_union = _any([f"_usable_{k}" for k in bands])
    # What each group reaches that the OTHER group does not. This is the pair
    # of numbers the fuzzy stage lives or dies by, and neither is visible in
    # the per-key table above.
    ex_only = int(base.select((
        pl.any_horizontal([pl.col(f"_usable_{k}") for k in exact])
        & ~pl.any_horizontal([pl.col(f"_usable_{k}") for k in bands])
    ).sum()).item()) if exact and bands else 0
    bd_only = int(base.select((
        pl.any_horizontal([pl.col(f"_usable_{k}") for k in bands])
        & ~pl.any_horizontal([pl.col(f"_usable_{k}") for k in exact])
    ).sum()).item()) if exact and bands else 0

    print(f"\nUNION (what stage A can reach) : {union / total:.2%}")
    print(f"UNION if buckets were unlimited: {uncapped / total:.2%}")
    print(f"  -> the bucket cap costs        {(uncapped - union) / total:.2%} "
          "of all true pairs")
    if bands:
        print(f"\n  exact keys alone    : {ex_union / total:>7.2%}   "
              "<- the ceiling every number in findings.md was measured against")
        print(f"  LSH bands alone     : {bd_union / total:>7.2%}")
        print(f"  bands add uniquely  : {bd_only / total:>7.2%}   "
              "<- true pairs NO exact key reaches at any cap")
        print(f"  exact adds uniquely : {ex_only / total:>7.2%}   "
              "<- why the exact keys stay")
        print(f"\nUNREACHABLE by ANY key         : {1 - uncapped / total:.2%}")
    else:
        print(f"UNREACHABLE by any exact key   : {1 - uncapped / total:.2%} "
              "<- needs fuzzy matching; run without --bands 0")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate blocking recall.")
    p.add_argument("--split", default="train", choices=paths.SPLITS)
    p.add_argument("--work-root", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--budget", type=int, default=20,
                   help="Budget blocking was run with, for the shortfall count.")
    p.add_argument(
        "--acronym-cap", type=int, default=None,
        help="The --acronym-cap BLOCKING was run with. The per-key table is "
        "recomputed from the normalized parquet, so it has no way to know "
        "which caps produced the candidate file and will describe the "
        "defaults unless told.",
    )
    p.add_argument(
        "--band-cap", type=int, default=None,
        help="The --band-cap BLOCKING was run with. See --acronym-cap.",
    )
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
        key_diagnostics(a.split, work, data, a.max_bucket, a.sample_pairs,
                    a.acronym_cap, a.band_cap)


if __name__ == "__main__":
    main()
