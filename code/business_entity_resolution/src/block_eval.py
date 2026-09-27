"""Scoring the blocking stage against ground truth.

Blocking sets a hard ceiling on everything downstream: a true pair that never
becomes a candidate cannot be recovered by any model, threshold or feature. So
this is the number to move before any of those are worth touching, and the
per-key diagnostics are what make a blocking change judgeable rather than a
guess.

Three sections, and they answer different questions.

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

**Missed pairs** starts from the true pairs that are NOT in the candidate set
and attributes each one to the stage that lost it, which the other two
sections cannot do: the first says how many were lost, the second says what
each key reaches against the true pairs rather than against the emitted file.
A pair can be `usable` there and still be absent here, evicted by the budget.

  * *country-mismatch* — every key is `<country>|value`, so no key can fire.
  * *no-key-agrees* — every key disagrees. Only a NEW key helps, and the
    similarity breakdown under it says which kind: a miss at high trigram
    Jaccard is one the LSH bands are supposed to reach, a miss at high
    containment wants a prefix or containment key, and a miss at J < 0.2
    shares almost no characters with its match at all.
  * *bucket-cap* — a key agreed and its target bucket was over cap.
  * *band-gate* — band-only evidence, below `--band-min-hits`.
  * *budget-evicted* — generated, ranked, and cut by `--budget`.
  * *unexplained* — reachable, for an entity that never filled its budget.
    Nothing in blocking drops those, so a non-zero count means the flags
    passed here are not the ones that built the candidate file. It is printed
    loudly rather than folded into `budget-evicted`, where it would look like
    a reason to raise the budget.

It then prints `--show-misses` of them in full — by default the
`no-key-agrees` ones, since every other reason is already actionable from the
table and that one is not. Each shows the raw AND normalized name and address
of both sides, the trigram Jaccard and containment, and what blocking returned
for that entity instead. `--miss-reason` picks a different bucket.

Usage:

    python -m src.block_eval --split train
    python -m src.block_eval --split train --no-key-diagnostics

    # the flags BLOCKING ran with have to be repeated, or the attribution
    # describes a run that never happened
    python -m src.block_eval --split train --budget 60 --band-cap 100 \
        --band-min-hits 2 --show-misses 20

    # the near-misses alone, or a different bucket
    python -m src.block_eval --split train --miss-order most-similar
    python -m src.block_eval --split train --miss-reason bucket-cap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from . import blocking, paths, progress


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

    # Processed ONE MERGE BUCKET AT A TIME, and the reason is the same one the
    # merge itself is bucketed for.
    #
    # This used to `pl.read_parquet` the whole candidate file eagerly and take
    # a GLOBAL SORT over it. At four exact keys and budget 20 that is 23.6M
    # rows and merely slow. With the LSH bands at budget 60 it is ~132M rows
    # and a global sort of them, which is the "concatenating every partial and
    # taking a global sort over ~100M rows wants 8-12GB" failure recorded in
    # the journal -- fixed in `_merge` in session 3 and sitting here unnoticed
    # ever since, in a diagnostic that runs on every single invocation.
    #
    # The sort is only ever needed per entity, and an entity's candidates are
    # entirely inside one bucket by construction, so bucketing makes the
    # ranking exact rather than approximate. Nothing is traded away.
    #
    # What accumulates across buckets is only the RANK OF EACH TRUE MATCH --
    # about 4.9M rows of two columns at present -- not the candidate set.
    cols = set(pl.scan_parquet(cand_path).collect_schema().names())
    if "bucket" in cols:
        bucket_ids = sorted(
            pl.scan_parquet(cand_path).select("bucket").unique()
            .collect().to_series().to_list()
        )
        def _read(b):
            return pl.scan_parquet(cand_path).filter(pl.col("bucket") == b).collect()
    else:
        # A candidate file written before blocking recorded the column. Fall
        # back to the old whole-file path rather than silently mis-ranking,
        # but say so, because on a large file it is the expensive route.
        print("  (candidate file predates the `bucket` column — reading it "
              "whole, which needs several GB at this size)")
        bucket_ids = [None]
        def _read(_):
            return pl.read_parquet(cand_path)

    matched_parts = []
    n_entities = 0
    total_rows = 0
    truth_eager = truth.collect()
    for b in bucket_ids:
        part = _read(b)
        if part.is_empty():
            continue
        total_rows += part.height
        n_entities += int(part["source1_entity_id"].n_unique())
        ranked_b = part.sort(
            ["block_score", "n_keys", "candidate_entity_id"],
            descending=[True, True, False],
        ).with_columns(
            pl.int_range(pl.len()).over("source1_entity_id").alias("rank")
        ).select("source1_entity_id", "candidate_entity_id", "rank")
        matched_parts.append(
            ranked_b.join(
                truth_eager, on=["source1_entity_id", "candidate_entity_id"],
                how="inner",
            ).select("source1_entity_id", "rank")
        )
        del part, ranked_b
    matched = (
        pl.concat(matched_parts, how="vertical") if matched_parts
        else pl.DataFrame(schema={"source1_entity_id": pl.Utf8, "rank": pl.Int64})
    )
    del matched_parts
    # Denominator: every entity that has a true match, whether or not blocking
    # found any of them. Taken from the ground truth rather than the candidate
    # file, because an entity with zero candidates must count as zero recall
    # and is absent from the candidate file entirely.
    n_true_per_entity = (
        truth_eager.group_by("source1_entity_id").len().rename({"len": "n_true"})
    )

    n_true_total = int(n_true_per_entity["n_true"].sum())
    # Per-entity candidate counts, once, so the "candidates" column at each cut
    # is sum(min(count, k)) rather than the height of a materialized cut.
    ent_sizes = (
        pl.scan_parquet(cand_path).group_by("source1_entity_id").len()
        .select(pl.col("len")).collect().to_series()
    )

    # The curve is clamped to what this FILE can answer, and always includes the
    # cap it was written at.
    #
    # Neither was true before, and the combination printed advice that
    # contradicted the data. The default cuts stop at 30; a budget-60 file was
    # therefore read only up to 30, the last two points still differed, and the
    # footer said "still climbing -- re-run blocking with a larger --budget"
    # about a run that had ALREADY been done at 60. Cuts at or below the file's
    # cap need no re-blocking, so the cap belongs in the curve. Cuts above it
    # are unanswerable from this file and used to be printed anyway, silently
    # repeating the cap's own row under a larger label.
    file_cap = int(ent_sizes.max() or 0)
    if file_cap:
        asked = tuple(cuts)
        cuts = tuple(sorted({c for c in asked if 0 < c <= file_cap} | {file_cap}))
        dropped = [c for c in asked if c > file_cap]
        if dropped:
            print(f"\n  NOTE: cuts {dropped} exceed this file's cap ({file_cap}) "
                  "and cannot be read off it — they need another blocking run. "
                  "Dropped.")

    _section("macro recall vs per-entity budget (same candidate file)")
    print(f"{'budget':>8} {'macro recall':>14} {'pair recall':>13} "
          f"{'candidates':>13} {'mean/entity':>12}")
    observed: list[tuple[int, float]] = []
    for k in cuts:
        found = (
            matched.filter(pl.col("rank") < k)
            .group_by("source1_entity_id").len().rename({"len": "n_found"})
        )
        per_entity = n_true_per_entity.join(
            found, on="source1_entity_id", how="left"
        ).with_columns(pl.col("n_found").fill_null(0))
        if per_entity.is_empty():
            continue
        # Candidate count at this cut without materializing the cut: every
        # entity contributes min(its own count, k).
        n_cut = int(ent_sizes.clip(upper_bound=k).sum())
        macro = float((per_entity["n_found"] / per_entity["n_true"]).mean())
        pair = int(per_entity["n_found"].sum()) / max(n_true_total, 1)
        observed.append((k, macro))
        print(f"{k:>8} {macro:>13.2%} {pair:>12.2%} {n_cut:>13,} "
              f"{n_cut / max(n_entities, 1):>11.2f}")

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
        print("  evicting true matches, so a larger --budget would recover some")
        print("  (cuts above this file's cap need another blocking run).")
        print("  Weigh it against `mean candidates per entity`, which is now a")
        print("  SCORED quantity: a smaller candidate set per entity ranks higher.")


def resolve_caps(
    max_bucket: int | None, acronym_cap: int | None, band_cap: int | None,
) -> dict[str, int]:
    """Every key's effective bucket cap, as BLOCKING would have applied it.

    Split out of `key_diagnostics` so the miss report cannot drift from it.
    Two functions computing "the cap" from the same flags by two copies of the
    same three-branch rule is exactly how `usable` came to describe a run that
    never happened, and the second copy would be the one nobody re-checks.
    """
    def _cap(key: str, default: int) -> int:
        if max_bucket:
            return max_bucket
        if key == "name_acronym" and acronym_cap:
            return acronym_cap
        if blocking.is_band(key) and band_cap:
            return band_cap
        return default

    return {k: _cap(k, c) for k, _, c in blocking.KEY_SPECS}


def annotate_agreement(
    base: pl.DataFrame, others: pl.LazyFrame, keys: list[str],
    caps: dict[str, int], min_lens: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Add `_agree_<key>` and `_usable_<key>` for every key, to a pair frame.

    `base` holds one row per pair with each key's value on the left under its
    own name and on the right under `<key>_r`, plus `country_norm_r`. Both
    `key_diagnostics` and `miss_report` need exactly this and they need it to
    mean the same thing, so it lives in one place.

    Each key is reduced to TWO booleans and its inputs dropped before the next
    one is joined. That matters at twenty-one keys: holding every key's value
    for both sides plus every bucket size over millions of pairs is several GB,
    and it would be spent at the end of a run that has already paid for
    blocking. The journal's "merge would have failed at the end of the longest
    job" entry is the same shape -- peak memory in a step that runs last.

    A key's own column and its `_r` twin are consumed here. A caller that needs
    a key's VALUE afterwards (the miss report needs `name_sorted` to measure
    how close a missed pair was) must copy it under another name first.
    """
    min_lens = min_lens or {}

    def _agrees(key: str) -> pl.Expr:
        """Both sides present and equal.

        The emptiness test is not cosmetic and is the same one the usability
        test makes: two records that both normalize to an empty name would
        otherwise "agree" on every name key at once and be counted as reach
        this stage does not have. For a band key the analogue is nullness — an
        empty trigram list gives a null signature — and `str.len_chars()` on a
        UInt64 raises rather than returning something wrong, which is how this
        was found.
        """
        eq = (pl.col(key) == pl.col(f"{key}_r")) & pl.col(key).is_not_null()
        if blocking.is_band(key):
            return eq & pl.col(f"{key}_r").is_not_null()
        min_len = min_lens.get(key, 0)
        ok = eq & (pl.col(key).str.len_chars() > 0)
        if min_len > 1:
            # `min_len` drops a value before blocking ever buckets it, on BOTH
            # sides — `_s1_for_key` and `_target_table` each filter with
            # `_usable(key, min_len)`. So a short acronym is not "agrees but
            # capped", it is a key that never fired at all, and counting it as
            # agreement would attribute a miss to the cap that the length rule
            # caused.
            ok = ok & (pl.col(f"{key}_r").str.len_chars() >= min_len)
        return ok

    for key in keys:
        min_len = min_lens.get(key, 0)
        # Target-side bucket sizes, so a key that agrees can be told apart from
        # one that agrees AND would have survived the cap. Filtered with
        # `blocking._usable` first, because `_target_table` filters before it
        # counts: without it a `min_len` run's buckets are measured over rows
        # blocking had already discarded.
        sizes = (
            others.filter(blocking._usable(key, min_len))
            .select(blocking.blocking_key(key))
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
    return base


def key_diagnostics(
    split: str, work: Path, data: Path, max_bucket: int | None,
    sample_pairs: int | None, acronym_cap: int | None = None,
    band_cap: int | None = None, acronym_min_len: int = 0,
) -> None:
    # The caps must be the ones BLOCKING ran with, not the KEY_SPECS defaults.
    # This function used to read the defaults only, so `--acronym-cap 500` (and
    # later `--band-cap`) changed the candidate file while `usable`, `cap cost`
    # and both UNION lines went on describing a run that never happened.
    # Measured: a 40x sweep of --band-cap moved macro recall from 81.48% to
    # 77.29% while `the bucket cap costs` sat at 1.25% for every point -- a
    # number that cannot move is not a measurement, which is the session-4
    # budget-curve lesson in a third place.
    caps = resolve_caps(max_bucket, acronym_cap, band_cap)
    keys = list(caps)
    min_lens = {"name_acronym": acronym_min_len} if acronym_min_len > 1 else {}

    _section("per-key diagnostics")
    print("  caps: " + ", ".join(f"{k}={v:,}" for k, v in caps.items()))
    if min_lens:
        print(f"  name_acronym min length: {acronym_min_len}")

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

    # Per-key agreement and usability, shared with the miss report so the two
    # cannot disagree about what "usable" means.
    base = annotate_agreement(base, others, keys, caps, min_lens)

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


# --------------------------------------------------------------------------
# which true pairs were missed, and why
# --------------------------------------------------------------------------

# Why a true pair can be absent from the candidate set. Ordered by how early
# in the pipeline the pair was lost, and evaluated in that order, because the
# reasons are not exclusive: a pair whose countries disagree also agrees on no
# key, and reporting it as "no key agrees" would send the reader looking for a
# better name key when no name key could ever have fired.
#
# The ordering is also the fix ordering. Everything above `budget` needs a
# change to blocking's reach; `budget` alone is a number.
# Trigram-Jaccard strata. ONE definition, used both by the similarity
# breakdown under the reason table and by `--miss-order stratified`, so the
# bands a reader is shown a cross-section of are the bands they were just
# given counts for. Two copies would drift, and the drift would be invisible:
# both would still print four plausible rows.
SIMILARITY_BANDS: tuple[tuple[float, float], ...] = (
    (0.8, 1.01), (0.5, 0.8), (0.2, 0.5), (0.0, 0.2),
)

MISS_REASONS: tuple[str, ...] = (
    "country-mismatch",   # keys are country-partitioned, so nothing can fire
    "no-key-agrees",      # every key disagrees: only a NEW key helps
    "bucket-cap",         # a key agreed, its target bucket was over cap
    "band-gate",          # band-only evidence, below --band-min-hits
    "budget-evicted",     # generated and admitted, then cut by --budget
    "unexplained",        # reachable, entity not at the cap: see the note below
)


def _raw_lookup(split: str, data: Path, sources: tuple[int, ...],
                ids: list[str]) -> dict[str, tuple[str, str, str]]:
    """{entity_id: (name, address, country)} for a handful of ids.

    Read from the source TSVs rather than the normalized parquet, which keeps
    only normalized keys — the point of an example is to see the string a
    human would see. `quote_char=None` for the reason `main.scan_source` gives:
    addresses contain stray quotes and letting polars interpret them merges
    rows.
    """
    if not ids:
        return {}
    wanted = list(dict.fromkeys(ids))
    frames = [
        pl.scan_csv(
            paths.source_tsv(split, s, data), separator="\t", quote_char=None,
            has_header=True,
            schema={c: pl.Utf8 for c in paths.SOURCE_COLUMNS},
        ).filter(pl.col("entity_id").is_in(wanted))
        for s in sources
    ]
    df = pl.concat(frames, how="vertical").collect()
    return {
        r["entity_id"]: (
            r["business_name"] or "", r["business_address"] or "",
            r["country"] or "",
        )
        for r in df.iter_rows(named=True)
    }


def _jaccard_pair(a: str, b: str) -> pl.Expr:
    """Jaccard of two deduplicated list columns.

    Union as `|a| + |b| - |a and b|` rather than a second set pass, the same
    identity `features._jaccard_and_containment` uses and valid for the same
    reason: `text_norm.trigram_expression` emits a UNIQUE list.
    """
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(a).list.len() + pl.col(b).list.len() - inter
    return pl.when(union == 0).then(0.0).otherwise(inter / union)


def _containment_pair(a: str, b: str) -> pl.Expr:
    """Overlap over the SMALLER of two deduplicated list columns.

    Jaccard and containment disagree in exactly the case a missed pair is most
    likely to be: `acme` against `acme foods private limited` has a low
    Jaccard because one set is much larger, and a containment of 1.0. See
    `features._jaccard_and_containment`, which makes the same distinction for
    the model.
    """
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    smaller = pl.min_horizontal(pl.col(a).list.len(), pl.col(b).list.len())
    return pl.when(smaller == 0).then(0.0).otherwise(inter / smaller)


def miss_report(
    split: str, work: Path, data: Path, cand_path: Path,
    max_bucket: int | None, acronym_cap: int | None, band_cap: int | None,
    acronym_min_len: int, band_min_hits: int, keys: tuple[str, ...] | None,
    n_examples: int, order: str, seed: int, sample_pairs: int | None = None,
    reason_filter: str = "no-key-agrees",
) -> None:
    """Classify every missed true pair, then print `n_examples` of them.

    The per-key table says what each key reaches. It cannot say what happened
    to the pairs that got away, because it is computed against the true pairs
    rather than against the emitted file — a pair can be `usable` there and
    still be absent from the candidate set, evicted by the per-entity budget.
    This function starts from the MISSES themselves and attributes each one to
    the stage that lost it, so "raise the budget", "loosen a cap" and "the keys
    do not reach these at all" stop being three guesses about one number.

    The caps, the length rule and the band gate must be the ones BLOCKING ran
    with, for the reason recorded above `key_diagnostics`: recomputed from the
    normalized parquet, this has no way to know what produced the candidate
    file and will describe the defaults unless told.
    """
    selected = list(keys) if keys is not None else list(blocking.default_keys(None))
    caps = resolve_caps(max_bucket, acronym_cap, band_cap)
    caps = {k: v for k, v in caps.items() if k in selected}
    min_lens = {"name_acronym": acronym_min_len} if acronym_min_len > 1 else {}

    truth = ground_truth_pairs(paths.ground_truth_tsv(data)).rename(
        {"matched_entity_id": "candidate_entity_id"}
    )
    cand = pl.scan_parquet(cand_path).select(
        "source1_entity_id", "candidate_entity_id"
    )
    # An anti-join, not a left join with a flag. The misses are the small side
    # here (a few percent of 7.6M against ~44M candidates) and they are all
    # this function reads; materializing the found ones too would be the
    # larger frame, held for nothing.
    missed = truth.join(
        cand, on=["source1_entity_id", "candidate_entity_id"], how="anti"
    ).collect()

    _section(f"missed true pairs [{split}] — {cand_path.name}")
    n_true = int(truth.select(pl.len()).collect().item())
    if missed.is_empty():
        print(f"  no misses: all {n_true:,} true pairs are in the candidate set.")
        return
    n_missed_total = missed.height
    print(f"true pairs                  : {n_true:,}")
    print(f"MISSED                      : {n_missed_total:,} "
          f"({n_missed_total / n_true:.2%})")
    print(f"  keys assumed for attribution: {len(selected)} "
          f"({'default set' if keys is None else 'from --keys/--bands'})")
    # Every missed pair carries both sides' trigram lists through two joins,
    # which is the memory shape of this section. At ~1.5M misses that is a few
    # hundred MB and fine; `--sample-pairs` is the escape hatch if the miss set
    # is larger than expected. SAMPLED, not `head`, unlike the per-key
    # diagnostics: the misses arrive in candidate-file order, which is bucket
    # order, which is a hash of the entity id — taking a prefix of that would
    # be a prefix of a hash partition rather than of nothing in particular.
    if sample_pairs and missed.height > sample_pairs:
        missed = missed.sample(n=sample_pairs, seed=seed, shuffle=True)
        print(f"  --sample-pairs: classifying a random {sample_pairs:,} of "
              "them; the percentages below are over the sample.")

    # The extra columns the sub-diagnosis reads. `name_trigrams` is not a
    # blocking key, so `annotate_agreement` leaves it alone; `name_sorted` and
    # `addr_sorted` are, and get consumed, so both are copied before the pass.
    extra = ["name_trigrams"]
    # Both are read as VALUES as well as being keys, and a `--keys` subset that
    # leaves one out would otherwise make this section fail on a column
    # blocking never needed. Requested explicitly and deduplicated, rather than
    # assumed to be in `selected`.
    cols = list(dict.fromkeys(
        ["entity_id", "country_norm", *selected, "name_sorted", "addr_sorted",
         *extra]
    ))
    s1 = pl.scan_parquet(paths.normalized_parquet(split, 1, work)).select(cols)
    others = pl.concat(
        [
            pl.scan_parquet(paths.normalized_parquet(split, s, work)).select(cols)
            for s in (2, 3)
        ],
        how="vertical",
    )
    base = (
        missed.lazy()
        .join(s1, left_on="source1_entity_id", right_on="entity_id", how="inner")
        # Copied BEFORE the agreement pass, which consumes every key column.
        # The NORMALIZED forms are what the examples have to show next to the
        # raw strings: when no key agrees, the question is always whether the
        # two records are genuinely different or whether normalization pulled
        # them apart, and only the normalized pair answers it.
        .with_columns(
            pl.col("name_sorted").alias("_ns"),
            pl.col("addr_sorted").alias("_as"),
        )
        .join(others, left_on="candidate_entity_id", right_on="entity_id",
              how="inner", suffix="_r")
        .with_columns(
            pl.col("name_sorted_r").alias("_ns_r"),
            pl.col("addr_sorted_r").alias("_as_r"),
        )
        .collect()
    )
    if base.is_empty():
        print("  none of the missed pairs resolved against the normalized "
              "parquet — check that it matches the candidate file's run.")
        return
    # Both joins are inner, so a missed pair whose either side is absent from
    # the normalized parquet drops out here. Under `--sample` that is expected
    # and large; on a full run it should be zero, and it has to be SAID either
    # way — every percentage below is over `base`, not over the misses, and a
    # silently smaller denominator is how a partial view gets read as a
    # complete one.
    if base.height != missed.height:
        lost = missed.height - base.height
        print(f"  NOTE: {lost:,} missed pair(s) ({lost / missed.height:.2%}) "
              "have a side that is not in the normalized parquet and are "
              "excluded from the attribution below. On a full run this should "
              "be 0 — if it is not, the parquet and the candidate file are "
              "from different runs.")

    # Similarity of the two names, measured BEFORE the key columns are
    # consumed. `tri_jac` is the number the LSH bands are a sampler of, so a
    # miss with a high `tri_jac` is a band configuration failure rather than a
    # hard pair — the distinction the aggregate below exists to make.
    base = base.with_columns(
        _jaccard_pair("name_trigrams", "name_trigrams_r").cast(pl.Float32)
        .alias("tri_jac"),
        _containment_pair("name_trigrams", "name_trigrams_r").cast(pl.Float32)
        .alias("tri_cont"),
        (
            pl.col("country_norm").fill_null("")
            == pl.col("country_norm_r").fill_null("")
        ).alias("_country_ok"),
    ).drop("name_trigrams", "name_trigrams_r")

    base = annotate_agreement(base, others, selected, caps, min_lens)

    # `annotate_agreement` compares key VALUES. The key blocking actually joins
    # on is `<country>|value`, so a cross-country pair agrees on nothing no
    # matter how identical the names are — and the first run of this printed
    # "keys agreeing: name_sorted, name_nospace, addr_sorted, name_acronym, 16
    # band(s)" under `reason=country-mismatch`, which reads as a contradiction
    # and invites exactly the wrong fix. Folded in here rather than inside the
    # helper, because `key_diagnostics` reports per-key reach and country is a
    # separate partition there, not part of a key's definition.
    base = base.with_columns([
        (pl.col(f"_agree_{k}") & pl.col("_country_ok")).alias(f"_agree_{k}")
        for k in selected
    ] + [
        (pl.col(f"_usable_{k}") & pl.col("_country_ok")).alias(f"_usable_{k}")
        for k in selected
    ])

    exact = [k for k in selected if not blocking.is_band(k)]
    bands = [k for k in selected if blocking.is_band(k)]
    n_ub_exact = (
        pl.sum_horizontal([pl.col(f"_usable_{k}").cast(pl.UInt8) for k in exact])
        if exact else pl.lit(0)
    )
    n_ub_band = (
        pl.sum_horizontal([pl.col(f"_usable_{k}").cast(pl.UInt8) for k in bands])
        if bands else pl.lit(0)
    )
    any_agree = pl.any_horizontal([pl.col(f"_agree_{k}") for k in selected])

    base = base.with_columns(
        n_ub_exact.cast(pl.UInt8).alias("_n_exact"),
        n_ub_band.cast(pl.UInt8).alias("_n_band"),
        any_agree.alias("_any_agree"),
    ).with_columns(
        pl.when(~pl.col("_country_ok")).then(pl.lit("country-mismatch"))
        .when(~pl.col("_any_agree")).then(pl.lit("no-key-agrees"))
        .when(pl.col("_n_exact") + pl.col("_n_band") == 0)
        .then(pl.lit("bucket-cap"))
        .when(
            (pl.col("_n_exact") == 0) & (pl.col("_n_band") < band_min_hits)
        ).then(pl.lit("band-gate"))
        .otherwise(pl.lit("budget-evicted"))
        .alias("reason")
    )

    # `budget-evicted` is a claim about the merge, and it is checkable: an
    # entity that did not fill its budget cannot have had a candidate evicted
    # by it. Any pair that lands there for an unsaturated entity means this
    # function's caps, key set or band gate are NOT the ones blocking ran with,
    # which is the failure the whole file is written against — so it is
    # relabelled and printed rather than absorbed into a plausible-looking
    # number.
    sizes = (
        pl.scan_parquet(cand_path).group_by("source1_entity_id").len()
        .rename({"len": "_n_cand"}).collect()
    )
    file_cap = int(sizes["_n_cand"].max() or 0)
    base = base.join(sizes, on="source1_entity_id", how="left").with_columns(
        pl.col("_n_cand").fill_null(0)
    ).with_columns(
        pl.when(
            (pl.col("reason") == "budget-evicted")
            & (pl.col("_n_cand") < file_cap)
        ).then(pl.lit("unexplained")).otherwise(pl.col("reason")).alias("reason")
    )

    # Ordered by MISS_REASONS — pipeline order — rather than by count. A table
    # whose rows move between runs cannot be diffed against the previous run,
    # and comparing two runs is the only thing this table is for.
    order_map = {r: i for i, r in enumerate(MISS_REASONS)}
    counts = (
        base.group_by("reason").len().rename({"len": "n"})
        .with_columns(
            pl.col("reason").replace_strict(
                order_map, default=len(MISS_REASONS), return_dtype=pl.UInt8
            ).alias("_ord")
        )
        .sort("_ord")
    )
    print(f"\n{'reason':<18} {'misses':>12} {'of misses':>11} {'of all true':>12}")
    for row in counts.iter_rows(named=True):
        # `of all true` is scaled from the classified sample back to every
        # true pair, so the column means the same thing sampled or not.
        share = row["n"] / base.height
        print(f"{row['reason']:<18} {row['n']:>12,} "
              f"{share:>10.2%} {share * n_missed_total / n_true:>11.2%}")
    print(f"\n  budget cap observed on the file: {file_cap}")
    print("  country-mismatch : every key is `<country>|value`, so these are")
    print("                     unreachable until the partition is relaxed.")
    print("  no-key-agrees    : needs a NEW key. The breakdown below says which.")
    print("  bucket-cap       : a key DID agree — loosening that cap recovers it.")
    print("  band-gate        : --band-min-hits rejected band-only evidence.")
    print("  budget-evicted   : generated, ranked, and cut. --budget recovers it.")
    unexplained = int(
        counts.filter(pl.col("reason") == "unexplained")["n"].sum() or 0
    )
    if unexplained:
        print(f"\n  WARNING: {unexplained:,} pair(s) are reachable under these "
              "caps "
              "for an entity that never filled its budget. Nothing in blocking "
              "drops those, so the caps/keys/--band-min-hits passed here are "
              "NOT the ones that built this file. Fix the flags before reading "
              "any row above.")

    # What the unreachable pairs actually look like. This is the half of the
    # report that answers "is there a pattern a key could exploit", and it is
    # deliberately split by trigram Jaccard: a miss at J >= 0.5 is one the LSH
    # bands are supposed to catch and did not, which is a band-parameter
    # finding. A miss at J < 0.2 shares almost no characters and no character
    # method will ever reach it.
    # `no-key-agrees` ONLY. A country-mismatch miss can sit at trigram J = 1.0
    # — identical names either side of the partition — and including it made
    # the first run print "the bands should reach these; they did not" about
    # pairs no band could ever reach. It has its own row in the table above;
    # mixing it in here answers a different question than the one asked.
    hard = base.filter(pl.col("reason") == "no-key-agrees")
    if not hard.is_empty():
        print(f"\n  `no-key-agrees` misses, by name similarity "
              f"({hard.height:,}) — same country, no key fires:")
        empties = int(
            hard.select(
                (
                    pl.col("_ns").fill_null("").str.len_chars().eq(0)
                    | pl.col("_ns_r").fill_null("").str.len_chars().eq(0)
                ).sum()
            ).item()
        )
        # Containment rather than a substring test. `acme` against `acme foods
        # private limited` is the case that matters — a truncated or
        # abbreviated name — and containment is how `features.py` already
        # measures it, over the same deduplicated trigram sets. A literal
        # substring test would also need `str.contains` against a COLUMN
        # pattern, which is a different and less portable thing to depend on.
        cont = int(hard.select((pl.col("tri_cont") >= 0.9).sum()).item())
        for lo, hi in SIMILARITY_BANDS:
            n = int(
                hard.select(
                    ((pl.col("tri_jac") >= lo) & (pl.col("tri_jac") < hi)).sum()
                ).item()
            )
            note = ""
            if lo >= 0.5:
                note = "  <- the bands should reach these; they did not"
            elif lo == 0.0:
                note = "  <- no character method reaches these"
            print(f"    trigram J {lo:.1f}-{hi if hi <= 1 else 1.0:.1f} : "
                  f"{n:>10,} ({n / hard.height:>6.2%}){note}")
        print(f"    one name empty        : {empties:>10,} "
              f"({empties / hard.height:.2%})  <- normalization, not blocking")
        print(f"    one name inside other : {cont:>10,} "
              f"({cont / hard.height:.2%})  <- trigram containment >= 0.9; a "
              "prefix or containment key would reach these")

    # ---- examples --------------------------------------------------------
    if n_examples <= 0:
        return

    # Filtered to ONE reason, and by default to `no-key-agrees`.
    #
    # The other five reasons are already actionable from the table above: a
    # `bucket-cap` miss says loosen that cap, a `budget-evicted` miss says
    # raise the budget, a `country-mismatch` miss says relax the partition.
    # None of them needs to be read. `no-key-agrees` is the only bucket where
    # the table cannot tell you what to do, because the answer is "some key
    # that does not exist yet" and the only way to find out which is to look
    # at the pairs. So that is what this prints unless asked otherwise.
    if reason_filter != "any":
        ex_pool = base.filter(pl.col("reason") == reason_filter)
        if ex_pool.is_empty():
            print(f"\n  (no `{reason_filter}` misses to show; "
                  "--miss-reason any prints from all of them)")
            return
    else:
        ex_pool = base

    # Ordered so that twenty rows SPAN the range rather than land in one
    # corner of it.
    #
    # This matters more than it looks. Sorted by similarity the first twenty
    # `no-key-agrees` misses are all near-duplicates and the reader concludes
    # "one prefix key fixes everything"; sampled at random from a distribution
    # that is mostly J < 0.2 they are all hopeless and the reader concludes
    # "this needs a different method". Both conclusions are drawn from the
    # sort order rather than from the data. `stratified` takes an equal share
    # from each similarity band, so the printout is a cross-section and the
    # TABLE above stays the thing that says how common each band is.
    if order == "stratified":
        per_band = max(1, n_examples // len(SIMILARITY_BANDS))
        parts = []
        for lo, hi in SIMILARITY_BANDS:
            part = ex_pool.filter(
                (pl.col("tri_jac") >= lo) & (pl.col("tri_jac") < hi)
            ).sort("tri_jac", descending=True)
            parts.append(part.head(per_band))
        picked = pl.concat(parts, how="vertical")
        # Bands can be empty or short, so the quota is a floor and the
        # remainder is topped up MOST-SIMILAR FIRST rather than at random.
        # That bias is deliberate: once every band is represented, the rows
        # worth spending the rest of the page on are the near-misses, because
        # those are the ones a new key could plausibly reach.
        if picked.height < n_examples:
            rest = ex_pool.join(
                picked.select("source1_entity_id", "candidate_entity_id"),
                on=["source1_entity_id", "candidate_entity_id"], how="anti",
            ).sort("tri_jac", descending=True)
            picked = pl.concat([picked, rest.head(n_examples - picked.height)],
                               how="vertical")
        picked = picked.sort("tri_jac", descending=True).head(n_examples)
    elif order == "most-similar":
        picked = ex_pool.sort("tri_jac", descending=True).head(n_examples)
    elif order == "most-missed":
        counts_e = (
            ex_pool.group_by("source1_entity_id").len()
            .rename({"len": "_n_miss_e"})
        )
        picked = (
            ex_pool.join(counts_e, on="source1_entity_id", how="left")
            .sort(["_n_miss_e", "source1_entity_id"], descending=[True, False])
            .head(n_examples)
        )
    elif order == "worst":
        # Entities that scored a FLAT ZERO: every true match missed. Not the
        # same set as "missed the most" — an entity missing 4 of 5 is a worse
        # miss count and a better score than one missing its only match.
        n_true_e = (
            ground_truth_pairs(paths.ground_truth_tsv(data))
            .group_by("source1_entity_id").len().rename({"len": "n_true"})
            .collect()
        )
        n_miss_e = (
            base.group_by("source1_entity_id").len()
            .rename({"len": "_n_miss_e"})
        )
        picked = (
            ex_pool.join(n_true_e, on="source1_entity_id", how="left")
            .join(n_miss_e, on="source1_entity_id", how="left")
            .with_columns(
                (pl.col("n_true") - pl.col("_n_miss_e")).alias("_found")
            )
            .sort(["_found", "n_true"], descending=[False, True])
            .head(n_examples)
        )
    else:
        picked = ex_pool.sample(
            n=min(ex_pool.height, n_examples), seed=seed, shuffle=True
        ).sort("tri_jac", descending=True)

    # Grouped by entity for DISPLAY only. The selection above is per pair, so
    # `--show-misses 20` prints exactly twenty pairs — the earlier version
    # added a whole entity at a time and overshot.
    ent_order: list[str] = []
    for eid in picked["source1_entity_id"].to_list():
        if eid not in ent_order:
            ent_order.append(eid)

    got = (
        pl.scan_parquet(cand_path)
        .filter(pl.col("source1_entity_id").is_in(ent_order))
        .select("source1_entity_id", "candidate_entity_id", "block_score",
                "n_keys")
        .collect()
        .sort("block_score", descending=True)
    )
    raw_l = _raw_lookup(split, data, (1,), ent_order)
    raw_r = _raw_lookup(
        split, data, (2, 3),
        picked["candidate_entity_id"].to_list()
        + got["candidate_entity_id"].to_list(),
    )

    agree_cols = [f"_agree_{k}" for k in selected]
    label_reason = "" if reason_filter == "any" else f"`{reason_filter}` "
    _section(f"{picked.height} {label_reason}misses across "
             f"{len(ent_order)} entities (--miss-order {order})")
    if reason_filter == "no-key-agrees":
        print("Normalized forms are shown under the raw ones: when no key "
              "agrees,\nthe question is whether the records really differ or "
              "whether\nnormalization pulled them apart, and only the "
              "normalized pair says.\n")
    for eid in ent_order:
        rows = picked.filter(pl.col("source1_entity_id") == eid)
        name, addr, country = raw_l.get(eid, ("<not in source1>", "", ""))
        n_cand = int(rows["_n_cand"][0])
        print(f"\n{'-' * 70}")
        print(f"source1 {eid}  [{country}]")
        print(f"  name : {name}")
        print(f"  norm : {rows['_ns'][0]}")
        print(f"  addr : {addr}")
        print(f"  anorm: {rows['_as'][0]}")
        print(f"  blocking returned {n_cand} candidates"
              f"{' (AT the cap)' if n_cand >= file_cap else ''}")
        for r in rows.iter_rows(named=True):
            cid = r["candidate_entity_id"]
            cname, caddr, ccountry = raw_r.get(cid, ("<not in source2/3>", "", ""))
            agreed = [k for k, c in zip(selected, agree_cols) if r[c]]
            band_agreed = sum(1 for k in agreed if blocking.is_band(k))
            shown_keys = [k for k in agreed if not blocking.is_band(k)]
            if band_agreed:
                shown_keys.append(f"{band_agreed} band(s)")
            print(f"  MISSED {cid}  [{ccountry}]  reason={r['reason']}  "
                  f"trigram J={r['tri_jac']:.2f} cont={r['tri_cont']:.2f}")
            print(f"    name : {cname}")
            print(f"    norm : {r['_ns_r']}")
            print(f"    addr : {caddr}")
            print(f"    anorm: {r['_as_r']}")
            print(f"    keys agreeing: "
                  f"{', '.join(shown_keys) if shown_keys else 'NONE'}")
        # What blocking returned instead. For a `no-key-agrees` miss this is
        # not what displaced the true match — nothing displaced it, it was
        # never generated — but it is still the fastest way to see what the
        # keys DID fire on for this entity, which is half of why they missed.
        top = got.filter(pl.col("source1_entity_id") == eid).head(3)
        if not top.is_empty():
            label = ("also returned" if reason_filter == "no-key-agrees"
                     else "kept instead")
            print(f"    {label} (top 3 by block_score):")
            for g in top.iter_rows(named=True):
                gname = raw_r.get(g["candidate_entity_id"], ("?", "", ""))[0]
                print(f"      {g['block_score']:.3f}  n_keys={g['n_keys']}  "
                      f"{gname}")


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
    p.add_argument(
        "--acronym-min-len", type=int, default=0,
        help="The --acronym-min-len BLOCKING was run with. See --acronym-cap. "
        "It drops a value on BOTH sides before bucketing, so a miss under it "
        "is not 'agrees but capped' — it is a key that never fired.",
    )
    p.add_argument(
        "--band-min-hits", type=int, default=1,
        help="The --band-min-hits BLOCKING was run with. Only the miss report "
        "reads it: it is what separates a pair the bands found and the gate "
        "rejected from one the budget evicted.",
    )
    p.add_argument(
        "--keys", default=None,
        help="The --keys BLOCKING was run with, comma-separated. The miss "
        "report defaults to blocking's own default set (which excludes "
        "name_core); the per-key table above still reports every key in "
        "KEY_SPECS.",
    )
    p.add_argument(
        "--bands", type=int, default=None,
        help="The --bands BLOCKING was run with. Mutually exclusive with "
        "--keys, as in blocking.",
    )
    p.add_argument(
        "--show-misses", type=int, default=20, metavar="N",
        help="Print N missed true pairs in full, with the raw name and "
        "address of both sides, why each was missed, and what blocking kept "
        "instead (default: %(default)s). 0 skips the whole section.",
    )
    p.add_argument(
        "--miss-reason", default="no-key-agrees",
        choices=(*MISS_REASONS, "any"),
        help="Which kind of miss to print examples of (default: "
        "%(default)s). The other reasons are already actionable from the "
        "summary table — a bucket-cap miss says loosen that cap, a "
        "budget-evicted one says raise the budget. `no-key-agrees` is the "
        "bucket the table cannot act on, because the answer is a key that "
        "does not exist yet and the only way to find out which is to read "
        "the pairs.",
    )
    p.add_argument(
        "--miss-order", default="stratified",
        choices=("stratified", "most-similar", "worst", "most-missed",
                 "random"),
        help="Which of them to print (default: %(default)s). `stratified` "
        "takes an equal share from each trigram-similarity band, so twenty "
        "rows span the range instead of landing in one corner of it — "
        "sorted by similarity they would all be near-duplicates, sampled at "
        "random they would all be hopeless, and either way the conclusion "
        "comes from the sort order rather than the data. `most-similar` is "
        "the near-misses alone, `worst` the entities that found none of "
        "their true matches, `most-missed` the largest miss counts.",
    )
    p.add_argument("--seed", type=int, default=0,
                   help="Sampling seed for --miss-order random.")
    p.add_argument("--max-bucket", type=int, default=None,
                   help="Global cap override for the diagnostics. Leave unset "
                        "to use each key's own cap from KEY_SPECS.")
    p.add_argument("--sample-pairs", type=int, default=None,
                   help="Cap the true pairs used for per-key diagnostics. For "
                        "memory only — it biases nothing, but it does make the "
                        "percentages noisier.")
    p.add_argument("--no-key-diagnostics", action="store_true")
    p.add_argument("--no-miss-report", action="store_true",
                   help="Skip the missed-pair classification and examples.")
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
    progress.add_arguments(p)
    a = p.parse_args()

    if a.keys is not None and a.bands is not None:
        p.error("pass --keys or --bands, not both, exactly as blocking "
                "requires: --keys already names the band subset.")
    if a.keys is not None:
        keys = tuple(k.strip() for k in a.keys.split(",") if k.strip())
        unknown = sorted(set(keys) - set(blocking.KEY_COLUMNS))
        if unknown:
            p.error(f"unknown key(s) {unknown}; available: "
                    f"{list(blocking.KEY_COLUMNS)}")
    elif a.bands is not None:
        # Validated by `default_keys`, which raises on an out-of-range band
        # count — the same check and the same message blocking gives.
        try:
            keys = blocking.default_keys(a.bands)
        except ValueError as exc:
            p.error(str(exc))
    else:
        keys = None

    work = paths.work_root(a.work_root)
    data = paths.data_root(a.data_root)
    progress.configure_from_args(a, work, label=f"block_eval:{a.split}")

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
        with progress.stage("eval/candidate-recall", budget=a.budget):
            candidate_recall(a.split, work, data, cand_path, a.budget)
        cuts = tuple(
            sorted({int(c) for c in a.budget_curve.split(",") if c.strip()})
        )
        if cuts:
            with progress.stage("eval/budget-curve", cuts=len(cuts)):
                budget_curve(a.split, work, data, cand_path, cuts)
        if not a.no_miss_report:
            with progress.stage("eval/miss-report", show=a.show_misses):
                miss_report(
                    a.split, work, data, cand_path, a.max_bucket,
                    a.acronym_cap, a.band_cap, a.acronym_min_len,
                    a.band_min_hits, keys, a.show_misses, a.miss_order,
                    a.seed, a.sample_pairs, a.miss_reason,
                )

    if not a.no_key_diagnostics:
        # The per-key diagnostics rebuild every key's target table, so this
        # section alone costs about as much as a blocking run. Timing it
        # separately is what stops that being attributed to the evaluation of
        # the candidate file.
        with progress.stage("eval/key-diagnostics"):
            key_diagnostics(a.split, work, data, a.max_bucket, a.sample_pairs,
                            a.acronym_cap, a.band_cap, a.acronym_min_len)
    progress.shutdown()


if __name__ == "__main__":
    main()
