"""Stage D: a learned scorer, and a per-entity rule for how much to predict.

This is the first model in the project. Every number recorded anywhere in the
docs before this describes either the candidate set or a hand-written
similarity rule, so there is no prior precision figure from a model to compare
against — only the 0.6070 the address-Jaccard scorer reached and the 0.7943
ceiling the candidate set allows.

Two decisions live here, and they are separable on purpose.

1. **How good is this pair?** LightGBM over the features from `features.py`.
   Gradient boosting rather than anything heavier because the inputs are ~30
   dense numeric columns with meaningful interactions and 23.6M rows: trees fit
   that shape in minutes on CPU inside 16GB, and the challenge's 8B-parameter
   licence limit is not remotely a constraint.

2. **How many of them should be predicted?** Not a threshold. See
   `select_matches`.

Honest measurement
------------------
The training rules in `text_norm` were built by reading training data and the
0.6070 was scored on that same split, so part of that number is fitting the
rules to the data they came from. A model trained and scored on the same
entities would compound exactly that error, and the problem statement asks for
a held-out split for this reason.

So: **grouped 5-fold cross-validation over source-1 entities.** Grouped
matters. An entity contributes ~10.7 candidate rows that share its name, its
address and all seven of its relative features; splitting by ROW would put near
duplicates of the same entity on both sides of the split and report a score
that no held-out data would reproduce. Splitting by entity — via a hash of the
entity ID, so the assignment is deterministic and reproducible without storing
a fold map — keeps each entity wholly on one side.

Every entity then receives a prediction from a model that never saw it, so the
macro F_0.5 printed here is over the FULL training split and is directly
comparable to 0.6070, with no in-sample component from the model. It says
nothing about the in-sample component in the normalization rules, which is a
separate problem that cross-validation cannot fix.

Early stopping uses an inner slice of the training folds, never the evaluation
fold. Using the evaluation fold to decide when to stop is a real leak — small,
but it is the kind that makes a cross-validated number quietly optimistic, and
the whole point of this stage is a number that is not.

Usage:

    python -m src.model --split train        # 5-fold CV, OOF score, save models
    python -m src.model --split test         # predict with the saved ensemble
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import paths
from .features import FEATURE_COLUMNS

# Fold assignment is a hash of the entity ID rather than a stored map or a
# shuffled index. Deterministic across runs and across machines, needs nothing
# persisted, and — unlike `random.shuffle` over a row index — cannot drift when
# the candidate file is rebuilt with a different row order.
FOLD_SEED = 0x5EED5A17

# The inner early-stopping split MUST NOT share a seed with the outer folds.
# With one seed, `hash % 20 == 0` implies `hash % 5 == 0`, so the inner slice
# would lie entirely inside outer fold 0 — and since the inner slice is carved
# out of the OTHER folds, fold 0 would be handed an empty early-stopping set
# and LightGBM would fail on an empty Dataset. Two independent hashes make the
# two splits independent.
INNER_SEED = 0x1F123BB5
# One twentieth of the training folds, held back to decide when to stop.
INNER_PARTS = 20

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 0,
}

ENT = "source1_entity_id"
CAND = "candidate_entity_id"


def fold_of(col: str, n_folds: int, seed: int = FOLD_SEED) -> pl.Expr:
    return (pl.col(col).hash(seed=seed) % n_folds).cast(pl.UInt8)


def _matrix(df: pl.DataFrame) -> np.ndarray:
    """Feature columns as a float32 matrix, in the manifest's order.

    Column ORDER is part of a saved booster's contract — LightGBM indexes
    features positionally, so a model applied to the same columns in a
    different order produces confident nonsense rather than an error. Selecting
    by `FEATURE_COLUMNS` here, and persisting that list beside the booster,
    makes the order one fact rather than two that have to agree.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"feature table is missing {missing}")
    return (
        df.select([pl.col(c).cast(pl.Float32) for c in FEATURE_COLUMNS])
        .to_numpy()
        .astype(np.float32, copy=False)
    )


# --------------------------------------------------------------------------
# how many matches to predict
# --------------------------------------------------------------------------

def select_matches(
    scored: pl.DataFrame,
    topk: int,
    coverage: float = 1.0,
    prob: str = "p",
    buckets: int = 16,
) -> pl.DataFrame:
    """Choose, per entity, the k that maximizes expected F_0.5.

    Why not a threshold. A single global cut on the probability throws away the
    thing that actually decides this problem: F_0.5 is macro-averaged per
    entity, so the question is never "is p > t" but "for THIS entity, does
    adding a fourth candidate help or hurt its own score". An entity whose best
    candidate is at 0.55 and whose next is at 0.12 wants one prediction; an
    entity with four candidates all at 0.55 wants all four. A threshold cannot
    express that, and tuning one is the sweep this project has twice agreed not
    to run.

    The rule instead. Sort an entity's candidates by probability, and for each
    prefix of length k treat the probabilities as the expected number of true
    matches it contains:

        E[TP@k] = sum of the top k probabilities
        P(k)    = E[TP@k] / k
        R(k)    = E[TP@k] / (expected true matches)
        F(k)    = 1.25 P R / (0.25 P + R)

    then take the k that maximizes F(k). The empty prediction is handled in the
    same currency rather than as a special case: an entity with no true matches
    scores exactly 1.0 for an empty prediction, so the expected score of
    predicting nothing is the probability that nothing matches, which under
    independence is the product of (1 - p) over ALL its candidates. If that
    beats the best F(k), predict nothing.

    This is a plug-in estimate, not a true expectation — F is a ratio of
    dependent random variables and E[F] is not F applied to the expectations.
    It is stated as an approximation rather than dressed up as optimal. What it
    does have is no tuned parameter, a per-entity decision derived from the
    model's own output, and the right behaviour at both ends: confident
    singletons predict nothing, confident clusters predict several.

    `coverage` corrects a known bias in R. The denominator is summed over the
    CANDIDATE set, but blocking only reaches 59.56% of true pairs, so recall is
    systematically over-estimated and the rule leans toward predicting too
    FEW — the direction that matters, since 173,911 entities already get an
    empty prediction despite having real matches. Pass the macro recall that
    `src.block_eval` reports to correct it. Defaulting to 1.0 keeps the
    uncorrected behaviour reachable so the correction's effect is attributable.

    `topk` caps k. The maximum true match count on the training split is 11 and
    the 99th percentile is 8, so predicting more than that is almost certainly
    wrong regardless of what the probabilities say.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")

    # Partitioned by a hash of the entity ID, which puts an entity's whole
    # candidate list in one partition — the precondition every step below
    # depends on. This is the last step of the longest job in the pipeline and
    # it sorts ~23.6M rows; taking that as one global sort is how a run dies
    # after forty minutes having written nothing, which is already in the
    # journal against the blocking merge. Bucket count changes peak memory
    # only, never the output.
    if buckets > 1:
        parts = [
            select_matches(
                scored.filter((pl.col(ENT).hash() % buckets) == b),
                topk, coverage, prob, buckets=1,
            )
            for b in range(buckets)
        ]
        parts = [p for p in parts if not p.is_empty()]
        out = (
            pl.concat(parts, how="vertical") if parts
            else scored.select(ENT, CAND).clear()
        )
        # Reported once, from the result, rather than sixteen times from the
        # partitions. An entity that predicts nothing is simply absent from
        # `out`, so it has to be counted by difference.
        n_ents = scored[ENT].n_unique()
        n_pred = out[ENT].n_unique()
        empty = n_ents - n_pred
        print(
            f"  selection: {n_ents:,} entities, {empty:,} "
            f"({empty / max(n_ents, 1):.2%}) predict nothing, "
            f"{out.height:,} predictions, mean {out.height / max(n_ents, 1):.2f} "
            f"per entity (coverage {coverage:g}, topk {topk})"
        )
        return out

    if scored.is_empty():
        return scored.select(ENT, CAND)

    # Clipped away from both ends: a probability of exactly 1.0 makes
    # `(1 - p).log()` below negative infinity and collapses the
    # empty-prediction term to zero for the whole entity, and a probability of
    # exactly 0 can make the recall denominator zero for an entity whose
    # candidates are all hopeless.
    #
    # Note on `_r`: it cannot exceed 1. At k = m the numerator equals
    # `expected_true`, so `_r` reaches exactly `coverage`, which is bounded by
    # 1 by the check above. A lower coverage therefore makes recall scarcer and
    # pushes the optimum toward LARGER k, which is the intended direction.
    d = scored.with_columns(pl.col(prob).cast(pl.Float64).clip(1e-9, 1 - 1e-9))

    # Eager sort before any positional or cumulative work. `cum_sum().over()`
    # reads the frame's current row order, and a lazy optimizer is free to
    # reorder a sort past the operation that depends on it — the exact hazard
    # already recorded against the blocking merge. Ties break on the candidate
    # ID so the chosen prefix is deterministic rather than join-order dependent.
    d = d.sort([ENT, prob, CAND], descending=[False, True, False])

    # Entity-wide quantities, computed BEFORE the top-k truncation below: the
    # recall denominator and the empty-prediction probability are statements
    # about all of an entity's candidates, not about the ones that survive the
    # cap.
    d = d.with_columns(
        (pl.int_range(pl.len()).over(ENT) + 1).alias("k"),
        pl.col(prob).cum_sum().over(ENT).alias("etp"),
        pl.col(prob).sum().over(ENT).alias("expected_true"),
        (1.0 - pl.col(prob)).log().sum().over(ENT).exp().alias("p_empty"),
    ).filter(pl.col("k") <= topk)

    d = d.with_columns(
        (pl.col("etp") / pl.col("k")).alias("_p"),
        (pl.col("etp") / (pl.col("expected_true") / coverage)).alias("_r"),
    ).with_columns(
        (
            1.25 * pl.col("_p") * pl.col("_r")
            / (0.25 * pl.col("_p") + pl.col("_r"))
        ).alias("_f")
    )

    best = d.group_by(ENT).agg(
        pl.col("_f").max().alias("f_best"),
        # Smallest k that attains the maximum. F_0.5 weights precision, so when
        # two prefixes score identically the shorter one is the right tiebreak
        # — and leaving it to row order is how a previous version of this
        # pipeline ended up selecting candidates alphabetically.
        pl.col("k").filter(pl.col("_f") == pl.col("_f").max()).min().alias("k_best"),
        pl.col("p_empty").first().alias("p_empty"),
    ).with_columns(
        pl.when(pl.col("p_empty") >= pl.col("f_best"))
        .then(pl.lit(0, dtype=pl.UInt32))
        .otherwise(pl.col("k_best").cast(pl.UInt32))
        .alias("keep")
    )

    return (
        d.join(best.select(ENT, "keep"), on=ENT, how="inner")
        .filter(pl.col("k") <= pl.col("keep"))
        .select(ENT, CAND)
    )


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def _entity_plan(
    src: Path, n_folds: int, train_entities: int | None, seed: int
) -> pl.DataFrame:
    """One row per source-1 entity: its outer fold and its inner-split part.

    Planning on the ENTITY table rather than the pair table is what keeps this
    stage inside 16GB. There are ~2.2M entities against ~23.6M pairs, so the
    plan is an order of magnitude smaller than the data it selects, and each
    fold then loads only the rows it needs instead of slicing a resident copy
    of everything.

    Sampling is over entities, never rows. A randomly sampled subset of ROWS
    would break the relative features outright — `block_score_rank` and
    `n_cand` describe a candidate list, and a partially sampled list describes
    a list that never existed — and it would let entities with many candidates
    dominate training, which is the sampling note carried in the handoff.
    """
    ents = pl.scan_parquet(src).select(ENT).unique().collect()
    ents = ents.with_columns(
        fold_of(ENT, n_folds).alias("_fold"),
        fold_of(ENT, INNER_PARTS, seed=INNER_SEED).alias("_inner"),
    )
    if train_entities:
        # A per-fold sample would have to be drawn inside the loop; drawing one
        # ordering here and taking a prefix per fold gives the same coverage
        # with one shuffle. `_pick` is a stable random ordering of the entities.
        ents = ents.with_columns(
            pl.int_range(pl.len()).shuffle(seed=seed).alias("_pick")
        )
    else:
        ents = ents.with_columns(pl.lit(0, dtype=pl.Int64).alias("_pick"))
    return ents


def train_cv(
    src: Path,
    n_folds: int,
    train_entities: int | None,
    rounds: int,
    seed: int,
    params: dict | None = None,
) -> tuple[list, pl.DataFrame]:
    """Grouped k-fold over source-1 entities. Returns (boosters, OOF scores).

    Every entity appears exactly once in the returned frame, scored by the one
    fold model that did not train on it.

    Rows are read from the parquet per fold rather than sliced from a resident
    table. The full feature set is ~23.6M rows over 30 float columns plus two
    ID strings — call it 5GB resident — and holding that alongside a training
    copy and a LightGBM Dataset is how a long job dies at the end, which this
    project has already recorded once against the blocking merge.
    """
    import lightgbm as lgb

    params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
    plan = _entity_plan(src, n_folds, train_entities, seed)
    n_ents = plan.height
    if n_ents < n_folds:
        raise RuntimeError(
            f"{n_ents} entities cannot be split into {n_folds} folds."
        )
    sizes = plan.group_by("_fold").agg(pl.len().alias("entities")).sort("_fold")
    print(f"  {n_ents:,} entities across {n_folds} folds: "
          + ", ".join(f"{r['entities']:,}" for r in sizes.iter_rows(named=True)))

    def rows_for(keep: pl.DataFrame) -> pl.DataFrame:
        """Load exactly the pairs belonging to a set of entities."""
        return (
            pl.scan_parquet(src)
            .join(keep.select(ENT).lazy(), on=ENT, how="semi")
            .collect()
        )

    boosters, oof_parts = [], []
    for f in range(n_folds):
        t0 = time.perf_counter()
        rest = plan.filter(pl.col("_fold") != f)
        fit_ents = rest.filter(pl.col("_inner") != 0)
        early_ents = rest.filter(pl.col("_inner") == 0)
        if early_ents.is_empty() or fit_ents.is_empty():
            raise RuntimeError(
                f"fold {f}: inner split produced {fit_ents.height} training and "
                f"{early_ents.height} early-stopping entities. With independent "
                "hash seeds this should not happen — check FOLD_SEED/INNER_SEED."
            )
        if train_entities:
            fit_ents = fit_ents.sort("_pick").head(train_entities)

        fit = rows_for(fit_ents)
        early = rows_for(early_ents)
        dtrain = lgb.Dataset(_matrix(fit), label=fit["label"].to_numpy())
        dvalid = lgb.Dataset(
            _matrix(early), label=early["label"].to_numpy(), reference=dtrain
        )
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            valid_sets=[dvalid],
            valid_names=["early"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        boosters.append(booster)
        n_fit, n_fit_ents = fit.height, fit_ents.height
        del fit, early, dtrain, dvalid

        hold = rows_for(plan.filter(pl.col("_fold") == f))
        oof_parts.append(
            hold.select(ENT, CAND).with_columns(
                pl.Series(
                    "p",
                    booster.predict(_matrix(hold), num_iteration=booster.best_iteration),
                    dtype=pl.Float32,
                )
            )
        )
        n_hold = hold.height
        del hold

        ap = next(iter(booster.best_score.get("early", {}).values()), float("nan"))
        print(
            f"    fold {f}: fit {n_fit:,} rows / {n_fit_ents:,} entities, "
            f"scored {n_hold:,} held-out rows, best iter {booster.best_iteration}, "
            f"AP {ap:.4f}   ({time.perf_counter() - t0:.1f}s)"
        )

    return boosters, pl.concat(oof_parts, how="vertical")


def predict_ensemble(
    boosters: list, src: Path, chunk: int = 2_000_000
) -> pl.DataFrame:
    """Average the fold models' probabilities over a feature parquet.

    No separate refit on all the data. Each fold model saw 4/5 of the entities
    and their errors are only partly shared, so the mean of the five is at
    least as good as any one of them and usually slightly better — for the cost
    of prediction, which is minutes, rather than a sixth training run.

    Averaging PROBABILITIES rather than raw scores is deliberate: the selection
    rule downstream reads them as probabilities and multiplies `1 - p` across
    an entity's candidates, which is only meaningful on the probability scale.

    Streamed with `pyarrow.ParquetFile.iter_batches`, which is a genuine
    sequential cursor. `pl.scan_parquet(src).slice(offset, chunk).collect()` in
    a loop is NOT — each collect re-reads from the start and discards
    everything before the offset, which at 23.6M rows is a dozen full passes
    over the file. That exact mistake is already in the journal against the
    normalization chunker, and it reappeared here while this function was being
    written.

    A batch boundary may cut an entity's candidate list in half, which is
    harmless HERE because scoring a pair does not depend on the entity's other
    pairs — the features that do were computed upstream. The selection rule
    downstream does depend on it, and partitions by entity accordingly.
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src)
    total = pf.metadata.num_rows
    parts = []
    seen = 0
    for batch in pf.iter_batches(batch_size=chunk):
        block = pl.from_arrow(batch)
        if block.is_empty():
            continue
        seen += block.height
        X = _matrix(block)
        p = np.mean(
            [b.predict(X, num_iteration=b.best_iteration) for b in boosters], axis=0
        )
        parts.append(
            block.select(ENT, CAND).with_columns(pl.Series("p", p, dtype=pl.Float32))
        )
        print(f"    predicted {seen:,}/{total:,}", end="\r")
    print()
    if not parts:
        raise RuntimeError(f"{src} yielded no rows to predict on.")
    return pl.concat(parts, how="vertical")


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def save_models(boosters: list, work: Path, params: dict) -> None:
    d = paths.model_dir(work)
    for i, b in enumerate(boosters):
        b.save_model(str(d / f"fold{i}.txt"), num_iteration=b.best_iteration)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "features": list(FEATURE_COLUMNS),
                "n_folds": len(boosters),
                "params": params,
            },
            indent=2,
        )
    )
    print(f"  saved {len(boosters)} boosters -> {d}")


def load_models(work: Path) -> list:
    import lightgbm as lgb

    d = paths.model_dir(work)
    manifest = d / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{manifest} missing — run `python -m src.model --split train` first."
        )
    saved = json.loads(manifest.read_text())["features"]
    # Positional, not by name: LightGBM will happily score a matrix whose
    # columns mean something different from the ones it was trained on. Compare
    # the whole ordered list, not the set.
    if saved != list(FEATURE_COLUMNS):
        raise RuntimeError(
            "the saved model was trained on different feature columns than "
            f"features.py now produces.\n  saved: {saved}\n  now:   "
            f"{list(FEATURE_COLUMNS)}\nRe-run `python -m src.model --split train`."
        )
    files = sorted(d.glob("fold*.txt"))
    if not files:
        raise FileNotFoundError(f"no fold*.txt boosters in {d}")
    return [lgb.Booster(model_file=str(f)) for f in files]


def report_importance(boosters: list, top: int = 15) -> None:
    """Mean gain per feature across folds.

    Worth reading rather than skipping: the point of this stage was that the
    scorer had one feature and needed independent evidence. If `addr_tok_jac`
    still dominates, the model has reproduced the circular scorer with extra
    steps, and the name features are not doing what they were added to do.
    """
    gains = np.mean(
        [b.feature_importance(importance_type="gain") for b in boosters], axis=0
    )
    order = np.argsort(gains)[::-1]
    total = gains.sum() or 1.0
    print(f"  feature importance (mean gain across {len(boosters)} folds):")
    for i in order[:top]:
        print(f"    {FEATURE_COLUMNS[i]:<22} {gains[i] / total:6.2%}")


# --------------------------------------------------------------------------

def run(
    split: str,
    work: Path,
    topk: int,
    coverage: float,
    n_folds: int,
    train_entities: int | None,
    rounds: int,
    seed: int,
) -> pl.DataFrame:
    """Train (train split) or load (test split), then predict and select."""
    src = paths.features_parquet(split, work)
    if not src.is_file():
        raise FileNotFoundError(
            f"{src} missing — run `python -m src.features --split {split}` first."
        )
    # `limit(0).collect().columns` rather than `collect_schema()`: the schema
    # API moved between polars versions and this project pins a range rather
    # than an exact build, so the column list is read the way the rest of the
    # codebase already reads frames.
    columns = pl.scan_parquet(src).limit(0).collect().columns
    n_rows = pl.scan_parquet(src).select(pl.len()).collect().item()
    print(f"  features: {n_rows:,} pairs, {len(FEATURE_COLUMNS)} columns")

    if split == "train":
        if "label" not in columns:
            raise RuntimeError(
                f"{src} has no `label` column. Rebuild it with "
                "`python -m src.features --split train`."
            )
        boosters, scored = train_cv(src, n_folds, train_entities, rounds, seed)
        save_models(boosters, work, DEFAULT_PARAMS)
        report_importance(boosters)
    else:
        boosters = load_models(work)
        print(f"  loaded {len(boosters)} boosters")
        scored = predict_ensemble(boosters, src)

    return select_matches(scored, topk, coverage)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage D: LightGBM scorer with grouped k-fold CV and a "
                    "per-entity expected-F0.5 selection rule."
    )
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--data-root", default=None)
    p.add_argument("--work-root", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--topk", type=int, default=8,
                   help="Hard cap on predictions per entity. The training "
                        "split's 99th percentile true match count is 8 and its "
                        "maximum is 11.")
    p.add_argument("--coverage", type=float, default=1.0,
                   help="Share of an entity's true matches the candidate set "
                        "is expected to contain, used to de-bias the recall "
                        "term of the selection rule. `src.block_eval` reports "
                        "this as macro recall (0.5956 at the last measurement). "
                        "Left at 1.0 the rule over-estimates recall and "
                        "predicts slightly too few.")
    p.add_argument("--folds", type=int, default=5,
                   help="Grouped CV folds over source-1 entities.")
    p.add_argument("--train-entities", type=int, default=500_000,
                   help="Entities sampled from the training folds per fold. "
                        "Entities, never rows — a partly sampled candidate "
                        "list makes the relative features describe a list that "
                        "never existed. 0 uses everything.")
    p.add_argument("--rounds", type=int, default=600,
                   help="Boosting round ceiling. Early stopping on an inner "
                        "slice of the training folds usually stops first.")
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()

    work = paths.work_root(a.work_root)
    data = paths.data_root(a.data_root)
    out = paths.output_root(a.out_dir)
    started = time.perf_counter()
    print(f"model [{a.split}]  work={work}")

    pairs = run(
        a.split, work, a.topk, a.coverage, a.folds,
        a.train_entities or None, a.rounds, a.seed,
    )

    # Imported here, not at module scope. `main` reaches into this module for
    # the `--scorer model` path, so a top-level import in the other direction
    # would be a cycle. Confined to the CLI entry point so the library half of
    # this file stays importable on its own.
    from . import main as pipeline

    preds = pipeline.write_submission(
        pairs.rename({ENT: "entity_id"}), a.split, data, None,
        out / "matching_results.tsv", "matched_entity_ids",
    )
    if a.split == "train":
        score = pipeline.macro_f05(preds, data, "matched_entity_ids")
        print(f"\n  OUT-OF-FOLD MACRO F_0.5 = {score:.4f}")
        print("  (every entity scored by a model that did not train on it)")
    print(f"  done in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
