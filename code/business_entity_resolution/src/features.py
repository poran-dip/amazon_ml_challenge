"""Stage C: turn each candidate pair into evidence a model can weigh.

Why this stage exists
---------------------
Multi-key blocking raised the candidate-set ceiling to 0.7943 while the scorer
stayed a single hand-written feature, so the realized score FELL to 0.5745 even
though the candidate set strictly improved. The gap between those two numbers —
**0.2198** — is the largest lever in the project, and it is entirely a scoring
problem.

The mechanism is specific. `addr_sorted` blocks on sorted address tokens and
the old scorer scores on address Jaccard, so any candidate reached through that
key is graded by the exam it wrote:

    Sharma Textiles  vs Kanha Logistics, both at 12 MG Road      -> 1.00
    Acme Foods       vs Zenith Motors,   400 Oak St Suite 5/9    -> 0.75

Two different businesses in one office block score at or near the maximum. No
threshold can fix that, because `sim` means opposite things for different
candidates: for a name-key candidate a high address score is independent
confirming evidence, and for an address-key candidate it is tautological. A
single global cut on a feature that is informative for some rows and circular
for others preferentially deletes true name-matches with imperfect addresses
while keeping every same-building false positive.

The fix is not a better threshold. It is giving the scorer evidence that does
not depend on which key produced the pair — and, crucially, telling it which
key DID produce the pair, so it can discount accordingly. That is what
`key_mask` and the name-side features below are for.

What it produces
----------------
One row per candidate pair, ~30 numeric columns, in four groups:

* **provenance** — `block_score`, `n_keys`, and one flag per blocking key
  unpacked from `key_mask`, plus `addr_only`, which is exactly the circular
  case above.
* **name evidence** — character-3-gram and token-set overlap on `name_core`,
  plus the three exact-match flags. This is the independent signal the old
  scorer never had. Character 3-grams specifically, because the 23.74% of true
  pairs no exact key reaches fail for character-level reasons (typos,
  concatenation, transliteration residue), not semantic ones.
* **address evidence** — token overlap, exact match, and separately the
  overlap of postal codes and of raw building numbers, which say different
  things: a shared postal code means the same locality, a shared building
  number means the same door.
* **relative** — each pair's standing WITHIN its own entity's candidate list:
  rank, and value as a fraction of the entity's best. A trigram similarity of
  0.5 means something very different when it is the entity's best candidate
  than when it is seventh of twenty. These are also what let the model decide
  that an entity has no good match at all, which matters because 222,965
  entities currently get an empty prediction and 173,911 of those have real
  matches.

Everything here is vectorized polars over hashed integer lists — see
`_attributes` for why the strings are hashed rather than carried.

Usage:

    python -m src.features --split train        # writes features + labels
    python -m src.features --split test         # features only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl

from . import blocking, block_eval, paths

# Hashing tokens/numbers for comparison only — never for reconstruction. Kept
# distinct from `text_norm.TRIGRAM_HASH_SEED` so that a trigram hash and a
# token hash of the same string do not collide into looking like evidence.
TOKEN_HASH_SEED = 0x85EBCA6B

# The model's input contract. `model.py` imports this rather than introspecting
# the parquet, so a column added here without being handled there fails loudly
# at training time instead of being silently dropped.
#
# Order is not meaningful to LightGBM, but it IS stable, and a saved model is
# only valid against the column order it was trained on — `model.py` persists
# this list alongside the booster and refuses a mismatch.
FEATURE_COLUMNS: tuple[str, ...] = (
    # provenance
    "block_score", "n_keys",
    *(f"hit_{k}" for k in blocking.KEY_COLUMNS),
    "addr_only", "n_band_hits",
    # name evidence
    "name_tri_jac", "name_tri_cont",
    "name_tok_jac", "name_tok_cont",
    "name_exact", "name_nospace_exact", "name_acronym_exact",
    "name_len_ratio", "name_ntok_e", "name_ntok_c", "name_ntok_absdiff",
    # address evidence
    "addr_tok_jac", "addr_tok_cont", "addr_exact",
    "addr_postal_overlap", "addr_number_overlap",
    "addr_ntok_e", "addr_ntok_c", "addr_blank_e", "addr_blank_c",
    # relative to the entity's own candidate list
    "n_cand",
    "block_score_frac", "block_score_rank",
    "name_tri_frac", "name_tri_rank",
    "addr_tok_frac", "addr_tok_rank",
)

ID_COLUMNS: tuple[str, ...] = ("source1_entity_id", "candidate_entity_id")


# --------------------------------------------------------------------------
# entity-side attributes
# --------------------------------------------------------------------------

def _hash32(e: pl.Expr) -> pl.Expr:
    """Low 32 bits of the polars hash.

    A plain narrowing cast is wrong here: strict mode raises on overflow and
    non-strict mode yields NULL, which would poison every set operation
    downstream without raising anything. Taking the low bits explicitly is
    defined for every input.
    """
    return (e.hash(seed=TOKEN_HASH_SEED) % 4294967296).cast(pl.UInt32)


def _hashed_tokens(col: str) -> pl.Expr:
    """Whitespace tokens of a normalized column, hashed, deduplicated.

    `list.unique()` is NOT optional. Union size is computed downstream as
    `|A| + |B| - |A and B|`, which is exact for sets and wrong for multisets,
    and it is used in place of a second `set_union` pass because this runs over
    23.6M pairs. A duplicate token would silently inflate the union and deflate
    every Jaccard on that row.
    """
    return (
        pl.col(col).fill_null("").str.split(" ")
        .list.eval(_hash32(pl.element().filter(pl.element().str.len_chars() > 0)))
        .list.unique()
    )


def _hashed_list(col: str) -> pl.Expr:
    """An existing list column, hashed and deduplicated.

    `addr_numbers` comes out of `str.extract_all` and is NOT unique — an
    address like "Plot 12, Block 12" yields ["12", "12"]. See `_hashed_tokens`
    for why that matters.
    """
    return pl.col(col).list.eval(_hash32(pl.element())).list.unique()


def _attributes(split: str, sources: tuple[int, ...], work: Path, id_alias: str) -> pl.DataFrame:
    """Everything the pair features need about one side, as narrow numbers.

    Every string except the entity ID is reduced to a hash or a hashed list
    before this frame is materialized. The reason is scale rather than taste:
    this table is resident while the whole candidate set streams past it, and
    at 10.3M target rows a `List(String)` of ~25 three-character grams costs
    roughly 2.5GB once per-string overhead is counted, against ~1.25GB as
    `List(UInt32)`. Integer set intersection is also materially faster, and it
    runs 23.6M times.

    Nothing downstream needs to read a name back, only to compare two of them,
    so the information loss is exactly zero for this purpose.
    """
    return pl.concat(
        [
            pl.scan_parquet(paths.normalized_parquet(split, s, work)).select(
                pl.col("entity_id").alias(id_alias),
                # Exact-match keys become scalar hashes: equality on hashes is
                # equality on strings up to a collision probability far below
                # anything that moves a score.
                _hash32(pl.col("name_sorted").fill_null("")).alias("h_name_sorted"),
                _hash32(pl.col("name_nospace").fill_null("")).alias("h_name_nospace"),
                _hash32(pl.col("name_acronym").fill_null("")).alias("h_name_acronym"),
                _hash32(pl.col("addr_sorted").fill_null("")).alias("h_addr_sorted"),
                # Trigrams are already hashed and uniqued by `text_norm`.
                pl.col("name_trigrams").alias("tri"),
                _hashed_tokens("name_core").alias("name_tok"),
                _hashed_tokens("addr_core").alias("addr_tok"),
                _hashed_list("addr_postal").alias("postal"),
                _hashed_list("addr_numbers").alias("numbers"),
                pl.col("name_token_count").cast(pl.UInt16).alias("n_name_tok"),
                pl.col("addr_token_count").cast(pl.UInt16).alias("n_addr_tok"),
                pl.col("name_nospace").fill_null("").str.len_chars()
                .cast(pl.UInt16).alias("name_len"),
                (pl.col("addr_core").fill_null("").str.len_chars() == 0)
                .alias("blank"),
                # Emptiness flags for the NAME side, for the same reason the
                # address has one. `prepare_full` deliberately does not drop
                # rows whose name normalizes to nothing — they are still
                # reachable through `addr_sorted` — so two such records exist,
                # hash equal, and would otherwise satisfy `name_exact`,
                # `name_nospace_exact` AND `name_acronym_exact` at once: three
                # independent-looking pieces of evidence manufactured out of
                # two blank strings.
                #
                # These cannot be derived from `name_token_count`. That column
                # is `split(" ").list.len()`, and polars returns `[""]` — a
                # list of length ONE — for an empty string, so the count is 1
                # for a blank name and a `> 0` test on it is always true. That
                # is the same `split` behaviour that made two empty addresses
                # score a Jaccard of 1.0 for two sessions, and a guard written
                # against the count instead of the string would be dead code
                # documented as a guard.
                (pl.col("name_core").fill_null("").str.len_chars() == 0)
                .alias("name_blank"),
                (pl.col("name_acronym").fill_null("").str.len_chars() == 0)
                .alias("acr_blank"),
            )
            for s in sources
        ],
        how="vertical",
    ).collect()


def _suffixed(df: pl.DataFrame, id_alias: str, suffix: str) -> pl.DataFrame:
    """Rename every attribute column with a side suffix, leaving the ID alone."""
    return df.rename(
        {c: f"{c}{suffix}" for c in df.columns if c != id_alias}
    )


# --------------------------------------------------------------------------
# pair features
# --------------------------------------------------------------------------

def _overlap(a: str, b: str) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """(intersection size, |a|, |b|) for two hashed, deduplicated list columns."""
    return (
        pl.col(a).list.set_intersection(pl.col(b)).list.len().cast(pl.UInt32),
        pl.col(a).list.len().cast(pl.UInt32),
        pl.col(b).list.len().cast(pl.UInt32),
    )


def _jaccard_and_containment(
    a: str, b: str, jac_name: str, cont_name: str
) -> list[pl.Expr]:
    """Both overlap ratios for one pair of set columns.

    Jaccard and containment disagree in exactly the case that matters for
    business names: `acme` against `acme foods private limited` has a low
    Jaccard because the second set is much larger, and a containment of 1.0
    because the first is wholly inside it. Truncated and abbreviated names are
    common enough in this data that the model should see both and decide.

    Union is `|a| + |b| - |a and b|` rather than a second `set_union` pass.
    That identity holds only for sets, which is why every list column reaching
    here is deduplicated at build time.
    """
    inter, la, lb = _overlap(a, b)
    union = la + lb - inter
    smaller = pl.min_horizontal(la, lb)
    return [
        pl.when(union == 0).then(0.0)
        .otherwise(inter / union).cast(pl.Float32).alias(jac_name),
        pl.when(smaller == 0).then(0.0)
        .otherwise(inter / smaller).cast(pl.Float32).alias(cont_name),
    ]


def _pair_features(df: pl.DataFrame) -> pl.DataFrame:
    """Everything computable from the two sides of a single pair.

    Deliberately excludes anything that depends on the entity's other
    candidates — those are added afterwards by `_relative_features`, which has
    a partitioning precondition this function does not.
    """
    return df.with_columns(
        # ---- provenance, unpacked from the bitmask ------------------------
        #
        # The mask is cast UP to the current dtype before any bit is tested.
        #
        # A candidate parquet written before the LSH bands existed carries a
        # UInt16 `key_mask`, which `--reuse-candidates` will happily read back
        # since it is explicitly not freshness-checked. Measured, rather than
        # assumed: `UInt16 & (1 << 20)` in polars raises nothing and returns
        # False. For a pre-band candidate file that answer is correct — such a
        # file really does contain no band hits — so this cast changes no
        # result today. It is here so the correctness stops being an accident
        # of two errors cancelling: the moment the mask dtype and the bit
        # positions drift apart again, silent False is the failure mode, and it
        # is indistinguishable from "the bands found nothing".
        *[
            (((pl.col("key_mask").cast(blocking.KEY_MASK_DTYPE))
              & blocking.KEY_BITS[k]) > 0)
            .cast(pl.UInt8).alias(f"hit_{k}")
            for k in blocking.KEY_COLUMNS
        ],
    ).with_columns(
        # The circular case, named. A pair found ONLY by the address key has no
        # name evidence behind it whatsoever, and the old scorer's address
        # Jaccard is guaranteed to be high for it by construction. The model
        # can learn this interaction from the individual flags, but it is the
        # single most important structure in the data and handing it over
        # directly costs one column.
        (
            (pl.col("hit_addr_sorted") > 0)
            & (
                pl.sum_horizontal(
                    [pl.col(f"hit_{k}") for k in blocking.KEY_COLUMNS
                     if k != "addr_sorted"]
                ) == 0
            )
        ).cast(pl.UInt8).alias("addr_only"),

        # How many LSH bands found this pair. This is not redundant with the
        # individual `hit_band_*` flags: the bands are exchangeable, so which
        # ones agreed carries no information and only the COUNT does — and a
        # tree would need sixteen splits to reconstruct a sum it can otherwise
        # read in one.
        #
        # It is also the most directly useful number the fuzzy stage produces.
        # P[a band agrees] is exactly the trigram Jaccard raised to `rows`, so
        # the count is an unbiased estimator of similarity, measured at
        # corr(n_bands, J) = 0.90. `name_tri_jac` computes that Jaccard
        # exactly, so the two overlap — but this one is defined for pairs the
        # trigram lists disagree about in ways Jaccard flattens, and it costs
        # one column.
        pl.sum_horizontal(
            [pl.col(f"hit_{k}") for k in blocking.BAND_KEYS]
        ).cast(pl.Float32).alias("n_band_hits"),

        # ---- name evidence ------------------------------------------------
        *_jaccard_and_containment("tri_e", "tri_c", "name_tri_jac", "name_tri_cont"),
        *_jaccard_and_containment(
            "name_tok_e", "name_tok_c", "name_tok_jac", "name_tok_cont"
        ),
        # All three name-equality flags are guarded on BOTH sides being
        # non-empty. Two blank names hash equal, so without this they would
        # each fire together and hand the model three fabricated pieces of
        # agreement. Latent on the training split in the same way the address
        # version was — and, like that one, one address-only record away from
        # firing on test, where nothing has been measured and France appears.
        (
            (pl.col("h_name_sorted_e") == pl.col("h_name_sorted_c"))
            & ~pl.col("name_blank_e") & ~pl.col("name_blank_c")
        ).cast(pl.UInt8).alias("name_exact"),
        (
            (pl.col("h_name_nospace_e") == pl.col("h_name_nospace_c"))
            & ~pl.col("name_blank_e") & ~pl.col("name_blank_c")
        ).cast(pl.UInt8).alias("name_nospace_exact"),
        (
            (pl.col("h_name_acronym_e") == pl.col("h_name_acronym_c"))
            & ~pl.col("acr_blank_e") & ~pl.col("acr_blank_c")
        ).cast(pl.UInt8).alias("name_acronym_exact"),
        (
            pl.min_horizontal("name_len_e", "name_len_c")
            # `clip(1)` and not a `when/otherwise`: two names that both
            # normalize to nothing give 0/0, and NaN reaches LightGBM as a
            # missing value it will happily split on — turning a degenerate row
            # into a category of its own.
            / pl.max_horizontal("name_len_e", "name_len_c").clip(1)
        ).cast(pl.Float32).alias("name_len_ratio"),
        pl.col("n_name_tok_e").cast(pl.Float32).alias("name_ntok_e"),
        pl.col("n_name_tok_c").cast(pl.Float32).alias("name_ntok_c"),
        (pl.col("n_name_tok_e").cast(pl.Int32) - pl.col("n_name_tok_c").cast(pl.Int32))
        .abs().cast(pl.Float32).alias("name_ntok_absdiff"),

        # ---- address evidence ---------------------------------------------
        *_jaccard_and_containment(
            "addr_tok_e", "addr_tok_c", "addr_tok_jac", "addr_tok_cont"
        ),
        (
            (pl.col("h_addr_sorted_e") == pl.col("h_addr_sorted_c"))
            # Two blank addresses hash equal and would read as an exact match.
            # This is the `_jaccard` empty-vs-empty bug — which scored a
            # Jaccard of 1.0 for two blank addresses and survived two sessions
            # because source 1 has no empty normalized addresses on TRAIN —
            # reappearing in a new column. Source 2 and 3 have 3.3% empties
            # today, and nothing about the test split has been measured.
            & ~pl.col("blank_e") & ~pl.col("blank_c")
        ).cast(pl.UInt8).alias("addr_exact"),
        pl.col("postal_e").list.set_intersection(pl.col("postal_c"))
        .list.len().cast(pl.Float32).alias("addr_postal_overlap"),
        pl.col("numbers_e").list.set_intersection(pl.col("numbers_c"))
        .list.len().cast(pl.Float32).alias("addr_number_overlap"),
        pl.col("n_addr_tok_e").cast(pl.Float32).alias("addr_ntok_e"),
        pl.col("n_addr_tok_c").cast(pl.Float32).alias("addr_ntok_c"),
        pl.col("blank_e").cast(pl.UInt8).alias("addr_blank_e"),
        pl.col("blank_c").cast(pl.UInt8).alias("addr_blank_c"),
    )


def _relative_features(df: pl.DataFrame) -> pl.DataFrame:
    """Each pair's standing within its own entity's candidate list.

    PRECONDITION: every candidate of an entity must be present in `df`. The
    caller partitions by `hash(source1_entity_id) % n_buckets`, which puts an
    entity's whole candidate list in exactly one partition, so this holds. If
    the partitioning ever changes to anything not keyed on the source-1 ID,
    every column here silently becomes a per-partition artefact rather than a
    per-entity fact. That is the same class of bug as scoring a truncated
    sample and reporting it as a full run.

    Why these matter: F_0.5 is macro-averaged per entity, so the decision is
    never "is this pair good in absolute terms" but "is this pair good for this
    entity, relative to its alternatives". An address Jaccard of 0.6 is weak
    evidence when three other candidates also score 0.6 and strong evidence
    when it is the only one above 0.2.
    """
    by = "source1_entity_id"

    def frac(col: str) -> pl.Expr:
        # Division by the entity's best. Guarded on the max rather than on the
        # value, because a whole entity can legitimately score zero on a
        # feature and 0/0 would produce NaN — which LightGBM treats as a
        # missing value and splits on, turning "this entity has no address
        # evidence at all" into a silent category.
        top = pl.col(col).max().over(by)
        return (
            pl.when(top <= 0).then(0.0)
            .otherwise(pl.col(col) / top).cast(pl.Float32)
        )

    def rank(col: str) -> pl.Expr:
        # "ordinal" and not "min": ties must produce distinct ranks, otherwise
        # the feature carries the tie structure instead of the ordering. The
        # arbitrariness within a tie is acceptable here — this is an input to a
        # model, not a selection cut — but it is the same mechanism that once
        # evicted true matches alphabetically, so it is worth being explicit.
        return (
            pl.col(col).rank("ordinal", descending=True).over(by)
            .cast(pl.Float32)
        )

    return df.with_columns(
        pl.len().over(by).cast(pl.Float32).alias("n_cand"),
        frac("block_score").alias("block_score_frac"),
        rank("block_score").alias("block_score_rank"),
        frac("name_tri_jac").alias("name_tri_frac"),
        rank("name_tri_jac").alias("name_tri_rank"),
        frac("addr_tok_jac").alias("addr_tok_frac"),
        rank("addr_tok_jac").alias("addr_tok_rank"),
    )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def build_features(
    split: str,
    work: Path,
    data: Path,
    merge_buckets: int = 16,
    cand_path: Path | None = None,
) -> Path:
    """Join attributes onto every candidate pair and write the feature table.

    Bucketed by `hash(source1_entity_id) % merge_buckets` for two independent
    reasons: it bounds peak memory the same way the blocking merge does, and it
    is the precondition `_relative_features` depends on. Bucket count does not
    change the output.

    On the training split a `label` column is attached from the ground truth.
    On test it is absent, and `model.py` refuses to train on a table without
    it rather than inventing zeros.
    """
    cand_path = cand_path or paths.candidates_parquet(split, work)
    if not cand_path.is_file():
        raise FileNotFoundError(
            f"{cand_path} missing — run `python -m src.blocking --split {split}` "
            "first."
        )

    started = time.perf_counter()
    s1 = _suffixed(_attributes(split, (1,), work, "source1_entity_id"), "source1_entity_id", "_e")
    tgt = _suffixed(
        _attributes(split, (2, 3), work, "candidate_entity_id"), "candidate_entity_id", "_c"
    )
    print(f"  attributes: {s1.height:,} source-1, {tgt.height:,} target")

    truth = None
    if split == "train":
        truth = (
            block_eval.ground_truth_pairs(paths.ground_truth_tsv(data))
            .rename({"matched_entity_id": "candidate_entity_id"})
            .with_columns(pl.lit(1, dtype=pl.UInt8).alias("label"))
            .collect()
        )
        print(f"  ground truth: {truth.height:,} true pairs")

    dst = paths.features_parquet(split, work)
    import pyarrow.parquet as pq

    writer = None
    total = 0
    positives = 0
    try:
        for b in range(merge_buckets):
            chunk = (
                pl.scan_parquet(cand_path)
                .filter((pl.col("source1_entity_id").hash() % merge_buckets) == b)
                .collect()
            )
            if chunk.is_empty():
                continue
            # Inner joins on both sides. A candidate whose attributes are
            # missing cannot be featurized at all, and silently carrying it
            # with null features would let LightGBM split on "this row is
            # broken". If the counts ever diverge, the normalized parquet and
            # the candidate file disagree and that is worth a loud failure
            # rather than a quiet one — checked after the loop.
            joined = chunk.join(s1, on="source1_entity_id", how="inner").join(
                tgt, on="candidate_entity_id", how="inner"
            )
            feats = _relative_features(_pair_features(joined))
            keep = list(ID_COLUMNS) + list(FEATURE_COLUMNS)
            if truth is not None:
                feats = feats.join(
                    truth, on=list(ID_COLUMNS), how="left"
                ).with_columns(pl.col("label").fill_null(0).cast(pl.UInt8))
                keep.append("label")
                positives += int(feats["label"].sum())
            out = feats.select(keep)
            table = out.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
            writer.write_table(table)
            total += out.height
            print(f"    bucket {b + 1}/{merge_buckets}   {total:,} pairs", end="\r")
    finally:
        if writer is not None:
            writer.close()
    print()

    n_cand = pl.scan_parquet(cand_path).select(pl.len()).collect().item()
    if total != n_cand:
        raise RuntimeError(
            f"featurized {total:,} pairs from {n_cand:,} candidates — "
            f"{n_cand - total:,} pairs reference an entity missing from the "
            "normalized parquet. The candidate file and the normalized parquet "
            "were built from different data; re-run --normalize-only and "
            "src.blocking together."
        )
    msg = f"  {total:,} pairs -> {dst.name}"
    if truth is not None:
        msg += (
            f"   {positives:,} positive ({positives / max(total, 1):.1%}), "
            f"{total - positives:,} negative"
        )
    print(f"{msg}   ({time.perf_counter() - started:.1f}s)")
    return dst


def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage C: per-candidate-pair features for the learned scorer."
    )
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--data-root", default=None)
    p.add_argument("--work-root", default=None)
    p.add_argument(
        "--merge-buckets", type=int, default=16,
        help="Hash partitions over the source-1 entity ID. Affects peak memory "
        "only, not the output. Must stay keyed on the source-1 ID — see "
        "_relative_features.",
    )
    a = p.parse_args()
    work = paths.work_root(a.work_root)
    data = paths.data_root(a.data_root)
    print(f"features [{a.split}]  work={work}")
    build_features(a.split, work, data, a.merge_buckets)


if __name__ == "__main__":
    main()
