"""Run bounded experiments for the standalone blocking strategies.

Run from the project directory with ``python -m src.test.test_blocking`` or
``python -m src.test_blocking``. These experiments use raw name columns because
the production ``prepare()`` function reduces records to ``entity_id``,
``block_key`` and ``addr``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Support execution both from root and as submodule
try:
    from .. import main, paths
except (ImportError, ValueError):
    try:
        from . import main, paths
    except (ImportError, ValueError):
        from src import main, paths


def _load_raw_source(
    split: str, source: int, data_root: Path, limit: int
) -> pl.DataFrame:
    """Read a bounded raw sample with the same schema as the pipeline."""
    frame = main.scan_source(split, source, data_root, limit).collect()
    return frame.select("entity_id", "business_name", "country")


def inspect_candidates(name: str, candidates: pl.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Number of candidate pairs: {candidates.height:,}")
    print("Columns:")
    print(candidates.columns)
    print("First 10 candidates:")
    print(candidates.head(10))


def load_ground_truth(
    data_root: Path,
    source1_ids: pl.DataFrame,
    target_ids: pl.DataFrame,
) -> pl.DataFrame:
    """
    Load training ground truth and keep only true matches that are
    present in the sampled Source-1 and target records.

    Returns:
        source1_entity_id
        candidate_entity_id
    """

    truth_path = (
        data_root
        / "dataset"
        / "train"
        / "train_ground_truth.tsv"
    )

    truth = pl.read_csv(
        truth_path,
        separator="\t",
        quote_char=None,
        schema={
            "source1_entity_id": pl.Utf8,
            "matched_entity_ids": pl.Utf8,
        },
    )

    # Keep only Source-1 entities included in our experiment.
    truth = truth.join(
        source1_ids.select("entity_id"),
        left_on="source1_entity_id",
        right_on="entity_id",
        how="inner",
    )

    # Convert the comma-separated ground-truth IDs into
    # one row per true match.
    truth = (
        truth
        .with_columns(
            pl.col("matched_entity_ids")
            .fill_null("")
            .str.split(",")
            .alias("candidate_entity_id")
        )
        .explode("candidate_entity_id")
        .with_columns(
            pl.col("candidate_entity_id").str.strip_chars()
        )
        .filter(
            pl.col("candidate_entity_id") != ""
        )
    )

    # Keep only true matches whose target record is actually present
    # in the sampled target set.
    truth = truth.join(
        target_ids.select("candidate_entity_id"),
        on="candidate_entity_id",
        how="inner",
    )

    return truth.select(
        "source1_entity_id",
        "candidate_entity_id",
    ).unique()


def evaluate_candidate_recall(
    name: str,
    candidates: pl.DataFrame,
    truth: pl.DataFrame,
    num_source1: int = 1,
    num_targets: int = 1,
) -> dict:
    """
    Measure how many sampled ground-truth matches survived
    the blocking stage, along with candidate reduction metrics.
    """

    total_true = truth.height

    if total_true == 0:
        recall = 0.0
        retained = 0
    else:
        retained = (
            candidates
            .join(
                truth,
                on=[
                    "source1_entity_id",
                    "candidate_entity_id",
                ],
                how="inner",
            )
            .unique()
            .height
        )

        recall = retained / total_true

    candidate_count = candidates.height
    total_comparisons = max(1, num_source1 * num_targets)
    reduction_ratio = 1.0 - (candidate_count / total_comparisons)
    avg_candidates_per_source = candidate_count / max(1, num_source1)

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)
    print(f"Candidate pairs           : {candidate_count:,}")
    print(f"Avg candidates per source : {avg_candidates_per_source:.1f}")
    print(f"Candidate reduction ratio : {reduction_ratio:.4%}")
    print(f"True matches in sample    : {total_true:,}")
    print(f"True matches retained     : {retained:,}")
    print(f"Candidate recall          : {recall:.4%}")

    return {
        "method": name,
        "candidates": candidate_count,
        "avg_per_source": avg_candidates_per_source,
        "reduction": reduction_ratio,
        "true_matches": total_true,
        "retained": retained,
        "recall": recall,
    }


def batched_sparse_tfidf_blocking(
    source1: pl.DataFrame,
    targets: pl.DataFrame,
    top_k: int = 20,
    batch_size: int = 500,
) -> pl.DataFrame:
    """
    Memory-efficient batched sparse TF-IDF character n-gram cosine retrieval.
    Evaluates across the full sampled universe without dense matrix blowup.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    s1_names = source1["business_name"].fill_null("").cast(pl.Utf8).to_list()
    tgt_names = targets["business_name"].fill_null("").cast(pl.Utf8).to_list()
    s1_ids = source1["entity_id"].cast(pl.Utf8).to_list()
    tgt_ids = targets["candidate_entity_id"].cast(pl.Utf8).to_list()

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        lowercase=True,
        min_df=2,
    )

    all_vecs = vectorizer.fit_transform(s1_names + tgt_names)
    s1_mat = all_vecs[: len(s1_names)]
    tgt_mat = all_vecs[len(s1_names) :]

    candidate_rows = []
    num_s1 = len(s1_names)

    for b in range(0, num_s1, batch_size):
        b_end = min(b + batch_size, num_s1)
        s1_sub = s1_mat[b:b_end]
        # Batched sparse dot-product -> (batch_size, num_targets)
        sims = (s1_sub * tgt_mat.T).toarray()

        for local_idx, global_idx in enumerate(range(b, b_end)):
            row = sims[local_idx]
            if len(row) > top_k:
                top_indices = np.argpartition(row, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(-row[top_indices])]
            else:
                top_indices = np.argsort(-row)

            for idx in top_indices:
                score = float(row[idx])
                if score > 0.0:
                    candidate_rows.append(
                        (s1_ids[global_idx], tgt_ids[idx], score)
                    )

    return pl.DataFrame(
        candidate_rows,
        schema=[
            "source1_entity_id",
            "candidate_entity_id",
            "tfidf_similarity",
        ],
        orient="row",
    )


def run_blocking_experiments(
    split: str = "train",
    data_root: str | None = None,
    source_sample: int = 5_000,
    target_sample_per_source: int = 40_000,
    ngram_n: int = 5,
    ngram_max_bucket: int = 200,
    prefix_length: int = 3,
    prefix_max_bucket: int = 500,
    tfidf_top_k: int = 20,
) -> None:
    """Run all experimental blockers on the exact same full sample universe."""
    root = paths.data_root(data_root)
    print(f"Loading {split} samples from {root}")

    source1 = _load_raw_source(split, 1, root, source_sample)
    target_frames = [
        _load_raw_source(split, source, root, target_sample_per_source)
        for source in (2, 3)
    ]
    targets = pl.concat(target_frames, how="vertical").rename(
        {"entity_id": "candidate_entity_id"}
    )

    num_s1 = source1.height
    num_tgt = targets.height

    print(f"Source 1 records: {num_s1:,}")
    print(f"Target records: {num_tgt:,}")
    print(f"Total possible unblocked pairs: {num_s1 * num_tgt:,}")

    # ----------------------------------------------------------------------
    # Ground truth for the sampled experiment
    # ----------------------------------------------------------------------

    source1_ids = source1.select("entity_id")
    target_ids = targets.select("candidate_entity_id")

    truth = load_ground_truth(
        root,
        source1_ids,
        target_ids,
    )

    print(f"True matches in sampled universe: {truth.height:,}")

    from src.main import character_ngram_blocking, prefix_blocking

    # 1. Character n-gram blocking
    candidates_ngram = character_ngram_blocking(
        source1, targets, n=ngram_n, max_bucket=ngram_max_bucket
    )
    inspect_candidates(f"Character n-gram blocking (n={ngram_n}, cap={ngram_max_bucket})", candidates_ngram)

    ngram_result = evaluate_candidate_recall(
        f"Char {ngram_n}-gram (cap={ngram_max_bucket})",
        candidates_ngram,
        truth,
        num_source1=num_s1,
        num_targets=num_tgt,
    )

    # 2. Prefix blocking
    candidates_prefix = prefix_blocking(
        source1, targets, prefix_length=prefix_length, max_bucket=prefix_max_bucket
    )
    inspect_candidates(f"Prefix blocking (len={prefix_length}, cap={prefix_max_bucket})", candidates_prefix)

    prefix_result = evaluate_candidate_recall(
        f"Prefix len={prefix_length} (cap={prefix_max_bucket})",
        candidates_prefix,
        truth,
        num_source1=num_s1,
        num_targets=num_tgt,
    )

    # 3. Full-Universe Sparse TF-IDF retrieval
    print(f"\nRunning Batched Sparse TF-IDF (top_k={tfidf_top_k}) across full universe...")
    try:
        candidates_tfidf = batched_sparse_tfidf_blocking(
            source1,
            targets,
            top_k=tfidf_top_k,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "sklearn":
            raise
        candidates_tfidf = None
        tfidf_result = None
        print("Skipped TF-IDF: install scikit-learn to run TF-IDF.")
    else:
        inspect_candidates(
            f"Sparse TF-IDF (top_k={tfidf_top_k})",
            candidates_tfidf,
        )

        tfidf_result = evaluate_candidate_recall(
            f"TF-IDF (top_k={tfidf_top_k})",
            candidates_tfidf,
            truth,
            num_source1=num_s1,
            num_targets=num_tgt,
        )

    # 4. Combined Union blocking (n-gram + prefix + TF-IDF)
    cand_list = [
        candidates_ngram.select("source1_entity_id", "candidate_entity_id"),
        candidates_prefix.select("source1_entity_id", "candidate_entity_id"),
    ]
    if candidates_tfidf is not None:
        cand_list.append(candidates_tfidf.select("source1_entity_id", "candidate_entity_id"))

    candidates_combined = (
        pl.concat(cand_list)
        .unique(subset=["source1_entity_id", "candidate_entity_id"])
    )
    inspect_candidates("Combined Union (n-gram + Prefix + TF-IDF)", candidates_combined)

    combined_result = evaluate_candidate_recall(
        "Combined (All Strategies)",
        candidates_combined,
        truth,
        num_source1=num_s1,
        num_targets=num_tgt,
    )

    print("\n" + "=" * 80)
    print("BLOCKING EXPERIMENT SUMMARY (EVALUATED ON THE EXACT SAME UNIVERSE)")
    print("=" * 80)

    print(
        f"{'Method':<28}"
        f"{'Candidates':>13}"
        f"{'Avg/Src':>10}"
        f"{'Reduction':>11}"
        f"{'True':>8}"
        f"{'Retained':>10}"
        f"{'Recall':>10}"
    )

    print("-" * 80)

    for res in [ngram_result, prefix_result, tfidf_result, combined_result]:
        if res is not None:
            print(
                f"{res['method']:<28}"
                f"{res['candidates']:>13,}"
                f"{res['avg_per_source']:>10.1f}"
                f"{res['reduction']:>10.2%}"
                f"{res['true_matches']:>8,}"
                f"{res['retained']:>10,}"
                f"{res['recall']:>9.2%}"
            )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=paths.SPLITS, default="train")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Folder containing dataset/ (defaults to AMLC_DATA or the repo default)",
    )
    parser.add_argument("--sample", type=_positive_int, default=5_000,
                        help="Source-1 rows to test (default: 5000)")
    parser.add_argument("--target-sample", type=_positive_int, default=40_000,
                        help="Rows per target source to test (default: 40000)")
    parser.add_argument("--ngram-n", type=_positive_int, default=5,
                        help="Character n-gram length (default: 5)")
    parser.add_argument("--ngram-max-bucket", type=_positive_int, default=200,
                        help="Max bucket cap for n-grams (default: 200)")
    parser.add_argument("--prefix-len", type=_positive_int, default=3,
                        help="Prefix length (default: 3)")
    parser.add_argument("--prefix-max-bucket", type=_positive_int, default=500,
                        help="Max bucket cap for prefix blocking (default: 500)")
    parser.add_argument("--tfidf-top-k", type=_positive_int, default=20,
                        help="Top-k retrieval count for TF-IDF (default: 20)")
    args = parser.parse_args()

    run_blocking_experiments(
        split=args.split,
        data_root=args.data_root,
        source_sample=args.sample,
        target_sample_per_source=args.target_sample,
        ngram_n=args.ngram_n,
        ngram_max_bucket=args.ngram_max_bucket,
        prefix_length=args.prefix_len,
        prefix_max_bucket=args.prefix_max_bucket,
        tfidf_top_k=args.tfidf_top_k,
    )


if __name__ == "__main__":
    main_cli()