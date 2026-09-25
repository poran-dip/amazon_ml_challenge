"""Step 2, stage B — fuzzy candidate generation with character n-grams.

Why this stage exists
---------------------
Stage A reached ~60% pair recall and the per-key diagnostics show the remainder
is not recoverable by any exact key: the failures are character-level by
construction. Typos (`fnesnius`/`fresenius`), concatenation (`dentcoffee`/`coffee
dent`), and transliteration residue (`kanstraksans`/`constructions`) all share
most of their character n-grams while sharing no token exactly.

Pipeline
--------
1. Character 3-gram TF-IDF over `name_core`, fitted on a sample.
2. Truncated SVD to `--dim` dense dimensions (default 128), also fitted on a
   sample, then applied in batches. Dense vectors are what an ANN index needs.
3. L2 normalization, so inner product is cosine similarity.
4. One FAISS IVFPQ index per country. Country agrees on 100% of true pairs, so
   partitioning costs no recall and shrinks each index.
5. Top-k search per Source-1 entity, filtered by `--min-sim`, merged with the
   stage A candidates and capped at the budget.

Memory
------
Embeddings are written to disk as memmaps and streamed into the index in chunks,
because holding 12.5M dense vectors in RAM would need double-digit gigabytes.
IVFPQ keeps only compressed codes resident (~16 bytes per vector), so a
6M-record country partition costs roughly 100MB rather than 3GB.

Usage:

    python -m src.fuzzy --split train
    python -m src.fuzzy --split train --dim 128 --topk 50 --min-sim 0.45
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl

from . import blocking, config

EMBED_TEXT = "name_core"


def embed_dir(work: Path) -> Path:
    d = work / "embed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def final_candidates_parquet(split: str, work: Path) -> Path:
    out = work / "candidates"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{split}_candidates_final.parquet"


def _fit_models(split: str, work: Path, dim: int, sample: int, seed: int):
    """Fit the TF-IDF vocabulary and SVD basis on a sample of all three sources."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts: list[str] = []
    per_source = max(sample // 3, 1)
    for s in config.SOURCES:
        col = (
            pl.scan_parquet(config.normalized_parquet(split, s, work))
            .select(EMBED_TEXT)
            .head(per_source)
            .collect()
            .to_series()
        )
        texts.extend(col.to_list())
    rng = np.random.default_rng(seed)
    rng.shuffle(texts)

    print(f"  fitting tf-idf on {len(texts):,} sampled names")
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 3), min_df=3, dtype=np.float32
    )
    X = vec.fit_transform(texts)
    print(f"    vocabulary: {len(vec.vocabulary_):,} 3-grams")

    # SVD cannot produce more components than the vocabulary has features. This
    # only bites on small corpora (tests, tiny country partitions), but a hard
    # crash there is worse than quietly using a smaller basis.
    usable = min(dim, max(X.shape[1] - 1, 1))
    if usable != dim:
        print(f"    vocabulary too small for {dim} dims — using {usable}")
    print(f"  fitting SVD -> {usable} dims")
    svd = TruncatedSVD(n_components=usable, random_state=seed)
    svd.fit(X)
    print(f"    explained variance: {svd.explained_variance_ratio_.sum():.1%}")
    return vec, svd


def _embed_source(
    split: str, source: int, work: Path, vec, svd, dim: int, batch: int
) -> tuple[Path, pl.DataFrame]:
    """Write L2-normalized embeddings to a memmap; return it with the id table."""
    path = config.normalized_parquet(split, source, work)
    meta = (
        pl.scan_parquet(path).select("entity_id", "country_norm").collect()
    )
    n = meta.height
    dst = embed_dir(work) / f"{split}_source{source}_d{dim}.f32"
    mm = np.lib.format.open_memmap(
        dst.with_suffix(".npy"), mode="w+", dtype=np.float32, shape=(n, dim)
    )
    lf = pl.scan_parquet(path).select(EMBED_TEXT)
    written = 0
    while written < n:
        chunk = lf.slice(written, batch).collect().to_series().to_list()
        if not chunk:
            break
        emb = svd.transform(vec.transform(chunk)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        np.divide(emb, np.maximum(norms, 1e-8), out=emb)
        mm[written : written + len(chunk)] = emb
        written += len(chunk)
    mm.flush()
    del mm
    print(f"    source{source}: {written:,} vectors -> {dst.with_suffix('.npy').name}")
    return dst.with_suffix(".npy"), meta


def _search_country(
    country: str,
    q_vecs: np.ndarray,
    q_ids: list[str],
    make_t_chunks,
    n_t: int,
    t_ids: list[str],
    dim: int,
    topk: int,
    min_sim: float,
    nprobe: int,
    add_batch: int,
) -> pl.DataFrame:
    """Search one country partition.

    Targets arrive as a *callable returning a fresh generator* of chunks rather
    than one array: materializing a 6M-record partition at 128 dims costs 3GB of
    RAM, which on a laptop means swapping, which is why this stage appeared to
    hang rather than run slowly.
    """
    import faiss

    empty = pl.DataFrame(
        schema={"source1_entity_id": pl.Utf8,
                "candidate_entity_id": pl.Utf8, "sim": pl.Float32}
    )
    n_q = q_vecs.shape[0]
    if n_t == 0 or n_q == 0:
        return empty

    if n_t < 10_000:
        index = faiss.IndexFlatIP(dim)  # too small for IVF to pay off
    else:
        # nlist drives IVF k-means training, which is O(nlist * train_n * dim)
        # per iteration. The textbook 4*sqrt(N) gives ~9800 lists at 6M records
        # and training then dominates total runtime; sqrt(N) with more probing
        # reaches the same recall for a fraction of the setup cost.
        nlist = max(min(int(np.sqrt(n_t)), n_t // 39), 1)
        m = max(dim // 8, 1)
        while dim % m:
            m -= 1
        train_n = min(n_t, max(nlist * 30, 10_000), 400_000)

        train_parts, have = [], 0
        for chunk in make_t_chunks():
            train_parts.append(chunk)
            have += chunk.shape[0]
            if have >= train_n:
                break
        train = np.vstack(train_parts)[:train_n]
        del train_parts
        print(f"    {country}: training IVFPQ nlist={nlist} m={m} on {train.shape[0]:,}")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(train)
        index.nprobe = min(nprobe, nlist)
        del train

    added = 0
    for chunk in make_t_chunks():
        index.add(chunk)
        added += chunk.shape[0]
        print(f"      indexed {added:,}/{n_t:,}", end="\r")

    sims, idx = [], []
    for s0 in range(0, n_q, add_batch):
        d, i = index.search(np.ascontiguousarray(q_vecs[s0 : s0 + add_batch]), topk)
        sims.append(d)
        idx.append(i)
        print(f"      searched {min(s0 + add_batch, n_q):,}/{n_q:,}", end="\r")
    sims = np.vstack(sims)
    idx = np.vstack(idx)

    rows = np.repeat(np.arange(n_q), topk)
    flat_i, flat_s = idx.ravel(), sims.ravel()
    keep = (flat_i >= 0) & (flat_s >= min_sim)
    if not keep.any():
        return empty
    q_arr = np.asarray(q_ids, dtype=object)
    t_arr = np.asarray(t_ids, dtype=object)
    out = pl.DataFrame({
        "source1_entity_id": pl.Series(q_arr[rows[keep]], dtype=pl.Utf8),
        "candidate_entity_id": pl.Series(t_arr[flat_i[keep]], dtype=pl.Utf8),
        "sim": pl.Series(flat_s[keep], dtype=pl.Float32),
    })
    print(f"    {country:<10} {n_q:>9,} queries x {n_t:>9,} targets "
          f"-> {out.height:,} pairs above {min_sim}")
    return out


def _entities_needing_topup(split: str, work: Path, budget: int) -> set[str] | None:
    """Source-1 ids that stage A left short of budget, or None if stage A is absent.

    Entities already at budget cannot gain anything from stage B — whatever it
    finds would be evicted by the cap — so querying them is pure waste. On the
    training set this skipped roughly half the queries.
    """
    path = blocking.candidates_parquet(split, work)
    if not path.is_file():
        return None
    counts = (
        pl.scan_parquet(path)
        .group_by("source1_entity_id")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") < budget)
        .select("source1_entity_id")
        .collect()
    )
    at_budget = set(counts["source1_entity_id"].to_list())
    all_s1 = set(
        pl.scan_parquet(config.normalized_parquet(split, 1, work))
        .select("entity_id").collect()["entity_id"].to_list()
    )
    covered = set(
        pl.scan_parquet(path).select("source1_entity_id").collect()
        ["source1_entity_id"].to_list()
    )
    # Entities stage A produced nothing at all for also need stage B.
    return at_budget | (all_s1 - covered)


def run(
    split: str, work: Path, dim: int, sample: int, batch: int, topk: int,
    min_sim: float, nprobe: int, budget: int, fuzzy_weight: float, seed: int,
    only_missing: bool,
) -> None:
    started = time.perf_counter()
    vec, svd = _fit_models(split, work, dim, sample, seed)
    dim = svd.n_components  # may have been clamped to the vocabulary size

    print("  embedding sources:")
    paths, metas = {}, {}
    for s in config.SOURCES:
        paths[s], metas[s] = _embed_source(split, s, work, vec, svd, dim, batch)

    needed = _entities_needing_topup(split, work, budget) if only_missing else None
    if needed is not None:
        print(f"  querying only entities stage A left short: {len(needed):,} "
              f"of {metas[1].height:,}")

    q_all = np.load(paths[1], mmap_mode="r")
    t_arrays = {s: np.load(paths[s], mmap_mode="r") for s in (2, 3)}

    print("  searching per country:")
    results = []
    for country in metas[1]["country_norm"].unique().to_list():
        q_sel = metas[1].with_row_index("_row").filter(
            pl.col("country_norm") == country
        )
        if needed is not None:
            q_sel = q_sel.filter(pl.col("entity_id").is_in(list(needed)))
        if q_sel.height == 0:
            continue
        q_rows = q_sel["_row"].to_numpy()
        q_vecs = np.ascontiguousarray(q_all[q_rows])
        q_ids = q_sel["entity_id"].to_list()

        t_index, t_ids, n_t = [], [], 0
        for s in (2, 3):
            sel = metas[s].with_row_index("_row").filter(
                pl.col("country_norm") == country
            )
            if sel.height == 0:
                continue
            t_index.append((s, sel["_row"].to_numpy()))
            t_ids.extend(sel["entity_id"].to_list())
            n_t += sel.height
        if n_t == 0:
            continue

        def make_chunks(_ti=t_index):
            for s, rows in _ti:
                arr = t_arrays[s]
                for start in range(0, len(rows), batch):
                    yield np.ascontiguousarray(arr[rows[start : start + batch]])

        results.append(
            _search_country(country, q_vecs, q_ids, make_chunks, n_t, t_ids,
                            dim, topk, min_sim, nprobe, batch)
        )
        del q_vecs

    fuzzy = (
        pl.concat(results, how="vertical") if results
        else pl.DataFrame(schema={"source1_entity_id": pl.Utf8,
                                  "candidate_entity_id": pl.Utf8, "sim": pl.Float32})
    )

    # Stage A scores are key-weight sums (up to ~3.6); fuzzy similarities are
    # 0-1, so they are scaled to sit below a solid multi-key hit but above a
    # lone acronym collision.
    stage_a_path = blocking.candidates_parquet(split, work)
    frames = [
        fuzzy.select(
            "source1_entity_id", "candidate_entity_id",
            (pl.col("sim") * fuzzy_weight).alias("block_score"),
            pl.lit(0, dtype=pl.UInt8).alias("n_keys"),
        )
    ]
    if stage_a_path.is_file():
        frames.append(
            pl.read_parquet(stage_a_path).select(
                "source1_entity_id", "candidate_entity_id", "block_score", "n_keys"
            )
        )
    else:
        print(f"  note: {stage_a_path.name} not found — writing fuzzy candidates only.")

    merged = (
        pl.concat(frames, how="vertical")
        .group_by("source1_entity_id", "candidate_entity_id")
        .agg(
            pl.col("block_score").sum().alias("block_score"),
            pl.col("n_keys").max().alias("n_keys"),
        )
        .sort(["block_score", "n_keys", "candidate_entity_id"],
              descending=[True, True, False])
        .filter(pl.int_range(pl.len()).over("source1_entity_id") < budget)
    )
    dst = final_candidates_parquet(split, work)
    merged.write_parquet(dst, compression="zstd")
    print(f"  wrote {merged.height:,} candidate pairs -> {dst.name} "
          f"({time.perf_counter() - started:.1f}s total)")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage B blocking: fuzzy n-gram search.")
    p.add_argument("--split", choices=config.SPLITS, required=True)
    p.add_argument("--work-root", default=None)
    p.add_argument("--dim", type=int, default=128, help="SVD dimensions.")
    p.add_argument("--sample", type=int, default=600_000,
                   help="Names sampled to fit tf-idf and SVD.")
    p.add_argument("--batch", type=int, default=200_000, help="Rows per embed/search batch.")
    p.add_argument("--topk", type=int, default=50, help="Neighbours retrieved per entity.")
    p.add_argument("--min-sim", type=float, default=0.45,
                   help="Cosine floor for keeping a fuzzy candidate.")
    p.add_argument("--nprobe", type=int, default=16,
                   help="IVF lists probed. Higher = better recall, slower.")
    p.add_argument("--budget", type=int, default=30, help="Final candidates per entity.")
    p.add_argument("--fuzzy-weight", type=float, default=0.85,
                   help="Scale applied to cosine before merging with stage A scores.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all-entities", action="store_true",
                   help="Query every Source-1 entity, not only those stage A left "
                        "short of budget. Slower; entities already at budget "
                        "cannot gain from stage B anyway.")
    a = p.parse_args()

    work = config.work_root(a.work_root)
    print(f"fuzzy [{a.split}]  work={work}  dim={a.dim}  topk={a.topk}  "
          f"min_sim={a.min_sim}  nprobe={a.nprobe}")
    run(a.split, work, a.dim, a.sample, a.batch, a.topk, a.min_sim,
        a.nprobe, a.budget, a.fuzzy_weight, a.seed, not a.all_entities)


if __name__ == "__main__":
    main()
