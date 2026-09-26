# Business Entity Resolution

`src/main.py` is the pipeline and the single entry point for every run: it reads the data, normalizes, blocks, scores, predicts, writes both submission files, and scores itself against the training ground truth. Every submission comes from here, and every improvement lands here.

## Setup

```bash
pip install -r requirements.txt
```

The data is not in this repo (`student_resource/` is gitignored, it is >1 GB). Unzip it at the repo root so `student_resource/dataset/train/train_source1.tsv` exists, or pass `--data-root`.

## Run

```bash
cd code/business_entity_resolution

python -m src.main --split train --sample 200000   # ~1 min smoke test
python -m src.main --split train                   # full run, prints macro F_0.5 + diagnostics
python -m src.main --split test                    # writes the submission
```

To iterate on blocking without paying for normalization every time, split the run. The normalized parquet is cached and reused until the source data, the rules in `text_norm.py`/`blocking.py`, or the row count change:

```bash
python -m src.main --split train --normalize-only   # once, ~3 min
python -m src.blocking   --split train --budget 20  # seconds
python -m src.block_eval --split train              # what that blocking bought
```

Outputs go to `output/matching_results.tsv` and `output/candidate_pairs.tsv` at the repo root, or wherever `--out-dir` points. **Both splits write to the same directory**, so run `--split test` last (or give it its own `--out-dir`) — a training run afterwards overwrites the submission with training predictions.

Validate before submitting, from the `student_resource/` directory:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## How it works

Five stages, each one function with an explicit input and output.

**Normalize** (`prepare`, rules in `text_norm.py`) folds accents and IAST diacritics, glues runs of single letters into one token so `S.B.I. ATM` does not become `south b i atm`, expands legal and street abbreviations by whole-token lookup, and strips pure legal-form markers to produce `name_core`. Rule ordering is load-bearing: initialism gluing must precede abbreviation expansion. Indic romanization (`translit.py`) is available behind `--translit` but off by default; see *Tuned settings*.

Accent handling deliberately strips only the Latin combining-diacritics block, never `\p{M}`. In Indic scripts the vowel signs and virama are themselves Mark-category characters, so a blanket strip deletes the vowels.

**Block** (`blocking.py`) takes the union of five exact keys — `name_sorted`, `name_core`, `name_nospace`, `addr_sorted` and `name_acronym`, each paired with country. No single key is enough: the best one agrees on about 53% of true pairs. But they are nearly free (a hash join, not a similarity search) and they disagree with each other, so the union reaches much further than the best of them. `--blocking single` reproduces the previous one-key configuration.

Each key carries a weight and its own bucket cap, and a candidate's score is the sum of the weights of the keys that found it, **divided by `log2(1 + bucket_size)`**. That inverse-frequency term is what makes the score discriminative: sharing a name with one other record is strong evidence, sharing it with 2000 is nearly none. Without it, tied candidates are evicted by the budget cap in effectively alphabetical order — worth 2.4 points of macro recall at an identical candidate count.

Country is folded into every key rather than compared separately: it agrees on 100% of true training pairs, so it is a free hard partition. It is treated as an opaque label and never enumerated, because the test set contains France and the training set does not.

**Fuzzy retrieval** (`minhash.py`) is what reaches the true pairs no exact key can. 23.74% of true pairs are unreachable by any exact key at any cap or budget, and the failures are character-level — typos (`fnesnius`/`fresenius`), concatenation (`dentcoffee`/`coffee dent`), transliteration residue (`kanstraksans`/`constructions`), web debris (`com jgsreal`/`jgs real`). MinHash reduces each entity's character-3-gram set to 48 integers such that two entities' MinHashes agree with probability exactly equal to their Jaccard similarity; grouping those into 16 bands of 3 and requiring a whole band to agree turns a *similarity* question into an *equality* question. So each band signature is an ordinary exact blocking key and the stage above handles it unchanged — same key-outer join, same bucket cap, same inverse-frequency weighting, same budget. No new dependency.

Two design choices worth knowing. An inverted index over the 3-grams was rejected on measurement, not taste: the vocabulary is 15,285 distinct grams over ~258M target postings, so mean document frequency is ~17,000 and there are no rare grams to invert on. And `P_MOD` is `2**31-1` rather than a larger prime because **polars wraps UInt64 silently** — at `2**61-1` a permutation overflows and yields a hash family that is neither the intended one nor provably anything, with no exception raised.

A property that falls out free: a genuinely similar pair agrees on many bands, and `block_score` sums the weight of every key that found a pair, so it becomes a monotone estimator of trigram similarity rather than a flat count (measured `corr(n_bands, Jaccard) = 0.90`). That is exactly what the per-entity budget needs to rank on.

`--bands N` blocks on the first N bands; `--bands 0` turns fuzzy retrieval off and reproduces the exact-key candidate set bit-for-bit, which is how a regression is attributed to this stage rather than bisected. The bands are **nested** — band *k* is the same permutation group at every setting — so a curve over `--bands` reads off one normalization pass instead of one per point.

**Score** (`_jaccard`) is the token-set overlap of `addr_core`. **Predict** (`generate`) keeps pairs at or above `--threshold`, best `--topk` per entity. **Evaluate** (`macro_f05`) reports macro F_0.5 plus macro precision, macro recall, the share of entities given an empty prediction, how many of those have real matches, and mean predicted versus true count.

## Knobs

`--threshold` is the address Jaccard floor. `--topk` caps predictions per entity. `--budget` caps *candidates* per entity before scoring. `--max-bucket` overrides every key's own cap with one global value — normally leave it unset, since the name keys need a loose cap and `name_acronym` a tight one. `--keys` restricts blocking to a subset, for attributing a change to one key. `--chunk` sets how many source-1 entities are processed per pass; `--merge-buckets` sets the hash partitioning of the merge and scoring passes, which affects peak memory only and not the output.

**`--budget` and `--topk` are not the same kind of number, and sizing the budget from true-match counts is a mistake.** The maximum true match count is 11 and the 99th percentile is 8, which bounds `--topk`. The budget has to be large enough that true matches are not crowded out by *false* candidates that outrank them, which is a question about ranking quality, not about match counts. `src.block_eval` reports macro recall at every cut of a single candidate file, which answers it in one run instead of a sweep.

`--sample N` matches only the first N Source-1 entities, against the **full** target set, so the reported F_0.5 stays representative — on a quarter of the entities it lands within a few thousandths of the full-run score. `--sample-targets N` additionally truncates Sources 2 and 3 to save memory, and prints a warning, because a Source-1 entity's true partners are scattered anywhere across 10M target rows: keeping only the first N puts most of them out of reach and the score then measures the truncation rather than the pipeline, in either direction.

`--legacy-norm` reproduces the superseded normalization bit-for-bit. It is a measurement reference for attributing a regression to the normalization stage, not a supported configuration.

## Tuned settings

Defaults are `--threshold 0.15 --topk 8 --max-bucket 100`. Macro F_0.5 on the full training split, against a do-nothing floor of 0.0558:

| normalization | macro F_0.5 | candidates | runtime |
| --- | --- | --- | --- |
| superseded (`--legacy-norm`) | 0.4670 | 14.0M | 27s |
| current | **0.6070** | 27.1M | 176s |
| current + `--translit` | 0.6075 | 27.4M | 302s |

Romanization is off by default: +0.0005 for +126s. Exact-key blocking requires *every* token to agree, and a partly-romanized name still differs on its Latin tokens. Its measured value is character-trigram similarity on cross-script pairs (0.000 → 0.326), which only pays once a fuzzy stage exists — so the code stays.

Multi-key blocking is new and **unmeasured on the real data** — every score in the table above is single-key. Run `src.block_eval` before trusting any weight or cap in `KEY_SPECS`; they were all derived under the superseded normalization, against key definitions that have since changed.

One thing to expect. Until now every candidate was an exact name match, so address Jaccard was *independent* confirming evidence and a low threshold was safe. That is no longer true: candidates reached through `addr_sorted` agree on their address tokens by construction, so scoring them by address overlap is close to circular and they will pass almost any threshold. Precision should be expected to fall and the right threshold to move. `block_score` and `n_keys` are carried through the candidate file so a scorer can use the name evidence too, but that is a separate change and is deliberately not bundled with this one.

### What is stale

`--threshold 0.15` and `--max-bucket 100` were both tuned against the superseded normalization and are now mis-tuned by an unmeasured amount. `addr_core` expands abbreviations and drops stopwords, so the Jaccard distribution has shifted. The threshold sweep below is retained for reference but **no longer describes the current pipeline**:

| threshold | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro F_0.5 (superseded norm) | 0.3860 | 0.4281 | 0.4496 | 0.4670 | 0.4664 | 0.4455 | 0.4127 | 0.3672 | 0.3056 | 0.2355 |

The precision and recall figures printed after each training run indicate which direction the threshold should move, which a single F_0.5 value cannot.

## Scale

Roughly 12.5M rows per split. Sources are read once and normalized in slices; the resident structure is the source-2/3 key table, about 600MB–1GB at full scale, which fits in 16GB alongside one slice.

Five resident key tables would be 3–5GB, so blocking loops key-outer instead: build one target table, join all of source 1 against it, spill the hits to parquet, free it, move on. Peak memory stays at about one key table, which is what the single-key pipeline already used. The merge is then the step that would otherwise die at the very end of the longest job — concatenating every partial and taking a global sort over ~100M rows wants 8–12GB — so it hashes the entity ID into `--merge-buckets` partitions and processes them one at a time. The budget is per entity and entities are independent, so this is exact rather than approximate; bucket count is verified not to change the output.

The reader avoids two tempting APIs on purpose. `scan_csv(...).slice(offset, chunk).collect()` in a loop re-reads the file from the start every time — 42 passes over a 10M-row TSV. `read_csv_batched` is deprecated in current polars, and its replacement `LazyFrame.collect_batches` does not exist in 1.18, is documented as unstable, and warns that it is slower than the alternatives. Reading once and slicing the DataFrame is zero-copy and version-agnostic; outputs are verified identical on polars 1.18.0 and 1.44.2.

## Working on it

One change at a time, measured against the run it replaces, with the previous behaviour reachable behind a flag rather than deleted. That is what makes a regression attributable instead of something to bisect.

The stage boundaries are where the next changes go. A fuzzy candidate stage appends to the candidate set before scoring, targeting the true pairs that no exact key reaches at any cap — and `--translit` becomes worth turning on at that moment and not before. A learned scorer replaces `_jaccard` and the threshold comparison with a model over several pair features, including `block_score` and `n_keys`; nothing has been trained yet, so no precision figure from a model exists.

`claude/findings.md` in the project holds the measured numbers behind all of the above.

The fuzzy stage is now MinHash/LSH in `minhash.py` (see **Block** above), not the FAISS n-gram stage that was on `feat/full-pipeline`. That approach was dropped on measurement rather than preference: its truncated-SVD basis captured 34.3% of variance, so retrieval ran against a lossy projection of the similarity it was measuring, and it needed two dependencies with no GPU wheels on Windows. LSH banding needs neither, and computes exact 3-gram agreement rather than an approximation of it.
